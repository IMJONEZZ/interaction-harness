"""Check every model role on its own, so a bad setting shows up before you start talking.

    foreground  streams, calls a tool when asked, and answers the end-of-turn prompt in one word
    background  streams with thinking on and calls a tool
    TTS -> ASR  speech made by the TTS comes back from the ASR as the same words
    VAD         hears that speech and not silence
"""
from __future__ import annotations

import re
import time

import numpy as np

from .config import Settings
from .llm import LLM, resolve_models
from .speech import MIC_SR, VAD_FRAME, build_speech
from .speech.base import resample
from .speech.silero import SileroVAD
from .tools import schemas

PHRASE = "The quick brown fox jumps over the lazy dog, and the interaction model decides who speaks next."

OK, WARN, FAIL = "ok", "warn", "FAIL"
_MARK = {OK: "✓", WARN: "!", FAIL: "✗"}


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, check: str, status: str, detail: str, ms: float | None = None):
        self.rows.append((check, status, detail, "" if ms is None else f"{ms:,.0f} ms"))
        print(f"  {_MARK[status]} {check:<26} {self.rows[-1][3]:>10}  {detail}", flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r[1] == FAIL)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def word_overlap(said: str, heard: str) -> float:
    a, b = _words(said), set(_words(heard))
    return sum(w in b for w in a) / max(1, len(a))


async def _chat_check(llm: LLM, report: Report, role: str, thinking: bool):
    # the same system prompt and tools a real session sends, so the check measures what you will get
    background = role == "background"
    prompt = llm.s.background_prompt if background else llm.s.foreground_prompt
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": "What time is it right now?"}]
    first = done = None
    calls, reasoning, text = [], 0, []
    t0 = time.perf_counter()
    try:
        async for ev in llm.stream_chat(msgs, tools=schemas(role), thinking=thinking, max_tokens=2000,
                                        background=background):
            if ev["type"] == "first_token":
                first = ev["ms"]
            elif ev["type"] == "reasoning":
                reasoning += 1
            elif ev["type"] == "content":
                text.append(ev["delta"])
            elif ev["type"] == "tool_calls":
                calls = ev["calls"]
            elif ev["type"] == "done":
                done = ev
    except Exception as e:  # noqa: BLE001
        report.add(f"{role}: chat", FAIL, f"{type(e).__name__}: {str(e)[:160]}", (time.perf_counter() - t0) * 1000)
        return
    report.add(f"{role}: first token", OK if first is not None else WARN,
               "streamed" if first is not None else "no tokens streamed", first)
    if any(c["name"] == "get_time" for c in calls):
        report.add(f"{role}: tool call", OK, "called get_time", done["ms"] if done else None)
    else:
        said = "".join(text).strip()[:80]
        report.add(f"{role}: tool call", WARN, f"did not call get_time for the time; it said {said!r}. If tools never "
                                               "fire, check the server passes them to the model", done["ms"] if done else None)
    if background:
        report.add(f"{role}: reasoning", OK if reasoning else WARN,
                   f"{reasoning} reasoning deltas" if reasoning else
                   "no reasoning streamed; check BACKGROUND_THINKING for this server")


async def _judge_check(llm: LLM, report: Report):
    for text, want in (("what time is it right now", "COMPLETE"), ("so I was wondering whether and", "INCOMPLETE")):
        try:
            complete, ms, raw = await llm.end_of_turn(text)
        except Exception as e:  # noqa: BLE001
            report.add("foreground: end-of-turn", FAIL, f"{type(e).__name__}: {str(e)[:160]}")
            return
        got = "COMPLETE" if complete else "INCOMPLETE"
        report.add(f"judge: {want.lower()}", OK if got == want else WARN,
                   f"{text!r} -> {raw or '(empty)'}" + ("" if got == want else f" (expected {want})"), ms)


async def speech_round_trip(settings: Settings, report: Report) -> None:
    try:
        speech = await build_speech(settings, log=lambda m: None)
    except Exception as e:  # noqa: BLE001
        report.add("speech: load", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
        return
    asr, tts = speech.asr, speech.tts
    report.add("asr: model", OK, f"{asr.model_id} ({asr.where})")
    report.add("tts: model", OK, f"{tts.model_id} ({tts.where}, voice {settings.tts_voice})")
    try:
        pcm, sr, ms = await tts.synth(PHRASE, settings.tts_voice, 1.0)
    except Exception as e:  # noqa: BLE001
        report.add("tts: synthesize", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
        return
    dur = len(pcm) / sr if sr else 0
    report.add("tts: synthesize", OK if dur > 1 else FAIL, f"{dur:.2f}s of audio at {sr} Hz (RTF {ms / 1000 / max(dur, 1e-6):.2f})", ms)
    audio16 = resample(pcm, sr, MIC_SR).astype(np.float32) / 32768.0
    try:
        text, ms = await asr.transcribe(audio16)
    except Exception as e:  # noqa: BLE001
        report.add("asr: transcribe", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
        return
    overlap = word_overlap(PHRASE, text)
    report.add("tts -> asr round trip", OK if overlap >= 0.8 else WARN if overlap >= 0.5 else FAIL,
               f"{overlap:.0%} of words back: {text[:80]!r}", ms)
    stream = await asr.stream()
    step = MIC_SR // 2
    t0 = time.perf_counter()
    partial = ""
    for i in range(0, len(audio16), step):
        partial, _ = await stream.feed(audio16[i:i + step])
    await stream.close()
    report.add("asr: partials", OK if word_overlap(PHRASE, partial) >= 0.5 else WARN,
               f"{len(range(0, len(audio16), step))} partials, last {partial[-50:]!r}", (time.perf_counter() - t0) * 1000)
    vad = SileroVAD()
    frames = (audio16 * 32767).astype(np.int16)
    probs = [vad.prob(frames[i:i + VAD_FRAME]) for i in range(0, len(frames) - VAD_FRAME, VAD_FRAME)]
    vad.reset()
    quiet = max(vad.prob(np.zeros(VAD_FRAME, dtype=np.int16)) for _ in range(10))
    speechy = float(np.mean(np.array(probs) > 0.5)) if probs else 0.0
    report.add("vad", OK if speechy > 0.4 and quiet < 0.2 else WARN,
               f"speech frames {speechy:.0%}, max p on silence {quiet:.2f}")


async def run(settings: Settings) -> int:
    report = Report()
    print("model roles", flush=True)
    for name, ep in settings.roles().items():
        print(f"  {name:<11} {ep.model or '(server default)':<44} {ep.host}", flush=True)
    print("\nforeground + background", flush=True)
    try:
        for note in await resolve_models(settings):
            report.add("model", OK, note)
    except Exception as e:  # noqa: BLE001
        report.add("model list", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
    if settings.foreground.model:
        llm = LLM(settings)
        await _chat_check(llm, report, "foreground", thinking=False)
        await _judge_check(llm, report)
        await _chat_check(llm, report, "background", thinking=True)  # thinking on: its own check even on one server
    print("\nspeech", flush=True)
    await speech_round_trip(settings, report)
    print(f"\n{'all checks passed' if not report.failed else f'{report.failed} check(s) failed'}"
          f"{'' if not any(r[1] == WARN for r in report.rows) else ' (with warnings)'}", flush=True)
    return 1 if report.failed else 0


async def warmup(settings: Settings) -> int:
    """Download and load the in-process models, then check they work together."""
    report = Report()
    await speech_round_trip(settings, report)
    return 1 if report.failed else 0
