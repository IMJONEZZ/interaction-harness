"""The interaction model, one per WebSocket connection.

This is the piece Thinking Machines' "interaction model" does natively and that a cascaded stack has to
build as a harness: a state machine that decides who has the floor, when the user's turn is over, when to
interrupt the agent, and when the agent may interject on its own.

    idle ──mic──▶ listen ──onset──▶ user_speaking ──silence──▶ endpointing ──EOT──▶ think ──audio──▶ speak
                    ▲                    ▲  (resumes)               │                                  │
                    │                    └──────────────────────────┘                                  │
                    └────────────────────────────── drained ◀─────────────────────────────────────────┘
                                          (barge-in: speak/think ──onset──▶ user_speaking)

Turn-taking modes (cfg.mode):
    ptt       - half-duplex; the client commits the turn (hold-to-talk)
    vad       - Silero VAD + fixed silence timeout
    semantic  - VAD + an LLM end-of-turn judge (short min silence, long max silence)
    duplex    - no state machine: a full-duplex speech model hears and talks at once (duplex.py)

Two models:
    foreground - thinking off, answers now, may call tools, may delegate
    background - thinking on, runs in the background, result is delivered proactively

Every step is emitted as a JSON event (and audio as binary frames) for the dashboard; docs/protocol.md
lists them.
"""
from __future__ import annotations

import asyncio
import itertools
import re
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import llm as llm_mod
from .config import Settings
from .duplex import DuplexMode
from .llm import LLM
from .prompts import FILLERS
from .speech import MIC_SR, VAD_FRAME, SentenceChunker, SpeechStack, strip_markdown
from .tools import ToolContext, run_tool, schemas

FRAME_MS = VAD_FRAME * 1000 / MIC_SR  # 32 ms
_BG_IDS = itertools.count(1)  # background task ids stay unique across sessions (the page may reconnect)
_BACKGROUND_SLOTS = asyncio.Semaphore(1)  # one background task at a time: the foreground and the judge keep their server slots
BG_HISTORY_MAX = 50


@dataclass
class Config:
    mode: str = "semantic"          # ptt | vad | semantic | duplex
    barge_in: bool = True
    filler: bool = True
    foreground_thinking: bool = False  # thinking on the foreground model (slower, shows reasoning)
    voice: str = "af_heart"
    speed: float = 1.05
    silence_ms: int = 700           # vad mode: end of turn after this much silence
    sem_min_ms: int = 300           # semantic mode: earliest judge call
    sem_max_ms: int = 2200          # semantic mode: give up waiting and end the turn
    onset_thresh: float = 0.5
    barge_thresh: float = 0.7
    barge_frames: int = 6           # ~190 ms of confident speech to interrupt the agent
    onset_frames: int = 3           # ~96 ms to open a user turn while listening

    def update(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k) and k != "update":
                cur = getattr(self, k)
                if isinstance(cur, bool) and not isinstance(v, bool):
                    v = str(v).lower() in ("1", "true", "yes", "on")  # bool("false") would be True
                setattr(self, k, type(cur)(v))


@dataclass
class TurnMetrics:
    turn: int
    source: str = "voice"                     # voice | typed | proactive
    t0: float = 0.0                           # end of user speech (perf_counter)
    marks: dict[str, float] = field(default_factory=dict)   # name -> ms after t0
    llm_ttft_ms: float = 0.0
    tokens: int = 0
    tps: float = 0.0
    tool_calls: int = 0
    tts_chunks: int = 0
    interrupted: bool = False

    def mark(self, name: str):
        self.marks[name] = round((time.perf_counter() - self.t0) * 1000, 1)

    def to_dict(self):
        d = asdict(self); d.pop("t0"); return d


_MARKER = re.compile(r"\s*\[(?:interrupted|event|cut off|note)[^\]]{0,80}\]", re.I)


class MarkerFilter:
    """Strip bracketed stage directions ("[interrupted by user]", "[EVENT] ...") from a token stream.

    Text after an unmatched "[" is held back until the bracket closes or 90 chars pass, so a marker
    split across deltas is still caught before anything reaches the screen or the TTS.
    """

    def __init__(self):
        self.buf = ""

    def push(self, delta: str) -> str:
        self.buf += delta
        self.buf = _MARKER.sub("", self.buf)
        i = self.buf.rfind("[")
        if i >= 0 and "]" not in self.buf[i:] and len(self.buf) - i < 90:
            out, self.buf = self.buf[:i], self.buf[i:]
        else:
            out, self.buf = self.buf, ""
        return out

    def flush(self) -> str:
        out, self.buf = _MARKER.sub("", self.buf), ""
        return out


class Session:
    def __init__(self, speech: SpeechStack, llm: LLM, settings: Settings,
                 send_json: Callable[[dict], Awaitable[None]], send_bytes: Callable[[bytes], Awaitable[None]]):
        self.sp = speech
        self.llm = llm
        self.s = settings
        self.vad = speech.new_vad()  # per-session VAD state
        self._send_json, self._send_bytes = send_json, send_bytes
        self.cfg = Config(voice=settings.tts_voice or "af_heart", speed=settings.tts_speed)
        self.state = "idle"
        self.history: list[dict] = [{"role": "system", "content": settings.foreground_prompt}]
        self.tools_ctx = ToolContext(on_timer=self._tool_timer, on_delegate=self._tool_delegate)
        self.partial_frames = max(1, round(settings.asr_partial_ms / FRAME_MS)) if settings.asr_partial_ms > 0 else 0

        # audio / endpointing
        self._pcm_rest = np.zeros(0, dtype=np.int16)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=12)  # ~380 ms before onset
        self.utter: list[np.ndarray] = []
        self.onset_run = 0
        self.silence_frames = 0
        self.last_speech_t = 0.0
        self.last_user_activity = time.perf_counter()
        self.ptt_down = False
        self._stream = None
        self._stream_fed = 0
        self._partial_task: asyncio.Task | None = None
        self._endpoint_task: asyncio.Task | None = None
        self._frame_i = 0

        # reply / playback
        self.turn_no = 0
        self.gen = 0                      # generation counter: bumps on interrupt
        self.reply_task: asyncio.Task | None = None
        self.speak_q: asyncio.Queue | None = None
        self.speaker_task: asyncio.Task | None = None
        self.chunk_id = 0
        self.chunks_sent: dict[int, str] = {}
        self.chunks_started: set[int] = set()
        self.chunks_ended: set[int] = set()
        self.pending_sentences = 0
        self.llm_done = False
        self.metrics: TurnMetrics | None = None
        self.filler_task: asyncio.Task | None = None

        # background
        self.proactive: deque[dict] = deque()
        self.bg: dict[int, dict] = {}
        self.interrupted_pending = False
        self.timers: dict[int, asyncio.Task] = {}
        self._bg_tasks: set[asyncio.Task] = set()
        self._proactive_task = asyncio.create_task(self._proactive_loop())

        # full-duplex: opened now, because loading its prompt takes seconds
        self.duplex = DuplexMode(self) if settings.duplex.base_url else None
        if self.duplex:
            self.duplex.start()

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    async def emit(self, type_: str, **data):
        data["type"] = type_
        data["t"] = round(time.time() * 1000)
        try:
            await self._send_json(data)
        except Exception:  # noqa: BLE001
            pass

    async def set_state(self, s: str, **extra):
        if s != self.state:
            self.state = s
            await self.emit("state", state=s, **extra)

    async def close(self):
        self._proactive_task.cancel()
        for t in list(self.timers.values()) + list(self._bg_tasks):
            t.cancel()
        await self._cancel_reply(reason="close")
        await self._close_stream()
        if self.duplex:
            await self.duplex.close()

    # ------------------------------------------------------------------ #
    # client messages
    # ------------------------------------------------------------------ #
    async def on_json(self, msg: dict):
        t = msg.get("type")
        if t == "config":
            old_mode = self.cfg.mode
            self.cfg.update({k: v for k, v in msg.items() if k != "type"})
            if self.cfg.mode == "duplex" and not self.duplex:
                self.cfg.mode = old_mode
                await self.emit("log", level="error", text="full-duplex needs DUPLEX_BASE_URL (see docs/duplex.md)")
            await self.emit("config", **asdict(self.cfg))
            if old_mode != self.cfg.mode and "duplex" in (old_mode, self.cfg.mode):
                await self._switch_duplex(entering=self.cfg.mode == "duplex")
            elif old_mode != self.cfg.mode and self.state in ("user_speaking", "endpointing"):
                await self._end_user_turn(reason="mode changed")
        elif t == "mic":
            if msg.get("on"):
                self.vad.reset()
                await self.set_state("listen")
            else:
                if self.state in ("user_speaking", "endpointing"):
                    await self._abandon_utterance()
                if self.state == "listen":
                    await self.set_state("idle")
        elif t == "ptt":
            await self._on_ptt(bool(msg.get("down")))
        elif t == "text":
            text = (msg.get("text") or "").strip()
            if text and self.cfg.mode == "duplex":
                await self.emit("log", level="warn", text="the full-duplex model only hears audio; typed turns "
                                                          "work in the other modes")
            elif text:
                await self._on_typed(text)
        elif t == "playback":
            await self._on_playback(msg)
        elif t == "stop":
            await self._interrupt(source="button")
        elif t == "reset":
            if self.duplex:
                await self.duplex.restart()  # a fresh model session: it forgets the conversation too
            await self._cancel_reply(reason="reset")
            self.history = [{"role": "system", "content": self.s.foreground_prompt}]
            self.tools_ctx.notes.clear()
            self.proactive.clear()
            await self.emit("log", level="info", text="conversation reset")
        elif t == "ping":
            await self.emit("pong")

    async def on_audio(self, data: bytes):
        pcm = np.frombuffer(data, dtype=np.int16)
        if len(self._pcm_rest):
            pcm = np.concatenate([self._pcm_rest, pcm])
        n = len(pcm) // VAD_FRAME
        self._pcm_rest = pcm[n * VAD_FRAME:].copy()
        for i in range(n):
            await self._on_frame(pcm[i * VAD_FRAME:(i + 1) * VAD_FRAME])

    # ------------------------------------------------------------------ #
    # per-frame turn-taking logic
    # ------------------------------------------------------------------ #
    async def _on_frame(self, frame: np.ndarray):
        self._frame_i += 1
        p = self.vad.prob(frame)
        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
        if self._frame_i % 2 == 0:
            await self.emit("vad", p=round(p, 3), rms=round(rms, 4))
        if self.cfg.mode == "duplex":
            await self.duplex.on_frame(frame, p)
            return
        speech = p >= self.cfg.onset_thresh
        now = time.perf_counter()
        if speech:
            self.last_speech_t = now

        if self.cfg.mode == "ptt":
            if self.ptt_down and self.state == "user_speaking":
                self.utter.append(frame)
                self.last_user_activity = now
                self._maybe_partial()
            else:
                self.pre_roll.append(frame)
            return

        if self.state in ("user_speaking", "endpointing"):
            self.utter.append(frame)
            self.last_user_activity = now
            if speech:
                self.silence_frames = 0
                if self.state == "endpointing":
                    await self._cancel_endpoint()
                    await self.set_state("user_speaking", resumed=True)
                    await self.emit("endpoint", phase="resumed")
            else:
                self.silence_frames += 1
                min_ms = self.cfg.silence_ms if self.cfg.mode == "vad" else self.cfg.sem_min_ms
                if self.state == "user_speaking" and self.silence_frames * FRAME_MS >= min_ms:
                    await self.set_state("endpointing")
                    self._endpoint_task = asyncio.create_task(self._endpoint())
            self._maybe_partial()
            if len(self.utter) * FRAME_MS > 60_000:
                await self._end_user_turn(reason="max length")
            return

        # listen / think / speak: look for an onset
        self.pre_roll.append(frame)
        if self.state == "idle":
            return
        if self.state == "listen":
            thresh, need = self.cfg.onset_thresh, self.cfg.onset_frames
        else:  # think / speak -> barge-in candidates
            if not self.cfg.barge_in:
                self.onset_run = 0
                return
            thresh, need = self.cfg.barge_thresh, self.cfg.barge_frames
        self.onset_run = self.onset_run + 1 if p >= thresh else 0
        if self.onset_run >= need:
            self.onset_run = 0
            if self.state in ("think", "speak"):
                await self._interrupt(source="voice")
            await self._start_utterance()

    async def _start_utterance(self):
        self.utter = list(self.pre_roll)
        self.pre_roll.clear()
        self.silence_frames = 0
        self.last_user_activity = time.perf_counter()
        self._stream_fed = 0
        await self.set_state("user_speaking")
        await self.emit("user.start")
        # open a streaming ASR session (partials)
        try:
            self._stream = await self.sp.asr.stream()
        except Exception as e:  # noqa: BLE001
            self._stream = None
            await self.emit("log", level="warn", text=f"asr stream unavailable: {e}")

    def _maybe_partial(self, force: bool = False):
        """Feed the new audio (every ASR_PARTIAL_MS) to the streaming ASR for a partial transcript."""
        n_new = len(self.utter) - self._stream_fed
        if self._stream is None or n_new < 1 or (self._partial_task and not self._partial_task.done()):
            return
        if not force and (not self.partial_frames or n_new < self.partial_frames):
            return
        chunk = np.concatenate(self.utter[self._stream_fed:]).astype(np.float32) / 32768.0
        self._stream_fed = len(self.utter)
        self._partial_task = asyncio.create_task(self._partial(chunk))

    async def _partial(self, chunk: np.ndarray):
        stream = self._stream
        if stream is None:
            return
        try:
            text, ms = await stream.feed(chunk)
        except Exception as e:  # noqa: BLE001
            await self.emit("log", level="warn", text=f"partial asr failed: {e}")
            return
        if stream is self._stream:
            await self.emit("asr.partial", text=text, ms=round(ms, 1),
                            audio_s=round(len(self.utter) * FRAME_MS / 1000, 2))

    async def _endpoint(self):
        """Runs while state == endpointing; decides when the user's turn is over."""
        try:
            if self.cfg.mode == "vad":
                await self.emit("endpoint", phase="silence", ms=self.cfg.silence_ms, verdict="COMPLETE (timeout)")
                await self._end_user_turn(reason=f"silence ≥ {self.cfg.silence_ms} ms")
                return
            # semantic: judge the transcript so far; re-judge as silence grows
            last_judged = None
            while self.state == "endpointing":
                sil_ms = self.silence_frames * FRAME_MS
                if sil_ms >= self.cfg.sem_max_ms:
                    await self.emit("endpoint", phase="judge", verdict="COMPLETE (max silence)", ms=0, text=last_judged or "")
                    await self._end_user_turn(reason=f"silence ≥ {self.cfg.sem_max_ms} ms")
                    return
                # make sure the partial transcript is current
                if self._partial_task and not self._partial_task.done():
                    await self._partial_task
                if self._stream is not None and self._stream_fed < len(self.utter) and not self._stream.text.strip():
                    self._maybe_partial(force=True)  # partials off or not due yet: transcribe now for the judge
                    if self._partial_task and not self._partial_task.done():
                        await self._partial_task
                text = self._stream.text if self._stream else ""
                if not text.strip():
                    # nothing transcribed yet: treat as a mid-turn pause, keep waiting
                    await asyncio.sleep(0.25)
                    continue
                if text != last_judged:
                    last_judged = text
                    complete, ms, raw = await self.llm.end_of_turn(text)
                    if self.state != "endpointing":
                        return
                    await self.emit("endpoint", phase="judge", text=text, verdict=raw or "?",
                                    ms=round(ms, 1), silence_ms=round(self.silence_frames * FRAME_MS))
                    if complete:
                        await self._end_user_turn(reason=f"semantic: {raw} after {round(self.silence_frames * FRAME_MS)} ms")
                        return
                await asyncio.sleep(0.3)
                if not self.partial_frames:
                    self._maybe_partial(force=True)  # no regular partials: refresh before the next judgement
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            await self.emit("log", level="error", text=f"endpoint error: {e}")
            await self._end_user_turn(reason="endpoint error")

    async def _cancel_endpoint(self):
        t, self._endpoint_task = self._endpoint_task, None
        # the endpoint task itself calls _end_user_turn -> never cancel the current task
        if t and not t.done() and t is not asyncio.current_task():
            t.cancel()

    async def _close_stream(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                await s.close()
            except Exception:  # noqa: BLE001
                pass

    async def _abandon_utterance(self):
        await self._cancel_endpoint()
        await self._close_stream()
        self.utter = []
        await self.set_state("listen")

    async def _end_user_turn(self, reason: str):
        if self.state not in ("user_speaking", "endpointing"):
            return
        await self._cancel_endpoint()
        t0 = self.last_speech_t or time.perf_counter()
        self.turn_no += 1
        m = TurnMetrics(turn=self.turn_no, source="voice", t0=t0)
        m.mark("eot")
        await self.set_state("think")
        await self.emit("user.end", reason=reason, audio_s=round(len(self.utter) * FRAME_MS / 1000, 2),
                        eot_ms=m.marks["eot"], turn=self.turn_no)
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        await self._close_stream()
        audio = np.concatenate(self.utter).astype(np.float32) / 32768.0 if self.utter else np.zeros(0, np.float32)
        self.utter = []
        if self.s.save_utterances:  # debug: keep what the ASR heard
            import soundfile as sf
            out = Path(self.s.save_utterances)
            out.mkdir(parents=True, exist_ok=True)
            sf.write(out / f"turn_{self.turn_no:03d}.wav", audio, MIC_SR)
        try:
            text, ms = await self.sp.asr.transcribe(audio)
        except Exception as e:  # noqa: BLE001
            await self.emit("log", level="error", text=f"asr error: {type(e).__name__}: {e}")
            text, ms = "", 0.0
        m.mark("asr")
        await self.emit("asr.final", text=text, ms=round(ms, 1), audio_s=round(len(audio) / MIC_SR, 2),
                        turn=self.turn_no, model=self.sp.asr.model_id)
        if not text.strip():
            await self.emit("log", level="info", text="empty transcript — ignoring")
            self.turn_no -= 1
            await self.set_state("listen")
            return
        self.reply_task = asyncio.create_task(self._respond([{"role": "user", "content": text}], m))

    # ------------------------------------------------------------------ #
    # PTT / typed
    # ------------------------------------------------------------------ #
    async def _on_ptt(self, down: bool):
        if self.cfg.mode != "ptt":
            return
        if down and not self.ptt_down:
            self.ptt_down = True
            if self.state in ("think", "speak"):
                await self._interrupt(source="ptt")
            await self._start_utterance()
        elif not down and self.ptt_down:
            self.ptt_down = False
            self.last_speech_t = time.perf_counter()
            await self._end_user_turn(reason="ptt released")

    async def _on_typed(self, text: str):
        if self.state in ("think", "speak"):
            await self._interrupt(source="typed")
        elif self.state in ("user_speaking", "endpointing"):
            await self._abandon_utterance()
        self.turn_no += 1
        m = TurnMetrics(turn=self.turn_no, source="typed", t0=time.perf_counter())
        m.mark("eot"); m.mark("asr")
        await self.set_state("think")
        await self.emit("asr.final", text=text, ms=0, audio_s=0, turn=self.turn_no, model="typed")
        self.reply_task = asyncio.create_task(self._respond([{"role": "user", "content": text}], m))

    # ------------------------------------------------------------------ #
    # the foreground model: respond, speak, handle tools
    # ------------------------------------------------------------------ #
    async def _respond(self, new_messages: list[dict], m: TurnMetrics):
        gen = self.gen
        self.metrics = m
        self.llm_done = False
        self.chunks_sent, self.chunks_started, self.chunks_ended = {}, set(), set()
        self.pending_sentences = 0
        self.speak_q = asyncio.Queue()
        self.speaker_task = asyncio.create_task(self._speaker(gen))
        if self.interrupted_pending and m.source != "proactive" and new_messages and new_messages[0]["role"] == "user":
            new_messages[0] = {"role": "user", "content": "(You were cut off mid-reply a moment ago; don't repeat yourself.) "
                               + new_messages[0]["content"]}
        self.interrupted_pending = False
        self.history.extend(new_messages)
        self._trim_history()
        spoken_text: list[str] = []
        await self.emit("turn.start", turn=m.turn, source=m.source,
                        lane="foreground", thinking=self.cfg.foreground_thinking)
        if self.cfg.filler and m.source != "proactive":
            self.filler_task = asyncio.create_task(self._filler(gen))
        try:
            for round_ in range(5):
                chunker = SentenceChunker()
                marks = MarkerFilter()
                content_parts: list[str] = []
                calls: list[dict] = []
                async for ev in self.llm.stream_chat(self.history, tools=schemas("foreground"),
                                                     thinking=self.cfg.foreground_thinking):
                    if gen != self.gen:
                        return
                    et = ev["type"]
                    if et == "first_token":
                        if round_ == 0:
                            m.llm_ttft_ms = round(ev["ms"], 1); m.mark("llm_first_token")
                        await self.emit("llm.first_token", ms=round(ev["ms"], 1), round=round_)
                    elif et == "reasoning":
                        await self.emit("llm.reasoning", delta=ev["delta"], lane="foreground")
                    elif et == "content":
                        clean = marks.push(ev["delta"])
                        if clean:
                            content_parts.append(clean)
                            await self.emit("llm.content", delta=clean, lane="foreground")
                            for s in chunker.push(clean):
                                await self._enqueue_speech(s, m)
                    elif et == "tool_calls":
                        calls = ev["calls"]
                    elif et == "done":
                        m.tokens += (ev["usage"].get("completion_tokens") or 0)
                        m.tps = round(ev["tps"], 1)
                        await self.emit("llm.done", lane="foreground", round=round_, ms=round(ev["ms"]),
                                        tokens=ev["usage"].get("completion_tokens"), tps=round(ev["tps"], 1),
                                        finish=ev["finish"])
                tail = marks.flush()
                if tail:
                    content_parts.append(tail)
                    await self.emit("llm.content", delta=tail, lane="foreground")
                    for s in chunker.push(tail):
                        await self._enqueue_speech(s, m)
                for s in chunker.flush():
                    await self._enqueue_speech(s, m)
                content = "".join(content_parts)
                if content.strip():
                    spoken_text.append(content)
                if not calls:
                    self.history.append({"role": "assistant", "content": content or ""})
                    break
                # tool round
                self.history.append(llm_mod.assistant_tool_message(content, calls))
                m.tool_calls += len(calls)
                for c in calls:
                    await self.emit("llm.tool_call", lane="foreground", id=c["id"], name=c["name"], arguments=c["arguments"])
                results = await asyncio.gather(*[run_tool(self.tools_ctx, c["name"], c["arguments"], "foreground")
                                                 for c in calls])
                if gen != self.gen:
                    return
                for c, (res, ms) in zip(calls, results, strict=True):
                    await self.emit("llm.tool_result", lane="foreground", id=c["id"], name=c["name"], result=res, ms=round(ms, 1))
                    self.history.append(llm_mod.tool_result_message(c["id"], c["name"], res))
            else:
                self.history.append({"role": "assistant", "content": " ".join(spoken_text) or "(no reply)"})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            await self.emit("log", level="error", text=f"llm error: {type(e).__name__}: {e}")
            await self._enqueue_speech("Sorry, I hit an error talking to the model.", m)
            self.history.append({"role": "assistant", "content": "(error)"})
        finally:
            if self.filler_task:
                self.filler_task.cancel()
            self.llm_done = True
            if gen == self.gen:
                await self.speak_q.put(None)  # sentinel: nothing more to say
                await self._maybe_finish_turn()

    async def _enqueue_speech(self, sentence: str, m: TurnMetrics):
        text = strip_markdown(sentence)
        if not text:
            return
        if "first_sentence" not in m.marks:
            m.mark("first_sentence")
        self.pending_sentences += 1
        await self.speak_q.put(text)
        await self.emit("tts.queued", text=text, turn=m.turn)

    async def _speaker(self, gen: int):
        """Serialises TTS: synthesize each sentence in order and ship it as a binary frame."""
        try:
            while True:
                text = await self.speak_q.get()
                if text is None or gen != self.gen:
                    break
                try:
                    await self._synth_and_send(text, gen, kind="reply")
                except Exception as e:  # noqa: BLE001
                    await self.emit("log", level="error", text=f"tts error: {type(e).__name__}: {e}")
                finally:
                    self.pending_sentences -= 1
                await self._maybe_finish_turn()
        except asyncio.CancelledError:
            pass

    async def _synth_and_send(self, text: str, gen: int, kind: str):
        pcm, sr, ms = await self.sp.tts.synth(text, self.cfg.voice, self.cfg.speed)
        if gen != self.gen or not len(pcm):
            return
        self.chunk_id += 1
        cid = self.chunk_id
        self.chunks_sent[cid] = text
        dur = len(pcm) / sr
        m = self.metrics
        if m and "tts_first" not in m.marks:
            m.mark("tts_first")
        if m:
            m.tts_chunks += 1
        await self.emit("tts.chunk", id=cid, text=text, synth_ms=round(ms, 1), audio_s=round(dur, 2),
                        rtf=round(ms / 1000 / dur, 2) if dur else 0, voice=self.cfg.voice, kind=kind)
        header = struct.pack("<II", cid, sr)
        await self._send_bytes(header + pcm.tobytes())
        if self.state == "think":
            await self.set_state("speak")

    async def _filler(self, gen: int):
        """If the first sentence isn't ready ~700 ms after the turn started, say something."""
        try:
            await asyncio.sleep(0.7)
            if gen != self.gen or self.chunks_sent or (self.metrics and "first_sentence" in self.metrics.marks):
                return
            cache = self.sp.filler_cache.setdefault(f"{self.cfg.voice}@{self.cfg.speed}", [])
            if not cache:
                for f in FILLERS:
                    pcm, sr, _ = await self.sp.tts.synth(f, self.cfg.voice, self.cfg.speed)
                    cache.append((pcm, sr))
                if gen != self.gen or self.chunks_sent:
                    return
            i = self.turn_no % len(cache)
            pcm, sr = cache[i]
            self.chunk_id += 1
            cid = self.chunk_id
            self.chunks_sent[cid] = FILLERS[i]
            await self.emit("tts.chunk", id=cid, text=FILLERS[i], synth_ms=0, audio_s=round(len(pcm) / sr, 2),
                            rtf=0, voice=self.cfg.voice, kind="filler")
            await self._send_bytes(struct.pack("<II", cid, sr) + pcm.tobytes())
            if self.state == "think":
                await self.set_state("speak")
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            await self.emit("log", level="warn", text=f"filler skipped: {type(e).__name__}: {e}")

    async def _switch_duplex(self, entering: bool):
        """Hand the floor to the full-duplex model, or take it back."""
        if entering:
            if self.state in ("user_speaking", "endpointing"):
                await self._abandon_utterance()
            await self._interrupt(source="mode")
            self.duplex.start()  # no-op when already open
            await self.set_state("listen")
        else:
            await self.emit("playback.stop")
            self.duplex.user_on = self.duplex.agent_on = False
            await self.set_state("listen")

    async def _on_playback(self, msg: dict):
        ev, cid = msg.get("event"), msg.get("id")
        if ev == "voice" or self.cfg.mode == "duplex":  # the duplex stream is reported as voiced/silent only
            if ev == "voice" and self.duplex and self.cfg.mode == "duplex":
                await self.duplex.on_voice(bool(msg.get("on")))
            return
        m = self.metrics
        if ev == "start":
            self.chunks_started.add(cid)
            if m and "first_audio" not in m.marks and self.chunks_sent.get(cid) not in FILLERS:
                m.mark("first_audio")
                await self.emit("metrics", **m.to_dict(), partial=True)
            elif m and "first_audio_any" not in m.marks:
                m.mark("first_audio_any")
        elif ev == "end":
            self.chunks_ended.add(cid)
            await self._maybe_finish_turn()
        elif ev == "drained":
            await self._maybe_finish_turn()

    async def _maybe_finish_turn(self):
        if self.state != "speak" and self.state != "think":
            return
        if not self.llm_done or self.pending_sentences > 0:
            return
        # every chunk we sent has finished playing in the browser
        if set(self.chunks_sent) - self.chunks_ended:
            return
        m = self.metrics
        if m:
            m.mark("turn_end")
            await self.emit("metrics", **m.to_dict(), partial=False)
        await self.emit("turn.end", turn=m.turn if m else self.turn_no)
        await self.set_state("listen")
        self.last_user_activity = time.perf_counter()

    HISTORY_MAX = 60   # messages (system excluded) before trimming
    HISTORY_KEEP = 36  # ...down to roughly this many, cut at a user-message boundary

    def _trim_history(self):
        """Bound the prompt. Without this every turn re-sends the whole conversation, so time-to-first-token
        grows for hours until the context window is hit. Cutting at a user message keeps tool-call /
        tool-result pairs intact."""
        body = self.history[1:]
        if len(body) <= self.HISTORY_MAX:
            return
        cut = len(body) - self.HISTORY_KEEP
        while cut < len(body) and body[cut]["role"] != "user":
            cut += 1
        dropped = body[:cut]
        self.history = [self.history[0]] + body[cut:]
        self.emit_later("log", level="info", text=f"history trimmed: dropped {len(dropped)} old messages")

    def emit_later(self, type_: str, **data):
        asyncio.get_running_loop().create_task(self.emit(type_, **data))

    # ------------------------------------------------------------------ #
    # interruption
    # ------------------------------------------------------------------ #
    async def _interrupt(self, source: str):
        if self.state not in ("think", "speak"):
            return
        spoken = [self.chunks_sent[c] for c in sorted(self.chunks_started) if c in self.chunks_sent
                  and self.chunks_sent[c] not in FILLERS]
        unspoken = [t for c, t in sorted(self.chunks_sent.items()) if c not in self.chunks_started]
        await self._cancel_reply(reason=f"interrupted ({source})")
        await self.emit("playback.stop")
        # truncate history to what was actually heard (drop partial assistant / tool-round messages)
        while len(self.history) > 1 and self.history[-1]["role"] != "user":
            self.history.pop()
        if spoken:
            self.history.append({"role": "assistant", "content": " ".join(spoken)})
        self.interrupted_pending = True
        if self.metrics:
            self.metrics.interrupted = True
            self.metrics.mark("interrupted")
            await self.emit("metrics", **self.metrics.to_dict(), partial=False)
        await self.emit("interrupt", source=source, spoken=" ".join(spoken), dropped=" ".join(unspoken),
                        turn=self.turn_no)
        await self.set_state("listen")

    async def _cancel_reply(self, reason: str):
        self.gen += 1
        for t in (self.reply_task, self.speaker_task, self.filler_task):
            if t and not t.done():
                t.cancel()
        self.reply_task = self.speaker_task = self.filler_task = None
        self.llm_done = True

    # ------------------------------------------------------------------ #
    # background model + proactive channel
    # ------------------------------------------------------------------ #
    async def _tool_timer(self, seconds: float, label: str) -> dict:
        seconds = max(1.0, min(seconds, 3600))
        tid = next(_BG_IDS)

        async def fire():
            await asyncio.sleep(seconds)
            await self.emit("timer.fired", id=tid, label=label)
            self.proactive.append({"kind": "timer", "id": tid, "label": label, "seconds": seconds})
            self.timers.pop(tid, None)

        self.timers[tid] = asyncio.create_task(fire())
        await self.emit("timer.set", id=tid, label=label, seconds=seconds)
        return {"ok": True, "timer_id": tid, "fires_in_s": seconds}

    async def _tool_delegate(self, task: str) -> dict:
        tid = next(_BG_IDS)
        self.bg[tid] = {"id": tid, "task": task, "status": "running", "started": time.time()}
        t = asyncio.create_task(self._run_background(tid, task))
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)
        return {"ok": True, "task_id": tid, "note": "running in the background; result will be delivered proactively"}

    def _context_package(self) -> str:
        lines = []
        for msg in self.history[1:]:
            if msg["role"] in ("user", "assistant") and msg.get("content"):
                lines.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(lines[-20:])

    async def _run_background(self, tid: int, task: str):
        if _BACKGROUND_SLOTS.locked():
            await self.emit("background.queued", id=tid, task=task)
        async with _BACKGROUND_SLOTS:
            await self._run_background_inner(tid, task)
        # bound the per-session record of finished tasks
        while len(self.bg) > BG_HISTORY_MAX:
            self.bg.pop(next(iter(self.bg)))

    async def _run_background_inner(self, tid: int, task: str):
        await self.emit("background.start", id=tid, task=task)
        messages = [{"role": "system", "content": self.s.background_prompt},
                    {"role": "user", "content": f"Conversation so far:\n{self._context_package()}\n\nDelegated task: {task}"}]
        t0 = time.perf_counter()
        final = ""
        try:
            for round_ in range(self.s.background_rounds):
                content_parts, calls = [], []
                async for ev in self.llm.stream_chat(messages, tools=schemas("background"), thinking=True,
                                                     max_tokens=6000, background=True):
                    et = ev["type"]
                    if et == "reasoning":
                        await self.emit("background.reasoning", id=tid, delta=ev["delta"])
                    elif et == "content":
                        content_parts.append(ev["delta"])
                        await self.emit("background.content", id=tid, delta=ev["delta"])
                    elif et == "tool_calls":
                        calls = ev["calls"]
                    elif et == "done":
                        await self.emit("background.round", id=tid, round=round_,
                                        tokens=ev["usage"].get("completion_tokens"),
                                        tps=round(ev["tps"], 1), ms=round(ev["ms"]))
                content = "".join(content_parts)
                if not calls:
                    final = content
                    break
                messages.append(llm_mod.assistant_tool_message(content, calls))
                for c in calls:
                    await self.emit("background.tool_call", id=tid, call_id=c["id"], name=c["name"], arguments=c["arguments"])
                results = await asyncio.gather(*[run_tool(self.tools_ctx, c["name"], c["arguments"], "background")
                                                 for c in calls])
                for c, (res, ms) in zip(calls, results, strict=True):
                    await self.emit("background.tool_result", id=tid, call_id=c["id"], name=c["name"], result=res, ms=round(ms, 1))
                    messages.append(llm_mod.tool_result_message(c["id"], c["name"], res))
            else:
                final = "".join(content_parts) or "I ran out of steps before finishing."
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001
            final = f"The background task failed: {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        self.bg[tid].update(status="done", result=final, ms=ms)
        await self.emit("background.done", id=tid, result=final, ms=round(ms))
        self.proactive.append({"kind": "background", "id": tid, "task": task, "result": final})

    async def _proactive_loop(self):
        """Deliver background results when the user isn't mid-sentence: the agent's own turn."""
        try:
            while True:
                await asyncio.sleep(0.25)
                if self.proactive and self.cfg.mode == "duplex":
                    if self.duplex.can_deliver():
                        item = self.proactive.popleft()
                        await self.emit("proactive", **item, waited_ms=0)
                        await self.duplex.deliver(item)
                    continue
                if not self.proactive or self.state not in ("listen", "idle") or self.ptt_down:
                    continue
                if time.perf_counter() - self.last_user_activity < 1.0:
                    continue
                item = self.proactive.popleft()
                await self.emit("proactive", **item, waited_ms=0)
                self.turn_no += 1
                m = TurnMetrics(turn=self.turn_no, source="proactive", t0=time.perf_counter())
                m.mark("eot"); m.mark("asr")
                await self.set_state("think")
                if item["kind"] == "timer":
                    ev = f"(Background event, not something the user said: the '{item['label']}' timer, {int(item['seconds'])} s, just fired. Tell the user.)"
                else:
                    ev = (f"(Background event, not something the user said: background task #{item['id']} finished.\n"
                          f"Task: {item['task']}\nResult: {item['result']}\n"
                          f"Tell the user the result now, conversationally, in at most three sentences. "
                          f"If the result says the task failed or did not finish, say so; never supply figures of your own.)")
                self.reply_task = asyncio.create_task(self._respond([{"role": "user", "content": ev}], m))
                # wait for that turn to finish before delivering the next
                while self.reply_task and not self.reply_task.done():
                    await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass
