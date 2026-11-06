"""End-to-end test: drive the server over its WebSocket with synthesized speech.

The configured TTS plays the user: each scripted utterance is synthesized up front, resampled to 16 kHz and
streamed as 32 ms mic frames in real time, with silence around it, so the real VAD / endpointing / ASR /
foreground / TTS path runs exactly as it would with a microphone. The test also plays the browser's part
(acknowledging playback so turns can end), and checks:

    * semantic endpointing ends the turn and the ASR transcribes the utterance
    * the foreground answers, calls a tool, and streams sentence-chunked TTS audio back
    * barge-in: speaking while the agent speaks interrupts it
    * a background task is delegated and delivered proactively, and a timer fires and is announced
    * push-to-talk and typed turns, with a note remembered and recalled
    * with DUPLEX_BASE_URL set: the full-duplex model answers (and calls a tool), and stops when talked over

Checks that depend on the model's choices rather than the harness (which tool it picked when two would do,
whether the background model streamed its reasoning) are warnings, not failures.

    interaction-harness selftest            # starts a server in-process unless one is already listening
    interaction-harness selftest --duplex   # only the full-duplex part
"""
from __future__ import annotations

import asyncio
import json
import struct
import time

import httpx
import numpy as np
import websockets

from .config import Settings
from .duplex import STREAM_SILENT, STREAM_VOICED
from .speech import MIC_SR, VAD_FRAME, local_backend, run_local
from .speech.base import resample

FRAME_S = VAD_FRAME / MIC_SR
UTTERANCES = {
    "time": "What time is it right now?",
    "lecture": "Without delegating explain in at least eight full sentences how a lighthouse lens works.",
    "interrupt": "Stop, actually what's the weather in Boston?",
    "two_things": ("Two things. First, set a timer for ten seconds called stretch. Second, in the background, work "
                   "out whether a 30 year mortgage at 6 percent or a 15 year at 5 and a half percent costs less in "
                   "total interest on 400 thousand dollars, and by how much."),
    "remember": "Remember that my favorite color is green.",
    "goldfish": "What's a good name for a goldfish?",
    "timer": "Remind me in fifteen seconds to stretch.",
    "delegate": ("In the background, work out whether a 15 year mortgage at 5 and a half percent or a 30 year at 6 "
                 "percent costs less interest on 400 thousand dollars."),
}
QUIET_EVENTS = ("vad", "llm.content", "llm.reasoning", "background.reasoning", "background.content", "tts.queued")


class Client:
    def __init__(self, ws, user_audio: dict[str, np.ndarray]):
        self.ws, self.user_audio = ws, user_audio
        self.events: list[dict] = []
        self.audio_chunks: list[tuple[int, int, np.ndarray]] = []
        self.state = "idle"
        self._play_tasks: set[asyncio.Task] = set()
        self.pending = 0
        self.voiced = False  # the full-duplex model's stream, as the browser would report it
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        async for msg in self.ws:
            if isinstance(msg, bytes):
                cid, sr = struct.unpack("<II", msg[:8])
                pcm = np.frombuffer(msg[8:], dtype=np.int16)
                if cid in (STREAM_SILENT, STREAM_VOICED):
                    if (cid == STREAM_VOICED) != self.voiced:
                        self.voiced = cid == STREAM_VOICED
                        await self.send({"type": "playback", "event": "voice", "on": self.voiced})
                    continue
                self.audio_chunks.append((cid, sr, pcm))
                t = asyncio.create_task(self._play(cid, len(pcm) / sr))
                self._play_tasks.add(t); t.add_done_callback(self._play_tasks.discard)
            else:
                ev = json.loads(msg)
                self.events.append(ev)
                if ev["type"] == "state":
                    self.state = ev["state"]
                if ev["type"] == "playback.stop":
                    for t in list(self._play_tasks):
                        t.cancel()
                    self.pending = 0
                if ev["type"] not in QUIET_EVENTS:
                    short = {k: (str(v)[:140]) for k, v in ev.items() if k not in ("type", "t")}
                    print(f"   ← {ev['type']:<22} {short}", flush=True)

    async def _play(self, cid: int, dur: float):
        # pretend to be the browser: start now (serialised), end after the audio's duration
        self.pending += 1
        try:
            await self.send({"type": "playback", "event": "start", "id": cid})
            await asyncio.sleep(dur)
            await self.send({"type": "playback", "event": "end", "id": cid})
        except asyncio.CancelledError:
            return
        finally:
            self.pending = max(0, self.pending - 1)
        if self.pending == 0:
            await self.send({"type": "playback", "event": "drained"})

    async def send(self, d: dict):
        await self.ws.send(json.dumps(d))

    async def say(self, key: str, pre_silence=0.4, post_silence=3.0):
        """Stream an utterance as real-time mic frames."""
        audio = self.user_audio[key]
        print(f"\n🗣  user: {UTTERANCES[key]!r}  ({len(audio) / MIC_SR:.1f}s)", flush=True)
        sil = np.zeros(int(pre_silence * MIC_SR), dtype=np.int16)
        await self.stream(np.concatenate([sil, audio]))
        await self.stream(np.zeros(int(post_silence * MIC_SR), dtype=np.int16))

    async def stream(self, audio: np.ndarray):
        t0 = time.perf_counter()
        for i in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME):
            await self.ws.send(audio[i:i + VAD_FRAME].tobytes())
            # pace to real time
            target = t0 + (i // VAD_FRAME + 1) * FRAME_S
            d = target - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)

    async def silence(self, seconds: float):
        await self.stream(np.zeros(int(seconds * MIC_SR), dtype=np.int16))

    async def wait_for(self, type_: str, timeout=60, since=0, pred=None) -> dict:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            for ev in self.events[since:]:
                if ev["type"] == type_ and (pred is None or pred(ev)):
                    return ev
            await asyncio.sleep(0.05)
        raise TimeoutError(f"no {type_} within {timeout}s")

    def count(self, type_: str, since=0) -> int:
        return sum(1 for e in self.events[since:] if e["type"] == type_)


class Checks:
    def __init__(self):
        self.failed = self.warned = 0

    def check(self, cond, msg):
        print(("✅ " if cond else "❌ ") + msg, flush=True)
        self.failed += 0 if cond else 1

    def warn(self, cond, msg):
        print(("✅ " if cond else "⚠️  ") + msg, flush=True)
        self.warned += 0 if cond else 1


async def conversation(c: Client, ck: Checks):
    await c.wait_for("hello", 10)
    await c.send({"type": "config", "mode": "semantic", "barge_in": True, "filler": False, "foreground_thinking": False})
    await c.send({"type": "mic", "on": True})
    await c.silence(0.5)

    # 1) simple tool turn ------------------------------------------------
    n = len(c.events)
    await c.say("time")
    fin = await c.wait_for("asr.final", 30, n)
    ck.check("time" in fin["text"].lower(), f"ASR final transcript: {fin['text']!r} ({fin['ms']} ms)")
    ep = [e for e in c.events[n:] if e["type"] == "endpoint" and e.get("phase") == "judge"]
    ck.check(bool(ep), f"semantic end-of-turn judge ran: {[e.get('verdict') for e in ep]}")
    tc = await c.wait_for("llm.tool_call", 30, n)
    ck.check(tc["name"] == "get_time", f"foreground called tool {tc['name']}")
    ch = await c.wait_for("tts.chunk", 40, n)
    ck.check(True, f"first TTS chunk: {ch['text']!r} synth {ch['synth_ms']} ms for {ch['audio_s']}s (RTF {ch['rtf']})")
    met = await c.wait_for("metrics", 60, n, pred=lambda e: not e.get("partial"))
    ck.check("first_audio" in met["marks"], f"turn metrics: {met['marks']}  tokens={met['tokens']} tps={met['tps']}")
    await c.wait_for("turn.end", 60, n)
    ck.check(c.state == "listen", f"back to listen after turn (state={c.state})")

    # 2) barge-in -------------------------------------------------------
    n = len(c.events)
    # the server counts silence in frames, so send enough to end the turn even if the judge says INCOMPLETE
    await c.say("lecture", post_silence=2.5)
    await c.wait_for("tts.chunk", 60, n)
    await c.wait_for("state", 30, n, pred=lambda e: e["state"] == "speak")
    await c.silence(1.0)  # let it talk a bit
    await c.say("interrupt", pre_silence=0.0)
    it = await c.wait_for("interrupt", 15, n)
    ck.check(it["source"] == "voice", f"barge-in interrupted the agent (spoken so far: {it['spoken'][:60]!r}…)")
    fin = await c.wait_for("asr.final", 30, n, pred=lambda e: "boston" in e["text"].lower() or "weather" in e["text"].lower())
    ck.check(True, f"post-interrupt transcript: {fin['text']!r}")
    await c.wait_for("turn.end", 90, n, pred=lambda e: e["turn"] == fin["turn"])

    # 3) delegate + timer -------------------------------------------------
    n = len(c.events)
    await c.say("two_things", post_silence=3.0)
    tset = await c.wait_for("timer.set", 60, n)
    ck.check(True, f"timer set: {tset['label']} {tset['seconds']}s")
    bs = await c.wait_for("background.start", 60, n)
    ck.check(True, f"background task delegated: {bs['task'][:80]!r}")
    await c.wait_for("turn.end", 90, n)
    # keep the mic quiet so proactive delivery can happen
    got_timer = got_bg = False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 300 and not (got_timer and got_bg):
        await c.silence(0.5)
        got_timer = any(e["type"] == "proactive" and e.get("kind") == "timer" for e in c.events[n:])
        got_bg = any(e["type"] == "proactive" and e.get("kind") == "background" for e in c.events[n:])
    ck.check(got_timer, "timer fired and was delivered proactively")
    dd = [e for e in c.events[n:] if e["type"] == "background.done"]
    ck.check(got_bg and dd, f"background task done in {dd[0]['ms'] if dd else '?'} ms and delivered proactively: "
                            f"{dd[0]['result'][:120] if dd else ''!r}")
    rs = c.count("background.reasoning", n)
    ck.warn(rs > 0, f"background model streamed {rs} reasoning deltas")
    # wait for the proactive turns to finish speaking
    await c.silence(1.0)
    for _ in range(60):
        if c.state == "listen" and c.pending == 0:
            break
        await c.silence(0.5)

    # 4) PTT + typed ------------------------------------------------------
    n = len(c.events)
    await c.send({"type": "config", "mode": "ptt"})
    await c.send({"type": "ptt", "down": True})
    await c.stream(c.user_audio["remember"])
    await c.send({"type": "ptt", "down": False})
    fin = await c.wait_for("asr.final", 20, n)
    ck.check("green" in fin["text"].lower(), f"push-to-talk transcript: {fin['text']!r}")
    await c.wait_for("turn.end", 60, n)
    n = len(c.events)
    await c.send({"type": "text", "text": "What is my favorite color? Use recall."})
    await c.wait_for("turn.end", 60, n)
    used = [e["name"] for e in c.events[n:] if e["type"] == "llm.tool_call"]
    ck.warn("recall" in used, f"typed turn used the recall tool (tools called: {used or 'none'}; the model may "
                              "answer from the conversation instead)")
    ans = "".join(e.get("delta", "") for e in c.events[n:] if e["type"] == "llm.content")
    ck.check("green" in ans.lower(), f"assistant recalled the note: {ans[:80]!r}")

    await c.send({"type": "mic", "on": False})


async def duplex_conversation(c: Client, ck: Checks):
    """The full-duplex model: no harness turn-taking, so check what the model itself did."""
    print("\n--- full-duplex ---", flush=True)
    await c.send({"type": "config", "mode": "duplex"})
    st = await c.wait_for("duplex.status", 300, 0, pred=lambda e: e["state"] in ("ready", "error"))
    ck.check(st["state"] == "ready", f"duplex model ready: {st.get('model', st.get('error'))} (session opened in "
                                     f"{st.get('load_ms')} ms; tool results {'on' if st.get('tool_outputs') else 'off'})")
    if st["state"] != "ready":
        return
    await c.send({"type": "mic", "on": True})
    await c.silence(4.0)  # it may greet you first

    def first_audio(since: int) -> float | None:
        met = [e for e in c.events[since:] if e["type"] == "metrics" and "first_audio" in e["marks"]]
        return met[0]["marks"]["first_audio"] if met else None

    n = len(c.events)
    await c.say("goldfish", post_silence=5.0)
    fa = first_audio(n)
    said = "".join(e.get("delta", "") for e in c.events[n:] if e["type"] == "llm.content").strip()
    ck.check(fa is not None and bool(said), "the model answered on its own" +
             (f", first audio {fa:.0f} ms after you stopped: {said[:80]!r}" if fa is not None else ""))

    n = len(c.events)
    await c.say("time", post_silence=8.0)
    fa = first_audio(n)
    said = "".join(e.get("delta", "") for e in c.events[n:] if e["type"] == "llm.content").strip()
    ck.check(fa is not None and bool(said), "it answered a tool question" +
             (f", first audio {fa:.0f} ms after you stopped: {said[:80]!r}" if fa is not None else ""))
    calls = [e["name"] for e in c.events[n:] if e["type"] == "llm.tool_call"]
    ck.warn("get_time" in calls, f"it called get_time (tools called: {calls or 'none'})")
    heard = [e["text"] for e in c.events[n:] if e["type"] == "asr.final"]
    ck.warn(any("time" in h.lower() for h in heard), f"its transcript of you: {heard}")

    n = len(c.events)
    await c.say("lecture", post_silence=0.0)
    try:
        await c.wait_for("state", 20, n, pred=lambda e: e["state"] == "speak")
    except TimeoutError:
        ck.check(False, "the model didn't start talking after a long request")
        return
    await c.silence(2.0)  # let it talk
    n = len(c.events)
    await c.say("interrupt", pre_silence=0.0, post_silence=4.0)
    y = [e for e in c.events[n:] if e["type"] == "duplex.yield"]
    ck.warn(bool(y) and not y[0]["kept_talking"],
            f"it stopped when talked over: {y[0]['ms']} ms after you cut in" if y and not y[0]["kept_talking"] else
            "it kept talking over you" if y else "no cut-in measured (it may have finished first)")
    await c.silence(3.0)

    # a timer and a background task: their results reach the model as tool responses, and it should say them
    n = len(c.events)
    await c.say("timer", post_silence=5.0)
    await c.say("delegate", post_silence=5.0)
    calls = [e["name"] for e in c.events[n:] if e["type"] == "llm.tool_call"]
    ck.warn("set_timer" in calls, f"it set a timer (tools called: {calls or 'none'})")
    ck.warn("delegate_background_task" in calls, f"it handed work to the background model (tools called: {calls or 'none'})")
    expected = {k for k, tool in (("background", "delegate_background_task"), ("timer", "set_timer")) if tool in calls}
    t0, delivered = time.perf_counter(), set()
    while time.perf_counter() - t0 < 240 and not expected <= delivered:
        await c.silence(0.5)
        delivered = {e["kind"] for e in c.events[n:] if e["type"] == "proactive"}
    await c.silence(10.0)  # time to say it
    for kind in sorted(expected):
        at = next((i for i, e in enumerate(c.events) if i >= n and e["type"] == "proactive" and e["kind"] == kind), None)
        said = "".join(e.get("delta", "") for e in c.events[at:] if e["type"] == "llm.content").strip() if at else ""
        ck.warn(bool(said), f"the {kind} result reached the model, and then it said {said[:100]!r}" if at else
                f"the {kind} result was never delivered")
    await c.send({"type": "mic", "on": False})


async def _listening(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2, verify=False) as h:
            return (await h.get(url)).status_code == 200
    except httpx.HTTPError:
        return False


async def run(settings: Settings, url: str | None = None, duplex_only: bool = False) -> int:
    import uvicorn

    from .server import create_app
    from .speech import load_tts

    ws_url = url or f"ws://127.0.0.1:{settings.port}/ws"
    status_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws", 1)[0] + "/api/status"
    server = task = None
    if await _listening(status_url):
        print(f"using the server at {ws_url}", flush=True)
        tts = await run_local(load_tts, settings, local_backend(settings)) if settings.tts.local \
            else load_tts(settings, local_backend(settings))
        voices = await tts.discover_voices() if hasattr(tts, "discover_voices") else tts.voices
    elif url:
        print(f"nothing is listening at {url}", flush=True)
        return 1
    else:
        print(f"starting a server on 127.0.0.1:{settings.port} for the test…", flush=True)
        app = create_app(settings)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=settings.port, log_level="warning"))
        task = asyncio.create_task(server.serve())
        while not server.started:
            if task.done():
                task.result()  # surfaces the startup error
                return 1
            await asyncio.sleep(0.2)
        tts, voices = app.state.speech.tts, app.state.speech.tts.voices
    agent_voice = settings.tts_voice or (voices[0] if voices else "")
    user_voice = next((v for v in ("am_michael", "am_puck", "bm_george") if v in voices),
                      next((v for v in voices if v != agent_voice), agent_voice))
    print(f"synthesizing the user's lines with voice {user_voice!r}…", flush=True)
    user_audio = {}
    for key, text in UTTERANCES.items():
        pcm, sr, _ = await tts.synth(text, user_voice, 1.0)
        user_audio[key] = resample(pcm, sr, MIC_SR)

    ck = Checks()
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            c = Client(ws, user_audio)
            try:
                if not duplex_only:
                    await conversation(c, ck)
                if settings.duplex.base_url:
                    await duplex_conversation(c, ck)
                elif duplex_only:
                    ck.check(False, "--duplex needs DUPLEX_BASE_URL")
            except TimeoutError as e:
                ck.check(False, f"timed out: {e}")
            finally:
                c.reader.cancel()
            chunks = len(c.audio_chunks)
    finally:
        if server is not None:
            server.should_exit = True
            await task
    verdict = "ALL PASSED" if not ck.failed else f"{ck.failed} CHECK(S) FAILED"
    print(f"\n{verdict}{f' ({ck.warned} warning(s))' if ck.warned else ''}  ({chunks} audio chunks received)", flush=True)
    return 1 if ck.failed else 0
