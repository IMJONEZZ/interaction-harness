"""Full-duplex mode: one speech-to-speech model hears the mic and talks back, every 80 ms.

In the other modes the harness decides who has the floor. Here the model does: it listens and speaks at the
same time, so it can answer before a pause would end a turn, talk over you, or stop when you cut in. ASR, the
foreground model, TTS and the end-of-turn judge are all out of the loop. The harness:

- streams mic audio to the DUPLEX endpoint, and the model's audio to the browser
- runs the model's tool calls, including the hand-off to the background model, and feeds the results back
- measures what the model did, with local VAD on your side and the playback level on its side: how long after
  you stopped it started talking (first audio), and how long it kept talking after you cut in (yield)

The endpoint speaks mlx-vlm's /v1/realtime protocol. `interaction-harness duplex-server` serves it with tool
results; mlx-vlm's own server works too, but its model can't hear what a tool returned.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import struct
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import NO_KEY, Endpoint
from .tools import run_tool, schemas

if TYPE_CHECKING:
    from .session import Session

DEFAULT_MODEL = "mlx-community/NemotronLabs-VoiceChat-11B-4bit"
STREAM_SILENT, STREAM_VOICED = 0xFFFFFFFE, 0xFFFFFFFF  # chunk ids for the model's continuous audio
VOICED_RMS = 0.01        # a model output frame louder than this is speech
HANGOVER_FRAMES = 5      # ...and it still counts as talking through 400 ms of quiet: words have gaps
HANGOVER_S = HANGOVER_FRAMES * 0.08
USER_END_MS = 500        # this much silence ends your utterance (for measuring only; the model decides)
YIELD_GIVE_UP_S = 4.0    # still talking this long after you cut in: it didn't yield
TEXT_IDLE_S = 0.8        # agent text and audio both quiet this long: its utterance is over

# NVIDIA's tool-calling protocol for this model (the system message is followed by the tools and these rules)
_TOOL_PROTOCOL = (
    'If you decide to call any tool(s), use the following format:\n'
    '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, {"name": "tool_name2", "arguments": "tool_args2"}]'
    '</TOOLCALL>\n\n'
    'The user will execute tool-calls and return responses from tool(s) in this format:\n'
    '<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>\n\n'
    'Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are '
    'found, or just respond to the user.')
_CALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def render_prompt(system: str, tools: list[dict]) -> str:
    """The system message plus tools, compacted: the model reads its prompt one token per 80 ms step."""
    if not tools:
        return system
    compact = []
    for t in tools:
        f = t.get("function", t)
        props = {k: {"type": v.get("type", "string")} for k, v in f["parameters"].get("properties", {}).items()}
        compact.append({"name": f["name"], "description": f["description"].split(". ")[0].rstrip(".") + ".",
                        "parameters": {"type": "object", "properties": props,
                                       "required": f["parameters"].get("required", [])}})
    return (f"{system}\n\nYou can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>["
            + ", ".join(json.dumps(t) for t in compact) + f"]</AVAILABLE_TOOLS>\n\n{_TOOL_PROTOCOL}")


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Every complete <TOOLCALL>[...]</TOOLCALL> in `text` as (name, arguments)."""
    out = []
    for body in _CALL.findall(text):
        try:
            items = json.loads(body.strip())
        except json.JSONDecodeError:
            continue
        for it in items if isinstance(items, list) else [items]:
            if not isinstance(it, dict) or not it.get("name"):
                continue
            args = it.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip().startswith("{") else {}
                except json.JSONDecodeError:
                    args = {}
            out.append((str(it["name"]), args if isinstance(args, dict) else {}))
    return out


def brief(result: Any, limit: int = 220) -> str:
    """A tool result as the model should read it: compact ASCII JSON, long values cut (each token costs 80 ms).
    A result with a "spoken" field sends only that: tools can say what matters in one line."""
    if isinstance(result, dict) and isinstance(result.get("spoken"), str):
        result = {"spoken": result["spoken"]}

    def trim(v):
        if isinstance(v, str):
            return v if len(v) <= 120 else v[:117] + "..."
        if isinstance(v, dict):
            return {k: trim(x) for k, x in list(v.items())[:8]}
        if isinstance(v, list):
            return [trim(x) for x in v[:4]]
        return v
    text = json.dumps(trim(result), ensure_ascii=True)
    return text if len(text) <= limit else json.dumps({"summary": text[: limit - 20]}, ensure_ascii=True)


def ungrounded(args: dict, heard: str) -> str | None:
    """NVIDIA's rule for this model: tool arguments are values the user spoke. Returns the first short text
    argument (a city, a label) that never came up in what the model heard, or None. A word counts as heard if
    a heard word starts the same way ("reminder" / "remind", and the ASR's "stress" for "stretch"). Longer
    text, like a task to hand off, is the model's paraphrase and isn't checked."""
    said = set(re.findall(r"[a-z0-9]+", heard.lower()))

    def near(w: str) -> bool:
        n = min(4, len(w))
        return any(h[:n] == w[:n] for h in said)

    for v in args.values():
        if isinstance(v, str) and 0 < len(v.split()) <= 3:
            words = [w for w in re.findall(r"[a-z0-9]+", v.split(",")[0].lower()) if len(w) >= 3]
            if words and not all(near(w) for w in words):
                return v
    return None


def realtime_url(ep: Endpoint) -> str:
    base = ep.base_url or ""
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http):]
    return base + "/realtime"


class DuplexLink:
    """A client for one /v1/realtime full-duplex session."""

    def __init__(self, ep: Endpoint):
        self.ep = ep
        self.ws = None
        self.info: dict = {}
        self.model = ep.model or DEFAULT_MODEL

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.ep.api_key}"} if self.ep.api_key != NO_KEY else {}

    async def resolve_model(self) -> str:
        """DUPLEX_MODEL, or the one model the server lists, or the default."""
        if not self.ep.model:
            import httpx
            url = realtime_url(self.ep).replace("ws", "http", 1).removesuffix("/realtime") + "/models"
            try:
                async with httpx.AsyncClient(timeout=3) as c:
                    data = (await c.get(url, headers=self._headers)).json().get("data") or []
                if len(data) == 1:
                    self.model = data[0]["id"]
            except Exception:  # noqa: BLE001
                pass
        return self.model

    @property
    def tool_outputs(self) -> bool:
        """Whether the server can feed tool results to the model (duplex-server can; mlx-vlm's can't yet)."""
        return bool(self.info.get("tool_outputs"))

    async def open(self, system_prompt: str, timeout: float = 180) -> dict:
        from websockets.asyncio.client import connect
        self.ws = await connect(realtime_url(self.ep), max_size=None, additional_headers=self._headers, open_timeout=10)
        await self.send({"type": "session.update", "session": {"model": self.model, "system_prompt": system_prompt}})
        while True:  # the prompt loads one token per step, so this can take a while
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout))
            if msg.get("type") == "session.updated":
                self.info = msg.get("session") or {}
                return self.info
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", {}).get("message", "duplex endpoint error"))

    async def send(self, d: dict) -> None:
        await self.ws.send(json.dumps(d))

    async def send_audio(self, pcm16: np.ndarray) -> None:
        await self.send({"type": "input_audio_buffer.append", "sample_rate": 16000,
                         "audio": base64.b64encode(pcm16.astype("<i2").tobytes()).decode("ascii")})

    async def send_tool_output(self, output: str, call: str | None = None) -> None:
        """A tool result. `call` (a JSON list, as in <TOOLCALL>) names the call it answers, when the model didn't
        just make one: a timer firing, a background task finishing."""
        item = {"type": "function_call_output", "output": output, **({"call": call} if call else {})}
        await self.send({"type": "conversation.item.create", "item": item})

    async def events(self):
        async for raw in self.ws:
            yield json.loads(raw)

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None


class DuplexMode:
    """What a Session does while cfg.mode == "duplex"."""

    def __init__(self, session: Session):
        self.s = session
        self.link: DuplexLink | None = None
        self.ready = False
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._out = np.zeros(0, np.int16)
        self._fn = ""
        # your side, from the session's VAD
        self.user_on = False
        self._run = self._quiet = 0
        self._user_last = 0.0
        self.heard = ""
        self._heard_before: deque[str] = deque(maxlen=8)  # your last few utterances, as the model heard them
        # the model's side, from playback in the browser
        self.agent_on = False
        self._voiced_out, self._quiet_out = False, 0
        self._agent_text = ""
        self._agent_t = 0.0
        self._agent_turn: int | None = None
        self._answered: set[int] = set()
        self._barge_t: float | None = None
        self.turn = None  # TurnMetrics of your last utterance, until the model answers it

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Open the session in the background: loading the prompt takes seconds, so do it before it's needed."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_link())

    async def close(self) -> None:
        for t in [self._task, *self._tasks]:
            if t and not t.done():
                t.cancel()
        if self.link:
            await self.link.close()
        self.link, self.ready = None, False

    async def restart(self) -> None:
        await self.close()
        self.start()

    async def _run_link(self) -> None:
        s = self.s
        link = DuplexLink(s.s.duplex)
        prompt = render_prompt(s.s.duplex_prompt, schemas("duplex"))
        await link.resolve_model()
        await s.emit("duplex.status", state="loading", model=link.model, host=s.s.duplex.host,
                     prompt_chars=len(prompt))
        t0 = time.perf_counter()
        try:
            await link.open(prompt)
        except Exception as e:  # noqa: BLE001
            await s.emit("duplex.status", state="error", error=f"{type(e).__name__}: {e}")
            await link.close()
            return
        self.link, self.ready = link, True
        await s.emit("duplex.status", state="ready", model=link.model, load_ms=round((time.perf_counter() - t0) * 1000),
                     tool_outputs=link.tool_outputs, frame_samples=link.info.get("frame_samples"))
        try:
            async for ev in link.events():
                await self._on_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            await s.emit("log", level="error", text=f"duplex link: {type(e).__name__}: {e}")
        self.ready = False
        await s.emit("duplex.status", state="closed")

    # ------------------------------------------------------------------ mic frames (every 32 ms)
    async def on_frame(self, frame: np.ndarray, p: float) -> None:
        s, now = self.s, time.perf_counter()
        if not self.ready:
            return
        self._out = np.concatenate([self._out, frame])
        if len(self._out) >= 1280:  # one model frame: 80 ms
            block, self._out = self._out[:1280], self._out[1280:]
            try:
                await self.link.send_audio(block)
            except Exception as e:  # noqa: BLE001
                self.ready = False
                await s.emit("log", level="error", text=f"duplex send failed: {e}")
                return
        speech = p >= s.cfg.onset_thresh
        if not self.user_on:
            self._run = self._run + 1 if p >= (s.cfg.barge_thresh if self.agent_on else s.cfg.onset_thresh) else 0
            if self._run >= (s.cfg.barge_frames if self.agent_on else s.cfg.onset_frames):
                await self._user_started(now)
        else:
            s.last_user_activity = now
            if speech:
                self._quiet, self._user_last = 0, now
            else:
                self._quiet += 1
                if self._quiet * 32 >= USER_END_MS:
                    await self._user_ended()
        if self._barge_t and self.agent_on and now - self._barge_t > YIELD_GIVE_UP_S:
            await s.emit("duplex.yield", ms=None, kept_talking=True, turn=self._agent_turn)
            self._barge_t = None
        await self._maybe_end_agent(now)

    async def _user_started(self, now: float) -> None:
        s = self.s
        onset = now - self._run * 0.032  # you started talking when the run of speech frames began
        self.user_on, self._run, self._quiet, self._user_last = True, 0, 0, now
        self.heard = ""
        s.last_user_activity = now
        await s.emit("user.start")
        if self.agent_on:  # you cut in: time how long the model keeps talking
            self._barge_t = onset
            await s.emit("duplex.overlap", who="you", text="you started talking while the model was talking")
        await s.set_state("user_speaking")

    async def _user_ended(self) -> None:
        from .session import TurnMetrics
        s = self.s
        self.user_on = False
        self._heard_before.append(self.heard)
        if self.heard.strip():  # the conversation history the background model (and the other modes) see
            s.history.append({"role": "user", "content": self.heard.strip()})
        s.turn_no += 1
        m = TurnMetrics(turn=s.turn_no, source="duplex", t0=self._user_last)
        m.mark("eot")
        s.metrics = self.turn = m
        await s.emit("asr.final", text=self.heard.strip() or "(nothing heard yet)", ms=0,
                     audio_s=0, turn=m.turn, model="duplex")
        if self.agent_on:  # it was already talking when you stopped
            m.marks["first_audio"] = 0.0
            m.marks["overlap"] = 1.0
            await s.emit("metrics", **m.to_dict(), partial=True)
        await s.set_state("speak" if self.agent_on else "listen")

    # ------------------------------------------------------------------ the browser's playback
    async def on_voice(self, on: bool) -> None:
        s, now = self.s, time.perf_counter()
        if on == self.agent_on:
            return
        self.agent_on = on
        if on:
            self._agent_t = now
            m = self.turn
            if m and "first_audio" not in m.marks:
                m.mark("first_audio")
                await s.emit("metrics", **m.to_dict(), partial=True)
            if self.user_on:
                await s.emit("duplex.overlap", who="model", text="the model started talking while you were talking")
            await s.set_state("speak")
        else:
            if self._barge_t is not None:  # it went quiet HANGOVER_S before the stream said so
                ms = max(0, round((now - HANGOVER_S - self._barge_t) * 1000))
                self._barge_t = None
                await s.emit("duplex.yield", ms=ms, kept_talking=False, turn=self._agent_turn)
                m = self.turn
                if m:
                    m.interrupted = True
                    m.marks["interrupted"] = round((now - m.t0) * 1000, 1)
            await s.set_state("user_speaking" if self.user_on else "listen")

    # ------------------------------------------------------------------ events from the model
    async def _on_event(self, ev: dict) -> None:
        s, kind = self.s, ev.get("type")
        if kind == "response.audio.delta":
            pcm = np.frombuffer(base64.b64decode(ev["delta"]), "<i2")
            rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2))) if len(pcm) else 0.0
            self._quiet_out = 0 if rms > VOICED_RMS else self._quiet_out + 1
            self._voiced_out = rms > VOICED_RMS or (self._voiced_out and self._quiet_out < HANGOVER_FRAMES)
            cid = STREAM_VOICED if self._voiced_out else STREAM_SILENT
            await s._send_bytes(struct.pack("<II", cid, int(ev.get("sample_rate") or 22050)) + pcm.tobytes())
            if cid == STREAM_VOICED:
                self._agent_t = time.perf_counter()
        elif kind == "conversation.item.input_audio_transcription.delta":
            self.heard += ev.get("delta") or ""
            await s.emit("asr.partial", text=self.heard.strip(), ms=0, audio_s=0, duplex=True)
        elif kind == "response.text.delta":
            delta = ev.get("delta") or ""
            if self._agent_turn is None:
                if self.turn is None or self.turn.turn in self._answered:
                    s.turn_no += 1
                    self._agent_turn = s.turn_no
                else:
                    self._agent_turn = self.turn.turn
                    self._answered.add(self.turn.turn)
                self._agent_text = ""
                await s.emit("turn.start", turn=self._agent_turn, source="duplex", lane="duplex", thinking=False)
            self._agent_text += delta
            self._agent_t = time.perf_counter()
            await s.emit("llm.content", delta=delta, lane="duplex")
        elif kind == "response.function.delta":
            self._fn += ev.get("delta") or ""
            if "</TOOLCALL>" in self._fn:
                calls, self._fn = parse_tool_calls(self._fn), self._fn[self._fn.rfind("</TOOLCALL>") + 11:]
                if calls:
                    t = asyncio.create_task(self._tools(calls))
                    self._tasks.add(t)
                    t.add_done_callback(self._tasks.discard)
        elif kind == "error":
            await s.emit("log", level="error", text=f"duplex: {ev.get('error', {}).get('message', ev)}")

    async def _maybe_end_agent(self, now: float) -> None:
        if self._agent_turn is None or self.agent_on or now - self._agent_t < TEXT_IDLE_S:
            return
        s, turn, text = self.s, self._agent_turn, self._agent_text.strip()
        self._agent_turn = None
        if text:
            s.history.append({"role": "assistant", "content": text})
        await s.emit("duplex.utterance", turn=turn, text=text)
        m = self.turn
        if m and m.turn == turn:
            m.mark("turn_end")
            await s.emit("metrics", **m.to_dict(), partial=False)
            self.turn = None
        await s.emit("turn.end", turn=turn)

    # ------------------------------------------------------------------ tools
    async def _tools(self, calls: list[tuple[str, dict]]) -> None:
        s = self.s
        if self.turn:
            self.turn.tool_calls += len(calls)
        outputs = []
        for i, (name, args) in enumerate(calls):
            cid = f"dx{s.turn_no}-{int(time.time() * 1000) % 100000}-{i}"
            arguments = json.dumps(args)
            await s.emit("llm.tool_call", lane="duplex", id=cid, name=name, arguments=arguments)
            unsaid = ungrounded(args, " ".join([*self._heard_before, self.heard]))
            if unsaid is not None:  # it invented an argument: don't act on it, and tell it why
                res, ms = {"error": f"blocked: the user never said {unsaid!r}; only call tools for what they ask"}, 0.0
            else:
                res, ms = await run_tool(s.tools_ctx, name, arguments, "duplex")
            await s.emit("llm.tool_result", lane="duplex", id=cid, name=name, result=res, ms=round(ms, 1))
            outputs.append(brief(res))
        await self._feed(", ".join(outputs))

    async def _feed(self, output: str, call: str | None = None) -> bool:
        if not (self.ready and self.link and self.link.tool_outputs):
            await self.s.emit("log", level="warn", text="the duplex server can't feed tool results to the model; "
                                                        "run interaction-harness duplex-server")
            return False
        await self.link.send_tool_output(output, call)
        await self.s.emit("duplex.tool_output", output=output, call=call)
        return True

    def can_deliver(self) -> bool:
        return (self.ready and self.link is not None and self.link.tool_outputs and not self.user_on
                and not self.agent_on and time.perf_counter() - self.s.last_user_activity > 1.0)

    async def deliver(self, item: dict) -> None:
        """A timer fired or a background task finished: hand it to the model as the answer to the call that
        started it, the pattern it was trained on (call, result, then speak)."""
        if item["kind"] == "timer":
            call = {"name": "set_timer", "arguments": {"seconds": int(item["seconds"]), "label": item["label"]}}
            payload = {"timer": item["label"], "status": "fired now"}
        else:
            call = {"name": "delegate_background_task", "arguments": {"task": item["task"][:80]}}
            try:
                summary = await self.s.llm.one_sentence(item["result"])
            except Exception:  # noqa: BLE001
                summary = item["result"][:200]
            payload = {"status": "done", "result": summary}
        await self._feed(brief(payload, limit=320), json.dumps([call]))
