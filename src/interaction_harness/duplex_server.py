"""A full-duplex speech model on a WebSocket, with tool results: `interaction-harness duplex-server`.

It serves NVIDIA's NemotronLabs VoiceChat on Apple Silicon through mlx-vlm (install the `duplex` extra):

    interaction-harness duplex-server                  # ws://127.0.0.1:8770/v1/realtime
    interaction-harness serve --duplex-base-url ws://127.0.0.1:8770/v1

The protocol is mlx-vlm's /v1/realtime (OpenAI Realtime-style event names), so a client written for one
works with the other. This server adds the one thing a tool-calling model needs and mlx-vlm doesn't have yet,
a tool result:

    {"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "<json>"}}

The model calls tools on its own function channel: <SPECIAL_20><TOOLCALL>[...]</TOOLCALL><SPECIAL_21>. A
result goes back the way NVIDIA's reference code does it, as <TOOL_RESPONSE>[...]</TOOL_RESPONSE> tokens fed
into that channel right after the call closes, one per 80 ms frame; the model then answers from it. In
training the result always comes next, so while a call waits for its result the model is held still (mic
frames are dropped) rather than stepped with nothing to read. A result nobody asked for (a timer that fired,
a finished background task) can name the call it answers, `"call": "[{...}]"`, and that call goes in first.

mlx-vlm's session has no hook for any of this, so the engine writes tokens into the session's
`_function_tokens` after each frame. That is why the `duplex` extra pins mlx-vlm to one version.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

DEFAULT_MODEL = "mlx-community/NemotronLabs-VoiceChat-11B-4bit"
DEFAULT_PORT = 8770
SOTC, EOTC, EOTR = "<SPECIAL_20>", "<SPECIAL_21>", "<SPECIAL_22>"  # start/end of a tool call, end of its result


def tool_response_text(output: str) -> str:
    """The tokens the model was trained to read a tool result from."""
    return f"<TOOL_RESPONSE>[{output}]</TOOL_RESPONSE>"


class ForcedChannel:
    """Feeds queued tokens into a model's function channel, one per frame, never inside an open tool call.

    `after_frame(tokens)` takes the session's function-token list right after a frame and may overwrite its
    last entry. It returns True when it did, so that frame's own prediction is hidden from clients. `hold()`
    says whether to skip the next frame instead: the model has closed a call and its result hasn't arrived.
    """

    MAX_CALL_FRAMES = 64  # a call that never closes stops blocking results after ~5 s
    MAX_WAIT_FRAMES = 25  # a result that never comes stops holding the model after 2 s
    COOLDOWN_FRAMES = 25  # after a result, 2 s in which the model may not start another call: it should speak

    def __init__(self, sotc_id: int, eotc_id: int, eotr_id: int | None = None, pad_id: int | None = None):
        self.sotc_id, self.eotc_id, self.eotr_id, self.pad_id = sotc_id, eotc_id, eotr_id, pad_id
        self.cooldown = 0
        self.queue: deque[int] = deque()
        self.in_call = False
        self.call_frames = 0
        self.forcing = False
        self.waiting = False
        self.wait_frames = 0

    def add(self, token_ids: list[int]) -> None:
        self.queue.extend(token_ids)

    def hold(self) -> bool:
        if not self.waiting:
            return False
        self.wait_frames += 1
        if self.queue or self.wait_frames > self.MAX_WAIT_FRAMES:
            self.waiting = False
        return self.waiting

    def after_frame(self, tokens: list[int]) -> bool:
        closed = masked = False
        inside = self.forcing and bool(self.queue)  # mid-result: the model's predictions are overwritten anyway
        if not inside:
            if self.cooldown > 0:
                self.cooldown -= 1
                if tokens[-1] == self.sotc_id and self.pad_id is not None:  # straight into another call: not yet
                    tokens[-1], masked = self.pad_id, True
            if tokens[-1] == self.sotc_id:  # the model's own prediction: track whether it is inside a call
                self.in_call, self.call_frames = True, 0
            elif tokens[-1] == self.eotc_id:
                self.in_call, closed = False, True  # the model reads its own end-of-call next; the result follows
                self.waiting, self.wait_frames = not self.queue, 0
            elif self.in_call:
                self.call_frames += 1
                self.in_call = self.call_frames < self.MAX_CALL_FRAMES
        self.forcing = bool(self.queue) and (inside or (not self.in_call and not closed))
        if self.forcing:
            tokens[-1] = self.queue.popleft()
            if not self.queue:  # the whole result is in: let the model speak before it calls again
                self.cooldown = self.COOLDOWN_FRAMES
        return self.forcing or masked


def _event(e) -> dict[str, Any]:
    common = {"frame_index": e.frame_index}
    if e.kind == "assistant_text_delta":
        return {"type": "response.text.delta", **common, "delta": e.delta, "text": e.text}
    if e.kind == "function_delta":
        return {"type": "response.function.delta", **common, "delta": e.delta, "text": e.text}
    if e.kind == "user_transcript_delta":
        return {"type": "conversation.item.input_audio_transcription.delta", **common,
                "delta": e.delta, "transcript": e.text}
    if e.kind == "audio":
        pcm = np.round(np.clip(np.asarray(e.samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
        return {"type": "response.audio.delta", **common, "delta": base64.b64encode(pcm.tobytes()).decode("ascii"),
                "format": "pcm16", "sample_rate": e.sample_rate, "channels": 1}
    return {"type": f"response.{e.kind}", **common}


class DuplexEngine:
    """The model and its one live session, all on one worker thread (MLX streams belong to their thread)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, log_functions: bool = False):
        self.model_id = model_id
        self.log_functions = log_functions
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duplex")
        self.parent = None
        self.session = None
        self.forced: ForcedChannel | None = None
        self.owner: str | None = None
        self.frame_ms: list[float] = []
        self.last_config: tuple[str | None, int] | None = None
        self.warm: tuple[tuple[str | None, int], Any] | None = None  # a session with the last prompt already read

    async def run(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._pool, partial(fn, *args, **kwargs))

    def load(self) -> None:
        from mlx_vlm.server.realtime import load_realtime_voicechat
        self.parent = load_realtime_voicechat(self.model_id)

    def open(self, system_prompt: str | None, seed: int = 0) -> dict:
        key = (system_prompt, seed)
        warm = self.warm is not None and self.warm[0] == key
        s = self.warm[1] if warm else self.parent.create_streaming_session(system_prompt=system_prompt, seed=seed)
        self.warm, self.last_config = None, key
        tok = s.tokenizer
        self.session = s
        self.forced = ForcedChannel(*(tok.convert_tokens_to_ids(t) for t in (SOTC, EOTC, EOTR)),
                                    pad_id=s.config.pad_token_id)
        self.frame_ms = []
        return {"frame_samples": s.frame_samples, "input_sample_rate": s.input_sample_rate,
                "output_sample_rate": s.output_sample_rate, "warm": warm}

    def prewarm(self) -> None:
        """Between connections, read the last prompt into a fresh session: the model reads a prompt one token per
        step (about 40 ms each on an M5 Max), so a reconnect would otherwise wait seconds."""
        if self.last_config and self.session is None and self.warm is None:
            prompt, seed = self.last_config
            self.warm = (self.last_config, self.parent.create_streaming_session(system_prompt=prompt, seed=seed))

    def push_frame(self, samples: np.ndarray) -> list[dict]:
        if self.forced.hold():  # a call is waiting for its result: don't step the model past it
            return []
        t0 = time.perf_counter()
        events = self.session.push_audio(samples, sample_rate=self.session.input_sample_rate)
        predicted = self.session._function_tokens[-1]
        hide = self.forced.after_frame(self.session._function_tokens)
        if self.log_functions:
            fed = self.session._function_tokens[-1]
            text = self.session._text_tokens[-1]
            if predicted != fed or fed not in (self.session.config.pad_token_id,) or self.forced.forcing or hide:
                tok = self.session.tokenizer
                print(f"frame {self.session.frame_index:5d}  predicted {tok.decode([predicted])!r:24}"
                      + (f"  fed {tok.decode([fed])!r:14}" if fed != predicted else " " * 19)
                      + f"  text {tok.decode([text])!r}", flush=True)
        self.frame_ms.append((time.perf_counter() - t0) * 1000)
        return [_event(e) for e in events if not (hide and e.kind == "function_delta")]

    def add_tool_output(self, output: str, call: str | None = None) -> int:
        tok, f = self.session.tokenizer, self.forced
        ids = [f.sotc_id, *tok.encode(f"<TOOLCALL>{call}</TOOLCALL>", add_special_tokens=False), f.eotc_id] if call else []
        ids += tok.encode(tool_response_text(output), add_special_tokens=False)
        ids += [f.eotr_id] if f.eotr_id is not None else []  # the result closes as it did in training
        f.add(ids)
        return len(ids)

    def close(self) -> None:
        self.session = None
        self.forced = None


def create_app(engine: DuplexEngine) -> FastAPI:
    app = FastAPI(title="interaction-harness duplex-server")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model"}]}

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        await ws.accept()
        sid = f"sess_{uuid.uuid4().hex[:16]}"

        async def send(d: dict):
            await ws.send_json({"event_id": f"event_{uuid.uuid4().hex[:16]}", **d})

        async def error(message: str, code: str = "invalid_request"):
            await send({"type": "error", "error": {"code": code, "message": message}})

        if engine.owner is not None:
            await error("another realtime session is already active", "server_busy")
            await ws.close(code=1013)
            return
        engine.owner = sid
        await send({"type": "session.created", "session": {"id": sid, "state": "configuring"}})
        configured, frame, pending = False, 1280, np.zeros(0, np.float32)
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type", "")
                if kind == "session.ping":
                    await send({"type": "session.pong"})
                elif kind == "session.update":
                    if configured:
                        await error("the active session cannot be reconfigured")
                        continue
                    cfg = msg.get("session") or {}
                    if cfg.get("model") not in (None, "", engine.model_id):
                        await error(f"this server runs {engine.model_id}, not {cfg['model']}")
                        continue
                    try:
                        info = await engine.run(engine.open, cfg.get("system_prompt"), int(cfg.get("seed", 0)))
                    except Exception as e:  # noqa: BLE001
                        await error(f"{type(e).__name__}: {e}", "session_initialization_failed")
                        continue
                    configured, frame = True, info["frame_samples"]
                    await send({"type": "session.updated", "session": {
                        "id": sid, "state": "ready", "model": engine.model_id, "frame_samples": frame,
                        "input_audio_format": {"type": "pcm16", "sample_rate": info["input_sample_rate"]},
                        "output_audio_format": {"type": "pcm16", "sample_rate": info["output_sample_rate"]},
                        "tool_outputs": True, "warm": info["warm"]}})
                elif not configured:
                    await error("send session.update before anything else")
                elif kind == "input_audio_buffer.append":
                    try:
                        raw = base64.b64decode(msg.get("audio", ""), validate=True)
                    except (binascii.Error, ValueError):
                        await error("audio must be base64 PCM16")
                        continue
                    pending = np.concatenate([pending, np.frombuffer(raw[: len(raw) // 2 * 2], "<i2") / 32768.0])
                    while len(pending) >= frame:  # one model step per 80 ms frame, so tool tokens land between
                        chunk, pending = pending[:frame].astype(np.float32), pending[frame:]
                        try:
                            events = await engine.run(engine.push_frame, chunk)
                        except Exception as e:  # noqa: BLE001
                            await error(f"{type(e).__name__}: {e}", "inference_error")
                            continue
                        for ev in events:
                            await send(ev)
                elif kind == "conversation.item.create":
                    item = msg.get("item") or {}
                    if item.get("type") != "function_call_output":
                        await error(f"unsupported item type {item.get('type')!r}")
                        continue
                    n = await engine.run(engine.add_tool_output, str(item.get("output", "")), item.get("call"))
                    await send({"type": "conversation.item.created", "item": {"type": "function_call_output",
                                                                               "tokens": n}})
                elif kind in ("input_audio_buffer.commit", "session.cancel", "response.cancel"):
                    break
                else:
                    await error(f"unsupported event type {kind!r}")
        except WebSocketDisconnect:
            pass
        finally:
            if configured:
                await engine.run(engine.close)
            engine.owner = None
            if configured:
                asyncio.get_running_loop().create_task(engine.run(engine.prewarm))

    return app


def serve(model: str = DEFAULT_MODEL, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          log_functions: bool = False) -> None:
    import uvicorn
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        raise SystemExit("duplex-server needs mlx-vlm on Apple Silicon: "
                         "uv tool install 'interaction-harness[duplex]'") from None
    engine = DuplexEngine(model, log_functions)
    print(f"loading {model} …", flush=True)
    t0 = time.perf_counter()
    asyncio.run(engine.run(engine.load))
    print(f"loaded in {time.perf_counter() - t0:.0f} s; ws://{host}:{port}/v1/realtime "
          f"(DUPLEX_BASE_URL=ws://{host}:{port}/v1)", flush=True)
    uvicorn.run(create_app(engine), host=host, port=port, log_level="warning")
