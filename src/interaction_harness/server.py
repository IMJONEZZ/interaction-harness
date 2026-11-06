"""FastAPI + WebSocket server: the dashboard at /, the session protocol at /ws.

Speech models load once at startup and are shared by every session; each browser tab gets its own Session
(state machine + history).

Wire protocol (docs/protocol.md has every event):
    client -> server   binary: int16 PCM @16 kHz mono (any chunk size)
                       json:   {"type":"config"|"mic"|"ptt"|"text"|"playback"|"stop"|"reset"|"ping"}
    server -> client   json:   typed events (see session.py)
                       binary: <u32 chunk_id><u32 sample_rate><int16 PCM>   (TTS audio)
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .llm import LLM, resolve_models
from .session import Session
from .speech import SpeechStack, build_speech, local_backend
from .tools import REGISTRY, load_tools_file

STATIC = Path(__file__).resolve().parent / "static"


def status(settings: Settings, speech: SpeechStack) -> dict:
    backend = local_backend(settings)

    def role(ep, model_id=None):
        return {"model": model_id or ep.model, "where": backend if ep.local else ep.host}

    return {"version": __version__,
            "roles": {"asr": role(settings.asr, speech.asr.model_id),
                      "tts": role(settings.tts, speech.tts.model_id),
                      "foreground": role(settings.foreground),
                      "background": role(settings.background)},
            "vad": speech.vad_name, "voices": speech.tts.voices, "voice": settings.tts_voice,
            "tools": {name: sorted(t.lanes) for name, t in REGISTRY.items()}}


def create_app(settings: Settings, speech: SpeechStack | None = None, llm: LLM | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for path in settings.tools_files:
            added = load_tools_file(path)
            print(f"tools from {path}: {', '.join(added) or '(none)'}", flush=True)
        for note in await resolve_models(settings):
            print(note, flush=True)
        app.state.speech = speech or await build_speech(settings)
        app.state.llm = llm or LLM(settings)
        fg, bg = settings.foreground, settings.background
        print(f"foreground {fg.model} @ {fg.host}  |  background {bg.model} @ {bg.host}", flush=True)
        yield

    app = FastAPI(title="interaction-harness", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def api_status():
        return status(settings, app.state.speech)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        lock = asyncio.Lock()

        async def send_json(d: dict):
            async with lock:
                await websocket.send_text(json.dumps(d, ensure_ascii=False))

        async def send_bytes(b: bytes):
            async with lock:
                await websocket.send_bytes(b)

        session = Session(app.state.speech, app.state.llm, settings, send_json, send_bytes)
        await session.emit("hello", **status(settings, app.state.speech))
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await session.on_audio(msg["bytes"])
                elif msg.get("text"):
                    try:
                        await session.on_json(json.loads(msg["text"]))
                    except Exception as e:  # noqa: BLE001
                        await session.emit("log", level="error", text=f"bad message: {e}")
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    return app
