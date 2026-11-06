"""Fakes for every model role, so the state machine can be tested without models, servers, or a network."""
from __future__ import annotations

import asyncio
import json
import struct
import time

import numpy as np
import pytest

from interaction_harness.config import load_settings
from interaction_harness.speech import SpeechStack
from interaction_harness.speech.base import WindowedStream

LOUD = 8000  # int16 amplitude the fake VAD calls speech


def make_settings(**env):
    base = {"FOREGROUND_BASE_URL": "http://fg.test/v1", "FOREGROUND_MODEL": "fake-model", "TTS_VOICE": "af_heart"}
    base.update(env)
    return load_settings(env=base, dotenv_path="/nonexistent/.env")


class FakeVAD:
    def prob(self, frame):
        return 1.0 if np.abs(frame.astype(np.int32)).mean() > LOUD / 2 else 0.0

    def reset(self):
        pass


class FakeASR:
    model_id, where = "fake-asr", "test"

    def __init__(self, text="what time is it"):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        return (self.text if len(audio) else ""), 1.0

    async def stream(self):
        return WindowedStream(self)


class FakeTTS:
    model_id, where = "fake-tts", "test"
    voices = ["af_heart"]
    sample_rate = 22_050  # not 24 kHz on purpose: the rate must travel with each chunk

    async def synth(self, text, voice, speed=1.0):
        return np.full(len(text) * 50, 100, dtype=np.int16), self.sample_rate, 1.0


def reply(text: str) -> list[dict]:
    return [{"type": "first_token", "ms": 1.0}, {"type": "content", "delta": text},
            {"type": "done", "content": text, "reasoning": "", "finish": "stop",
             "usage": {"completion_tokens": 5}, "ms": 5.0, "tps": 100.0}]


def call(name: str, **args) -> list[dict]:
    return [{"type": "first_token", "ms": 1.0},
            {"type": "tool_calls", "calls": [{"id": f"call_{name}", "name": name, "arguments": json.dumps(args)}]},
            {"type": "done", "content": "", "reasoning": "", "finish": "tool_calls",
             "usage": {"completion_tokens": 5}, "ms": 5.0, "tps": 100.0}]


class FakeLLM:
    """Scripted turns: each stream_chat call pops the next script for its role (or says 'ok')."""

    def __init__(self, foreground=None, background=None, complete=True):
        self.foreground, self.background = list(foreground or []), list(background or [])
        self.complete = complete
        self.judged: list[str] = []
        self.seen: list[list[dict]] = []

    async def stream_chat(self, messages, *, tools=None, thinking=False, temperature=0.7, max_tokens=1200,
                          background=False):
        self.seen.append([dict(m) for m in messages])
        script = self.background if background else self.foreground
        for ev in (script.pop(0) if script else reply("ok.")):
            await asyncio.sleep(0)
            yield ev

    async def end_of_turn(self, transcript):
        self.judged.append(transcript)
        return self.complete, 2.0, "COMPLETE" if self.complete else "INCOMPLETE"


class Harness:
    """A Session wired to fakes, with the browser's side played: every audio chunk is 'played' instantly."""

    def __init__(self, session_cls, llm, asr=None, settings=None, auto_play=True):
        self.events: list[dict] = []
        self.chunks: list[tuple[int, int, int]] = []  # (chunk id, sample rate, samples)
        self.auto_play = auto_play
        self.speech = SpeechStack(asr=asr or FakeASR(), tts=FakeTTS(), new_vad=FakeVAD, vad_name="fake")
        self.llm = llm
        self.session = session_cls(self.speech, llm, settings or make_settings(), self._json, self._bytes)

    async def _json(self, d):
        self.events.append(d)

    async def _bytes(self, b):
        cid, sr = struct.unpack("<II", b[:8])
        self.chunks.append((cid, sr, (len(b) - 8) // 2))
        if self.auto_play:
            asyncio.get_running_loop().create_task(self.play(cid))

    async def play(self, cid):
        await self.session.on_json({"type": "playback", "event": "start", "id": cid})
        await self.session.on_json({"type": "playback", "event": "end", "id": cid})

    def types(self):
        return [e["type"] for e in self.events]

    async def wait_for(self, type_, pred=None, timeout=5.0, since=0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            for e in self.events[since:]:
                if e["type"] == type_ and (pred is None or pred(e)):
                    return e
            await asyncio.sleep(0.01)
        raise AssertionError(f"no {type_} event within {timeout}s; saw {self.types()[-15:]}")

    async def frames(self, n, loud):
        frame = np.full(512, LOUD if loud else 0, dtype=np.int16)
        for _ in range(n):
            await self.session.on_audio(frame.tobytes())


@pytest.fixture
def settings():
    return make_settings()
