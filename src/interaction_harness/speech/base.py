"""What the session needs from speech, whatever runs it.

Audio conventions:
    mic  -> 16 kHz mono int16, 512-sample frames (32 ms), which is what Silero VAD wants
    ASR  <- 16 kHz float32
    TTS  -> int16 PCM at the backend's own rate; the rate travels with every chunk to the browser

ASR and TTS are async so an in-process model and an HTTP endpoint look the same to the session.
In-process models run on one worker thread (`run_local`) so the event loop never blocks and two
models never race each other for the GPU. VAD stays synchronous: it is cheap and runs on every frame.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Protocol

import numpy as np

MIC_SR = 16_000
VAD_FRAME = 512  # samples @16k = 32 ms
# Kokoro voices offered first, in this order, when a backend has them; af_heart is the default
PREFERRED_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_michael", "am_fenrir",
                    "am_puck", "bf_emma", "bm_george", "bm_lewis"]

_local_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech")


async def run_local(fn, *args, **kwargs):
    """Run a blocking in-process model call on the dedicated worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_local_pool, partial(fn, *args, **kwargs))


class VAD(Protocol):
    def prob(self, frame_int16: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class ASRStream(Protocol):
    text: str

    async def feed(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        """Add audio; return (transcript so far, ms)."""
        ...

    async def close(self) -> None: ...


class ASR(Protocol):
    model_id: str
    where: str   # "mlx", "onnx", or the endpoint host

    async def transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        """Whole utterance. Returns (text, ms)."""
        ...

    async def stream(self) -> ASRStream: ...


class TTS(Protocol):
    model_id: str
    where: str
    voices: list[str]

    async def synth(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int, float]:
        """One chunk of text. Returns (int16 mono PCM, sample rate, ms)."""
        ...


class WindowedStream:
    """Partial transcripts for an ASR that has no streaming mode: keep the audio and re-transcribe it.

    Each feed() transcribes at most the last `window_s` seconds, so a long turn costs a bounded amount per
    partial. The session decides how often to feed (ASR_PARTIAL_MS), which sets the cost.
    """

    def __init__(self, asr: ASR, window_s: float = 20.0):
        self.asr = asr
        self.max_samples = int(window_s * MIC_SR)
        self.audio = np.zeros(0, dtype=np.float32)
        self.text = ""

    async def feed(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        self.audio = np.concatenate([self.audio, audio_f32_16k])
        text, ms = await self.asr.transcribe(self.audio[-self.max_samples:])
        self.text = text
        return text, ms

    async def close(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """float [-1, 1] (or int16 already) -> int16."""
    if audio.dtype == np.int16:
        return audio
    return np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Linear resampling; good enough for feeding speech to an ASR or a VAD."""
    if sr_from == sr_to or not len(audio):
        return audio
    x = audio.astype(np.float32)
    n = int(len(x) * sr_to / sr_from)
    out = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return out.astype(audio.dtype) if audio.dtype == np.int16 else out.astype(np.float32)
