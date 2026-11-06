"""ASR and TTS on any OpenAI-compatible endpoint.

    ASR  POST {base}/audio/transcriptions   multipart: file=<16 kHz WAV>, model, response_format=json[, language]
    TTS  POST {base}/audio/speech           json: model, input, voice, speed, response_format=wav|pcm

Plain HTTP rather than the SDK, so servers that only implement the documented fields still work, and a
server that answers a json request with plain text is still understood. The transcriptions API has no
live-audio mode, so partials re-post the turn so far (WindowedStream); TTS makes one request per sentence.
"""
from __future__ import annotations

import io
import time

import httpx
import numpy as np
import soundfile as sf

from ..config import Endpoint
from .base import MIC_SR, PREFERRED_VOICES, WindowedStream, to_int16


def _client(ep: Endpoint, transport: httpx.AsyncBaseTransport | None, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=ep.base_url, transport=transport, timeout=timeout,
                             headers={"Authorization": f"Bearer {ep.api_key}"})


def _error(r: httpx.Response) -> str:
    return f"HTTP {r.status_code} from {r.request.url}: {r.text[:300]}"


class EndpointASR:
    def __init__(self, ep: Endpoint, language: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not ep.model:
            raise ValueError("ASR_BASE_URL is set but ASR_MODEL is not; name the server's transcription model")
        self.ep, self.language = ep, language
        self.model_id, self.where = ep.model, ep.host
        self.http = _client(ep, transport, timeout=60)

    async def transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        if len(audio_f32_16k) < MIC_SR // 10:  # <100 ms: nothing to say
            return "", 0.0
        buf = io.BytesIO()
        sf.write(buf, to_int16(audio_f32_16k), MIC_SR, format="WAV", subtype="PCM_16")
        data = {"model": self.ep.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        try:
            r = await self.http.post("/audio/transcriptions", data=data,
                                     files={"file": ("utterance.wav", buf.getvalue(), "audio/wav")})
        except httpx.HTTPError as e:
            raise RuntimeError(f"ASR request to {self.ep.host} failed ({type(e).__name__}: {e}); "
                               "the server's own log usually says why") from e
        if r.status_code >= 400:
            raise RuntimeError(_error(r))
        try:
            text = r.json().get("text", "")
        except ValueError:
            text = r.text  # a server that ignores response_format and returns text/plain
        return (text or "").strip(), (time.perf_counter() - t0) * 1000

    async def stream(self) -> WindowedStream:
        return WindowedStream(self)


class EndpointTTS:
    def __init__(self, ep: Endpoint, fmt: str = "wav", pcm_rate: int = 24_000, voices: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not ep.model:
            raise ValueError("TTS_BASE_URL is set but TTS_MODEL is not; name the server's speech model")
        self.ep, self.fmt, self.pcm_rate = ep, fmt, pcm_rate
        self.model_id, self.where = ep.model, ep.host
        self.voices = list(voices or [])
        self.http = _client(ep, transport, timeout=60)

    async def discover_voices(self) -> list[str]:
        """Best effort: GET {base}/audio/voices (mlx-audio and Kokoro-FastAPI serve it; OpenAI does not)."""
        if self.voices:
            return self.voices
        try:
            r = await self.http.get("/audio/voices", params={"model": self.ep.model}, timeout=30)
            body = r.json() if r.status_code < 400 else []
        except (httpx.HTTPError, ValueError):
            body = []
        items = body.get("voices", body.get("data", [])) if isinstance(body, dict) else body
        names = [v if isinstance(v, str) else (v.get("id") or v.get("name") or "") for v in items or []]
        names = [n for n in names if n]
        # the same default voice as the in-process backends when the server has it
        self.voices = [v for v in PREFERRED_VOICES if v in names] + [n for n in names if n not in PREFERRED_VOICES]
        return self.voices

    async def synth(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16), self.pcm_rate, 0.0
        try:
            r = await self.http.post("/audio/speech", json={"model": self.ep.model, "input": text, "voice": voice,
                                                            "speed": speed, "response_format": self.fmt})
        except httpx.HTTPError as e:
            raise RuntimeError(f"TTS request to {self.ep.host} failed ({type(e).__name__}: {e}); "
                               "the server's own log usually says why") from e
        if r.status_code >= 400:
            raise RuntimeError(_error(r))
        if self.fmt == "pcm":
            pcm, sr = np.frombuffer(r.content, dtype=np.int16).copy(), self.pcm_rate
        else:
            pcm, sr = sf.read(io.BytesIO(r.content), dtype="int16", always_2d=True)
            pcm = pcm.mean(axis=1).astype(np.int16) if pcm.shape[1] > 1 else pcm[:, 0]
        return pcm, int(sr), (time.perf_counter() - t0) * 1000
