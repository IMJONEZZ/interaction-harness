"""Speech for the session: ASR and TTS (each in-process or on an endpoint) and a VAD per connection."""
from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from .base import ASR, MIC_SR, TTS, VAD, VAD_FRAME, run_local
from .silero import SileroVAD
from .text import SentenceChunker, strip_markdown

__all__ = ["ASR", "TTS", "VAD", "MIC_SR", "VAD_FRAME", "SpeechStack", "SentenceChunker", "strip_markdown",
           "build_speech", "local_backend", "run_local"]


@dataclass
class SpeechStack:
    """Models shared by every session (loaded once). VAD is stateful, so each session makes its own."""

    asr: ASR
    tts: TTS
    new_vad: Callable[[], VAD] = SileroVAD
    vad_name: str = SileroVAD.name
    filler_cache: dict[str, list[tuple[np.ndarray, int]]] = field(default_factory=dict)


def mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def local_backend(settings: Settings) -> str:
    if settings.backend == "auto":
        return "mlx" if mlx_available() else "onnx"
    if settings.backend == "mlx" and not mlx_available():
        raise RuntimeError("BACKEND=mlx needs an Apple Silicon Mac; use BACKEND=onnx or auto")
    return settings.backend


def load_asr(settings: Settings, backend: str) -> ASR:
    if not settings.asr.local:
        from .endpoint import EndpointASR
        return EndpointASR(settings.asr, settings.asr_language)
    if backend == "mlx":
        from .mlx import ParakeetMLX
        return ParakeetMLX(settings.asr.model or None)
    from .onnx import ParakeetONNX
    return ParakeetONNX(settings.asr.model or None, device=settings.device,
                        quantization=None if settings.parakeet_onnx_variant == "fp32" else settings.parakeet_onnx_variant)


def load_tts(settings: Settings, backend: str) -> TTS:
    if not settings.tts.local:
        from .endpoint import EndpointTTS
        return EndpointTTS(settings.tts, settings.tts_format, settings.tts_sample_rate, settings.tts_voices)
    if backend == "mlx":
        from .mlx import KokoroMLX
        return KokoroMLX(settings.tts.model or None, settings.tts_voices or None)
    from .onnx import KokoroONNX
    return KokoroONNX(settings.kokoro_onnx_variant, settings.device, settings.tts_voices or None)


async def build_speech(settings: Settings, log: Callable[[str], None] = print) -> SpeechStack:
    """Load (or connect) ASR and TTS, pick a voice, and warm both up so the first real turn isn't slow."""
    backend = local_backend(settings)
    t0 = time.perf_counter()
    asr = await run_local(load_asr, settings, backend) if settings.asr.local else load_asr(settings, backend)
    tts = await run_local(load_tts, settings, backend) if settings.tts.local else load_tts(settings, backend)
    if hasattr(tts, "discover_voices"):
        await tts.discover_voices()
    if not settings.tts_voice:
        if not tts.voices:
            raise RuntimeError(f"the TTS server at {settings.tts.host} does not list its voices; set TTS_VOICE "
                               "to one of its voice names")
        settings.tts_voice = tts.voices[0]
    if settings.tts_voice not in tts.voices:
        tts.voices.insert(0, settings.tts_voice)
    await tts.synth("Ready.", settings.tts_voice, settings.tts_speed)
    await asr.transcribe(np.zeros(MIC_SR, dtype=np.float32))
    SileroVAD()
    log(f"speech ready in {time.perf_counter() - t0:.1f}s  |  ASR {asr.model_id} ({asr.where})  |  "
        f"TTS {tts.model_id} ({tts.where}, voice {settings.tts_voice})  |  VAD {SileroVAD.name}")
    return SpeechStack(asr=asr, tts=tts)
