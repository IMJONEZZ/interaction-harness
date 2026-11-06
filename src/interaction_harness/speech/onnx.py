"""In-process speech anywhere onnxruntime runs: the same Parakeet and Kokoro models as the MLX backend.

ASR is Parakeet TDT 0.6B v3 through onnx-asr (weights from Hugging Face on first use). TTS is Kokoro-82M
through kokoro-onnx (model and voices from the kokoro-onnx GitHub release on first use, cached under
~/.cache/interaction-harness). CPU is the default; DEVICE=cuda uses onnxruntime-gpu when it is installed.

Parakeet on ONNX has no streaming mode, so partial transcripts come from re-transcribing the turn so far
(see WindowedStream).
"""
from __future__ import annotations

import time

import numpy as np

from .base import MIC_SR, PREFERRED_VOICES, WindowedStream, cache_dir, download, run_local, to_int16

DEFAULT_ASR = "nemo-parakeet-tdt-0.6b-v3"
MAX_PASS_S = 20  # Parakeet handles about 20-30 s per pass; longer turns go in windows
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = {"fp32": "kokoro-v1.0.onnx", "fp16": "kokoro-v1.0.fp16.onnx", "int8": "kokoro-v1.0.int8.onnx"}
KOKORO_VOICES_FILE = "voices-v1.0.bin"


def providers(device: str) -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("DEVICE=cuda but onnxruntime has no CUDA provider; install the onnx-gpu extra "
                               "(pip install 'interaction-harness[onnx-gpu]') or use DEVICE=cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ParakeetONNX:
    where = "onnx"

    def __init__(self, model_id: str | None = None, device: str = "cpu", quantization: str | None = None):
        import onnx_asr
        self.model_id = model_id or DEFAULT_ASR
        # a plain directory, not the Hugging Face cache: onnxruntime refuses external weight files that are
        # symlinks into another folder, which is how the cache stores the fp32 encoder's .data file
        local = cache_dir() / (self.model_id.replace("/", "--") + (f"-{quantization}" if quantization else ""))
        self.model = onnx_asr.load_model(self.model_id, local, quantization=quantization, providers=providers(device))

    def _transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        if len(audio_f32_16k) < MIC_SR // 10:  # <100 ms: nothing to say
            return "", 0.0
        step = MAX_PASS_S * MIC_SR
        parts = [self.model.recognize(audio_f32_16k[i:i + step].astype(np.float32), sample_rate=MIC_SR)
                 for i in range(0, len(audio_f32_16k), step)]
        text = " ".join(p.strip() for p in parts if p and p.strip())
        return text, (time.perf_counter() - t0) * 1000

    async def transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        return await run_local(self._transcribe, audio_f32_16k)

    async def stream(self) -> WindowedStream:
        return WindowedStream(self)


class KokoroONNX:
    where = "onnx"
    sample_rate = 24_000

    def __init__(self, variant: str = "fp32", device: str = "cpu", voices: list[str] | None = None):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        model = download(f"{KOKORO_RELEASE}/{KOKORO_FILES[variant]}", cache_dir() / KOKORO_FILES[variant])
        voices_bin = download(f"{KOKORO_RELEASE}/{KOKORO_VOICES_FILE}", cache_dir() / KOKORO_VOICES_FILE)
        session = ort.InferenceSession(str(model), providers=providers(device))
        self.kokoro = Kokoro.from_session(session, str(voices_bin))
        self.model_id = f"Kokoro-82M ({variant})"
        available = set(self.kokoro.get_voices())
        self.voices = voices or [v for v in PREFERRED_VOICES if v in available] or sorted(available)

    def _synth(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16), self.sample_rate, 0.0
        audio, sr = self.kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        return to_int16(audio), int(sr), (time.perf_counter() - t0) * 1000

    async def synth(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int, float]:
        return await run_local(self._synth, text, voice, speed)
