"""In-process speech on Apple Silicon (MLX): Parakeet TDT 0.6B v3 for ASR, Kokoro-82M for TTS.

MLX streams belong to the thread that created them, so construct these on the speech worker
(`await run_local(ParakeetMLX)`), as build_speech does; every later call already runs there.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import shutil
import sys
import time
import zipfile

import numpy as np

from .base import MIC_SR, PREFERRED_VOICES, cache_dir, download, run_local, to_int16

DEFAULT_ASR = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_TTS = "prince-canuma/Kokoro-82M"
# Kokoro's English phonemizer (misaki) needs this spaCy model
SPACY_MODEL = "en_core_web_sm"
SPACY_WHEEL = ("https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/"
               "en_core_web_sm-3.8.0-py3-none-any.whl")
SPACY_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"


def ensure_spacy_model(url: str = SPACY_WHEEL, sha256: str = SPACY_SHA256) -> None:
    """Make en_core_web_sm importable before misaki looks for it.

    misaki pip-installs the model on first use, which fails wherever there is no pip (a `uv tool` install),
    and a PyPI package can't depend on it directly. So fetch the wheel once, check its hash, unpack it into
    the cache and put that folder on sys.path; the wheel's .dist-info makes spaCy see it as installed.
    """
    if importlib.util.find_spec(SPACY_MODEL) is not None:
        return
    folder = cache_dir() / "spacy"
    folder.mkdir(exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    target = folder / name.removesuffix(".whl")
    if not target.exists():
        wheel = download(url, folder / name)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != sha256:
            wheel.unlink()
            raise RuntimeError(f"{name}: sha256 {digest} does not match the expected {sha256}")
        tmp = target.with_name(target.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(wheel) as z:
            z.extractall(tmp)
        tmp.rename(target)
        wheel.unlink()
    if str(target) not in sys.path:
        sys.path.append(str(target))
    importlib.invalidate_caches()


class ParakeetMLX:
    """Parakeet via parakeet-mlx. Batch transcription plus a native streaming session for partials."""

    where = "mlx"

    def __init__(self, model_id: str | None = None):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        self._mx = mx
        self._get_logmel = get_logmel
        self.model_id = model_id or DEFAULT_ASR
        self.model = from_pretrained(self.model_id)

    def _transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        if len(audio_f32_16k) < MIC_SR // 10:  # <100 ms: nothing to say
            return "", 0.0
        mel = self._get_logmel(self._mx.array(audio_f32_16k), self.model.preprocessor_config)
        result = self.model.generate(mel)[0]
        self._mx.clear_cache()  # release MLX's buffer cache so a long session doesn't hold peak memory
        return result.text.strip(), (time.perf_counter() - t0) * 1000

    async def transcribe(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        return await run_local(self._transcribe, audio_f32_16k)

    async def stream(self) -> _ParakeetStream:
        return await run_local(_ParakeetStream, self)


class _ParakeetStream:
    def __init__(self, asr: ParakeetMLX, context_size=(256, 256), depth=1):
        self.asr = asr
        self._ctx = asr.model.transcribe_stream(context_size=context_size, depth=depth)
        self._session = self._ctx.__enter__()
        self.text = ""

    def _feed(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        self._session.add_audio(self.asr._mx.array(audio_f32_16k))
        self.text = self._session.result.text.strip()
        return self.text, (time.perf_counter() - t0) * 1000

    async def feed(self, audio_f32_16k: np.ndarray) -> tuple[str, float]:
        return await run_local(self._feed, audio_f32_16k)

    def _close(self):
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    async def close(self) -> None:
        await run_local(self._close)


class KokoroMLX:
    """Kokoro-82M via mlx-audio. One call per sentence -> 24 kHz int16."""

    where = "mlx"
    sample_rate = 24_000

    def __init__(self, model_id: str | None = None, voices: list[str] | None = None):
        logging.getLogger("phonemizer").setLevel(logging.ERROR)  # "words count mismatch" on every call otherwise
        ensure_spacy_model()
        from mlx_audio.tts.utils import load_model
        self.model_id = model_id or DEFAULT_TTS
        self.model = load_model(self.model_id)
        self.voices = voices or list(PREFERRED_VOICES)

    def _synth(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16), self.sample_rate, 0.0
        parts = [np.array(res.audio, dtype=np.float32)
                 for res in self.model.generate(text, voice=voice, speed=speed, lang_code="a",
                                                split_pattern=r"\n+")]
        if not parts:
            return np.zeros(0, dtype=np.int16), self.sample_rate, 0.0
        return to_int16(np.concatenate(parts)), self.sample_rate, (time.perf_counter() - t0) * 1000

    async def synth(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int, float]:
        return await run_local(self._synth, text, voice, speed)
