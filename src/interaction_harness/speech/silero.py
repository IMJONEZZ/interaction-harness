"""Silero VAD on onnxruntime + numpy, so voice detection needs no torch on any platform.

The model file ships with this package (MIT, see assets/SILERO_VAD_LICENSE). This mirrors silero-vad's
own OnnxWrapper at 16 kHz: each 512-sample frame is prefixed with the previous frame's last 64 samples,
and the recurrent state carries across calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import MIC_SR, VAD_FRAME

MODEL = Path(__file__).parent / "assets" / "silero_vad.onnx"
CONTEXT = 64
_session = None


def _shared_session():
    # one inference session for every connection; each SileroVAD keeps its own state
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


class SileroVAD:
    """Stateful: feed consecutive 512-sample int16 frames at 16 kHz."""

    name = "silero-vad (onnx, in-process)"

    def __init__(self):
        self.session = _shared_session()
        self._sr = np.array(MIC_SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame_int16: np.ndarray) -> float:
        if len(frame_int16) != VAD_FRAME:
            raise ValueError(f"VAD frames must be {VAD_FRAME} samples, got {len(frame_int16)}")
        x = (frame_int16.astype(np.float32) / 32768.0)[None, :]
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT:]
        return float(out[0][0])
