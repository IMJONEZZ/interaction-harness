# In-process speech: MLX and ONNX

When `ASR_BASE_URL` or `TTS_BASE_URL` is blank (or `local`), that role runs inside the harness. Both
backends run the same two models:

| | MLX (`BACKEND=mlx`) | ONNX (`BACKEND=onnx`) |
|---|---|---|
| Where | Apple Silicon Macs | anywhere onnxruntime runs: Linux, Windows, Intel Macs, Apple Silicon |
| ASR | Parakeet TDT 0.6B v3 via [parakeet-mlx](https://github.com/senstella/parakeet-mlx), native streaming partials | Parakeet TDT 0.6B v3 via [onnx-asr](https://github.com/istupakov/onnx-asr); partials re-transcribe the turn so far |
| TTS | Kokoro-82M via [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Kokoro-82M via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) |
| Weights | Hugging Face cache | `~/.cache/interaction-harness` (or `$XDG_CACHE_HOME`) |

`BACKEND=auto` (the default) picks MLX on Apple Silicon and ONNX everywhere else. You can also force
ONNX on a Mac to compare the two.

## ONNX model variants

| Setting | Values | Default | Download |
|---|---|---|---|
| `PARAKEET_ONNX_VARIANT` | `int8`, `fp32` | `int8` | 639 MB (fp32: 2.4 GB) |
| `KOKORO_ONNX_VARIANT` | `int8`, `fp16`, `fp32` | `int8` | 97 MB + 27 MB of voices (fp32: 321 MB) |

On an M5 Max CPU, the int8 variants gave the same transcripts as fp32 in our checks:

| Model | Audio | fp32 | int8 |
|---|---|---|---|
| Parakeet | 4.3 s | 115 ms | 100 ms |
| Kokoro | 2.7 s | 545 ms | 231 ms |

On small CPUs, TTS is what limits latency. On GitHub's standard Linux runner (x86_64), Kokoro int8 took 3.9 s
to synthesize 5.6 s of audio (RTF 0.69). That is faster than real time, but slow to the first word. On a
machine like that, point `TTS_BASE_URL` at a GPU-backed speech server.

## GPUs with ONNX

CPU is the default and the tested path. The GPU path is **untested**. To try it:

1. Install the `onnx-gpu` extra.
2. Make sure only `onnxruntime-gpu` is left in the environment. It and `onnxruntime` both install a module
   named `onnxruntime`, so uninstall the CPU package.
3. Pass `--device cuda`.

The harness stops with a clear message if onnxruntime has no CUDA provider. On a GPU box, an endpoint is
often simpler: point `ASR_BASE_URL` / `TTS_BASE_URL` at a speech server that already uses the GPU.

## Voice activity detection

Silero VAD always runs in-process. Its ONNX model ships inside the package (MIT, see
`src/interaction_harness/speech/assets/SILERO_VAD_LICENSE`). It runs on onnxruntime with a small numpy
wrapper that mirrors silero-vad's own, so neither backend needs torch for VAD. The unit tests check the
wrapper against probabilities from silero-vad's reference implementation.

## Writing another backend

A backend is two small classes; see `speech/base.py` for the Protocols:

- **ASR** has `model_id`, `where`, `async transcribe(audio_f32_16k) -> (text, ms)`, and
  `async stream() -> ASRStream`. If your model has no streaming mode, `WindowedStream(self)` gives you
  partials.
- **TTS** has `model_id`, `where`, `voices`, and
  `async synth(text, voice, speed) -> (int16 pcm, sample_rate, ms)`.

Blocking model calls should go through `run_local` (one worker thread), so the event loop stays free and
two models never fight over the GPU. Add the new backend to `load_asr` / `load_tts` in
`speech/__init__.py`.

## Troubleshooting

- **`Error processing file '…/phontab'`.** espeak-ng, which Kokoro uses to turn text into phonemes, can't
  use a data path longer than about 160 characters. Install somewhere with a shorter path; the default
  `uv tool install` location is about 120.
- **First run is slow.** In-process models download on first use. Run `interaction-harness warmup` once
  ahead of time.
