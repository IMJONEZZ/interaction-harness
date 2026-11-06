# Endpoints: the model roles

Every model the harness uses is a *role*, and every role is an OpenAI-compatible endpoint:

| Role | Calls | Used for |
|---|---|---|
| `FOREGROUND` | `POST {base}/chat/completions` (streaming, tools) | every reply, quick tool calls, hand-offs, and the end-of-turn judge (3 tokens, thinking off) |
| `BACKGROUND` | `POST {base}/chat/completions` (streaming, tools, thinking on) | delegated tasks; up to `BACKGROUND_ROUNDS` model calls each, with web search |
| `ASR` | `POST {base}/audio/transcriptions` (multipart: 16 kHz WAV, `model`, `response_format=json`, optional `language`) | the final transcript of each turn, and partial transcripts while you talk |
| `TTS` | `POST {base}/audio/speech` (JSON: `model`, `input`, `voice`, `speed`, `response_format`) | one request per sentence chunk |
| `DUPLEX` (optional) | WebSocket `{base}/realtime`, OpenAI Realtime-style events | full-duplex mode: one speech-to-speech model in place of ASR, foreground and TTS. See [duplex.md](duplex.md) |

`{base}` is the role's `<ROLE>_BASE_URL`, and includes the `/v1`.

## Settings for each role

| Setting | Meaning |
|---|---|
| `<ROLE>_BASE_URL` | the endpoint. For ASR and TTS, blank or `local` means in-process (see [backends.md](backends.md)) |
| `<ROLE>_API_KEY` | sent as `Authorization: Bearer …`. Blank means `not-needed` |
| `<ROLE>_MODEL` | the `model` field. For the chat roles, blank means the first model the server lists at `GET {base}/models` |

Fallbacks:

- `FOREGROUND_*` falls back to `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
- `BACKGROUND_*` falls back to the foreground field by field. Set only `BACKGROUND_MODEL` to use a different
  model on the same server.
- **Keys never cross hosts.** A role without its own key reuses the foreground key only when it points at
  the same scheme, host and port.

## Switching thinking

The foreground runs with thinking off, unless you tick "foreground thinking" in the dashboard. The
background runs with thinking on. Servers differ in how that is asked for, so each chat role has a
setting:

| `*_THINKING` | Sends | Servers |
|---|---|---|
| `chat_template_kwargs` (default) | `{"chat_template_kwargs": {"enable_thinking": true/false}}` | vLLM, llama.cpp, SGLang serving Qwen3-, GLM- or DeepSeek-style templates |
| `reasoning_effort` | `"reasoning_effort": "high"` on, `"low"` off | OpenAI-style reasoning models |
| `off` | nothing | servers that reject unknown fields, or models without a switch |

Reasoning is read from `reasoning_content` (llama.cpp) or `reasoning` (vLLM) in the stream, and shown in
the dashboard.

## ASR endpoints

- **Partial transcripts.** The transcriptions API takes a whole file, so the harness re-posts the turn so
  far every `ASR_PARTIAL_MS` (640 by default), using at most the last 20 s. That is one request per
  partial: a 5-second turn costs about 8 requests plus the final one. `ASR_PARTIAL_MS=0` turns partials
  off; the end-of-turn judge then asks for a transcript only when it needs one.
- **Responses.** The response should be `{"text": "..."}`. A server that answers in plain text is
  understood too.

## TTS endpoints

- **Formats.** `TTS_FORMAT=wav` (default) is decoded at whatever sample rate the server uses. `pcm`
  means raw 16-bit mono at `TTS_SAMPLE_RATE` (24000 by default, OpenAI's rate).
- **Voices.** The voice list comes from `GET {base}/audio/voices?model=…` when the server has it
  (mlx-audio and Kokoro-FastAPI do). Otherwise set `TTS_VOICE`, and optionally `TTS_VOICES` for the
  dashboard's picker.

## Server notes

These are tested here:

- **llama.cpp / Unsloth Studio.** Start `llama-server` with `--jinja` so tool calls reach the model, and
  with enough parallel slots (`-np 4`) for the foreground, the judge and the background at once.
- **vLLM.** Run with a tool-call parser and a reasoning parser for your model.
- **mlx-audio server.** Install it with `pip install "mlx-audio[server]" "misaki[en]"`; Kokoro needs misaki.
  Run `python -m mlx_audio.server --port 8790`, then use `ASR_MODEL=mlx-community/parakeet-tdt-0.6b-v3` and
  `TTS_MODEL=prince-canuma/Kokoro-82M`.

Ollama and LM Studio are built on llama.cpp, so the llama.cpp results above cover them. Anything else,
including OpenAI, Speaches and Kokoro-FastAPI, is untested here. Run `interaction-harness doctor` against
it, which checks each role on its own, and please file a
[server report](https://github.com/IMJONEZZ/interaction-harness/issues/new?template=server-report.yml)
with what you find.
