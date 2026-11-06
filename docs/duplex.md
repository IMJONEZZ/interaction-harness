# Full-duplex mode

The other three modes are a cascade: the harness decides when you've finished, transcribes you, asks a
model, and speaks its answer. In **full-duplex** mode one speech-to-speech model listens and talks at the
same time, every 80 ms, and decides for itself when to speak, when to stop, and when to talk over you.

You can switch between the modes mid-conversation, on the same microphone and dashboard, and compare.

## Run it

Full-duplex runs on Apple Silicon today, with NVIDIA's
[NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) through
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm), in a 4-bit build of about 9 GB. It is the first open
full-duplex model that calls tools.

```bash
uv tool install "interaction-harness[duplex]"
interaction-harness duplex-server                 # downloads the model once; ws://127.0.0.1:8770/v1/realtime
interaction-harness serve --duplex-base-url ws://127.0.0.1:8770/v1 --open
```

Then pick **Full-duplex** in the toolbar. Wear headphones: the model hears whatever reaches the microphone,
its own voice included.

`DUPLEX` is a fifth, optional role, with the same three settings as the others:

| Setting | Meaning |
|---|---|
| `DUPLEX_BASE_URL` | the server's base, e.g. `ws://127.0.0.1:8770/v1` (`http://` works too). The harness opens `{base}/realtime` |
| `DUPLEX_MODEL` | blank: the one model the server lists, else `mlx-community/NemotronLabs-VoiceChat-11B-4bit` |
| `DUPLEX_API_KEY` | sent as `Authorization: Bearer …` if set |
| `DUPLEX_PROMPT` | the system prompt (see `prompts.py`); keep it short, see below |

The foreground model still has to be configured: it shortens background results for the duplex model (below).

## What the harness does in this mode

ASR, the foreground model, TTS and the end-of-turn judge are out of the loop. The harness:

- streams your microphone to the model in 80 ms blocks, and the model's audio to the browser, which plays
  it gaplessly behind a 120 ms jitter buffer
- shows what the model heard (its own streaming transcript) and what it said (its text channel)
- runs the model's tool calls and feeds the results back (below)
- still runs Silero VAD on your side, to *measure* the model, not to steer it

The latency chart gets a ⇄ row per utterance:

- **first audio**: from the end of your speech to the first audio of the model's answer, as you hear it. If
  the model was already talking when you stopped, the row says *overlap*.
- **yield**: when you start talking while the model is talking, how long until it goes quiet. If it
  keeps going for 4 s, the Brain panel says it kept talking over you.

## Tools

The model calls tools on its own output channel, in the format NVIDIA trained it on:

```
<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Boston"}}]</TOOLCALL>
```

The harness runs the call and sends the result back. The model reads the result as
`<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` tokens fed into that same channel, one per 80 ms step, starting right
after it closes the call. That is how NVIDIA's reference code does it. `interaction-harness duplex-server`
adds this to mlx-vlm's realtime server as one extra client event:

```json
{"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "{\"spoken\": \"…\"}"}}
```

mlx-vlm's own `mlx_vlm.server` speaks the same protocol without that event. The harness works with it, but
the model never hears a tool result, so it guesses.

Fed naively, results made the model unreliable: it chained calls nobody asked for (the weather in Tokyo,
then London) and never answered. The server keeps the exchange as close to NVIDIA's training data as it can,
and `duplex-server --log-functions` prints the channel frame by frame if you want to see why:

- **The result comes right after the call.** If a tool takes time (the weather API, about a second), the
  model is held still until the result arrives, at most 2 s, rather than stepped with nothing to read. Mic
  frames during the hold are dropped.
- **The result is closed** with the end-of-result token the model was trained on.
- **It speaks before it calls again.** For 2 s after a result, a new start-of-call token is replaced with a
  pad, which leaves the model its turn to say the answer.
- **Arguments have to be things you said.** NVIDIA's prompt says so, and the harness enforces it for short
  values like a city or a label: a call for the weather in a city nobody mentioned is not run, and the model
  gets an error saying why. The dashboard shows the blocked call.

Because every token of a result costs 80 ms, results are kept short:

- A tool can return a `spoken` field, and the duplex model gets only that line. `get_time` and
  `get_weather` do.
- Anything else is sent as compact JSON, with long values cut, at most about 220 characters.
- A background task's answer is shortened to one sentence by the foreground model before it is sent.

Tools are opt-in for this mode, with `lanes=["duplex"]`, because each one makes the prompt longer. Four are
in by default: `get_time`, `get_weather`, `set_timer` and `delegate_background_task`. See
[tools.md](tools.md).

**Timers and the background model** work as in the other modes, except for delivery. When a timer fires
or a background task finishes, and you and the model are both quiet, the harness sends the result as the
answer to the call that started it: the server feeds that call in first (`"call": "[{...}]"` on the same
event), then the result, the pattern the model was trained on, and the model tells you. What you and the
model say in this mode also goes into the conversation history, so the background model gets the context,
and switching back to a cascade mode picks the conversation up.

## The prompt takes seconds to load

The default prompt is NVIDIA's own system message for this model's tool calling (from their reference code,
Apache-2.0), plus the four tools. Shorter prompts load faster, but with them the model called tools nobody
asked for. Set `DUPLEX_PROMPT` to change it.

The model reads its system prompt one token per step, about 40 ms each, so the default takes about 35 s. So:

- The harness opens the duplex session as soon as the dashboard connects, in the background. The pill
  shows *loading its prompt…*, then *ready*.
- After a client disconnects, `duplex-server` reads the same prompt into a fresh session, so a reload or
  reconnect is instant.

## Measured

On an M5 Max MacBook Pro, plugged in, with `selftest --duplex`. The user's lines are synthesized speech
streamed in real time.

| | Result |
|---|---|
| Model speed | 59 ms per 80 ms frame (1.3× real time) |
| First audio, plain question ("What's a good name for a goldfish?") | 670–940 ms |
| First audio, tool question ("What time is it right now?") | 5.5–5.7 s: the call, then the result tokens, then the answer |
| Yield after a cut-in | 0.3–2 s |

Over the same scripted conversation, run after run, it answered the time and the weather from the tool
results, set a reminder ("Reminder set. You will stretch in fifteen seconds.") and announced it when it
fired. It handed the mortgage question to the background model in some runs and not in others, and it
sometimes starts answering before you finish.

After a tool result it nearly always tries to call another tool at once: in 15 of the 17 results in our
logs, the very next step was a start-of-call token. The harness holds that off for 2 s, which leaves the
model its turn to say the answer, but it doesn't always recover. In the demo recording, after setting the
timer it went on about cities nobody had mentioned (the recording is cut there). Scripted runs haven't
reproduced that: with synthesized voices, with and without the real room noise from that recording, it
answered the time, the timer and the weather cleanly every time. So far the rambling has only shown up in
live conversation.

Tool descriptions leave out examples, because this model repeats them: with "such as 'remind me in 30
seconds to stretch'" in `set_timer`'s description, it offered to set that reminder when it was only asked
the time.

The cascade's median on the same machine is about 0.8–0.9 s (see the README), so on plain questions the
duplex model answers faster. A tool turn is slower, because the call and its result travel through the
model's 80 ms steps.

## Limits

- **Apple Silicon only, for now.** NVIDIA also ships the model as a NIM container and through NeMo for
  NVIDIA GPUs. Neither is wired in or tested here.
- **One session at a time** per server, and one built-in voice (Aria). English only.
- **No echo cancellation in the model.** Use headphones, or it hears itself.
- **Typed turns don't work** in this mode: the model only hears audio.
- **The model's own judgment.** It can start talking before you finish, and it picks tools less reliably
  than a large text model (NVIDIA reports 82.5% tool-selection F1 on Full-Duplex-Bench 3.0, and says
  end-to-end tool execution needs improvement). The dashboard shows what it did, which is the point.
- **License.** The weights are under the OpenMDW License Agreement 1.1 (see the
  [model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)). Nothing here ships them;
  `duplex-server` downloads them from Hugging Face.
- **mlx-vlm is pinned.** `duplex-server` writes tool tokens into a private field of mlx-vlm's session, so the
  `duplex` extra pins mlx-vlm to the version it was tested with.

## Protocol

`{base}/realtime` is a WebSocket with OpenAI Realtime-style event names, as mlx-vlm defines them:

| Direction | `type` | Fields |
|---|---|---|
| → | `session.update` | `session: {model, system_prompt, seed}` (once, first) |
| → | `input_audio_buffer.append` | `audio`: base64 PCM16, 16 kHz mono |
| → | `conversation.item.create` | `item: {type: "function_call_output", output}` (duplex-server only) |
| ← | `session.updated` | `session: {frame_samples, input_audio_format, output_audio_format, tool_outputs, warm}` |
| ← | `response.audio.delta` | `delta`: base64 PCM16, `sample_rate` (22,050), one per 80 ms frame |
| ← | `response.text.delta` | what the model is saying |
| ← | `response.function.delta` | its tool-call channel |
| ← | `conversation.item.input_audio_transcription.delta` | what it heard you say |
| ← | `error` | `error: {code, message}`; `server_busy` when another session is open |
