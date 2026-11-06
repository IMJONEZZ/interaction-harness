# The WebSocket protocol

The dashboard is one client of `ws://<host>:<port>/ws`. Anything that speaks this protocol can drive a
session: another UI, a phone bridge, a test harness (`selftest.py` is one). Each connection is its own
session, with its own state machine, history and VAD.

## Client → server

**Binary frames:** microphone audio as 16 kHz mono int16 PCM, any chunk size. The server cuts it into
512-sample (32 ms) frames.

**JSON messages:**

| `type` | Fields | Effect |
|---|---|---|
| `config` | any field of `Config` in `session.py` | change the interaction model live (see below) |
| `mic` | `on: bool` | start or stop listening |
| `ptt` | `down: bool` | push-to-talk press and release (push-to-talk mode only) |
| `text` | `text: str` | a typed user turn |
| `playback` | `event: "start" \| "end" \| "drained"`, `id` | report that a TTS chunk started or finished playing, or that the queue is empty. Turns end, and barge-in decides what was heard, from these |
| `playback` | `event: "voice"`, `on: bool` | full-duplex mode: the model's stream became audible or went quiet, at the moment you hear it |
| `stop` | | interrupt the agent (same as barge-in) |
| `reset` | | clear the history and notes |
| `ping` | | the server answers `pong` |

`Config` fields:

| Field | Default | Meaning |
|---|---|---|
| `mode` | `semantic` | turn-taking: `ptt`, `vad`, `semantic`, or `duplex` (needs `DUPLEX_BASE_URL`; see [duplex.md](duplex.md)) |
| `barge_in` | `true` | interrupt the agent when you talk |
| `filler` | `true` | say a cached filler line if the reply is slow |
| `foreground_thinking` | `false` | turn thinking on for the foreground model |
| `voice` | | TTS voice |
| `speed` | `1.05` | TTS speed |
| `silence_ms` | `700` | VAD mode: silence that ends a turn |
| `sem_min_ms` | `300` | semantic mode: silence before the judge is first asked |
| `sem_max_ms` | `2200` | semantic mode: silence that ends a turn whatever the judge says |
| `onset_thresh` | `0.5` | VAD probability that counts as speech |
| `onset_frames` | `3` | frames of speech that open a user turn |
| `barge_thresh` | `0.7` | VAD probability that counts toward barge-in |
| `barge_frames` | `6` | frames needed to barge in (about 190 ms) |

## Server → client

**Binary frames:** TTS audio, `<u32 chunk_id><u32 sample_rate><int16 mono PCM>`, little-endian. The
sample rate comes with every chunk, because endpoint TTS servers differ. Play the chunks in order, and
send `playback` start/end for each.

In full-duplex mode the model's audio arrives as one continuous stream of 80 ms frames, with chunk id
`0xFFFFFFFE` (silent) or `0xFFFFFFFF` (voiced). Play them back to back, without `start`/`end`; send
`playback` `voice` when playback goes from silent to voiced or back.

**JSON events:** every event has `type` and `t` (milliseconds since the epoch).

| `type` | Fields | When |
|---|---|---|
| `hello` | `version`, `roles{asr,tts,foreground,background,duplex?: {model, where}}`, `vad`, `voices`, `voice`, `tools` | on connect (`duplex` only when configured) |
| `config` | the full `Config` | after every `config` message |
| `state` | `state`: `idle`, `listen`, `user_speaking`, `endpointing`, `think` or `speak` | on every transition |
| `vad` | `p`, `rms` | every other 32 ms frame |
| `user.start` | | the user's turn opens |
| `asr.partial` | `text`, `ms`, `audio_s` | partial transcript |
| `endpoint` | `phase`: `silence`, `judge` or `resumed`; `verdict`, `text`, `ms`, `silence_ms` | each end-of-turn decision |
| `user.end` | `reason`, `audio_s`, `eot_ms`, `turn` | the user's turn is over |
| `asr.final` | `text`, `ms`, `audio_s`, `turn`, `model` | final transcript (`model: "typed"` for typed turns, `"duplex"` for what the duplex model heard) |
| `turn.start` | `turn`, `source` (`voice`, `typed`, `proactive` or `duplex`), `lane`, `thinking` | the agent's turn begins |
| `llm.first_token` | `ms`, `round` | foreground time to first token |
| `llm.reasoning` / `llm.content` | `delta`, `lane` | streamed foreground tokens |
| `llm.tool_call` / `llm.tool_result` | `id`, `name`, `arguments` / `result`, `ms` | foreground tools |
| `llm.done` | `round`, `ms`, `tokens`, `tps`, `finish` | one foreground model call finished |
| `tts.queued` | `text`, `turn` | a sentence is waiting for TTS |
| `tts.chunk` | `id`, `text`, `synth_ms`, `audio_s`, `rtf`, `voice`, `kind` (`reply` or `filler`) | audio follows as a binary frame |
| `playback.stop` | | stop playback now (barge-in) |
| `interrupt` | `source`, `spoken`, `dropped`, `turn` | what was heard and what was dropped |
| `metrics` | `turn`, `source`, `marks{eot, asr, llm_first_token, first_sentence, tts_first, first_audio, turn_end, …}`, `partial` | the turn's latency marks, in ms from the end of speech |
| `turn.end` | `turn` | every chunk has played |
| `timer.set` / `timer.fired` | `id`, `label`, `seconds` | timers |
| `background.queued` / `background.start` | `id`, `task` | a delegated task (one runs at a time) |
| `background.reasoning` / `background.content` | `id`, `delta` | streamed background tokens |
| `background.tool_call` / `background.tool_result` | `id`, `call_id`, `name`, `arguments` / `result`, `ms` | background tools |
| `background.round` | `id`, `round`, `tokens`, `tps`, `ms` | one background model call finished |
| `background.done` | `id`, `result`, `ms` | the result, waiting for a gap |
| `proactive` | `kind` (`timer` or `background`), `id`, … | the agent takes its own turn to deliver it |
| `duplex.status` | `state` (`loading`, `waiting` while the server has another session open, `ready`, `error`, `closed`), `model`, `load_ms`, `tool_outputs`, `error` | the full-duplex session |
| `duplex.utterance` | `turn`, `text` | the duplex model finished saying something |
| `duplex.overlap` | `who` (`you` or `model`), `text` | someone started talking over the other |
| `duplex.yield` | `ms`, `kept_talking`, `turn` | how long the model kept talking after you cut in |
| `duplex.tool_output` | `output` | a tool result (or a timer or background result) sent to the duplex model |
| `log` | `level`, `text` | anything worth showing |
| `pong` | | answer to `ping` |

`GET /api/status` returns the same body as `hello`.
