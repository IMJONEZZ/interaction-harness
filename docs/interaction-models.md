# Interaction models: background reading

Most of what makes a voice conversation feel alive isn't the words. It's the **timing**:

- knowing when you've finished talking
- letting you cut in
- being able to say "I'll get back to you" and actually doing it

Those decisions are the *interaction model*. This harness makes each of them visible, measurable, and
switchable.

## What "interaction model" means

The phrase is used three ways. This harness is about the second and third.

| Sense | Who uses it | Meaning |
|---|---|---|
| Voice-interface schema | Amazon Alexa Skills Kit | Intents, sample utterances and slots: the *vocabulary* a skill accepts ([docs](https://developer.amazon.com/en-US/docs/alexa/ask-overviews/voice-interaction-models.html)) |
| **Model class** | Thinking Machines Lab, May 2026 | Models that "handle interaction natively rather than through external scaffolding". They take in continuous audio, video and text, respond in real time, and work in **micro-turns** that interleave 200 ms of input with 200 ms of output. They delegate to an asynchronous **background model** whose results are "interleaved into the conversation at a moment appropriate to what the user is currently doing" ([blog](https://thinkingmachines.ai/blog/interaction-models/)) |
| **Turn-taking regime** | LiveKit, Pipecat, OpenAI, Google, Deepgram, and the June 2026 survey by Lu et al. | *How* human and agent trade the floor (push-to-talk, VAD, semantic endpointing, full-duplex), plus policies for barge-in, backchannels and fillers ([survey](https://arxiv.org/html/2606.19453v1)) |

The survey formalizes the third sense as an ontology of **temporal relation × user intent × system
response**, governed by a five-state machine (Idle, Listen, Speak, Wait, Dual). The options practitioners
actually configure:

| Interaction model | Mechanism | What it costs | Who ships it |
|---|---|---|---|
| **Push-to-talk** (half-duplex) | the client commits the turn | no endpointing latency, no false triggers, no barge-in | OpenAI Realtime `turn_detection: null`; LiveKit `manual`; Gemini Live `activityStart/End` |
| **VAD silence timeout** | Silero VAD plus a fixed silence | adds the whole timeout (700 ms of timeout ≈ +700 ms) | LiveKit `vad`; OpenAI `server_vad` (`silence_duration_ms`) |
| **STT endpointing** | the ASR emits end-of-utterance | faster than raw silence | LiveKit `stt`; Deepgram Flux `EndOfTurn` (`eot_threshold`), with eager EOT for a speculative LLM start |
| **Semantic end-of-turn** | a model reads the transcript (and/or audio) and predicts "done?" | 12–160 ms per judgement; can end a turn after about 300 ms of silence | LiveKit turn detector (0.1B distilled Qwen2.5, `min_delay` 0.3 s / `max_delay` 2.5 s); Pipecat Smart Turn v3 (12.6 ms on CPU); OpenAI `semantic_vad` (`eagerness`); Kyutai Unmute |
| **Full-duplex** | the model listens and speaks at once, every frame | about 160–400 ms turn latency; needs a native model | Moshi (Kyutai), TML-Interaction-Small (0.40 s on FD-Bench), Gemini native audio, NemotronLabs VoiceChat 11B (open, with tool calls; this harness's `duplex` mode) |
| **Barge-in** | cancel playback, truncate context to what was heard | must tell a backchannel ("uh-huh") from a claim on the floor | OpenAI `conversation.item.truncate`; LiveKit `interruption.min_duration/min_words`; Vapi `stopSpeakingPlan` |
| **Filler / "thinking out loud"** | say something while the reasoner works | hides time to first token; can sound fake if overused | ConvFill (talker plus streamed reasoner); LiveKit `preemptive_generation` |

**Latency budgets people cite.** Humans hand over the floor with a gap of about 200 ms. Voice agents
target 500–800 ms from the end of speech to first audio; ElevenLabs cites P50 < 800 ms and P95 < 1.5 s.

A well-tuned cascade spends:

- about 150–300 ms on turn detection
- 150–400 ms on the LLM's time to first token
- 100–200 ms on TTS

Untuned production cascades commonly run 1.5–3 s. The waterfall in the dashboard measures exactly these
segments, turn by turn.

**Cascades versus speech-to-speech.** The harness runs both: the three cascade modes, and a full-duplex
mode with a native model ([duplex.md](duplex.md)), on the same microphone and dashboard.

| | Cascade (ptt, vad, semantic) | Speech-to-speech / full-duplex model |
|---|---|---|
| Swap a component | yes, any one | no; you can't swap the brain |
| Model | any LLM | the native model |
| Tools | easy to call | few models can; VoiceChat 11B calls them, and reads each result one token per 80 ms |
| Paralinguistics (prosody, tone) | lost | kept |
| Latency | a serial chain of steps | saves 100–200 ms |
| Backchannel or interject | no | yes |

The Thinking Machines post argues that VAD-style harnesses "are meaningfully less intelligent than the
model itself", and that's fair. It's why semantic mode puts the *LLM itself* in the end-of-turn decision,
rather than relying on a silence timer alone.

## How the harness implements it

```
              mic (16 kHz int16, 32 ms frames)                      speakers
                     │                                                  ▲
                     ▼                                                  │  binary frames
 ┌────────── Silero VAD ──────────┐                     ┌────── TTS (sentence chunks) ──────┐
 │ onset · silence · barge-in     │                     │ synth ms · RTF · queue · dropped   │
 └───────────────┬────────────────┘                     └─────────────────▲──────────────────┘
                 ▼                                                        │
    ┌────── ASR ───────┐     ┌── end-of-turn ──┐     ┌──── foreground model (thinking off) ────┐
    │ partials + final │ ──▶ │ ptt | vad | LLM │ ──▶ │ streams tokens · calls tools · delegates │
    └──────────────────┘     │ judge (semantic)│     └──────────┬───────────────────▲───────────┘
                             └─────────────────┘                │ delegate          │ background event
                                                                ▼                   │
                                                 ┌──── background model (thinking on) ────┐
                                                 │ conversation so far · tools · reasoning │
                                                 │ result → proactive queue → spoken in a gap │
                                                 └──────────────────────────────────────────┘
```

**The state machine** (`session.py`) runs `idle → listen → user_speaking → endpointing → think → speak → listen`.
On barge-in it goes from `speak` or `think` back to `user_speaking`; if you start talking again during a
pause, it goes from `endpointing` back to `user_speaking`.

**Turn-taking modes** (toolbar):

- **Push-to-talk.** Hold the button or the space bar. Half-duplex: the browser commits the turn.
- **VAD timeout.** The turn ends after `silence` ms (default 700) without speech.
- **Semantic EOT.**
  - After only 300 ms of silence, the partial transcript goes to the foreground model (thinking off,
    3 tokens) with the question "COMPLETE or INCOMPLETE?".
  - `COMPLETE` ends the turn at once. `INCOMPLETE` waits and asks again as the transcript grows. A 2.2 s
    ceiling ends the turn regardless.
  - A trailing "and…" or "so…", an unfinished list, or half a question keeps the floor with you.

**Barge-in.**

- *What triggers it:* while the agent is thinking or speaking, about 190 ms of confident speech (6 frames
  at ≥ 0.7).
- *What it stops:* the model stream, the TTS queue, and playback in the browser.
- *What history keeps:* only what you actually heard. Your next turn carries a short note that the reply
  was cut off, so the model neither repeats itself nor quotes what you never heard.
- *A filter:* one also strips bracketed stage directions from anything the model says, so no marker is
  ever spoken.
- Turn barge-in off if you're on speakers without echo cancellation.

**Filler.** If the first sentence isn't ready 700 ms after your turn ends, a cached filler line plays
("Let me think about that."). It's marked as filler in the Mouth panel and left out of the
time-to-first-audio figure.

**The foreground and background models.** They can be the same model on the same server, called two
ways (thinking off versus on), or two different endpoints.

- *Foreground.* Keeps replies to one to three sentences and has the instant tools plus `set_timer` and
  `delegate_background_task`.
- *Background.* Gets the instant tools plus `web_search` and `fetch_page`.
- *Coming back later.* The two async tools return at once and re-enter the conversation through the
  **proactive channel**. A loop waits until you're listening and have been quiet for a second, then gives
  the agent its own turn. That's the Thinking Machines "background model" pattern, built as a harness.

**The latency waterfall.** Every turn is measured from the last frame of your speech, in five segments:

1. endpointing (silence plus judge)
2. ASR final
3. the foreground model's first sentence
4. the first TTS chunk
5. playback start

The total is **time to first audio**. Typed turns (⌨) and proactive turns (★) appear too, and interrupted
turns are marked ✂.

## Running it for a long time

- **History is bounded.** After 60 messages, the oldest turns are dropped down to about 36. The cut
  always falls at a user message, so tool-call / result pairs stay intact. Without this, every turn
  re-sends the whole conversation and time to first token creeps up for hours. Memory across a trim is
  what `remember` is for.
- **One background task at a time.** Background tasks queue, so the foreground and the judge always have a
  free server slot. Finished tasks are capped at 50 per session.
- **Reconnects start a fresh session**, with new history and new state, and the page clears its panels.
  Background task ids are unique per server process.
- **The page bounds its own DOM** (conversation, transcripts, TTS chunks, background cards, event log).
  On MLX, the server releases the buffer cache after each transcription.
