# Contributing

Issues and pull requests are welcome. The most useful contributions are:

- **Server reports.** Run `interaction-harness doctor` against a server not yet in the README's "Tested
  with" table (OpenAI, Speaches, Kokoro-FastAPI, …) and file a
  [server report](https://github.com/IMJONEZZ/interaction-harness/issues/new?template=server-report.yml)
  with the output.
- **Backends.** In-process speech on other hardware or with other models. See
  [docs/backends.md](docs/backends.md).
- **Interaction-model experiments.** New end-of-turn policies, backchannel handling, or better barge-in
  rules. `session.py` is the one place the policy lives.

## Setup

```bash
git clone https://github.com/IMJONEZZ/interaction-harness && cd interaction-harness
uv sync                      # Python 3.11-3.13
uv run pytest                # unit tests: fakes and stubs, no models, no network
uv run ruff check src tests examples
```

With a model server, check your change end to end:

```bash
uv run interaction-harness doctor   --foreground-base-url http://127.0.0.1:8080/v1
uv run interaction-harness selftest --foreground-base-url http://127.0.0.1:8080/v1
```

`selftest` talks to the agent with synthesized speech, checking endpointing, tools, barge-in, background
delivery, timers, push-to-talk and typed turns. Some checks depend on the model's choices, such as which
tool it picks; those are warnings, not failures.

## Where things are

| Path | What it holds |
|---|---|
| `src/interaction_harness/session.py` | the interaction model: state machine, endpointing, barge-in, both models, proactive delivery |
| `src/interaction_harness/config.py` | settings and the four roles |
| `src/interaction_harness/llm.py` | the chat roles, the thinking switch, the end-of-turn judge |
| `src/interaction_harness/duplex.py` | full-duplex mode: the client, tool calls and results, and what it measures |
| `src/interaction_harness/duplex_server.py` | `duplex-server`: VoiceChat on mlx-vlm, plus tool results |
| `src/interaction_harness/speech/` | VAD, and ASR and TTS on MLX, ONNX or an endpoint |
| `src/interaction_harness/tools/` | the tool registry, built-in tools, web search and fetch |
| `src/interaction_harness/static/` | the dashboard: vanilla JS, no build step |
| `tests/` | unit tests; `conftest.py` has fakes for every model role |

## Style

Keep the dashboard dependency-free, and keep the state machine readable. If a change affects latency,
say how you measured it; the waterfall and `doctor` are there for that.
