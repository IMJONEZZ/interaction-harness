# Tools

Both models can call tools. Each tool belongs to one or both *lanes*: `foreground`, `background`, or both
(the default).

## Built in

| Tool | Lanes | What it does |
|---|---|---|
| `get_time` | both | local date and time |
| `calculate` | both | arithmetic, evaluated safely (no `eval`) |
| `remember` / `recall` | both | notes for this session |
| `get_weather` | both | current weather from Open-Meteo (no key) |
| `wikipedia` | both | a short Wikipedia summary |
| `set_timer` | foreground | returns at once; when it fires, the agent tells you in its own turn |
| `delegate_background_task` | foreground | hands a task to the background model and returns at once; the result is delivered in its own turn |
| `web_search` | background | DuckDuckGo results as Title / URL / Snippet |
| `fetch_page` | background | a page as Markdown (main content, up to 16,000 characters), PDFs as text. Refuses private addresses on every redirect |

The background model cannot delegate again. Only one background task runs at a time, so the foreground
and the end-of-turn judge keep their server slots; others queue, and the dashboard says so.

## Your own

```python
from typing import Annotated
from interaction_harness.tools import tool

@tool
def stock_level(sku: Annotated[str, "the product code"], store: str = "main") -> dict:
    """How many of a product are in stock.

    Only the first paragraph of the docstring becomes the description."""
    return {"sku": sku, "store": store, "in_stock": 12}

@tool(lanes=["background"], name="tickets")
async def search_tickets(ctx, query: str, limit: int = 5) -> list[dict]:
    """Search the support ticket system."""
    ctx.notes.append(f"searched tickets for {query}")
    ...
```

Load it with `--tools my_tools.py`, `TOOLS=my_tools.py`, or `tools = ["my_tools.py"]` under `[server]` in
`interaction-harness.toml`.

Rules:

- **Types.** `str`, `int`, `float`, `bool`, `list` and `dict` become JSON types. `Optional[...]` / `X | None`
  and parameters with defaults are optional.
- **Descriptions.** `Annotated[type, "description"]` gives a parameter a description.
- **Full control.** `@tool(parameters={...})` takes a JSON schema as-is, and `description=` overrides the
  docstring.
- **`ctx`.** A parameter named `ctx` receives the session's `ToolContext` and is hidden from the model. It
  has `notes`, plus `on_timer(seconds, label)` and `on_delegate(task)` for tools that should come back later.
- **Sync and async.** Sync functions run in a thread, so a slow tool doesn't block audio; async functions
  are awaited.
- **Return values.** Return anything JSON-serializable, or a string.
- **Errors.** An exception becomes `{"error": "..."}` for the model; it never reaches the session.

Voice answers should be short, so keep tool results small and to the point. The foreground prompt tells
the model it has no clock or senses and must call tools for the time, weather and arithmetic. Adjust
`FOREGROUND_PROMPT` to say when to use yours.

## How the proactive channel works

The two async tools, `set_timer` and `delegate_background_task`, return at once. When the timer fires or
the task finishes, the result goes into a queue. A loop delivers it as soon as all three hold:

- the agent is listening or idle
- you haven't spoken for a second
- you aren't holding push-to-talk

It delivers by giving the foreground model a turn of its own. The model sees a
`(Background event, not something the user said: …)` message with the result and says it in a sentence or
two. Delivered turns are marked ★ in the waterfall and *proactive* in the Brain panel.
