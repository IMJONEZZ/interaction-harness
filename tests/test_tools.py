"""The tool registry, schema inference, lanes, and the built-in calculator."""
from __future__ import annotations

import json
from typing import Annotated

import pytest

from interaction_harness.tools import REGISTRY, ToolContext, load_tools_file, run_tool, schemas, tool
from interaction_harness.tools.builtin import safe_eval


async def _noop(*a):
    return {}


@pytest.fixture
def ctx():
    return ToolContext(on_timer=_noop, on_delegate=_noop)


@pytest.fixture
def scratch_tools():
    before = set(REGISTRY)
    yield
    for name in set(REGISTRY) - before:
        del REGISTRY[name]


def test_lanes():
    fg = {s["function"]["name"] for s in schemas("foreground")}
    bg = {s["function"]["name"] for s in schemas("background")}
    assert {"set_timer", "delegate_background_task"} <= fg and not {"web_search", "fetch_page"} & fg
    assert {"web_search", "fetch_page"} <= bg and "delegate_background_task" not in bg  # no re-delegation
    assert {"get_time", "calculate", "remember", "recall"} <= fg & bg


@pytest.mark.parametrize("expr, value", [("2+3*4", 14), ("2^10", 1024), ("sqrt(16)", 4), ("round(7/3, 2)", 2.33)])
def test_safe_eval(expr, value):
    assert safe_eval(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('true')", "open('x')", "[1, 2]"])
def test_safe_eval_refuses_anything_else(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_schema_inferred_from_signature(scratch_tools):
    @tool(lanes=["background"])
    def lookup(ctx, query: Annotated[str, "words to look for"], limit: int = 5, exact: bool | None = None) -> list:
        """Search the ticket system.

        More detail the model does not need."""
        return []

    t = REGISTRY["lookup"]
    assert t.description == "Search the ticket system."
    assert t.wants_ctx and t.lanes == {"background"}
    assert t.parameters == {"type": "object", "required": ["query"], "properties": {
        "query": {"type": "string", "description": "words to look for"},
        "limit": {"type": "integer"}, "exact": {"type": "boolean"}}}


async def test_run_tool_errors_go_back_to_the_model(ctx, scratch_tools):
    @tool
    def boom() -> None:
        raise RuntimeError("nope")

    assert (await run_tool(ctx, "boom", "{}"))[0] == {"error": "RuntimeError: nope"}
    assert "bad JSON" in (await run_tool(ctx, "calculate", "{oops"))[0]["error"]
    assert "unknown tool" in (await run_tool(ctx, "no_such_tool", "{}"))[0]["error"]
    # a background-only tool is not callable from the foreground
    assert "unknown tool" in (await run_tool(ctx, "web_search", '{"query": "x"}', "foreground"))[0]["error"]


async def test_notes_live_in_the_session_context(ctx):
    await run_tool(ctx, "remember", json.dumps({"note": "favorite color is green"}))
    res, _ = await run_tool(ctx, "recall", "")
    assert res == {"notes": ["favorite color is green"]}


async def test_load_tools_file(tmp_path, ctx, scratch_tools):
    f = tmp_path / "my_tools.py"
    f.write_text("from interaction_harness.tools import tool\n\n"
                 "@tool\nasync def shout(text: str) -> str:\n    'Say it louder.'\n    return text.upper()\n")
    assert load_tools_file(f) == ["shout"]
    assert "shout" in {s["function"]["name"] for s in schemas("foreground")}
    assert (await run_tool(ctx, "shout", '{"text": "hi"}'))[0] == "HI"
