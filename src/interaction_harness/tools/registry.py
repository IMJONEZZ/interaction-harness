"""Tool registry: register plain Python functions with @tool and both models can call them.

    from interaction_harness.tools import tool

    @tool
    def flip_coin() -> str:
        "Flip a coin."
        return random.choice(["heads", "tails"])

    @tool(lanes=["foreground", "duplex"])   # also offered to the full-duplex model
    def lights(on: bool) -> str:
        "Turn the lights on or off."
        ...

    @tool(lanes=["background"])
    async def search_tickets(query: Annotated[str, "words to look for"], limit: int = 5) -> list[dict]:
        "Search the support ticket system."
        ...

The schema comes from the signature: str/int/float/bool/list/dict become JSON types, parameters without a
default are required, Annotated[..., "text"] adds a description, and the docstring's first paragraph is the
tool's description. A parameter named `ctx` receives the session's ToolContext and is not shown to the model.
Load a file of tools with `--tools my_tools.py`.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import time
import types
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LANES = ("foreground", "background", "duplex")
DEFAULT_LANES = ("foreground", "background")  # duplex is opt-in: each tool lengthens its prompt, and so its load time
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    lanes: frozenset[str]
    wants_ctx: bool

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


REGISTRY: dict[str, Tool] = {}


@dataclass
class ToolContext:
    """What tools may use from the session: notes, and hooks that re-enter the conversation later."""

    on_timer: Callable[[float, str], Awaitable[dict]]
    on_delegate: Callable[[str], Awaitable[dict]]
    notes: list[str] = field(default_factory=list)


def _json_type(tp: Any) -> tuple[dict, bool]:
    """(JSON schema for a type hint, whether it is optional)."""
    desc = None
    if typing.get_origin(tp) is typing.Annotated:
        tp, *meta = typing.get_args(tp)
        desc = next((m for m in meta if isinstance(m, str)), None)
    optional = False
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        optional = len(args) < len(typing.get_args(tp))
        tp = args[0] if len(args) == 1 else str
        origin = typing.get_origin(tp)
    base = origin or tp
    schema: dict[str, Any] = {"type": _JSON_TYPES.get(base, "string")}
    if desc:
        schema["description"] = desc
    return schema, optional


def infer_parameters(fn: Callable) -> tuple[dict, bool]:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    props, required, wants_ctx = {}, [], False
    for name, p in sig.parameters.items():
        if name == "ctx":
            wants_ctx = True
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        schema, optional = _json_type(hints.get(name, str))
        props[name] = schema
        if p.default is p.empty and not optional:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}, wants_ctx


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None,
         parameters: dict | None = None, lanes: typing.Iterable[str] = DEFAULT_LANES):
    """Register a function as a tool. Usable bare (@tool) or with options (@tool(lanes=["background"]))."""

    def register(f: Callable) -> Callable:
        inferred, wants_ctx = infer_parameters(f)
        doc = (inspect.getdoc(f) or "").split("\n\n")[0].replace("\n", " ").strip()
        bad = set(lanes) - set(LANES)
        if bad:
            raise ValueError(f"unknown lane(s) {sorted(bad)}; use {LANES}")
        t = Tool(name=name or f.__name__, description=description or doc or (name or f.__name__),
                 parameters=parameters or inferred, fn=f, lanes=frozenset(lanes), wants_ctx=wants_ctx)
        REGISTRY[t.name] = t
        return f

    return register(fn) if fn is not None else register


def schemas(lane: str) -> list[dict]:
    return [t.schema() for t in REGISTRY.values() if lane in t.lanes]


async def run_tool(ctx: ToolContext, name: str, arguments: str, lane: str | None = None) -> tuple[Any, float]:
    """Execute one tool call. Never raises: errors go back to the model as text."""
    t0 = time.perf_counter()
    try:
        args = json.loads(arguments or "{}") if arguments else {}
    except json.JSONDecodeError:
        return {"error": f"bad JSON arguments: {arguments[:80]}"}, 0.0
    t = REGISTRY.get(name)
    if t is None or (lane and lane not in t.lanes):
        return {"error": f"unknown tool {name}"}, (time.perf_counter() - t0) * 1000
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}, 0.0
    try:
        kwargs = {"ctx": ctx, **args} if t.wants_ctx else args
        if inspect.iscoroutinefunction(t.fn):
            res = await t.fn(**kwargs)
        else:
            res = await asyncio.to_thread(t.fn, **kwargs)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"}
    return res, (time.perf_counter() - t0) * 1000


def load_tools_file(path: str | Path) -> list[str]:
    """Import a Python file so its @tool functions register. Returns the names it added."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"tools file not found: {path}")
    before = set(REGISTRY)
    spec = importlib.util.spec_from_file_location(f"interaction_harness_user_tools_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [n for n in REGISTRY if n not in before]
