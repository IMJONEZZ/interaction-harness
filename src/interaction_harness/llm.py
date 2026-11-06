"""The two chat roles, each on its own OpenAI-compatible endpoint.

    foreground : thinking off by default -> low-latency replies, quick tools, hand-offs
    background : thinking on             -> tasks the foreground delegates
    judge      : the foreground endpoint, thinking off, 3 tokens -> semantic end-of-turn

When both roles point at the same server and model, the only difference is how they are called.
How thinking is switched depends on the server (FOREGROUND_THINKING / BACKGROUND_THINKING):

    chat_template_kwargs : {"chat_template_kwargs": {"enable_thinking": on}}  (vLLM, llama.cpp, SGLang)
    reasoning_effort     : reasoning_effort "high" when on, "low" when off    (OpenAI-style servers)
    off                  : send nothing; the server's default applies

The stream yields typed events so the session can forward reasoning / content / tool-call deltas
straight to the browser.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .config import Endpoint, Settings


def thinking_body(mode: str, enabled: bool) -> dict:
    if mode == "chat_template_kwargs":
        return {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}
    if mode == "reasoning_effort":
        return {"reasoning_effort": "high" if enabled else "low"}
    return {}


async def list_models(ep: Endpoint, timeout: float = 5.0) -> list[str]:
    """GET {base}/models. Many local servers serve exactly one model, so a blank MODEL can use it."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{ep.base_url}/models", headers={"Authorization": f"Bearer {ep.api_key}"})
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", []) if m.get("id")]


async def resolve_models(settings: Settings) -> list[str]:
    """Fill in a blank FOREGROUND_MODEL / BACKGROUND_MODEL from the server's model list. Returns notes."""
    notes = []
    for ep in (settings.foreground, settings.background):
        if ep.model:
            continue
        models = await list_models(ep)
        if not models:
            raise RuntimeError(f"{ep.role}: {ep.base_url} lists no models; set {ep.role.upper()}_MODEL")
        ep.model = models[0]
        notes.append(f"{ep.role}: no model set, using {ep.model!r} (first of {len(models)} at {ep.host})")
    return notes


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        fg, bg = settings.foreground, settings.background
        self.client = AsyncOpenAI(base_url=fg.base_url, api_key=fg.api_key, timeout=120)
        self.background_client = (self.client if (bg.base_url, bg.api_key) == (fg.base_url, fg.api_key)
                                  else AsyncOpenAI(base_url=bg.base_url, api_key=bg.api_key, timeout=300))

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        thinking: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        background: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream one assistant turn.

        Yields events:
            {"type": "first_token", "ms": float}                      once
            {"type": "reasoning", "delta": str}
            {"type": "content",   "delta": str}
            {"type": "tool_calls", "calls": [{"id","name","arguments"}]}  at the end, if any
            {"type": "done", "content": str, "reasoning": str, "finish": str,
             "usage": {...}, "ms": float, "tps": float}
        """
        s = self.s
        ep = s.background if background else s.foreground
        mode = s.background_thinking if background else s.foreground_thinking
        t0 = time.perf_counter()
        first = None
        content, reasoning = [], []
        calls: dict[int, dict] = {}
        finish = None
        usage: dict[str, Any] = {}
        kwargs: dict[str, Any] = dict(model=ep.model, messages=messages, stream=True,
                                      temperature=s.background_temperature if background else temperature,
                                      max_tokens=max_tokens, extra_body=thinking_body(mode, thinking),
                                      stream_options={"include_usage": True})
        if tools:
            kwargs["tools"] = tools
        stream = await (self.background_client if background else self.client).chat.completions.create(**kwargs)
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage.model_dump()
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                d = ch.delta
                extra = d.model_extra or {}   # llama.cpp sends reasoning_content; vLLM sends reasoning
                rc = getattr(d, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
                if rc:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                        yield {"type": "first_token", "ms": first}
                    reasoning.append(rc)
                    yield {"type": "reasoning", "delta": rc}
                if d.content:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                        yield {"type": "first_token", "ms": first}
                    content.append(d.content)
                    yield {"type": "content", "delta": d.content}
                for tc in d.tool_calls or []:
                    if first is None:  # a turn that opens with a tool call still has a time to first token
                        first = (time.perf_counter() - t0) * 1000
                        yield {"type": "first_token", "ms": first}
                    slot = calls.setdefault(tc.index, {"id": tc.id or "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
                if ch.finish_reason:
                    finish = ch.finish_reason
        finally:
            await stream.close()
        ms = (time.perf_counter() - t0) * 1000
        if calls:
            yield {"type": "tool_calls", "calls": [calls[i] for i in sorted(calls)]}
        ctoks = usage.get("completion_tokens") or 0
        yield {"type": "done", "content": "".join(content), "reasoning": "".join(reasoning),
               "finish": finish or ("tool_calls" if calls else "stop"), "usage": usage, "ms": ms,
               "tps": (ctoks / (ms / 1000)) if ms > 0 and ctoks else 0.0}

    async def end_of_turn(self, transcript: str) -> tuple[bool, float, str]:
        """Semantic end-of-turn judge on the foreground endpoint. Returns (is_complete, ms, raw_answer)."""
        t0 = time.perf_counter()
        if not transcript.strip():
            return False, 0.0, ""
        resp = await self.client.chat.completions.create(
            model=self.s.foreground.model,
            messages=[{"role": "system", "content": self.s.end_of_turn_prompt},
                      {"role": "user", "content": f"User so far: \"{transcript.strip()}\"\nAnswer:"}],
            max_tokens=3, temperature=0.0, extra_body=thinking_body(self.s.foreground_thinking, False))
        raw = (resp.choices[0].message.content or "").strip().upper()
        ms = (time.perf_counter() - t0) * 1000
        return raw.startswith("COMPLETE"), ms, raw


def tool_result_message(call_id: str, name: str, result: Any) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name,
            "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)}


def assistant_tool_message(content: str | None, calls: list[dict]) -> dict:
    return {"role": "assistant", "content": content or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                           for c in calls]}
