"""Built-in tools.

  * instant - time, calculator, notes, weather, Wikipedia (either model may call these inline)
  * async   - set_timer, delegate_background_task: they return immediately and re-enter the
              conversation later through the session's proactive channel. That's the
              interaction-model point: the agent can speak when it has something, not only when asked.
  * web     - web_search, fetch_page (background only; see web.py)
"""
from __future__ import annotations

import ast
import operator as op
from datetime import datetime

import httpx

from .. import __version__
from . import web
from .registry import ToolContext, tool

USER_AGENT = f"interaction-harness/{__version__} (+https://github.com/IMJONEZZ/interaction-harness)"


def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos, ast.FloorDiv: op.floordiv}
_FUNCS = {"sqrt": lambda x: x ** 0.5, "round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported: {ast.dump(n)[:60]}")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval"))


# --------------------------------------------------------------------------- #
# instant
# --------------------------------------------------------------------------- #
@tool(description="Current local date and time.", parameters=_p({}, []), lanes=["foreground", "background", "duplex"])
def get_time() -> dict:
    now = datetime.now()
    return {"iso": now.isoformat(timespec="seconds"), "spoken": now.strftime("%A, %B %-d, %-I:%M %p")}


@tool(description="Evaluate an arithmetic expression (numbers, + - * / ** %, parentheses, sqrt, round).",
      parameters=_p({"expression": {"type": "string"}}, ["expression"]))
def calculate(expression: str = "") -> dict:
    return {"expression": expression, "result": safe_eval(expression)}


@tool(description="Save a short note/fact about the user for later.",
      parameters=_p({"note": {"type": "string"}}, ["note"]))
def remember(ctx: ToolContext, note: str = "") -> dict:
    ctx.notes.append(note.strip())
    return {"ok": True, "notes": len(ctx.notes)}


@tool(description="List everything remembered so far.", parameters=_p({}, []))
def recall(ctx: ToolContext) -> dict:
    return {"notes": ctx.notes}


_SKY = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
        65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "showers",
        81: "showers", 82: "violent showers", 95: "thunderstorm", 96: "thunderstorm w/ hail"}


@tool(description="Current weather for a city (Open-Meteo, no key).",
      parameters=_p({"city": {"type": "string"}}, ["city"]), lanes=["foreground", "background", "duplex"])
async def get_weather(city: str = "") -> dict:
    async with httpx.AsyncClient(timeout=8) as c:
        res = []
        for name in dict.fromkeys([city, city.split(",")[0].strip()]):  # "Boston, USA" -> "Boston" if needed
            g = await c.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 1})
            g.raise_for_status()
            res = g.json().get("results") or []
            if res:
                break
        if not res:
            return {"error": f"no such place: {city}"}
        loc = res[0]
        w = await c.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"})
        w.raise_for_status()
        cur = w.json().get("current", {})
        sky = _SKY.get(cur.get("weather_code"), "unknown")
        return {"place": f"{loc.get('name')}, {loc.get('admin1', '')} {loc.get('country_code', '')}".strip(),
                "temp_f": cur.get("temperature_2m"), "feels_like_f": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"), "wind_mph": cur.get("wind_speed_10m"),
                "sky": sky,
                "spoken": f"{loc.get('name')}: {round(cur.get('temperature_2m') or 0)} degrees Fahrenheit, {sky}, "
                          f"wind {round(cur.get('wind_speed_10m') or 0)} miles per hour"}


@tool(description="Look up a topic on Wikipedia and return a short summary.",
      parameters=_p({"topic": {"type": "string"}}, ["topic"]))
async def wikipedia(topic: str = "") -> dict:
    async with httpx.AsyncClient(timeout=8, headers={"User-Agent": USER_AGENT}) as c:
        s = await c.get("https://en.wikipedia.org/w/api.php",
                        params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"})
        s.raise_for_status()
        titles = s.json()[1]
        if not titles:
            return {"error": f"nothing found for {topic}"}
        r = await c.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{titles[0].replace(' ', '_')}")
        r.raise_for_status()
        j = r.json()
        return {"title": j.get("title"), "summary": j.get("extract"),
                "url": j.get("content_urls", {}).get("desktop", {}).get("page")}


# --------------------------------------------------------------------------- #
# async: these come back later through the proactive channel
# --------------------------------------------------------------------------- #
@tool(description="Set a timer or a reminder, such as 'remind me in 30 seconds to stretch'. When it fires, the "
                  "assistant will proactively tell the user.",
      parameters=_p({"seconds": {"type": "number"}, "label": {"type": "string"}}, ["seconds", "label"]),
      lanes=["foreground", "duplex"])
async def set_timer(ctx: ToolContext, seconds: float = 0, label: str = "timer") -> dict:
    return await ctx.on_timer(float(seconds), str(label))


@tool(description="Hand a task that needs real thinking (analysis, planning, comparison, calculation, writing, "
                  "multi-step reasoning, research) to the background model. Returns immediately; the result "
                  "is delivered proactively when ready. Use this instead of answering slowly yourself.",
      parameters=_p({"task": {"type": "string", "description": "Self-contained description of the task"}},
                    ["task"]),
      lanes=["foreground", "duplex"])
async def delegate_background_task(ctx: ToolContext, task: str = "") -> dict:
    return await ctx.on_delegate(str(task))


# --------------------------------------------------------------------------- #
# web: the background model can browse; the foreground hands off instead
# --------------------------------------------------------------------------- #
@tool(description="Search the web with DuckDuckGo and return titles, URLs and short snippets. "
                  "Pass url instead of query to fetch that page's full content.",
      parameters=_p({"query": {"type": "string"},
                     "max_results": {"type": "integer", "description": "1-10, default 5"},
                     "url": {"type": "string", "description": "optional: fetch this URL instead of searching"}}, []),
      lanes=["background"])
async def web_search(query: str = "", max_results: int = 5, url: str | None = None) -> str:
    return await web.web_search(query, max_results, url)


@tool(description="Fetch a web page (HTML, PDF or text) and return its readable content as Markdown, "
                  "main content only, up to 16,000 characters.",
      parameters=_p({"url": {"type": "string"},
                     "max_chars": {"type": "integer", "description": "2000-16000, default 16000"}}, ["url"]),
      lanes=["background"])
async def fetch_page(url: str = "", max_chars: int | None = None) -> str:
    return await web.fetch_page(url, max_chars)
