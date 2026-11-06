"""Example tools: run with  interaction-harness serve --tools examples/my_tools.py

Any function decorated with @tool becomes callable by the models. The JSON schema comes from the signature,
the description from the docstring's first paragraph. A `ctx` parameter receives the session's ToolContext
(notes, and hooks that re-enter the conversation later) and is hidden from the model.
"""
from __future__ import annotations

import random
from typing import Annotated

from interaction_harness.tools import tool


@tool
def roll_dice(sides: Annotated[int, "how many sides each die has"] = 6, count: int = 1) -> dict:
    """Roll dice and return each result and the total."""
    rolls = [random.randint(1, max(2, sides)) for _ in range(max(1, min(count, 20)))]
    return {"rolls": rolls, "total": sum(rolls)}


@tool(lanes=["background"])
async def summarize_notes(ctx) -> str:
    """Everything the user asked to remember, as one paragraph. Background model only."""
    return " ".join(ctx.notes) or "No notes yet."
