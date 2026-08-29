"""propose(): the LLM call, structured output. Task A3.

Two-tier routing: Opus 4.5 proposes and reflects, Haiku 4.5 repairs.
Log usage.input_tokens / usage.output_tokens on EVERY call — total tokens
are a scored deliverable (Feasibility, 15%).
"""

from __future__ import annotations

from agent.schema import Action

PROPOSE_MODEL = "claude-opus-4-5"
REPAIR_MODEL = "claude-haiku-4-5-20251001"


def propose(history: list[dict], knowledge: str, parent: dict) -> tuple[Action, dict]:
    """Return (action, usage). Raises only on unrecoverable API failure.

    usage = {"in": int, "out": int, "model": str} — goes straight into the
    node record's `tokens` field.
    """
    raise NotImplementedError("A3")


def repair(action: Action, error: dict) -> tuple[Action, dict]:
    """One bounded repair attempt on a failed action, routed to Haiku."""
    raise NotImplementedError("A5")
