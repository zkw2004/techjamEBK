"""propose(): the LLM call, structured output. Task A3.

Two-tier routing: Opus proposes and reflects, Haiku repairs.
Log usage.input_tokens / usage.output_tokens on EVERY call — total tokens
are a scored deliverable (Feasibility, 15%).

Design notes:

* **Structured output, not prose parsing.** `client.messages.parse()` with
  `output_format=Action` returns a validated `Action`. A malformed proposal
  becomes a schema error we can re-ask on, never a traceback in the loop.
* **The client is injectable.** CI has no API key, and the acceptance
  criterion is "given a fake node history, returns a valid Action" — so the
  transport is swapped, not mocked at the HTTP layer.
* **Cache-friendly prompt order.** The system prompt is the frozen
  instructions plus `knowledge.md`, marked `cache_control`; everything
  volatile (node history, parent) goes in the user turn, after the
  breakpoint. Caching is a prefix match, so stable-content-first is what
  keeps the cache warm across 40+ iterations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.schema import FAMILIES, Action

# Current model IDs. AGENT_PLAN.md Section 12 names claude-opus-4-5 and
# claude-haiku-4-5; Opus 5 supersedes the former. The id lands in every node's
# `tokens.model`, so the resource report stays accurate.
PROPOSE_MODEL = "claude-opus-5"
REPAIR_MODEL = "claude-haiku-4-5"

MAX_TOKENS = 16000
MAX_ATTEMPTS = 2  # one re-ask on an inconsistent Action, then give up

PROMPTS_DIR = Path(__file__).parent / "prompts"
KNOWLEDGE_PATH = Path(__file__).parent / "knowledge.md"

_client: Any | None = None


class ProposeError(RuntimeError):
    """The LLM could not produce a usable Action.

    ``usage`` is retained when the API returned a response that was billed but
    whose Action could not be used. The loop can therefore log failed proposal
    and repair calls without undercounting the run's token total.
    """

    def __init__(self, message: str, *, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}


def set_client(client: Any | None) -> None:
    """Inject the Anthropic client (or a stand-in). Used by tests and by A6."""
    global _client
    _client = client


def _get_client() -> Any:
    """Lazily construct a real client, so importing this module never needs a key."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


def load_knowledge() -> str:
    return KNOWLEDGE_PATH.read_text()


def _usage(response: Any, model: str) -> dict:
    """Extract the node record's `tokens` field (Section 8.7).

    `in`/`out`/`model` are the frozen keys. The cache counters are additive:
    cached input bills at ~0.1x, so a resource report that ignores them
    overstates cost.
    """
    u = response.usage
    return {
        "in": getattr(u, "input_tokens", 0) or 0,
        "out": getattr(u, "output_tokens", 0) or 0,
        "model": model,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def summarise_history(history: list[dict], limit: int = 40) -> list[dict]:
    """Compress node records for the prompt.

    A full node carries config, segments, gates and a traceback. Feeding 40 of
    those back costs thousands of tokens per iteration for information the
    proposer does not need. Keep what drives the next decision: what was tried,
    whether it worked, and why it failed.
    """
    out = []
    for node in history[-limit:]:
        metrics = node.get("metrics") or {}
        row = {
            "id": node.get("id"),
            "parent": node.get("parent"),
            "family": node.get("family"),
            "hypothesis": node.get("hypothesis"),
            "status": node.get("status"),
            "accepted": node.get("accepted", False),
            "primary": metrics.get("primary"),
        }
        if node.get("status") == "error":
            errors = node.get("errors") or []
            row["error_class"] = errors[0].get("error_class") if errors else None
        if node.get("fold_primaries"):
            row["fold_primaries"] = node["fold_primaries"]
        out.append({k: v for k, v in row.items() if v is not None})
    return out


def _families_covered(history: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(FAMILIES, 0)
    for node in history:
        fam = node.get("family")
        if fam in counts:
            counts[fam] += 1
    return counts


def check_action_consistency(action: Action) -> None:
    """Reject Actions the schema accepts but `execute()` cannot run.

    Section 8.6 types every optional field but cannot express "a tune action
    needs a search space". Catching it here turns a crash in A4 into a re-ask
    that costs one cheap call.
    """
    if action.type == "config" and action.config is None:
        raise ProposeError("type='config' requires a config block")
    if action.type == "tune" and not action.search_space:
        raise ProposeError("type='tune' requires a non-empty search_space")
    if action.type == "code" and not action.code:
        raise ProposeError("type='code' requires Python source in `code`")
    if action.type == "blend":
        if action.config is None:
            raise ProposeError("type='blend' requires a config block")
        if action.config.model != "blend":
            raise ProposeError("type='blend' requires config.model == 'blend'")
        if len(action.config.parents) < 2:
            raise ProposeError("type='blend' requires at least two parents")


def check_repair_consistency(original: Action, repaired: Action) -> None:
    """Enforce the repair prompt's evidence-preservation contract in code.

    A repair may change the mechanism that failed, but changing the claim or
    branch metadata turns it into a different experiment. Prompt instructions
    are not a sufficient trust boundary, so these three fields are checked
    after structured parsing as well.
    """
    for field in ("hypothesis", "family", "parent"):
        before = getattr(original, field)
        after = getattr(repaired, field)
        if after != before:
            raise ProposeError(
                f"repair must preserve {field}: expected {before!r}, got {after!r}"
            )


def _call(model: str, system: list[dict], user: str, **kwargs: Any) -> Any:
    client = _get_client()
    try:
        return client.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=Action,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as ProposeError to the loop
        raise ProposeError(f"{model} call failed: {exc}") from exc


def propose(history: list[dict], knowledge: str, parent: dict) -> tuple[Action, dict]:
    """Return (action, usage). Raises ProposeError only on unrecoverable failure.

    usage = {"in": int, "out": int, "model": str, ...} — goes straight into the
    node record's `tokens` field.
    """
    # Stable prefix first, so the cache survives across iterations.
    system = [
        {
            "type": "text",
            "text": load_prompt("propose") + "\n\n---\n\n" + knowledge,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    context = {
        "best_so_far": parent,
        "families_covered": _families_covered(history),
        "history": summarise_history(history),
    }
    user = (
        "Here is the run so far. Propose exactly one next experiment.\n\n"
        + json.dumps(context, indent=2, default=str)
    )

    total = {"in": 0, "out": 0, "model": PROPOSE_MODEL, "cache_read": 0, "cache_write": 0}
    last_error: str | None = None
    rejected_action: Action | None = None

    for _attempt in range(MAX_ATTEMPTS):
        prompt = user
        if last_error:
            prompt += (
                f"\n\nYour previous proposal was rejected: {last_error}\n"
                "Return a corrected Action. Keep the same hypothesis."
            )
            if rejected_action is not None:
                prompt += "\n\nRejected Action:\n" + rejected_action.model_dump_json(indent=2)

        response = _call(
            PROPOSE_MODEL, system, prompt, thinking={"type": "adaptive"}
        )
        usage = _usage(response, PROPOSE_MODEL)
        for key in ("in", "out", "cache_read", "cache_write"):
            total[key] += usage[key]

        action = response.parsed_output
        if action is None:
            last_error = "response did not contain a parseable Action"
            continue
        if rejected_action is not None and action.hypothesis != rejected_action.hypothesis:
            last_error = (
                "corrected Action changed the hypothesis from "
                f"{rejected_action.hypothesis!r} to {action.hypothesis!r}"
            )
            continue
        try:
            check_action_consistency(action)
        except ProposeError as exc:
            last_error = str(exc)
            rejected_action = action
            continue
        # Every attempt is billed, so report the running total, not the last call.
        return action, total

    raise ProposeError(
        f"no valid Action after {MAX_ATTEMPTS} attempts; last error: {last_error}",
        usage=total,
    )


def repair(action: Action, error: dict) -> tuple[Action, dict]:
    """One bounded repair attempt on a failed action, routed to Haiku.

    Cheap model on purpose: a repair fixes a mechanism that already failed with
    a concrete traceback, which is a much smaller problem than proposing. When
    to call this — and how many times — is A5's policy, not this function's.
    """
    system = [
        {
            "type": "text",
            "text": load_prompt("repair"),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user = json.dumps(
        {
            "failed_action": action.model_dump(),
            "error_class": error.get("error_class"),
            "stage": error.get("stage"),
            "traceback": (error.get("traceback") or "")[-4000:],
        },
        indent=2,
        default=str,
    )

    response = _call(REPAIR_MODEL, system, user)
    usage = _usage(response, REPAIR_MODEL)

    repaired = response.parsed_output
    if repaired is None:
        raise ProposeError("repair returned no parseable Action", usage=usage)
    try:
        check_action_consistency(repaired)
        check_repair_consistency(action, repaired)
    except ProposeError as exc:
        raise ProposeError(str(exc), usage=usage) from exc
    return repaired, usage
