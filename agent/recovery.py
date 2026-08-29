"""Per-error-class recovery policy. Task A5.

Failures are graded output (design commitment 5): classify, recover, and log
them rather than hiding them.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral

from agent import store

# error_class -> policy
POLICY = {
    "syntax": "one repair attempt via Haiku, then abandon the node",
    "schema": "one repair attempt, then abandon",
    "oom": "retry once at reduced batch size",
    "timeout": "retry once at a lower fidelity tier",
    "transient": "exponential backoff, up to 3 attempts",
    "leak_suspected": "quarantine, record lineage, never retry silently",
}

MAX_CONSECUTIVE_DEAD_NODES = 3  # then force a branch to a different parent
MAX_TRANSIENT_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 4096
MIN_BATCH_SIZE = 1


def _error(node: dict) -> dict:
    """Return A4's first structured error without trusting a malformed node."""
    errors = node.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return errors[0]
    return {}


def _event(node: dict, error_class: str, attempt: int, decision: str, **extra) -> None:
    """Persist every recovery decision, including an explicit decision to stop."""
    store.append_event(
        {
            "event": "recovery",
            "node": node.get("id"),
            "error_class": error_class,
            "attempt": attempt,
            "decision": decision,
            "manual_intervention": False,
            **extra,
        }
    )


def _base_plan(node: dict, operation: str, **extra) -> dict:
    """Preserve the original experiment identity while changing its mechanism."""
    return {
        "operation": operation,
        "fidelity": node.get("fidelity", "smoke"),
        "config": deepcopy(node.get("config") or {}),
        "backoff_s": 0.0,
        **extra,
    }


def _reduced_batch_config(node: dict) -> dict | None:
    config = deepcopy(node.get("config") or {})
    hparams = config.get("hparams")
    if not isinstance(hparams, dict):
        return None

    current = hparams.get("batch_size", DEFAULT_BATCH_SIZE)
    if isinstance(current, bool) or not isinstance(current, Integral) or current <= MIN_BATCH_SIZE:
        return None

    hparams["batch_size"] = max(MIN_BATCH_SIZE, current // 2)
    config["hparams"] = hparams
    return config


def should_force_branch(history: list[dict]) -> bool:
    """True after three consecutive failed nodes (A5's anti-stall guard)."""
    dead = 0
    for node in reversed(history):
        if node.get("status") != "error":
            break
        dead += 1
        if dead >= MAX_CONSECUTIVE_DEAD_NODES:
            return True
    return False


def recover(node: dict, attempt: int) -> dict | None:
    """Return a bounded recovery plan, or ``None`` when the node is terminal.

    The plan deliberately does not invoke the LLM. A6 owns the original
    ``Action`` and consumes ``operation="repair"`` by calling A3's Haiku
    repair function; this pure policy layer only decides whether that is
    allowed. Every invocation records an append-only recovery event.
    """
    if isinstance(attempt, bool) or not isinstance(attempt, Integral) or attempt < 0:
        _event(node, "schema", int(attempt) if isinstance(attempt, Integral) else -1, "abandon")
        return None

    error = _error(node)
    error_class = error.get("error_class")
    if error_class not in POLICY:
        _event(node, str(error_class), attempt, "abandon")
        return None

    if error_class in {"syntax", "schema"}:
        if attempt > 0:
            _event(node, error_class, attempt, "abandon")
            return None
        plan = _base_plan(
            node,
            "repair",
            error=deepcopy(error),
            repair_attempted=True,
        )
        _event(node, error_class, attempt, "repair", fidelity=plan["fidelity"])
        return plan

    if error_class == "oom":
        if attempt > 0:
            _event(node, error_class, attempt, "abandon")
            return None
        config = _reduced_batch_config(node)
        if config is None:
            _event(node, error_class, attempt, "abandon")
            return None
        plan = _base_plan(node, "retry", config=config)
        _event(
            node,
            error_class,
            attempt,
            "retry",
            fidelity=plan["fidelity"],
            batch_size=config["hparams"]["batch_size"],
        )
        return plan

    if error_class == "timeout":
        lower_fidelity = {"confirm": "full", "full": "screen", "screen": "smoke"}
        target = lower_fidelity.get(node.get("fidelity"))
        if attempt > 0 or target is None:
            _event(node, error_class, attempt, "abandon")
            return None
        plan = _base_plan(node, "retry", fidelity=target)
        _event(node, error_class, attempt, "retry", fidelity=target)
        return plan

    if error_class == "transient":
        if attempt >= MAX_TRANSIENT_ATTEMPTS:
            _event(node, error_class, attempt, "abandon")
            return None
        backoff_s = float(2**attempt)
        plan = _base_plan(node, "retry", backoff_s=backoff_s)
        _event(
            node,
            error_class,
            attempt,
            "retry",
            fidelity=plan["fidelity"],
            backoff_s=backoff_s,
        )
        return plan

    # `leak_suspected`: never retry or repair a candidate that may have read a
    # same-row outcome. Its failure node remains in the ledger for lineage.
    _event(node, error_class, attempt, "quarantine")
    return None
