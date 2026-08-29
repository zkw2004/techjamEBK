"""Per-error-class recovery policy. Task A5.

Failures are graded output (design commitment 5): classify, recover, and log
them rather than hiding them.
"""

from __future__ import annotations

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


def recover(node: dict, attempt: int) -> dict | None:
    """Return a modified action/fidelity to retry with, or None to give up.

    Every attempt is logged to the event stream, including the ones that fail.
    """
    raise NotImplementedError("A5")
