"""The unattended A6 research loop.

The loop owns orchestration, not modelling: proposals come from ``propose``,
execution comes from ``execute``, and A5 decides bounded recovery.  Every
attempt, including cheap smoke and screen pilots, is persisted before the next
decision so an interrupted run can be inspected and resumed safely.
"""

from __future__ import annotations

import math
import random as random_module
import time
from collections.abc import Callable
from copy import deepcopy
from statistics import median
from typing import Any

from agent import execute, manifest, propose, recovery, store
from agent.schema import FAMILIES, Action

EPSILON = 0.002
NO_IMPROVEMENT_ITERATIONS = 3

# A7: cover all five families before refining any one of them twice; over 40
# iterations, >=15% of nodes must branch from a non-best parent.
EPSILON_GREEDY = 0.15

ROOT_PARENT = {
    "id": "n000",
    "family": "baseline",
    "hypothesis": "run root",
    "status": "ok",
    "metrics": {},
    "fold_primaries": [],
}


def _branchable(history: list[dict]) -> list[dict]:
    """Nodes a proposal can legitimately branch from.

    Restricted to completed full/confirm evaluations with a finite primary —
    the same filter A6's placeholder used. A smoke or screen pilot never
    became the branch point: it was screened cheaply precisely so it would
    not consume attention (or official validation) as if it were evidence.
    """
    return [
        node
        for node in history
        if node.get("status") == "ok"
        and node.get("fidelity") in {"full", "confirm"}
        and isinstance(node.get("metrics", {}).get("primary"), (int, float))
    ]


def _family_counts(history: list[dict]) -> dict[str, int]:
    """How many attempts (any fidelity, any outcome) exist per family.

    Matches ``propose._families_covered``: a family is being explored the
    moment it is attempted, not only once it produces an accepted node.
    """
    counts = dict.fromkeys(FAMILIES, 0)
    for node in history:
        family = node.get("family")
        if family in counts:
            counts[family] += 1
    return counts


def select_parent(nodes: list[dict], *, rng: Any = None) -> dict:
    """Family-diverse + epsilon-greedy parent selection.

    Two failure modes this guards against, both real risks in a tree search
    this short: always extending the best node turns the loop into local
    hill-climbing that never revisits a family it tried once and abandoned,
    and picking randomly wastes iterations on lineages the run has already
    made a case against.

    With probability ``1 - EPSILON_GREEDY`` (exploit): the best branchable
    node, exactly as A6's placeholder chose it.

    With probability ``EPSILON_GREEDY`` (explore): a *non-best* branchable
    node, sampled with probability inversely proportional to how many times
    its family has already been attempted. This is what the acceptance
    criterion's "≥15% of nodes branch from a non-best parent" is checking,
    and the family weighting is what makes that 15% land on the families
    that most need attention rather than uniformly everywhere.

    One honest limit: ``family`` is a field the LLM sets on the proposed
    ``Action``, not something this function can dictate — select_parent
    chooses which existing node to anchor on, and biases exploration toward
    under-covered families, but full coverage is a joint outcome of this
    weighting and the "prefer an uncovered family" instruction already sent
    to the proposer in ``propose.py``.
    """
    rng = rng if rng is not None else random_module
    candidates = _branchable(nodes)
    if not candidates:
        return dict(ROOT_PARENT)

    best = max(candidates, key=lambda node: float(node["metrics"]["primary"]))
    others = [node for node in candidates if node.get("id") != best.get("id")]

    if others and rng.random() < EPSILON_GREEDY:
        counts = _family_counts(nodes)
        weights = [1.0 / (1 + counts.get(node.get("family"), 0)) for node in others]
        return rng.choices(others, weights=weights, k=1)[0]

    return best


def converged(history: list[dict]) -> bool:
    """Return whether three full evaluations failed to improve by ``EPSILON``.

    Smoke and screen pilots deliberately do not count: only full/confirm
    experiments consume official-validation evidence and therefore participate
    in the Section 4.5 stopping rule.
    """
    primaries = [
        float(node["metrics"]["primary"])
        for node in history
        if node.get("status") == "ok"
        and node.get("fidelity") in {"full", "confirm"}
        and isinstance(node.get("metrics", {}).get("primary"), (int, float))
        and math.isfinite(float(node["metrics"]["primary"]))
    ]
    window = NO_IMPROVEMENT_ITERATIONS
    if len(primaries) <= window:
        return False
    prior_best = max(primaries[:-window])
    return all(score <= prior_best + EPSILON for score in primaries[-window:])


def _persist(node: dict, *, tokens: dict | None = None, repair_attempted: bool = False) -> dict:
    """Write one completed attempt and return its normalised ledger record."""
    record = deepcopy(node)
    record["manual_intervention"] = False
    if tokens is not None:
        record["tokens"] = tokens
    if repair_attempted:
        record["repair_attempted"] = True
    path = store.write(record)
    return store.read(path.stem)


def _repair_failure_node(action: Action, fidelity: str, error: BaseException) -> dict:
    """Record an unavailable repair model as an ordinary failed attempt."""
    config = action.config.model_dump() if action.config is not None else {}
    return {
        "parent": action.parent,
        "family": action.family,
        "hypothesis": action.hypothesis,
        "reasoning": action.reasoning,
        "action_type": action.type,
        "fidelity": fidelity,
        "config": config,
        "diff": "repair proposal could not be executed",
        "status": "error",
        "metrics": {},
        "fold_primaries": [],
        "segments": {},
        "accepted": False,
        "errors": [
            {
                "stage": "repair",
                "error_class": "transient",
                "traceback": str(error),
            }
        ],
        "seconds": 0.0,
        "gpu_seconds": 0.0,
    }


def _with_retry_config(action: Action, config: dict) -> Action:
    """Apply A5's bounded config change without mutating its source Action."""
    return Action.model_validate({**action.model_dump(), "config": config})


def _execute_with_recovery(
    action: Action,
    fidelity: str,
    *,
    tokens: dict | None,
    timeout_s: int,
    sleep_fn: Callable[[float], None],
) -> tuple[dict, Action, list[dict]]:
    """Execute one tier and all recovery attempts allowed by A5."""
    current_action = action
    current_fidelity = fidelity
    current_tokens = tokens
    repair_attempted = False
    attempt = 0
    records: list[dict] = []

    while True:
        record = _persist(
            execute.execute(current_action, fidelity=current_fidelity, timeout_s=timeout_s),
            tokens=current_tokens,
            repair_attempted=repair_attempted,
        )
        records.append(record)
        if record.get("status") == "ok":
            return record, current_action, records

        plan = recovery.recover(record, attempt)
        if plan is None:
            return record, current_action, records
        if plan.get("backoff_s", 0.0):
            sleep_fn(float(plan["backoff_s"]))

        attempt += 1
        current_fidelity = str(plan["fidelity"])
        current_tokens = None  # an already-recorded LLM call must not be double counted
        if plan["operation"] == "retry":
            current_action = _with_retry_config(current_action, plan["config"])
            continue

        # A5 permits one syntax/schema repair.  A3 enforces that the repair
        # retains the original hypothesis, family, and parent.
        try:
            current_action, current_tokens = propose.repair(current_action, plan["error"])
            repair_attempted = True
        except propose.ProposeError as exc:
            failed_repair = _persist(
                _repair_failure_node(current_action, current_fidelity, exc),
                tokens=exc.usage,
                repair_attempted=True,
            )
            records.append(failed_repair)
            return failed_repair, current_action, records


def _screen_survives(screen: dict, parent: dict) -> bool:
    """Apply the Section 6.2 internal-fold promotion rule.

    The first viable candidate establishes an internal baseline.  Thereafter a
    screen run must have a positive median delta against its parent across the
    three expanding folds; errors or malformed/non-finite folds never promote.
    """
    folds = screen.get("fold_primaries", [])
    if screen.get("status") != "ok" or len(folds) != 3:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in folds):
        return False

    parent_folds = parent.get("fold_primaries", [])
    if len(parent_folds) != 3 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in parent_folds
    ):
        return True
    deltas = [
        float(candidate) - float(reference)
        for candidate, reference in zip(folds, parent_folds, strict=True)
    ]
    return median(deltas) > 0.0


def _proposal_failure(error: propose.ProposeError) -> None:
    """Keep billed/unavailable proposal attempts visible without forging a node."""
    store.append_event(
        {
            "event": "proposal",
            "status": "error",
            "error_class": "transient",
            "error": str(error),
            "tokens": error.usage,
            "manual_intervention": False,
        }
    )


def run(
    max_iterations: int = 50,  # K23 (02_REQUIREMENTS.md): hard per-run cap
    *,
    timeout_s: int = 1800,
    knowledge: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Run autonomous candidates through smoke → screen → full.

    ``max_iterations`` counts proposed candidates, rather than individual
    pilot nodes; a candidate can produce several persisted records.  The
    function returns the records written during this invocation for callers
    that want a live view, while durable state remains the append-only store.
    """
    bad_iterations = (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    )
    if bad_iterations:
        raise ValueError("max_iterations must be a positive integer")
    bad_timeout = isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 1
    if bad_timeout:
        raise ValueError("timeout_s must be a positive integer")

    manifest.preflight()
    knowledge = propose.load_knowledge() if knowledge is None else knowledge
    written: list[dict] = []

    for _ in range(max_iterations):
        history = store.list_nodes()
        if converged(history):
            break
        parent = select_parent(history)
        try:
            action, usage = propose.propose(history, knowledge, parent)
        except propose.ProposeError as exc:
            _proposal_failure(exc)
            continue

        smoke, current_action, records = _execute_with_recovery(
            action,
            "smoke",
            tokens=usage,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
        )
        written.extend(records)
        if smoke.get("status") != "ok" or smoke.get("fidelity") != "smoke":
            continue

        screen, current_action, records = _execute_with_recovery(
            current_action,
            "screen",
            tokens=None,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
        )
        written.extend(records)
        if screen.get("fidelity") != "screen" or not _screen_survives(screen, parent):
            continue

        full, _, records = _execute_with_recovery(
            current_action,
            "full",
            tokens=None,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
        )
        written.extend(records)
        store.append_event(
            {
                "event": "promotion",
                "from": "screen",
                "to": "full",
                "node": full["id"],
                "status": full["status"],
                "manual_intervention": False,
            }
        )

    return written


def main(max_iterations: int = 50) -> None:
    """CLI entry point; detailed records remain in ``logs/nodes``."""
    run(max_iterations=max_iterations)


if __name__ == "__main__":
    main()
