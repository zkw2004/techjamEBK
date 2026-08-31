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

from agent import execute, gate, manifest, propose, recovery, store
from agent.schema import FAMILIES, Action

EPSILON = 0.002
NO_IMPROVEMENT_ITERATIONS = 3

# The bar every full-tier candidate must clear, not merely "whatever ran
# first". Sourced from the run manifest (D2) so the gate and the recorded
# contract can never disagree; pipeline/evidence.py carries the same number
# for its standalone records.
BASELINE_VALIDATION_PRIMARY = manifest.BASELINE_VALIDATION

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

    Restricted to completed full/confirm evaluations with a finite primary,
    that also cleared the bootstrap accept gate (Section 6.6, wired in run()'s
    accept_fn). A smoke or screen pilot never became the branch point: it was
    screened cheaply precisely so it would not consume attention (or official
    validation) as if it were evidence. A full-tier result that lost to the
    reference on a 95% CI is the same case one tier up: real evidence against
    the mechanism, not a candidate to build the next hypothesis on.
    """
    return [
        node
        for node in history
        if node.get("status") == "ok"
        and node.get("fidelity") in {"full", "confirm"}
        and node.get("accepted") is True
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
    """Write one completed attempt and return its normalised ledger record.

    ``accepted`` is not decided here — it is baked into ``node`` already, by
    C1's ``execute()`` applying ``accept_fn`` to the raw result before this
    function ever sees it (the Section 8.7 node shape has dropped the score
    arrays accept/reject needs by this point).
    """
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
    accept_fn: Callable[[dict], bool] | None = None,
) -> tuple[dict, Action, list[dict]]:
    """Execute one tier and all recovery attempts allowed by A5.

    ``accept_fn``, when given, is forwarded to C1's ``execute()`` only while
    the attempt is still running at the originally requested fidelity — never
    on a fallback from a fidelity-reducing retry, which won't carry
    full-validation score alignment. C1 applies it to the raw result (which
    still has the score arrays the Section 8.7 node shape drops) before
    persisting, so accept/reject is decided once, at write time, matching the
    ledger's append-only contract.
    """
    current_action = action
    current_fidelity = fidelity
    current_tokens = tokens
    repair_attempted = False
    attempt = 0
    records: list[dict] = []

    while True:
        active_accept_fn = accept_fn if current_fidelity == fidelity else None
        record = _persist(
            execute.execute(
                current_action,
                fidelity=current_fidelity,
                timeout_s=timeout_s,
                accept_fn=active_accept_fn,
            ),
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


def _screen_survives(screen: dict, reference_folds: list[float] | None) -> bool:
    """Apply the Section 6.2 internal-fold promotion rule.

    A screen run must have a positive median delta against its reference
    across the three expanding folds; errors or malformed/non-finite folds
    never promote. With no usable reference, the candidate promotes — the
    "first viable candidate" convention.

    ``reference_folds`` must come from a run at the *same* fidelity. Screen
    budgets are capped (``pipeline.train.SCREEN_BUDGET_CAPS``) while full-tier
    folds are not, so comparing a capped candidate against an uncapped parent
    asks it to win while handicapped and rejects genuine improvements as
    regressions.
    """
    folds = screen.get("fold_primaries", [])
    if screen.get("status") != "ok" or len(folds) != 3:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in folds):
        return False

    parent_folds = reference_folds or []
    if len(parent_folds) != 3 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in parent_folds
    ):
        return True
    deltas = [
        float(candidate) - float(reference)
        for candidate, reference in zip(folds, parent_folds, strict=True)
    ]
    return median(deltas) > 0.0


def _baseline_anchor_action() -> Action:
    """The official FM baseline, as an Action the ordinary executor can run.

    This is the organiser configuration the whole project is measured against
    (K12): FM, k=16, lr=0.001, over the five official ``FIELDS``. Spelling the
    features out matters — ``Config.features`` defaults to two fields, so a
    config that omits them silently trains ``user_id x video_id`` and lands
    near the popularity baseline instead of 0.6016.
    """
    from pipeline.data import FIELDS

    return Action(
        hypothesis=(
            "The official FM baseline (five fields, k=16, lr=0.001) reproduces "
            "validation primary 0.6016 and is the reference every later "
            "candidate must beat by at least the minimum meaningful delta."
        ),
        reasoning=(
            "Seeded by the loop, not proposed. A run with no baseline node has "
            "no valid comparison point: whichever candidate happened to run "
            "first would otherwise become the bar, and a below-baseline first "
            "result would make later non-improvements look like wins."
        ),
        type="config",
        family="model",
        parent=ROOT_PARENT["id"],
        config={"model": "fm", "features": list(FIELDS), "hparams": {"k": 16, "lr": 0.001}},
    )


SCREEN_REFERENCE_HYPOTHESIS = (
    "Reference screen of the incumbent, for a like-for-like fold comparison."
)


def _screen_reference_action(parent: dict) -> Action | None:
    """An Action that re-runs the parent's own config at screen fidelity."""
    config = parent.get("config") or {}
    if not config.get("model"):
        return None
    family = parent.get("family")
    return Action(
        hypothesis=SCREEN_REFERENCE_HYPOTHESIS,
        reasoning=(
            "Not a candidate: the screen gate needs the incumbent measured under "
            "the same capped budget as the candidate it is judging."
        ),
        type="config",
        family=family if family in FAMILIES else "model",
        parent=parent.get("id") or ROOT_PARENT["id"],
        config=config,
    )


def _has_baseline_anchor(history: list[dict]) -> bool:
    """Whether this ledger already carries a seeded, successful anchor."""
    return any(
        node.get("status") == "ok"
        and node.get("fidelity") in {"full", "confirm"}
        and node.get("gates", {}).get("baseline_anchor") is True
        for node in history
    )


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


# Set at the end of every run() call; read by callers (cli.py) that want to
# report *why* a run stopped without re-deriving it from the node history.
LAST_STOP_REASON: str | None = None


def run(
    max_iterations: int = 50,  # K23 (02_REQUIREMENTS.md): hard per-run cap
    *,
    timeout_s: int = 1800,
    max_hours: float | None = None,  # K24 (02_REQUIREMENTS.md): wall-clock ceiling
    knowledge: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
    seed_baseline: bool = True,
) -> list[dict]:
    """Run autonomous candidates through smoke → screen → full.

    ``max_iterations`` counts proposed candidates, rather than individual
    pilot nodes; a candidate can produce several persisted records.  The
    function returns the records written during this invocation for callers
    that want a live view, while durable state remains the append-only store.

    Stops on whichever comes first: convergence (Section 4.5), the iteration
    cap, or ``max_hours`` of wall-clock time — checked before each new
    candidate starts, never mid-candidate, so a long-running full evaluation
    is never killed partway through. ``LAST_STOP_REASON`` records which one
    fired: ``"converged"``, ``"iteration_cap"``, or ``"time_cap"``.

    ``seed_baseline`` runs the official FM baseline once, before iteration 1,
    and installs it as the incumbent. It costs one full-tier evaluation and is
    skipped when the ledger already carries an anchor, so a resumed run does
    not repeat it. Turn it off only to isolate other behaviour under test:
    without an anchor there is no incumbent score vector, so the first
    candidate is gated on the baseline margin alone.
    """
    global LAST_STOP_REASON
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
    if max_hours is not None and (isinstance(max_hours, bool) or max_hours <= 0):
        raise ValueError("max_hours must be a positive number when given")

    manifest.preflight()
    knowledge = propose.load_knowledge() if knowledge is None else knowledge
    written: list[dict] = []
    deadline = time_fn() + max_hours * 3600 if max_hours is not None else None

    # In-memory reference for the accept gate (Section 6.6, D3). Node records
    # don't persist `seed`, so a past node's raw per-row scores cannot be
    # reliably re-derived across a resumed run() call; scoped to this
    # invocation.
    best_full: dict[str, Any] | None = None

    def _accept_full(raw: dict) -> dict:
        """Gate one full-tier result and return the verdict plus its evidence.

        Two absolute requirements, both of which the previous implementation
        skipped for the first full node it ever saw:

        * the candidate must beat ``max(official baseline, best so far)`` by
          ``MIN_DELTA_FLOOR``. Accepting the first full result unconditionally
          made whatever ran first the permanent bar — a run anchored on a node
          scoring *below* the shipped baseline reports later results as wins
          when they are not.
        * the bootstrap CI on the delta must exclude zero (Section 6.6). Skipped
          for the first node, this is what lets seed noise through as evidence.

        There is no score vector for the official baseline, so its gate is the
        margin check alone; the CI check applies from the second full node on,
        against the incumbent's actual per-row scores.
        """
        nonlocal best_full
        primary = raw.get("primary")
        val_scores = raw.get("val_scores")
        val_user_ids = raw.get("val_user_ids")
        if val_scores is None or val_user_ids is None or primary is None:
            # Nothing to gate on; never silently accept.
            return {"accepted": False, "gates": {"statistical": False}}

        incumbent = best_full["primary"] if best_full is not None else None
        reference = max(BASELINE_VALIDATION_PRIMARY, incumbent or 0.0)
        delta = float(primary) - reference
        clears_margin = delta >= gate.MIN_DELTA_FLOOR

        ci: tuple[float, float] | None = None
        if best_full is None:
            ci_excludes_zero = True  # no incumbent score vector to bootstrap against
        else:
            ci_excludes_zero, ci = gate.accept(val_scores, best_full["scores"], val_user_ids)

        accepted = bool(clears_margin and ci_excludes_zero)
        if accepted:
            best_full = {"scores": val_scores, "user_ids": val_user_ids, "primary": primary}
        return {
            "accepted": accepted,
            "delta_vs_best": delta,
            "ci_95": list(ci) if ci is not None else None,
            "gates": {
                "statistical": bool(ci_excludes_zero),
                "min_delta": bool(clears_margin),
            },
        }

    def _accept_anchor(raw: dict) -> dict:
        """Install the seeded baseline as the incumbent, without gating it.

        The anchor is the reference by construction, so it is not asked to
        beat itself by MIN_DELTA_FLOOR. Recording it as the incumbent is what
        gives every later candidate a real score vector to bootstrap against,
        so the CI check applies from the very first proposed node rather than
        being skipped for it.
        """
        nonlocal best_full
        val_scores = raw.get("val_scores")
        val_user_ids = raw.get("val_user_ids")
        primary = raw.get("primary")
        if val_scores is None or val_user_ids is None or primary is None:
            return {"accepted": False, "gates": {"baseline_anchor": False}}
        best_full = {"scores": val_scores, "user_ids": val_user_ids, "primary": primary}
        return {
            "accepted": True,
            "delta_vs_best": float(primary) - BASELINE_VALIDATION_PRIMARY,
            "gates": {"baseline_anchor": True},
        }

    # Screen-budget fold scores per parent id, so the screen gate compares like
    # with like. Memoised: this costs one extra screen run the first time a
    # parent is used as a reference, and nothing thereafter. It is deliberately
    # not persisted — a reference measurement is not an experiment, and the
    # ledger is the run-log deliverable.
    reference_folds_by_parent: dict[str, list[float] | None] = {}

    def _reference_folds(parent: dict) -> list[float] | None:
        parent_id = parent.get("id")
        if parent_id is None:
            return None
        if parent_id not in reference_folds_by_parent:
            action = _screen_reference_action(parent)
            folds = None
            if action is not None:
                result = execute.execute(action, fidelity="screen", timeout_s=timeout_s)
                if result.get("status") == "ok":
                    folds = result.get("fold_primaries") or None
            reference_folds_by_parent[parent_id] = folds
        return reference_folds_by_parent[parent_id]

    if seed_baseline and not _has_baseline_anchor(store.list_nodes()):
        anchor, _action, anchor_records = _execute_with_recovery(
            _baseline_anchor_action(),
            "full",
            tokens=None,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
            accept_fn=_accept_anchor,
        )
        written.extend(anchor_records)
        store.append_event(
            {
                "event": "baseline_anchor",
                "node": anchor.get("id"),
                "status": anchor.get("status"),
                "primary": anchor.get("metrics", {}).get("primary"),
                "manual_intervention": False,
            }
        )

    LAST_STOP_REASON = "iteration_cap"
    for _ in range(max_iterations):
        history = store.list_nodes()
        if converged(history):
            LAST_STOP_REASON = "converged"
            break
        if deadline is not None and time_fn() >= deadline:
            LAST_STOP_REASON = "time_cap"
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
        if screen.get("fidelity") != "screen" or not _screen_survives(
            screen, _reference_folds(parent)
        ):
            continue

        full, _, records = _execute_with_recovery(
            current_action,
            "full",
            tokens=None,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
            accept_fn=_accept_full,
        )
        written.extend(records)
        store.append_event(
            {
                "event": "promotion",
                "from": "screen",
                "to": "full",
                "node": full["id"],
                "status": full["status"],
                "accepted": full.get("accepted", False),
                "manual_intervention": False,
            }
        )

    return written


def main(max_iterations: int = 50, max_hours: float | None = None) -> None:
    """CLI entry point; detailed records remain in ``logs/nodes``."""
    run(max_iterations=max_iterations, max_hours=max_hours)


if __name__ == "__main__":
    main()
