"""C3b: hypothesis-ledger evidence — every result is checkable, every claim cited.

Owner: Workstream C (Ethan). Additive module; no frozen contract changes.

`agent.schema.Action` already requires a hypothesis, and `agent.store` already
keeps an append-only node ledger. What was missing is the evidence bridge: a
full/confirm result packaged so a stated hypothesis can be checked against a
stated disproof condition, with the config/fidelity/seed triple that
reproduces it. A failed experiment recorded this way is still evidence — it
rules out a mechanism instead of vanishing into a leaderboard.

Records are pure functions of (config, fidelity, seed): C1 guarantees the
runner is deterministic, and `canonical()` strips the wall-clock telemetry so
two records of the same run compare equal.
"""

from __future__ import annotations

import copy
from numbers import Real

from pydantic import ValidationError

from agent.gate import MIN_DELTA_FLOOR
from agent.schema import Config

# Official validation-window baseline (Section 3). The default referee: a
# hypothesis whose experiment does not beat this by MIN_DELTA_FLOOR has not
# demonstrated an improvement, whatever its author hoped.
BASELINE_VALIDATION_PRIMARY = 0.6016

REQUIRED_RECORD_FIELDS = (
    "hypothesis", "disproof_condition", "config", "fidelity", "seed",
    "metrics", "fold_primaries", "segments", "decision",
)


def evidence_record(
    hypothesis: str,
    disproof_condition: str,
    config: dict,
    fidelity: str,
    seed: int,
    result: dict,
    baseline_primary: float | None = BASELINE_VALIDATION_PRIMARY,
    min_delta: float = MIN_DELTA_FLOOR,
) -> dict:
    """Package one runner result as a ledger-ready evidence record.

    `baseline_primary` is the score the hypothesis must beat — pass the parent
    node's primary when branching, or leave the official baseline default.
    """
    if not str(hypothesis).strip():
        raise ValueError("a falsifiable hypothesis is required — it is graded (Section 8.6)")
    if not str(disproof_condition).strip():
        raise ValueError(
            "a disproof condition is required: state the result that would "
            "reject the hypothesis before seeing the outcome"
        )
    try:
        parsed_config = Config.model_validate(config).model_dump()
    except ValidationError as exc:
        raise ValueError(f"config does not validate against the frozen schema: {exc}") from exc

    record = {
        "hypothesis": str(hypothesis),
        "disproof_condition": str(disproof_condition),
        "config": parsed_config,
        "fidelity": fidelity,
        "seed": int(seed),
        "metrics": {},
        "fold_primaries": [],
        "segments": {},
        "decision": {},
        "reproduce": {
            "config": copy.deepcopy(parsed_config), "fidelity": fidelity, "seed": int(seed),
        },
        "telemetry": {
            "seconds": result.get("seconds"),
            "gpu_seconds": result.get("gpu_seconds"),
            "peak_rss_mb": result.get("peak_rss_mb"),
        },
    }

    if result.get("status") != "ok":
        trace = str(result.get("traceback", "")).strip().splitlines()
        record["decision"] = {
            "passed": False,
            "outcome": "error",
            "error_class": result.get("error_class"),
            "evidence": (
                f"run failed at stage {result.get('stage')!r} with class "
                f"{result.get('error_class')!r}: {trace[-1] if trace else 'no traceback'}. "
                "A failed run is recorded evidence, not a discarded attempt"
            ),
        }
        return record

    if result.get("fidelity") not in ("full", "confirm"):
        raise ValueError(
            "evidence records require a full or confirm result; screen and smoke "
            f"runs are pilots, got fidelity {result.get('fidelity')!r}"
        )

    from pipeline.audit import audit_evidence

    evidence_finding = audit_evidence(result)
    record["metrics"] = {
        "gauc": float(result["gauc"]),
        "ndcg": float(result["ndcg"]),
        "primary": float(result["primary"]),
    }
    record["fold_primaries"] = [float(value) for value in result["fold_primaries"]]
    record["segments"] = {
        key: float(value) for key, value in dict(result["segments"]).items()
    }

    if not evidence_finding["passed"]:
        record["decision"] = {
            "passed": False,
            "outcome": "insufficient_evidence",
            "evidence": evidence_finding["detail"],
        }
        return record

    primary = record["metrics"]["primary"]
    if baseline_primary is None or not isinstance(baseline_primary, Real):
        raise ValueError("baseline_primary must be a number the candidate is judged against")
    delta = primary - float(baseline_primary)
    passed = delta >= min_delta
    blend_detail = ""
    if parsed_config["model"] == "blend":
        blend_accepted = result.get("blend_accepted") is True
        record["blend"] = {
            "accepted": blend_accepted,
            "gates": copy.deepcopy(result.get("blend_gates", {})),
            "parent_correlation": result.get("parent_correlation"),
            "weight": result.get("blend_weight"),
            "confirmation_weights": result.get("blend_weights"),
            "parent_primaries": result.get("parent_primaries"),
        }
        passed = passed and blend_accepted
        blend_detail = f"; C7 parent/fold/bootstrap acceptance: {blend_accepted}"
    record["decision"] = {
        "passed": passed,
        "outcome": "pass" if passed else "fail",
        "baseline_primary": float(baseline_primary),
        "delta": delta,
        "min_delta": min_delta,
        "evidence": (
            f"primary {primary:.4f} vs baseline {float(baseline_primary):.4f} "
            f"(delta {delta:+.4f}, floor {min_delta:.4f}); "
            f"fold primaries {[round(v, 4) for v in record['fold_primaries']]}; "
            f"{len(record['segments'])} segment metrics recorded{blend_detail}"
        ),
    }
    return record


def run_with_evidence(
    hypothesis: str,
    disproof_condition: str,
    config: dict,
    fidelity: str = "full",
    seed: int = 42,
    baseline_primary: float | None = BASELINE_VALIDATION_PRIMARY,
    timeout_s: int = 1800,
) -> dict:
    """Run one experiment and return its evidence record.

    Deterministic end to end: C1 makes the result a pure function of
    (config, fidelity, seed), and the record adds nothing run-dependent
    outside `telemetry`.
    """
    from pipeline.train import run_experiment

    result = run_experiment(config, fidelity=fidelity, seed=seed, timeout_s=timeout_s)
    return evidence_record(
        hypothesis, disproof_condition, config, fidelity, seed, result,
        baseline_primary=baseline_primary,
    )


def canonical(record: dict) -> dict:
    """The reproducible core of a record: everything except wall-clock telemetry."""
    return {key: value for key, value in record.items() if key != "telemetry"}


def to_node(
    record: dict,
    parent: str,
    family: str,
    action_type: str = "config",
    reasoning: str = "",
    **extra,
) -> dict:
    """Shape an evidence record for agent.store.write().

    Only Section 8.7 keys plus the C3b evidence extras — the store's
    normalise() keeps extra keys, so `disproof_condition` and `decision`
    ride along without touching the frozen node schema.
    """
    return {
        "parent": parent,
        "family": family,
        "action_type": action_type,
        "hypothesis": record["hypothesis"],
        "reasoning": reasoning,
        "fidelity": record["fidelity"],
        "config": record["config"],
        "metrics": record["metrics"],
        "fold_primaries": record["fold_primaries"],
        "segments": record["segments"],
        "status": "ok" if record["decision"].get("outcome") not in ("error",) else "error",
        "accepted": bool(record["decision"].get("passed")),
        "seconds": record["telemetry"].get("seconds") or 0.0,
        "gpu_seconds": record["telemetry"].get("gpu_seconds") or 0.0,
        "disproof_condition": record["disproof_condition"],
        "decision": record["decision"],
        "seed": record["seed"],
        **({"blend": copy.deepcopy(record["blend"])} if "blend" in record else {}),
        **extra,
    }
