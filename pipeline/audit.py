"""C1b: self-audit battery — the runner proves it cannot cheat itself.

Owner: Workstream C (Ethan). Additive module; no frozen contract changes.

Automated experiment loops fail by optimising into leakage: a leaked feature
scores brilliantly and means nothing on the hidden test. This module gives the
agent (and CI) four checks, each returning a structured finding with an
actionable detail message:

* result-schema validity   — a malformed runner result is caught, not stored
* determinism              — same (config, fidelity, seed) twice, same output
* label-shuffle canary     — retrain on permuted labels; the score MUST
                             collapse to chance. A score that survives label
                             shuffling is reading the answer somewhere.
* evaluation evidence      — full/confirm results carry the fold, segment,
                             and alignment evidence the ledger needs (C3b)

The label-shuffle canary is the strongest cheap detector: it catches leaks the
static source scan cannot see, because it tests the *score*, not the code.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

# Chance-level primary is ~0.475 (Random reference, Section 3). Real signal
# starts near the popularity reference 0.5715. A shuffled-label run scoring
# above this threshold cannot be explained by a legitimately trained model.
SHUFFLE_CANARY_THRESHOLD = 0.55
RANDOM_REFERENCE_PRIMARY = 0.4753


def _finding(check: str, passed: bool, detail: str) -> dict:
    return {"check": check, "passed": bool(passed), "detail": detail}


def _error_detail(result: dict) -> str:
    trace = str(result.get("traceback", "")).strip().splitlines()
    last = trace[-1] if trace else "no traceback"
    return (
        f"runner returned status=error at stage {result.get('stage')!r} "
        f"(class {result.get('error_class')!r}): {last}"
    )


def audit_result_schema(result: dict, fidelity: str) -> dict:
    """Validate a runner result against the Section 8.5 shape."""
    from pipeline.train import _result_schema_issue

    if not isinstance(result, dict):
        return _finding(
            "result_schema", False,
            f"result must be a dict, got {type(result).__name__}",
        )
    try:
        issue = _result_schema_issue(result, fidelity)
    except (TypeError, ValueError) as exc:
        issue = f"{type(exc).__name__}: {exc}"
    if issue is not None:
        return _finding(
            "result_schema", False,
            f"{issue} — fix the runner output before trusting any score from it",
        )
    return _finding("result_schema", True, f"result matches the {fidelity} contract shape")


def audit_determinism(
    config: dict, fidelity: str = "smoke", seed: int = 42, timeout_s: int = 600
) -> dict:
    """Run the same experiment twice; every score and metric must match.

    Nondeterminism silently breaks the ledger: a recorded result that cannot
    be reproduced from (config, fidelity, seed) is not evidence.
    """
    from pipeline.train import run_experiment

    first = run_experiment(config, fidelity=fidelity, seed=seed, timeout_s=timeout_s)
    second = run_experiment(config, fidelity=fidelity, seed=seed, timeout_s=timeout_s)
    for result in (first, second):
        if result.get("status") != "ok":
            return _finding("determinism", False, _error_detail(result))

    for name in ("gauc", "ndcg", "primary"):
        if first[name] != second[name]:
            return _finding(
                "determinism", False,
                f"{name} differs across identical runs ({first[name]!r} vs "
                f"{second[name]!r}); find and seed the unseeded randomness "
                "(model init, sampling, thread-order reductions)",
            )
    for name in ("val_scores", "test_scores", "fold_primaries"):
        a = np.asarray(first[name], dtype=float)
        b = np.asarray(second[name], dtype=float)
        if a.shape != b.shape or not np.array_equal(a, b):
            return _finding(
                "determinism", False,
                f"{name} differs across identical runs "
                f"(max abs diff {np.max(np.abs(a - b)) if a.shape == b.shape else 'n/a'}); "
                "same (config, fidelity, seed) must reproduce bit-identical scores",
            )
    return _finding(
        "determinism", True,
        f"two {fidelity} runs at seed {seed} produced identical metrics and scores",
    )


def _with_shuffled_label(frame, label: str, rng: np.random.Generator):
    if isinstance(frame, dict):
        out = dict(frame)
        out[label] = rng.permutation(np.asarray(frame[label]))
        return out
    out = frame.copy()
    out[label] = rng.permutation(out[label].to_numpy())
    return out


def audit_label_shuffle(
    config: dict,
    seed: int = 42,
    threshold: float = SHUFFLE_CANARY_THRESHOLD,
    timeout_s: int = 1800,
) -> dict:
    """Retrain on permuted training labels; the score must collapse to chance.

    Fold-training labels are shuffled, fold-validation labels are left intact
    for scoring, and the run uses the screen tier — internal folds only, so
    the canary never spends the official validation window. A model fitted to
    noise cannot rank; if the primary stays high, some input is carrying the
    label (a target-side read, a same-row outcome, a future aggregate).
    """
    import pipeline.data as data
    from pipeline import train

    real_load_folds = train._load_folds
    label = data.LABEL

    def shuffled_folds():
        rng = np.random.default_rng(seed)
        return [
            (_with_shuffled_label(fold_train, label, rng), fold_validation)
            for fold_train, fold_validation in real_load_folds()
        ]

    # In-process patch; the fork-started child inherits it (see
    # train._process_context, which prefers fork for exactly this reason).
    with mock.patch.object(train, "_load_folds", shuffled_folds):
        result = train.run_experiment(config, fidelity="screen", seed=seed, timeout_s=timeout_s)

    if result.get("status") != "ok":
        if result.get("error_class") == "leak_suspected":
            return _finding(
                "label_shuffle", False,
                "shuffled-label run STILL tripped the >0.75 canary — the pipeline "
                "is reading the label from somewhere other than training labels",
            )
        return _finding("label_shuffle", False, _error_detail(result))

    primary = float(result["primary"])
    if primary > threshold:
        return _finding(
            "label_shuffle", False,
            f"shuffled-label primary {primary:.4f} exceeds {threshold:.2f}; a "
            f"leak-free pipeline collapses to ~{RANDOM_REFERENCE_PRIMARY:.4f}. Some "
            "feature or model input encodes the target row's outcome — audit the "
            "feature list with pipeline.features.leakage_check and quarantine the run",
        )
    return _finding(
        "label_shuffle", True,
        f"shuffled-label primary {primary:.4f} is at chance level "
        f"(threshold {threshold:.2f}); no input is carrying the label",
    )


def audit_evidence(result: dict) -> dict:
    """A full/confirm result must carry the evidence the hypothesis ledger needs.

    Without per-fold scores, segment metrics, and aligned validation vectors, a
    hypothesis cannot be checked against its disproof condition (C3b).
    """
    if result.get("status") != "ok":
        return _finding("evidence", False, _error_detail(result))
    fidelity = result.get("fidelity")
    if fidelity not in ("full", "confirm"):
        return _finding(
            "evidence", False,
            f"evidence is only required (and only complete) at full/confirm, got {fidelity!r}",
        )
    problems = []
    fold_primaries = np.asarray(result.get("fold_primaries", []), dtype=float)
    if len(fold_primaries) != 3 or not np.isfinite(fold_primaries).all():
        problems.append("fold_primaries must hold three finite internal-fold scores")
    segments = result.get("segments")
    if not isinstance(segments, dict) or not segments:
        problems.append(
            "segments is empty — segment metrics (activity/popularity/day) are "
            "required evidence for full runs; ensure the frames carry date, "
            "user_id, and video_id columns so the runner can populate them"
        )
    val_scores = np.asarray(result.get("val_scores", []))
    val_user_ids = np.asarray(result.get("val_user_ids", []))
    if len(val_scores) == 0 or len(val_scores) != len(val_user_ids):
        problems.append("val_scores and val_user_ids must be non-empty and aligned")
    if problems:
        return _finding("evidence", False, "; ".join(problems))
    return _finding(
        "evidence", True,
        "result carries fold primaries, segment metrics, and aligned validation scores",
    )


def self_audit(config: dict, seed: int = 42, shuffle_timeout_s: int = 1800) -> dict:
    """Run the pre-flight battery for one experiment config.

    Returns {"passed": bool, "checks": [findings]}. Runs the cheap checks
    first; the label-shuffle canary (a real screen-tier training run) goes
    last so an already-broken config fails fast with a cheaper message.
    """
    from pipeline.train import run_experiment

    checks: list[dict] = []
    smoke = run_experiment(config, fidelity="smoke", seed=seed, timeout_s=600)
    if smoke.get("status") != "ok":
        checks.append(_finding("result_schema", False, _error_detail(smoke)))
        return {"passed": False, "checks": checks}
    checks.append(audit_result_schema(smoke, "smoke"))
    checks.append(audit_determinism(config, fidelity="smoke", seed=seed))
    checks.append(audit_label_shuffle(config, seed=seed, timeout_s=shuffle_timeout_s))
    return {"passed": all(item["passed"] for item in checks), "checks": checks}
