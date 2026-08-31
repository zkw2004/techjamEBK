"""Independent leakage canaries run before autonomous experimentation (D13)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

GAUC_CENTER = 0.5
GAUC_TOLERANCE = 0.02
DEFAULT_RESULT_PATH = Path("logs/leak_preflight.json")


class LeakPreflightError(RuntimeError):
    """A leakage canary failed closed."""


def permute_labels_within_user(frame: Any, *, seed: int = 42):
    """Return a copy with each user's label multiset independently permuted."""
    from pipeline.data import LABEL

    if LABEL not in frame.columns or "user_id" not in frame.columns:
        raise LeakPreflightError(f"canary frame requires 'user_id' and {LABEL!r}")
    permuted = frame.copy()
    labels = permuted[LABEL].to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    positions = permuted.groupby("user_id", sort=False).indices
    for indices in positions.values():
        take = np.asarray(indices, dtype=np.int64)
        labels[take] = rng.permutation(labels[take])
    permuted[LABEL] = labels
    return permuted


def deliberately_leaky_feature(_train_df: Any, target_df: Any) -> np.ndarray:
    """Fixture-only future-label feature that the production guard must reject."""
    from pipeline.data import LABEL

    return target_df[LABEL].to_numpy(dtype=float)


def _clean_canary(
    train_frame: Any,
    validation_frame: Any,
    *,
    seed: int,
    fit_predict: Callable[[dict, Any, Any, list[Any], int], list[np.ndarray]],
    evaluate: Callable[[Any, np.ndarray], dict[str, float]],
) -> float:
    """Retrain a small FM after within-user label permutation and return GAUC."""
    from pipeline.data import FIELDS

    config = {
        "model": "fm",
        "loss": "pointwise",
        "features": list(FIELDS),
        "negative_sampling": "all",
        "hparams": {"k": 4, "lr": 0.001, "max_epochs": 3, "patience": 1},
        "parents": [],
        "blend_method": "rank_avg",
        "seed": seed,
    }
    permuted = permute_labels_within_user(train_frame, seed=seed)
    (scores,) = fit_predict(config, permuted, validation_frame, [validation_frame], seed)
    return float(evaluate(validation_frame, scores)["gauc"])


def run_preflight(
    *,
    seed: int = 42,
    frames: tuple[Any, Any] | None = None,
    fit_predict: Callable | None = None,
    evaluate: Callable | None = None,
    leakage_guard: Callable | None = None,
) -> dict[str, Any]:
    """Run both D13 canaries and raise unless the clean/leaky controls pass."""
    from pipeline import features, train
    from pipeline.data import load

    if frames is None:
        train_frame, validation_frame, _ = load()
    else:
        train_frame, validation_frame = frames
    fit_predict = fit_predict or train._fit_and_predict  # noqa: SLF001
    evaluate = evaluate or train._evaluate  # noqa: SLF001
    leakage_guard = leakage_guard or features.leakage_check

    gauc = _clean_canary(
        train_frame,
        validation_frame,
        seed=seed,
        fit_predict=fit_predict,
        evaluate=evaluate,
    )
    clean_passed = abs(gauc - GAUC_CENTER) <= GAUC_TOLERANCE
    leaky_rejected = leakage_guard(
        deliberately_leaky_feature, train_frame, validation_frame
    ) is False
    result = {
        "status": "passed" if clean_passed and leaky_rejected else "failed",
        "seed": seed,
        "clean_permuted_gauc": gauc,
        "clean_expected_range": [GAUC_CENTER - GAUC_TOLERANCE, GAUC_CENTER + GAUC_TOLERANCE],
        "clean_passed": clean_passed,
        "leaky_fixture_rejected": leaky_rejected,
    }
    if not clean_passed:
        raise LeakPreflightError(
            f"permuted-label GAUC {gauc:.6f} is outside "
            f"[{GAUC_CENTER - GAUC_TOLERANCE:.2f}, {GAUC_CENTER + GAUC_TOLERANCE:.2f}]"
        )
    if not leaky_rejected:
        raise LeakPreflightError("deliberately leaky future-label feature was not rejected")
    return result


def write_result(result: dict[str, Any], path: Path | str = DEFAULT_RESULT_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def record_preflight(result: dict[str, Any]) -> None:
    """Put the once-per-run verdict in the append-only run-log header."""
    from agent import store

    store.append_event(
        {
            "event": "leakage_preflight",
            **result,
            "manual_intervention": False,
        }
    )


def ensure_preflight(*, seed: int = 42) -> dict[str, Any]:
    """Run and record D13 once for the current ledger, before iteration 1."""
    from agent import store

    prior = [event for event in store.read_events() if event.get("event") == "leakage_preflight"]
    if prior and prior[-1].get("status") == "passed":
        return prior[-1]
    result = run_preflight(seed=seed)
    write_result(result)
    record_preflight(result)
    return result
