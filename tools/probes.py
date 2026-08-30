"""The five Day-1 probes. Task D5.

Single command runs all five and prints a comparison table. THE OUTPUT
DETERMINES THE SECTION 6.7 MODEL LADDER ORDERING — do not assume an ordering
in advance (Section 6.4; Dacrema 2019, Rendle 2020).

Total compute budget: under one hour.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable
from typing import Any

BASE_FEATURES = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
AGGREGATE_FEATURES = [
    "user_ctr", "video_ctr", "video_impressions", "user_activity", "user_tag_affinity",
]
ALL_FEATURES = BASE_FEATURES + AGGREGATE_FEATURES

PROBES: dict[str, dict[str, Any]] = {
    "P1": {
        "question": "Does feature engineering alone beat the FM baseline?",
        "config": {"model": "fm", "features": ALL_FEATURES},
    },
    "P2": {
        "question": "Does pointwise GBDT beat FM on the same features?",
        "config": {"model": "lgbm", "loss": "pointwise", "features": ALL_FEATURES},
    },
    "P3": {
        "question": "Does a ranking loss help LightGBM?",
        "config": {"model": "lgbm", "loss": "lambdarank", "features": ALL_FEATURES},
    },
    "P4": {
        "question": "Is the neural branch worth pursuing?",
        "config": {
            "model": "deepfm", "features": ALL_FEATURES,
            "hparams": {"emb_dim": 16, "mlp": [128, 64], "max_epochs": 3, "patience": 1},
        },
    },
    "P5": {
        "question": "Was the FM baseline simply undertuned?",
        "config": {"model": "fm", "features": BASE_FEATURES},
    },
}


def _tune_fm(
    runner: Callable[..., dict], fidelity: str, trials: int, seed: int, storage: str,
) -> dict:
    """Reuse C6's bounded, resumable internal-fold study, then evaluate once.

    Even with --fidelity full the official validation labels never enter the
    objective. The same seed is used across trials to compare configurations.
    """
    from pipeline.train import _cache_code_fingerprint
    from pipeline.tune import run_study

    started = time.monotonic()
    study = run_study(
        PROBES["P5"]["config"],
        {"k": ["categorical", [8, 16, 32, 64]],
         "lr": ["loguniform", 1e-4, 1e-2],
         "l2": ["loguniform", 1e-6, 1e-3],
         "max_epochs": ["int", 1, 3]},
        budget=trials, seed=seed, storage=storage,
        study_name=f"d5-p5-{seed}-{_cache_code_fingerprint()[:16]}",
    )
    if study["best_value"] is None:
        return {"status": "error", "error_class": "transient", "primary": None,
                "seconds": time.monotonic() - started, "tuning": study}
    config = {**PROBES["P5"]["config"], "hparams": study["best_params"]}
    result = runner(config, fidelity=fidelity, seed=seed)
    return {**result, "trials": trials, "best_hparams": study["best_params"],
            "seconds": time.monotonic() - started, "tuning": study}


def run_probes(
    runner: Callable[..., dict], *, fidelity: str = "screen", fm_trials: int = 30, seed: int = 42,
    study_storage: str = "sqlite:///logs/optuna/probes.db",
) -> list[dict[str, Any]]:
    """Run P1-P5 in fixed order; failures remain visible in the table."""
    rows = []
    for index, (probe_id, spec) in enumerate(PROBES.items()):
        if probe_id == "P5":
            started = time.monotonic()
            try:
                result = _tune_fm(runner, fidelity, fm_trials, seed + index, study_storage)
            except Exception:
                # A locked/unavailable study is not a failed model trial. Keep
                # P1-P4 visible and let an interrupted process still exit.
                result = {"status": "error", "error_class": "transient",
                          "seconds": time.monotonic() - started,
                          "traceback": traceback.format_exc()}
        else:
            result = runner(spec["config"], fidelity=fidelity, seed=seed + index)
        rows.append(
            {
                "probe": probe_id,
                "question": spec["question"],
                "status": result.get("status", "error"),
                "gauc": result.get("gauc"),
                "ndcg": result.get("ndcg"),
                "primary": result.get("primary"),
                "seconds": result.get("seconds", 0.0),
                "error_class": result.get("error_class", ""),
                **({"traceback": result["traceback"]} if "traceback" in result else {}),
                **({"tuning": result["tuning"]} if "tuning" in result else {}),
            }
        )
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    def metric(row, name):
        return "—" if row[name] is None else f"{float(row[name]):.6f}"

    headers = ("Probe", "Status", "GAUC", "nDCG", "Primary", "Seconds", "Question")
    body = []
    for row in rows:
        status = row["status"]
        if row["error_class"]:
            status = f"{status}:{row['error_class']}"
        body.append(
            (row["probe"], status, metric(row, "gauc"), metric(row, "ndcg"),
             metric(row, "primary"),
             f"{float(row['seconds']):.1f}", row["question"])
        )
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in body))
        for i in range(len(headers))
    ]

    def line(row):
        return " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))

    return "\n".join([line(headers), "-+-".join("-" * width for width in widths),
                      *(line(row) for row in body)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", choices=("screen", "full"), default="screen")
    parser.add_argument("--fm-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-storage", default="sqlite:///logs/optuna/probes.db")
    args = parser.parse_args(argv)
    if args.fm_trials <= 0:
        parser.error("--fm-trials must be positive")
    from pipeline.train import run_experiment

    print(render_table(run_probes(
        run_experiment, fidelity=args.fidelity, fm_trials=args.fm_trials, seed=args.seed,
        study_storage=args.study_storage,
    )))


if __name__ == "__main__":
    main()
