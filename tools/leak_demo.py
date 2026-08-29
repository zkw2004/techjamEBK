"""C1b/C4b demo: the agent cannot promote its own leak.

Runs two generated features through the full containment gauntlet
(syntax -> schema -> leakage audit -> smoke -> screen -> full):

* ``user_author_affinity``        — a safe temporal feature with a strict
  per-row date cutoff. Expected: ACCEPTED.
* ``user_author_affinity_leaky``  — its twin, which blends the target row's
  own outcome into the score through a runtime-assembled column name, so a
  static source grep cannot see the read. Expected: QUARANTINED by the
  dynamic outcome probe.

By default the demo runs on a small synthetic dataset so it works from a
clean clone with no data download; pass ``--real`` to run against the local
KuaiRand-Pure extract instead.

Usage:
    python -m tools.leak_demo [--real]

Exit code 0 iff the safe feature is accepted AND the leaky twin is
quarantined — so CI can keep the containment honest.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from unittest import mock

import numpy as np
import pandas as pd

from pipeline.codegen import (
    LEAKY_TWIN_SOURCE,
    USER_AUTHOR_AFFINITY_SOURCE,
    vet_generated_feature,
)

USERS = 12
AUTHORS = 6


def _synthetic_frame(rows: int, dates: list[int], seed: int) -> pd.DataFrame:
    """Impressions where users genuinely favour particular authors, so the
    affinity feature has real signal to find — legitimately."""
    rng = np.random.default_rng(seed)
    users = rng.integers(0, USERS, rows)
    authors = rng.integers(0, AUTHORS, rows)
    affinity = ((users + authors) % 3 == 0).astype(float)
    labels = (rng.random(rows) < 0.15 + 0.55 * affinity).astype(int)
    return pd.DataFrame(
        {
            "date": rng.choice(dates, rows),
            "user_id": users,
            "video_id": rng.integers(0, 300, rows),
            "author_id": authors,
            "tab": rng.integers(0, 2, rows),
            "duration_ms": rng.integers(1_000, 60_000, rows),
            "long_view": labels,
        }
    )


def _synthetic_data():
    train = _synthetic_frame(4_000, list(range(20220408, 20220422)), seed=1)
    validation = _synthetic_frame(800, list(range(20220422, 20220429)), seed=2)
    test = _synthetic_frame(800, list(range(20220429, 20220509)), seed=3)
    return train, validation, test


def _synthetic_folds():
    windows = [
        (list(range(20220408, 20220416)), list(range(20220416, 20220418))),
        (list(range(20220408, 20220418)), list(range(20220418, 20220420))),
        (list(range(20220408, 20220420)), list(range(20220420, 20220422))),
    ]
    return [
        (
            _synthetic_frame(2_500, fit_dates, seed=10 + i),
            _synthetic_frame(500, val_dates, seed=20 + i),
        )
        for i, (fit_dates, val_dates) in enumerate(windows)
    ]


def _print_report(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    for stage in report["stages"]:
        mark = "PASS" if stage["passed"] else "FAIL"
        print(f"  [{mark}] {stage['stage']:<8} {stage['detail']}")
    print(f"  -> {report['status'].upper()}: {report['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real", action="store_true",
        help="run against the local KuaiRand-Pure extract instead of synthetic data",
    )
    args = parser.parse_args(argv)

    from pipeline import train

    patches = contextlib.ExitStack()
    if not args.real:
        patches.enter_context(mock.patch.object(train, "_load_data", _synthetic_data))
        patches.enter_context(mock.patch.object(train, "_load_folds", _synthetic_folds))
        print("Running on synthetic data (pass --real for the KuaiRand-Pure extract).")

    with patches:
        safe = vet_generated_feature(
            "user_author_affinity", USER_AUTHOR_AFFINITY_SOURCE, log_events=False
        )
        leaky = vet_generated_feature(
            "user_author_affinity_leaky", LEAKY_TWIN_SOURCE, log_events=False
        )

    _print_report("safe feature: user_author_affinity", safe)
    _print_report("leaky twin: user_author_affinity_leaky", leaky)

    ok = safe["status"] == "accepted" and leaky["status"] == "quarantined"
    print(
        "\nContainment "
        + (
            "HELD: the safe feature was accepted and the leak was quarantined "
            "before it could report a score."
            if ok
            else "FAILED: expected safe=accepted and leaky=quarantined, got "
            f"safe={safe['status']!r} leaky={leaky['status']!r}."
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
