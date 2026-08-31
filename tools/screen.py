"""Within-user variance screen. Task B12.

GAUC and nDCG@5 are computed *per user, then averaged* (AGENT_PLAN.md
Section 5): a feature that takes the same value for every row belonging to
one user can never change that user's internal ranking. It is invisible to
the metric except through an interaction with something that *does* vary
within-user (Section 5.3; README's own corrections log: "Pure user-side
first-order terms contribute exactly zero, since ranking is within-user").

This module answers, for every feature currently in `pipeline.features.
FEATURES`: does it actually vary within a user's own impression list on the
official validation split, or is it constant there (``metric_inert``)?

**Registry-agnostic by design.** `screen_features()` takes a `{name: fn}`
mapping and a validation DataFrame as plain arguments — it does not import
or hardcode anything about which features exist. It iterates whatever is
registered at call time. B10's and B11's feature packs are separate,
currently-unmerged PRs; once merged, their entries in
`pipeline.features.FEATURES` are picked up automatically, with zero changes
to this file. The one feature name this module's docstring/tests mention by
name (`user_activity`) is only for the acceptance criterion's worked
example — nothing in the implementation depends on it.

--------------------------------------------------------------------------
Report schema (documented in Section 8 style — a frozen-shaped dict literal
with inline comments; this is B12's analogue of Section 8.3/8.7, not itself
a frozen contract, but the shape a future consumer such as A10's ablation
prompt injection should read):

    {
      "schema_version": "1",            # bump only on a breaking shape change
      "threshold": 1e-9,                # metric_inert cutoff used this run
      "ddof": 0,                        # variance convention, see per_user_variance()
      "split": "official_validation",   # which frame was screened
      "timestamp": "2026-08-31T00:00:00+00:00",  # UTC ISO-8601, report build time
      "features": {
        "user_activity": {
          "status": "ok",                    # "ok" | "error"
          "mean_within_user_variance": 0.0,   # None when status == "error"
          "metric_inert": true,               # None when status == "error"
          "n_users_considered": 812,          # users with >=2 val rows; None on error
          "n_users_total": 900,               # every distinct user_id in the split
          "error": None,                      # "<ExceptionType>: <message>" on error
        },
        "video_ctr": {
          "status": "ok",
          "mean_within_user_variance": 0.0143,
          "metric_inert": false,
          "n_users_considered": 812,
          "n_users_total": 900,
          "error": None,
        },
        # ... one entry per name in the screened features mapping ...
      },
    }

--------------------------------------------------------------------------
Design decisions:

1. **Variance convention: ddof=0 (population variance), configurable.**
   Each user's validation impression list *is* the population we care
   about — we are asking "did this feature actually differ across the rows
   this user was shown", not estimating a variance parameter of some larger
   hypothetical population beyond that list. ddof=0 also never divides by
   zero for a 2-row user, unlike ddof=1. Callers who want the unbiased
   sample-variance convention instead can pass ``ddof=1`` to any function
   here; the report records whichever was used.

2. **Single-impression users are excluded from the mean**, mirroring GAUC's
   own "Pinned convention" (Section 5.1): GAUC counts only users with
   `0 < positives < impressions`, degenerate per-user groups are excluded
   rather than assigned an arbitrary value. A user with exactly one
   validation row has an undefined "did it vary" question — there is
   nothing to compare it to — so it is dropped from both the per-user
   variance average and `n_users_considered`, while still counting toward
   `n_users_total` so the report is honest about how much of the split
   actually informed the number.

3. **A feature can legitimately fail to compute** (KeyError on a missing
   column because B10/B11 haven't merged yet, a malformed dependency, a
   wrong-length return array, ...). `screen_feature()` catches any
   exception, records `status: "error"` plus a short message, and moves on
   — one bad feature must not abort the screen for every other feature
   (this repo's general "fail per-component, not the whole run"
   philosophy; compare `tools/data_usage.py`'s per-file accounting and
   `pipeline.features.leakage_check`'s per-feature guard).

4. **Exit-code philosophy: this tool never exits non-zero for what it
   finds.** Unlike `tools/data_usage.py` (B13), whose non-zero exit marks a
   real correctness gap (an unaccounted-for shipped file), B12's findings
   are informational: `metric_inert: true` is an expected, useful signal
   this tool exists to produce, not a defect, and a per-feature `status:
   "error"` is recorded in the report rather than treated as a screen
   failure — the whole point is that the loop and a human can see it
   without the run being flagged red. `main()` therefore always returns 0
   once it can build a report at all; a failure to load the data or
   feature registry in the first place surfaces as an ordinary uncaught
   exception (non-zero exit from the interpreter), which is the correct
   signal for "this environment cannot run the screen", not something this
   module should swallow.

5. **Computation and I/O are separate.** `screen_features()` /
   `build_report()` take a features mapping and a DataFrame and return a
   plain dict — no disk access, so `test_screen.py` exercises them against
   small synthetic frames without the real ~125K-row validation split or
   `logs/` existing or being writable. `write_report()` is the only
   function that touches disk, with a `logs/screen_report.json` default
   that a caller can override; the CLI entry point wires the two together
   and is what calls `pipeline.data.load()` for real use.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 1e-9
DEFAULT_DDOF = 0
SCHEMA_VERSION = "1"
DEFAULT_SPLIT_NAME = "official_validation"
DEFAULT_REPORT_PATH = Path("logs") / "screen_report.json"
DEFAULT_USER_ID_COLUMN = "user_id"


def per_user_variance(
    values: np.ndarray,
    user_ids: np.ndarray,
    *,
    ddof: int = DEFAULT_DDOF,
) -> tuple[float, int, int]:
    """Mean within-user variance of ``values``, grouped by ``user_ids``.

    Returns ``(mean_variance, n_users_considered, n_users_total)``.

    Users with exactly one row are excluded from the mean — see design
    decision 2 in the module docstring — but still counted in
    ``n_users_total``. If every user has fewer than two rows,
    ``mean_variance`` is ``0.0`` and ``n_users_considered`` is ``0`` (there
    is genuinely nothing to average; this is not treated as an error, since
    a caller may legitimately screen a tiny synthetic split).
    """
    values = np.asarray(values, dtype=np.float64)
    user_ids = np.asarray(user_ids)
    if values.shape[0] != user_ids.shape[0]:
        raise ValueError("values and user_ids must have matching length")
    if values.shape[0] == 0:
        raise ValueError("per_user_variance requires at least one row")
    if not np.isfinite(values).all():
        raise ValueError("feature values must be finite to compute within-user variance")

    frame = pd.DataFrame({"_user": user_ids, "_value": values})
    grouped = frame.groupby("_user", sort=False)["_value"]
    counts = grouped.size()
    n_users_total = int(counts.shape[0])

    eligible = counts.to_numpy() >= 2
    n_users_considered = int(eligible.sum())
    if n_users_considered == 0:
        return 0.0, 0, n_users_total

    variances = grouped.var(ddof=ddof)
    mean_variance = float(variances.to_numpy()[eligible].mean())
    return mean_variance, n_users_considered, n_users_total


def screen_feature(
    name: str,
    fn: Callable[[Any, Any], np.ndarray],
    train_df: Any,
    val_df: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ddof: int = DEFAULT_DDOF,
    user_id_column: str = DEFAULT_USER_ID_COLUMN,
) -> dict[str, Any]:
    """Screen one feature. Never raises — failures are captured in the result.

    Calls ``fn(train_df, val_df)`` per the frozen feature-builder signature
    (Section 8.4), groups the returned values by ``val_df[user_id_column]``,
    and reports the mean within-user variance plus the ``metric_inert``
    flag (``mean_within_user_variance < threshold``).
    """
    if user_id_column not in val_df.columns:
        return {
            "status": "error",
            "mean_within_user_variance": None,
            "metric_inert": None,
            "n_users_considered": None,
            "n_users_total": None,
            "error": f"ValueError: val_df is missing the {user_id_column!r} column",
        }

    try:
        raw = fn(train_df, val_df)
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(val_df):
            raise ValueError(
                f"feature {name!r} must return a 1-D array of len(val_df) "
                f"({len(val_df)}); got shape {values.shape}"
            )
        mean_variance, n_considered, n_total = per_user_variance(
            values, val_df[user_id_column].to_numpy(), ddof=ddof
        )
    except Exception as exc:  # noqa: BLE001 - a feature may fail for any reason
        return {
            "status": "error",
            "mean_within_user_variance": None,
            "metric_inert": None,
            "n_users_considered": None,
            "n_users_total": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "ok",
        "mean_within_user_variance": mean_variance,
        "metric_inert": bool(mean_variance < threshold),
        "n_users_considered": n_considered,
        "n_users_total": n_total,
        "error": None,
    }


def screen_features(
    features: Mapping[str, Callable[[Any, Any], np.ndarray]],
    train_df: Any,
    val_df: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ddof: int = DEFAULT_DDOF,
    user_id_column: str = DEFAULT_USER_ID_COLUMN,
) -> dict[str, dict[str, Any]]:
    """Screen every ``(name, fn)`` pair in ``features``. Registry-agnostic:
    iterates whatever mapping is passed at call time (see module docstring).
    Order-preserving: results come back in the same order as ``features``.
    """
    return {
        name: screen_feature(
            name, fn, train_df, val_df,
            threshold=threshold, ddof=ddof, user_id_column=user_id_column,
        )
        for name, fn in features.items()
    }


def build_report(
    feature_results: Mapping[str, dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ddof: int = DEFAULT_DDOF,
    split: str = DEFAULT_SPLIT_NAME,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Wrap per-feature screen results in the top-level schema (see module
    docstring). Pure function of its arguments — no disk, no clock unless
    ``now`` is omitted, so it is trivially unit-testable.
    """
    timestamp = (now or datetime.now(UTC)).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "threshold": threshold,
        "ddof": ddof,
        "split": split,
        "timestamp": timestamp,
        "features": dict(feature_results),
    }


def write_report(report: Mapping[str, Any], path: str | Path = DEFAULT_REPORT_PATH) -> Path:
    """Write ``report`` as JSON to ``path``, creating parent directories as
    needed. Separate from computation (design decision 5) so tests can pass
    a ``tmp_path`` instead of touching the real ``logs/`` directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    return path


def render_report(report: Mapping[str, Any]) -> str:
    """Human/judge-readable rendering of a report dict, mirroring the style
    of ``tools/data_usage.py``'s ``render_report`` (B13)."""
    features: dict[str, dict[str, Any]] = report.get("features", {})
    inert = sorted(name for name, r in features.items() if r.get("metric_inert"))
    errored = sorted(name for name, r in features.items() if r.get("status") == "error")

    lines = [
        "=== B12 within-user variance screen ===",
        "",
        f"split: {report.get('split')}   threshold: {report.get('threshold')}   "
        f"ddof: {report.get('ddof')}   timestamp: {report.get('timestamp')}",
        "",
        "Features:",
    ]
    for name, result in features.items():
        if result.get("status") == "error":
            lines.append(f"  [error       ] {name}: {result.get('error')}")
            continue
        flag = "INERT" if result.get("metric_inert") else "ok"
        variance = result.get("mean_within_user_variance")
        lines.append(
            f"  [{flag:12s}] {name}: mean_within_user_variance="
            f"{variance:.3e} (n_users_considered="
            f"{result.get('n_users_considered')}/{result.get('n_users_total')})"
        )

    lines.append("")
    if inert:
        lines.append(f"metric_inert ({len(inert)}): {', '.join(inert)}")
    else:
        lines.append("metric_inert (0): none")
    if errored:
        lines.append(f"errored ({len(errored)}): {', '.join(errored)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"metric_inert cutoff on mean within-user variance (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--ddof", type=int, default=DEFAULT_DDOF, choices=(0, 1),
        help="variance convention: 0 = population (default), 1 = sample",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_REPORT_PATH),
        help=f"report output path (default {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="print the report without writing it to disk",
    )
    args = parser.parse_args(argv)

    from pipeline import data as pipeline_data
    from pipeline.features import FEATURES

    train, val, _ = pipeline_data.load()
    results = screen_features(FEATURES, train, val, threshold=args.threshold, ddof=args.ddof)
    report = build_report(results, threshold=args.threshold, ddof=args.ddof)

    print(render_report(report))

    if not args.no_write:
        out_path = write_report(report, args.out)
        print(f"\nWritten to {out_path}")

    # See module docstring, design decision 4: this tool never exits non-zero
    # over what it finds. metric_inert flags and per-feature error statuses
    # are both informational content of the report, not run failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
