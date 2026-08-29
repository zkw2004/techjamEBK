"""Feature registry, leakage guard, decay/EB helpers.

Contract: AGENT_PLAN.md Section 8.4 (FROZEN). Reference impl: Appendix A.1, A.2.
Owner: Workstream B (Malvika). Tasks B3-B7.

Every feature fits statistics on train_df ONLY and applies them to target_df.
That shape makes leakage the exception rather than the default.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FEATURES: dict[str, Callable] = {}


def feature(name: str):
    """Decorator registering a feature builder."""

    def wrap(fn):
        if name in FEATURES:
            raise ValueError(f"duplicate feature name: {name}")
        FEATURES[name] = fn
        return fn

    return wrap


def get(name: str) -> Callable:
    """Resolve a feature by name. Raises before any training starts (B3)."""
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; registered: {sorted(FEATURES)}")
    return FEATURES[name]


# Every feature has this exact signature:
#     def my_feature(train_df, target_df) -> np.ndarray  # len(target_df)

# --- Leakage policy (Section 6.8, trap 2 and trap 3) ------------------------
#
# CORRECTED vs AGENT_PLAN.md Section 8.4. The plan says the scored label is
# `click` and lists `long_view` as forbidden. The shipped code disagrees:
#
#     kuairand-starter-kit/data.py:5   LABEL = 'long_view'
#     baseline_scores.json             "label": "long_view"
#
# So the list is inverted from the plan: `long_view` IS the label, and
# `is_click` is a same-row post-exposure signal that must not be an input.
LABEL = "long_view"

FORBIDDEN_SAME_ROW = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter", "is_click_pause",
    "play_time_ms", "profile_stay_time",
]
# Legal as auxiliary TRAINING TARGETS and as HISTORICAL AGGREGATES
# over a user's past rows. Never as same-row input features.
#
# NOT forbidden, despite appearing in the plan's list: `duration_ms`. It is a
# video property known before exposure and is one of the five official
# baseline fields (as `dur_bucket`). Excluding it would drop a field the
# baseline we must beat already uses.

EXCLUDED_SOURCES = ["item_statistics_monthly"]  # see 6.8


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse integer, float-promoted, or ISO-formatted calendar dates."""
    date_strings = (
        values.astype("string")
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    return pd.to_datetime(date_strings, format="%Y%m%d", errors="coerce")


def _assert_historical_cutoff(train_df, target_df) -> None:
    """Fail closed unless every fitting row is strictly earlier than target."""
    if train_df.empty:
        raise ValueError("cannot fit aggregate features on an empty training frame")
    if target_df.empty:
        return

    train_dates = _parse_dates(train_df["date"])
    target_dates = _parse_dates(target_df["date"])
    if train_dates.isna().any() or target_dates.isna().any():
        raise ValueError("historical aggregate features require valid train and target dates")
    if train_dates.max() >= target_dates.min():
        raise ValueError("all training rows must be strictly earlier than all target rows")


def _numeric_labels(train_df) -> pd.Series:
    """Validate and normalise the scored training label once per aggregate."""
    labels = pd.to_numeric(train_df[LABEL], errors="coerce")
    if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"training column {LABEL!r} must contain only finite numbers")
    return labels.astype(np.float64)


def _group_rate(train_df, target_df, key: str) -> np.ndarray:
    """Map a train-only group rate to target rows in their original order."""
    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_labels(train_df)
    global_rate = float(labels.mean())
    fitting_rows = pd.DataFrame({key: train_df[key].to_numpy(), "_label": labels.to_numpy()})
    rates = fitting_rows.groupby(key, sort=False)["_label"].mean()
    return target_df[key].map(rates).fillna(global_rate).to_numpy(dtype=np.float64)


def _group_count(train_df, target_df, key: str) -> np.ndarray:
    """Map a train-only impression count, using zero for unseen groups."""
    _assert_historical_cutoff(train_df, target_df)

    counts = train_df.groupby(key, sort=False).size()
    return target_df[key].map(counts).fillna(0).to_numpy(dtype=np.float64)


def _split_tags(value) -> tuple[str, ...]:
    """Normalise one static comma-separated tag field without double-counting."""
    if pd.isna(value):
        return ()
    return tuple(dict.fromkeys(tag.strip() for tag in str(value).split(",") if tag.strip()))


@feature("user_ctr")
def user_ctr(train_df, target_df) -> np.ndarray:
    """Raw historical long-view rate by user; unseen users use the train mean."""
    return _group_rate(train_df, target_df, "user_id")


@feature("video_ctr")
def video_ctr(train_df, target_df) -> np.ndarray:
    """Raw historical long-view rate by video; unseen videos use the train mean."""
    return _group_rate(train_df, target_df, "video_id")


@feature("video_impressions")
def video_impressions(train_df, target_df) -> np.ndarray:
    """Historical impression count by video; unseen videos receive zero."""
    return _group_count(train_df, target_df, "video_id")


@feature("user_activity")
def user_activity(train_df, target_df) -> np.ndarray:
    """Historical impression count by user; unseen users receive zero."""
    return _group_count(train_df, target_df, "user_id")


@feature("user_tag_affinity")
def user_tag_affinity(train_df, target_df) -> np.ndarray:
    """Mean train-only long-view rate for a user's target-video tags.

    Multi-tag videos contribute once to each individual tag rate. At apply
    time, their per-tag rates are averaged. Missing or unseen user-tag pairs
    use the global training rate, keeping this feature neutral while the
    separate ``user_ctr`` feature carries general user propensity.
    """
    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_labels(train_df)
    global_rate = float(labels.mean())

    exploded_train = pd.DataFrame(
        {
            "user_id": train_df["user_id"].to_numpy(),
            LABEL: labels.to_numpy(),
            "_tag": train_df["tag"].map(_split_tags).to_numpy(),
        }
    ).explode("_tag", ignore_index=True)
    exploded_train = exploded_train.loc[exploded_train["_tag"].notna()]

    result = np.full(len(target_df), global_rate, dtype=np.float64)
    if exploded_train.empty or target_df.empty:
        return result

    rates = exploded_train.groupby(["user_id", "_tag"], sort=False)[LABEL].mean()
    exploded_target = pd.DataFrame(
        {
            "_position": np.arange(len(target_df)),
            "user_id": target_df["user_id"].to_numpy(),
            "_tag": target_df["tag"].map(_split_tags).to_numpy(),
        }
    ).explode("_tag", ignore_index=True)
    exploded_target = exploded_target.loc[exploded_target["_tag"].notna()]

    if exploded_target.empty:
        return result

    pairs = pd.MultiIndex.from_frame(exploded_target[["user_id", "_tag"]])
    exploded_target["_affinity"] = rates.reindex(pairs).fillna(global_rate).to_numpy()
    row_affinity = exploded_target.groupby("_position", sort=False)["_affinity"].mean()
    result[row_affinity.index.to_numpy(dtype=int)] = row_affinity.to_numpy(dtype=np.float64)
    return result


@feature("hour_of_day")
def hour_of_day(train_df, target_df) -> np.ndarray:
    """Pre-impression hour from HHMM context; invalid or missing values use -1."""
    del train_df  # Context-only feature: no statistics are fitted.
    hourmins = pd.to_numeric(target_df["hourmin"], errors="coerce")
    hours = hourmins.floordiv(100)
    minutes = hourmins.mod(100)
    valid = hourmins.eq(hourmins.round()) & hours.between(0, 23) & minutes.between(0, 59)
    hours = hours.where(valid, -1)
    return hours.to_numpy(dtype=np.float64)


@feature("day_of_week")
def day_of_week(train_df, target_df) -> np.ndarray:
    """Pre-impression weekday (Monday=0); invalid or missing dates use -1."""
    del train_df  # Context-only feature: no statistics are fitted.
    dates = _parse_dates(target_df["date"])
    return dates.dt.dayofweek.fillna(-1).to_numpy(dtype=np.float64)


def eb_smooth(clicks, impressions, global_rate, alpha: float = 20.0):
    """Empirical-Bayes shrinkage toward the global rate.

    Sparse groups (small n) collapse to global_rate; dense groups keep
    their own rate. Fit alpha on internal folds only. (B5)
    """
    raise NotImplementedError("B5")


def decay_weights(dates, cutoff, half_life_days: float = 7.0) -> np.ndarray:
    """Exponential recency weights: 0.5 ** (age_days / half_life_days).

    Recent behaviour predicts the near future better, and the test window
    sits 8-17 days out. Half-life fitted on internal folds only. (B5)
    """
    raise NotImplementedError("B5")


# --- Leakage guard (B6, Appendix A.2) ---------------------------------------

_PROBE_TRAIN_ROWS = 2_000
_PROBE_TARGET_ROWS = 1_000
_PROBE_SEED = 20220429  # first hidden-test day; fixed so the guard is deterministic


def _frame_length(frame) -> int:
    if hasattr(frame, "__len__") and not isinstance(frame, dict):
        return len(frame)
    lengths = {len(v) for v in frame.values()}
    return lengths.pop() if lengths else 0


def _frame_head(frame, rows: int):
    if isinstance(frame, dict):
        return {name: np.asarray(values)[:rows].copy() for name, values in frame.items()}
    return frame.head(rows).copy()


def _frame_columns(frame) -> tuple[str, ...]:
    if isinstance(frame, dict):
        return tuple(frame)
    return tuple(getattr(frame, "columns", ()))


def _with_shuffled(frame, columns: list[str], rng: np.random.Generator):
    """Copy of `frame` with each named column independently permuted."""
    if isinstance(frame, dict):
        out = dict(frame)
        for name in columns:
            out[name] = rng.permutation(np.asarray(frame[name]))
        return out
    out = frame.copy()
    for name in columns:
        out[name] = rng.permutation(out[name].to_numpy())
    return out


def _feature_source(fn: Callable) -> str:
    """Source text for the static scan. Fails closed when none is recoverable.

    Generated features (exec'd from a code string) have no file for
    `inspect.getsource`; the codegen path attaches the emitted source as
    `fn.__leak_source__` so they stay auditable.
    """
    import inspect

    source = getattr(fn, "__leak_source__", None)
    if isinstance(source, str) and source.strip():
        return source
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "leakage_check requires the feature's source (a file-backed function "
            "or a __leak_source__ attribute); refusing an unauditable feature"
        ) from exc


def _static_leak_scan(source: str) -> None:
    """Reject source that reads the label or a post-exposure column off target_df."""
    import re

    for column in [LABEL, *FORBIDDEN_SAME_ROW]:
        pattern = (
            rf"""target_df\s*(?:\[\s*['"]{column}['"]"""
            rf"""|\.{column}\b|\.get\(\s*['"]{column}['"])"""
        )
        match = re.search(pattern, source)
        if match:
            raise ValueError(
                f"feature source reads same-row outcome {column!r} off target_df "
                f"({match.group(0)!r}); post-exposure signals are never inputs (trap 2)"
            )
    for name in EXCLUDED_SOURCES:
        if name in source:
            raise ValueError(
                f"feature source references excluded source {name!r}; its aggregation "
                "window may span the test period (trap 3)"
            )


def _outputs_match(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True)


def leakage_check(fn: Callable, train_df, target_df) -> bool:
    """Static + probe leakage guard. See Appendix A.2. (B6)

    1. Static source check: no target_df["long_view"], no FORBIDDEN_SAME_ROW
       column read off target_df, no EXCLUDED_SOURCES. Fails closed when the
       feature's source cannot be recovered.
    2. Label-independence probe: shuffle train labels; if output is
       unchanged the feature is label-free (safe) or reading labels off
       target_df (unsafe) — the static check disambiguates.
    3. Target-outcome probe: shuffle the label and every forbidden column
       present on a target sample; any output change proves the feature is
       reading same-row outcomes, however the read is spelled.

    Probes run on fixed-size head samples so the guard stays cheap on the
    1.1M-row training frame. The final layer is the >0.75 primary canary
    inside run_experiment (C1).

    Raises ValueError (or lets a feature's own AssertionError propagate) on
    a detected leak; returns True when every check passes.
    """
    _static_leak_scan(_feature_source(fn))

    train_sample = _frame_head(train_df, min(_frame_length(train_df), _PROBE_TRAIN_ROWS))
    target_sample = _frame_head(target_df, min(_frame_length(target_df), _PROBE_TARGET_ROWS))
    if _frame_length(train_sample) == 0 or _frame_length(target_sample) == 0:
        return True
    rng = np.random.default_rng(_PROBE_SEED)

    baseline = np.asarray(fn(train_sample, target_sample), dtype=float)

    # Probe 2 (informational, per A.2): label-dependence on the training side
    # is legal — historical aggregates are supposed to use past labels.
    if LABEL in _frame_columns(train_sample):
        fn(_with_shuffled(train_sample, [LABEL], rng), target_sample)

    # Probe 3: reading outcomes off the target rows is never legal.
    outcome_columns = [
        name
        for name in (LABEL, *FORBIDDEN_SAME_ROW)
        if name in _frame_columns(target_sample)
    ]
    if outcome_columns:
        changed_columns: list[str] = []
        shuffled_target = target_sample
        for _ in range(8):  # a draw can coincidentally reproduce a short column
            shuffled_target = _with_shuffled(target_sample, outcome_columns, rng)
            changed_columns = [
                name
                for name in outcome_columns
                if not _outputs_match(
                    np.asarray(target_sample[name]), np.asarray(shuffled_target[name])
                )
            ]
            if changed_columns:
                break
        if changed_columns and not _outputs_match(baseline, fn(train_sample, shuffled_target)):
            raise ValueError(
                "feature output changed when same-row outcomes "
                f"{changed_columns!r} were permuted on target rows; it is reading "
                "post-exposure signals (trap 2)"
            )
    return True
