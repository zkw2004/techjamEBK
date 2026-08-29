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
INITIAL_RATE = 0.5


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


def _event_times(frame) -> pd.Series:
    """Return the pre-impression event timestamp used for in-sample history.

    The raw logs are not stored in chronological row order, so their positional
    order cannot establish a historical cutoff. ``time_ms`` is the impression
    timestamp; records sharing it intentionally see the same prior state and
    never one another's label.
    """
    if "time_ms" not in frame:
        raise ValueError("in-sample historical features require a time_ms column")
    times = pd.to_numeric(frame["time_ms"], errors="coerce")
    if times.isna().any() or not np.isfinite(times.to_numpy(dtype=np.float64)).all():
        raise ValueError("in-sample historical features require finite time_ms values")
    return times.astype(np.int64)


def _prior_global_rates(labels: pd.Series, times: pd.Series) -> np.ndarray:
    """Global long-view rate strictly before each event timestamp."""
    source = pd.DataFrame(
        {
            "_position": np.arange(len(labels)),
            "_time": times.to_numpy(),
            "_label": labels.to_numpy(),
        }
    )
    by_time = source.groupby("_time", sort=True)["_label"].agg(["sum", "count"])
    by_time["_prior_sum"] = by_time["sum"].cumsum() - by_time["sum"]
    by_time["_prior_count"] = by_time["count"].cumsum() - by_time["count"]
    states = source.merge(
        by_time[["_prior_sum", "_prior_count"]].reset_index(),
        on="_time",
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_position", kind="stable")
    prior_count = states["_prior_count"].to_numpy(dtype=np.float64)
    prior_sum = states["_prior_sum"].to_numpy(dtype=np.float64)
    return np.divide(
        prior_sum,
        prior_count,
        out=np.full(len(source), INITIAL_RATE, dtype=np.float64),
        where=prior_count > 0,
    )


def _prior_group_stats(
    frame,
    labels: pd.Series,
    times: pd.Series,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Label sum/count for each key tuple strictly before its event time."""
    source = pd.DataFrame(
        {
            "_position": np.arange(len(frame)),
            "_time": times.to_numpy(),
            "_label": labels.to_numpy(),
            **{key: frame[key].to_numpy() for key in keys},
        }
    )
    group_keys = [*keys, "_time"]
    grouped = source.groupby(group_keys, sort=True, dropna=False)["_label"].agg(["sum", "count"])
    key_levels = list(range(len(keys)))
    grouped["_prior_sum"] = grouped.groupby(level=key_levels, sort=False)["sum"].cumsum()
    grouped["_prior_sum"] -= grouped["sum"]
    grouped["_prior_count"] = grouped.groupby(level=key_levels, sort=False)["count"].cumsum()
    grouped["_prior_count"] -= grouped["count"]
    states = source.merge(
        grouped[["_prior_sum", "_prior_count"]].reset_index(),
        on=group_keys,
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_position", kind="stable")
    return (
        states["_prior_sum"].to_numpy(dtype=np.float64),
        states["_prior_count"].to_numpy(dtype=np.float64),
    )


def _in_sample_group_rate(train_df, key: str) -> np.ndarray:
    labels = _numeric_labels(train_df)
    times = _event_times(train_df)
    global_rates = _prior_global_rates(labels, times)
    prior_sum, prior_count = _prior_group_stats(train_df, labels, times, [key])
    return np.divide(prior_sum, prior_count, out=global_rates, where=prior_count > 0)


def _in_sample_group_count(train_df, key: str) -> np.ndarray:
    labels = _numeric_labels(train_df)
    _, prior_count = _prior_group_stats(train_df, labels, _event_times(train_df), [key])
    return prior_count


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
    if train_df is target_df:
        return _in_sample_group_rate(train_df, "user_id")
    return _group_rate(train_df, target_df, "user_id")


@feature("video_ctr")
def video_ctr(train_df, target_df) -> np.ndarray:
    """Raw historical long-view rate by video; unseen videos use the train mean."""
    if train_df is target_df:
        return _in_sample_group_rate(train_df, "video_id")
    return _group_rate(train_df, target_df, "video_id")


@feature("video_impressions")
def video_impressions(train_df, target_df) -> np.ndarray:
    """Historical impression count by video; unseen videos receive zero."""
    if train_df is target_df:
        return _in_sample_group_count(train_df, "video_id")
    return _group_count(train_df, target_df, "video_id")


@feature("user_activity")
def user_activity(train_df, target_df) -> np.ndarray:
    """Historical impression count by user; unseen users receive zero."""
    if train_df is target_df:
        return _in_sample_group_count(train_df, "user_id")
    return _group_count(train_df, target_df, "user_id")


@feature("user_tag_affinity")
def user_tag_affinity(train_df, target_df) -> np.ndarray:
    """Mean train-only long-view rate for a user's target-video tags.

    Multi-tag videos contribute once to each individual tag rate. At apply
    time, their per-tag rates are averaged. Missing or unseen user-tag pairs
    use the global training rate, keeping this feature neutral while the
    separate ``user_ctr`` feature carries general user propensity.
    """
    if train_df is target_df:
        return _in_sample_user_tag_affinity(train_df)

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


def _in_sample_user_tag_affinity(train_df) -> np.ndarray:
    """Per-row user-tag affinity using only impressions at earlier times."""
    labels = _numeric_labels(train_df)
    times = _event_times(train_df)
    global_rates = _prior_global_rates(labels, times)
    result = global_rates.copy()

    exploded = pd.DataFrame(
        {
            "_position": np.arange(len(train_df)),
            "_time": times.to_numpy(),
            "user_id": train_df["user_id"].to_numpy(),
            LABEL: labels.to_numpy(),
            "_tag": train_df["tag"].map(_split_tags).to_numpy(),
        }
    ).explode("_tag", ignore_index=True)
    exploded = exploded.loc[exploded["_tag"].notna()].copy()
    if exploded.empty:
        return result

    group_keys = ["user_id", "_tag", "_time"]
    grouped = exploded.groupby(group_keys, sort=True, dropna=False)[LABEL].agg(["sum", "count"])
    grouped["_prior_sum"] = grouped.groupby(level=[0, 1], sort=False)["sum"].cumsum()
    grouped["_prior_sum"] -= grouped["sum"]
    grouped["_prior_count"] = grouped.groupby(level=[0, 1], sort=False)["count"].cumsum()
    grouped["_prior_count"] -= grouped["count"]
    states = exploded.merge(
        grouped[["_prior_sum", "_prior_count"]].reset_index(),
        on=group_keys,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    fallback = global_rates[states["_position"].to_numpy(dtype=int)]
    prior_count = states["_prior_count"].to_numpy(dtype=np.float64)
    states["_affinity"] = np.divide(
        states["_prior_sum"].to_numpy(dtype=np.float64),
        prior_count,
        out=fallback,
        where=prior_count > 0,
    )
    per_row = states.groupby("_position", sort=False)["_affinity"].mean()
    result[per_row.index.to_numpy(dtype=int)] = per_row.to_numpy(dtype=np.float64)
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


def leakage_check(fn: Callable, train_df, target_df) -> bool:
    """Static + probe leakage guard. See Appendix A.2. (B6)

    1. Label-independence probe: shuffle train labels; if output is
       unchanged the feature is label-free (safe) or reading labels off
       target_df (unsafe) — the static check disambiguates.
    2. Static source check: no target_df["long_view"], no FORBIDDEN_SAME_ROW
       column read off target_df, no EXCLUDED_SOURCES.

    The third layer is the >0.75 primary canary inside run_experiment (C1).
    """
    raise NotImplementedError("B6")
