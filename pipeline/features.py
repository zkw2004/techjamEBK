"""Feature registry, leakage guard, decay/EB helpers.

Contract: AGENT_PLAN.md Section 8.4 (FROZEN). Reference impl: Appendix A.1, A.2.
Owner: Workstream B (Malvika). Tasks B3-B7.

Every feature fits statistics on train_df ONLY and applies them to target_df.
That shape makes leakage the exception rather than the default.
"""

from __future__ import annotations

import inspect
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
    """Parse integer, float-promoted, ISO, or pandas calendar dates."""
    if isinstance(values.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(
        values.dtype
    ):
        return pd.to_datetime(values, errors="coerce")

    date_strings = (
        values.astype("string")
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    parsed = pd.to_datetime(date_strings, format="%Y%m%d", errors="coerce")

    # Timestamp strings with a time component are not YYYYMMDD, but are still
    # valid inputs. Never apply this fallback to numeric calendar values:
    # pandas would otherwise interpret an invalid 20220431 as epoch nanoseconds.
    numeric = pd.to_numeric(values, errors="coerce")
    fallback_mask = parsed.isna() & values.notna() & numeric.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            values.loc[fallback_mask],
            format="mixed",
            errors="coerce",
        )
    return parsed


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
    their own rate. ``clicks`` and ``impressions`` may be fractional after
    time weighting. Callers must select alpha on internal folds only. (B5)
    """
    try:
        alpha = float(alpha)
        global_rate = float(global_rate)
        click_values = np.asarray(clicks, dtype=np.float64)
        impression_values = np.asarray(impressions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("EB inputs must be numeric") from exc

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")
    if not np.isfinite(global_rate) or not 0 <= global_rate <= 1:
        raise ValueError("global_rate must be finite and between zero and one")
    if isinstance(clicks, pd.Series) and isinstance(impressions, pd.Series):
        if not clicks.index.equals(impressions.index):
            raise ValueError("click and impression Series must have identical indexes")
    if click_values.ndim and impression_values.ndim:
        if click_values.shape != impression_values.shape:
            raise ValueError("clicks and impressions must have matching shapes")

    try:
        click_values, impression_values = np.broadcast_arrays(click_values, impression_values)
    except ValueError as exc:
        raise ValueError("clicks and impressions are not broadcast-compatible") from exc

    if not np.isfinite(click_values).all() or not np.isfinite(impression_values).all():
        raise ValueError("clicks and impressions must be finite")
    if (click_values < 0).any() or (impression_values < 0).any():
        raise ValueError("clicks and impressions must be non-negative")
    if (click_values > impression_values).any():
        raise ValueError("clicks cannot exceed impressions")

    result = (click_values + alpha * global_rate) / (impression_values + alpha)
    if isinstance(clicks, pd.Series):
        return pd.Series(result, index=clicks.index, name=clicks.name)
    if isinstance(impressions, pd.Series):
        return pd.Series(result, index=impressions.index, name=impressions.name)
    if result.ndim == 0:
        return float(result)
    return result


def decay_weights(dates, cutoff, half_life_days: float = 7.0) -> np.ndarray:
    """Exponential recency weights: 0.5 ** (age_days / half_life_days).

    Recent behaviour predicts the near future better, and the test window
    sits 8-17 days out. Callers must select the half-life on internal folds
    only. Dates after the fitting cutoff are rejected. (B5)
    """
    try:
        half_life_days = float(half_life_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("half_life_days must be numeric") from exc
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be finite and greater than zero")

    if isinstance(dates, pd.Series):
        date_values = dates.reset_index(drop=True)
    elif np.isscalar(dates):
        date_values = pd.Series([dates])
    else:
        date_values = pd.Series(dates)

    parsed_dates = _parse_dates(date_values)
    parsed_cutoff = _parse_dates(pd.Series([cutoff])).iloc[0]
    if pd.isna(parsed_cutoff):
        raise ValueError("cutoff must be a valid calendar date")
    if parsed_dates.isna().any():
        raise ValueError("dates must contain only valid calendar dates")

    age_days = (parsed_cutoff - parsed_dates).dt.total_seconds() / 86_400.0
    if (age_days < 0).any():
        raise ValueError("dates cannot be later than cutoff")
    return np.power(0.5, age_days.to_numpy(dtype=np.float64) / half_life_days)


@feature("user_ctr_decayed")
def user_ctr_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical long-view rate per user.

    The train-wide maximum date is the decay cutoff. The global prior is the
    unweighted training label mean, matching Appendix A.1. Alpha and half-life
    use helper defaults here; alternative values must be screened on B2's
    internal folds before a caller adopts them.
    """
    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_labels(train_df)
    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError("user_ctr_decayed requires valid training dates")

    weights = decay_weights(dates, dates.max())
    fitting_rows = pd.DataFrame(
        {
            "user_id": train_df["user_id"].to_numpy(),
            "_weighted_positive": weights * labels.to_numpy(),
            "_weight": weights,
        }
    )
    grouped = fitting_rows.groupby("user_id", sort=False)
    weighted_positives = grouped["_weighted_positive"].sum()
    effective_impressions = grouped["_weight"].sum()
    global_rate = float(labels.mean())
    rates = eb_smooth(weighted_positives, effective_impressions, global_rate)
    return target_df["user_id"].map(rates).fillna(global_rate).to_numpy(dtype=np.float64)


# --- Leakage guard (B6, Appendix A.2) ---------------------------------------

_PROBE_TRAIN_ROWS = 2_000
_PROBE_TARGET_ROWS = 1_000


def _feature_source(fn: Callable) -> str:
    """Source text for the static scan. Fails closed when none is recoverable.

    Generated features (exec'd from an emitted code string, C4b) have no file
    behind them for `inspect.getsource`; the codegen path attaches the source
    as `fn.__leak_source__` so they stay auditable. A callable with neither is
    refused outright — an unauditable feature is an unavailable feature
    (Section 6.8, governing rule).
    """
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


def _reads_target_column(src: str, column: str) -> bool:
    """True if `src` indexes target_df by `column`, either quote style."""
    return f'target_df["{column}"]' in src or f"target_df['{column}']" in src


def leakage_check(fn: Callable, train_df, target_df) -> bool:
    """Static + dynamic leakage guard. Returns True iff `fn` is safe. (B6)

    CORRECTED vs Appendix A.2's pseudocode. There, the label-independence
    probe runs first and short-circuits to `return True` whenever shuffling
    train_df's label leaves the output unchanged — with the static check
    reached only on the other branch. But a feature that reads
    target_df[LABEL] directly never looks at train_df's label column either,
    so shuffling it produces the identical "unchanged" signal and the
    leak sails through the very probe meant to disambiguate it. This is
    exactly the class of trap Section 11 warns about: it looks like a
    reasonable check and silently passes the thing it exists to catch.

    Two independent layers, both must pass:

    1. Static source check (authoritative). `fn`'s source must not index
       target_df by the label, a FORBIDDEN_SAME_ROW column, or reference an
       EXCLUDED_SOURCES name. Runs unconditionally, first, before `fn` is
       ever executed on real data. The source is recovered from the file or
       from `__leak_source__` (generated features); recovery failure raises
       ValueError rather than passing an unauditable feature.
    2. Dynamic corruption probe. With the label and every FORBIDDEN_SAME_ROW
       column on target_df replaced with NaN, `fn`'s output must not change.
       This catches indirection the source scan cannot see — a column name
       held in a variable, `.loc` access, a helper called under another
       name — exactly the kind of code Tier 2 (agent-generated features,
       Section 7.3) might produce. Legitimate historical aggregates never
       read those target-row columns at all, so they are untouched by the
       corruption and pass.

    The probe runs on fixed-size head samples so the guard stays cheap on
    the 1.1M-row training frame. The third layer is the >0.75 primary
    canary inside run_experiment (C1).
    """
    if train_df.empty or target_df.empty:
        raise ValueError("leakage_check requires non-empty train_df and target_df")

    src = _feature_source(fn)
    if _reads_target_column(src, LABEL):
        return False
    for column in FORBIDDEN_SAME_ROW:
        if _reads_target_column(src, column):
            return False
    for source_name in EXCLUDED_SOURCES:
        if source_name in src:
            return False

    train_sample = train_df.head(_PROBE_TRAIN_ROWS)
    target_sample = target_df.head(_PROBE_TARGET_ROWS)
    baseline = np.asarray(fn(train_sample, target_sample))
    corrupted_target = target_sample.copy()
    for column in (LABEL, *FORBIDDEN_SAME_ROW):
        if column in corrupted_target.columns:
            corrupted_target[column] = np.nan
    corrupted = np.asarray(fn(train_sample, corrupted_target))

    return bool(np.array_equal(baseline, corrupted, equal_nan=True))
