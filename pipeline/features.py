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

    in_sample = train_df is target_df
    train_sample = train_df.head(_PROBE_TRAIN_ROWS)
    if in_sample:
        # In-sample aggregate builders intentionally consume earlier labels
        # from this same frame. Slicing train and target independently loses
        # the identity signal they use to select their strictly historical
        # implementation. Static checks still run above, and every real
        # validation/test application takes the corruption-probe path below.
        np.asarray(fn(train_sample, train_sample))
        return True
    target_sample = target_df.head(_PROBE_TARGET_ROWS)
    baseline = np.asarray(fn(train_sample, target_sample))
    corrupted_target = target_sample.copy()
    for column in (LABEL, *FORBIDDEN_SAME_ROW):
        if column in corrupted_target.columns:
            corrupted_target[column] = np.nan
    corrupted = np.asarray(fn(train_sample, corrupted_target))

    return bool(np.array_equal(baseline, corrupted, equal_nan=True))


# --- Duration-bias feature pack (B10) ---------------------------------------
#
# `video_duration`, `duration_bucket`, `pcr_hist`, `long_view_rate_by_duration_
# group`. Citations (Appendix D idea bank): duration-quantile grouping is
# D2Q (Zhan et al., KDD 2022) -- duration confounds watch-time labels, and
# `long_view` is duration-derived, so raw duration and a naive "did they
# finish it" rate both need to be read relative to a video's own duration
# bucket rather than in absolute terms. The completion-ratio aggregate is
# from the PCR baseline family / CWM (Zhao et al., KDD 2024) -- completion
# behaviour predicts `long_view` beyond raw impression/click counts.
#
# `pipeline/train.py::_matrix` already special-cases the *inline* baseline
# field `dur_bucket` (one of the five official FM fields): 10 quantile
# buckets computed from `train_frame["duration_ms"]` via
# `np.quantile(..., np.linspace(0, 1, 11)[1:-1])` + `np.searchsorted`. The
# registered feature below is named `duration_bucket` -- deliberately not
# `dur_bucket` -- so it never collides with that inline special case or the
# baseline field name, while sharing the identical quantile-edge recipe via
# `_duration_bucket_edges` so the two are consistent by construction rather
# than by coincidence.

DURATION_BUCKET_COUNT = 10  # D2Q's spirit: 10 quantile groups by default.


def _duration_bucket_edges(train_df, n_buckets: int = DURATION_BUCKET_COUNT) -> np.ndarray:
    """Interior quantile edges for D2Q-style duration bucketing (B10).

    Fit on `train_df["duration_ms"]` ONLY -- `n_buckets - 1` interior edges
    via `np.linspace(0, 1, n_buckets + 1)[1:-1]`, matching the inline
    `dur_bucket` recipe in `pipeline/train.py::_matrix` at the default
    `n_buckets=10`. Callers apply these same edges to any other frame with
    `_apply_duration_buckets`; this function never sees or touches target
    data, so the edges cannot be refit on a validation/test distribution.
    `n_buckets` is configurable here (unit-testable independently) even
    though the registered `duration_bucket`/`long_view_rate_by_duration_
    group` features call it with the fixed default, since the registry
    signature (8.4) has no room for a per-call parameter.
    """
    if train_df.empty:
        raise ValueError("cannot fit duration bucket edges on an empty training frame")
    if not isinstance(n_buckets, (int, np.integer)) or n_buckets < 1:
        raise ValueError("n_buckets must be a positive integer")

    durations = pd.to_numeric(train_df["duration_ms"], errors="coerce")
    duration_values = durations.to_numpy(dtype=np.float64)
    if durations.isna().any() or not np.isfinite(duration_values).all():
        raise ValueError("training column 'duration_ms' must contain only finite numbers")

    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    return np.quantile(duration_values, quantiles)


def _apply_duration_buckets(frame, edges: np.ndarray) -> np.ndarray:
    """Assign integer bucket indices in `[0, len(edges)]` from fixed `edges`.

    Never fits: `edges` must already be computed (by `_duration_bucket_
    edges` on a train frame) and are applied here unchanged, which is what
    the B10 acceptance criterion ("quantile bucket edges computed on train
    are reused unchanged for validation/test") requires.
    """
    durations = pd.to_numeric(frame["duration_ms"], errors="coerce")
    duration_values = durations.to_numpy(dtype=np.float64)
    if durations.isna().any() or not np.isfinite(duration_values).all():
        raise ValueError("column 'duration_ms' must contain only finite numbers")
    return np.searchsorted(edges, duration_values)


@feature("video_duration")
def video_duration(train_df, target_df) -> np.ndarray:
    """Raw pre-exposure video duration (`duration_ms`) for each target row.

    A static video property known before exposure -- not forbidden (see the
    `duration_ms` note near `FORBIDDEN_SAME_ROW` above) and one of the five
    official baseline fields (as `dur_bucket`). No statistics are fitted, so
    this needs no train-only cutoff assertion, exactly like `hour_of_day`/
    `day_of_week`.
    """
    del train_df  # Context-only feature: no statistics are fitted.
    durations = pd.to_numeric(target_df["duration_ms"], errors="coerce")
    duration_values = durations.to_numpy(dtype=np.float64)
    if durations.isna().any() or not np.isfinite(duration_values).all():
        raise ValueError("column 'duration_ms' must contain only finite numbers")
    return duration_values


@feature("duration_bucket")
def duration_bucket(train_df, target_df) -> np.ndarray:
    """Train-quantile duration bucket group, D2Q-style (default 10 buckets).

    Edges are fit once on `train_df["duration_ms"]` via `_duration_bucket_
    edges` and applied unchanged to `target_df` via `_apply_duration_
    buckets` -- target's own duration distribution never influences the
    edges. `duration_ms` is a static, pre-exposure property (like
    `video_duration`), so -- as with `hour_of_day`/`day_of_week` -- this
    does not need the in-sample dual path used by label-dependent
    aggregates below: fitting edges on `train_df` and applying them to
    `target_df` is safe and well-defined even when they are literally the
    same frame (`train_df is target_df`, the in-sample matrix-construction
    call from `pipeline/train.py::_matrix`), because no outcome/label
    information is read at any point.
    """
    edges = _duration_bucket_edges(train_df, DURATION_BUCKET_COUNT)
    return _apply_duration_buckets(target_df, edges).astype(np.float64)


def _completion_ratio(train_df) -> pd.Series:
    """Per-row play-completion ratio `play_time_ms / duration_ms`, clipped
    to `[0, 1]`; `duration_ms == 0` maps to a ratio of 0 rather than
    dividing by zero.

    `play_time_ms` is in `FORBIDDEN_SAME_ROW`, but this helper is only ever
    called with `train_df` (the fitting side of the split), never
    `target_df` -- exactly the same legal pattern as reading `long_view`
    (the LABEL) out of `train_df` throughout this file.
    """
    play_time = pd.to_numeric(train_df["play_time_ms"], errors="coerce")
    play_values = play_time.to_numpy(dtype=np.float64)
    if play_time.isna().any() or not np.isfinite(play_values).all():
        raise ValueError("training column 'play_time_ms' must contain only finite numbers")

    duration = pd.to_numeric(train_df["duration_ms"], errors="coerce")
    duration_values = duration.to_numpy(dtype=np.float64)
    if duration.isna().any() or not np.isfinite(duration_values).all():
        raise ValueError("training column 'duration_ms' must contain only finite numbers")

    ratio = np.divide(
        play_values,
        duration_values,
        out=np.zeros(len(train_df), dtype=np.float64),
        where=duration_values > 0,
    )
    return pd.Series(np.clip(ratio, 0.0, 1.0), index=train_df.index)


def _in_sample_pcr(train_df) -> np.ndarray:
    """In-sample (`train_df is target_df`) leak-safe completion-ratio rate.

    Mirrors `_in_sample_group_rate`'s shape exactly, generalised from a
    binary label to the continuous completion ratio: strictly-prior sum
    over `_prior_group_stats`, divided by strictly-prior count, falling
    back to the strictly-prior running mean ratio (`_prior_global_rates`)
    for a user's first-ever row. Time-decay is intentionally not applied
    here, unlike the cross-frame branch of `pcr_hist` below -- none of the
    existing in-sample helpers (`_in_sample_group_rate`,
    `_in_sample_user_tag_affinity`) decay either, and this path only ever
    runs while `pipeline/train.py::_matrix` builds the training design
    matrix itself, never for out-of-sample scoring of validation/test rows.
    """
    ratio = _completion_ratio(train_df)
    times = _event_times(train_df)
    global_rates = _prior_global_rates(ratio, times)
    prior_sum, prior_count = _prior_group_stats(train_df, ratio, times, ["user_id"])
    return np.divide(prior_sum, prior_count, out=global_rates, where=prior_count > 0)


@feature("pcr_hist")
def pcr_hist(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical play-completion ratio per user.

    Design decision (B10), resolving a genuine ambiguity in AGENT_PLAN.md
    Section 9.2: the plan names one feature, `pcr_hist`, described as
    "historical play_time/duration ratio aggregates per user AND per
    video" -- but that same row caps the pack at "all four registered"
    features total, which leaves room for exactly one `pcr_hist` signal,
    not two. This implementation picks USER-level completion-ratio
    history, mirroring `user_ctr_decayed`'s exact shape (decay cutoff =
    train max date, helper-default alpha/half-life) but aggregating the
    per-row completion ratio from `_completion_ratio` instead of the
    binary `long_view` label. A video-level twin (e.g. `video_pcr_hist`)
    is a natural, obvious follow-up outside this task's four-feature
    scope and is not attempted here.

    `_assert_historical_cutoff` fails closed on the (train_df, target_df)
    application path; the `train_df is target_df` case (used by
    `pipeline/train.py::_matrix` to build the training matrix itself, per
    Section 9.2's dual-path requirement) is delegated to `_in_sample_pcr`,
    which never reads a row's own or a future row's `play_time_ms`.
    """
    if train_df is target_df:
        return _in_sample_pcr(train_df)

    _assert_historical_cutoff(train_df, target_df)
    ratio = _completion_ratio(train_df)
    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError("pcr_hist requires valid training dates")

    weights = decay_weights(dates, dates.max())
    fitting_rows = pd.DataFrame(
        {
            "user_id": train_df["user_id"].to_numpy(),
            "_weighted_ratio": weights * ratio.to_numpy(),
            "_weight": weights,
        }
    )
    grouped = fitting_rows.groupby("user_id", sort=False)
    weighted_ratio = grouped["_weighted_ratio"].sum()
    effective_impressions = grouped["_weight"].sum()
    global_rate = float(ratio.mean())
    rates = eb_smooth(weighted_ratio, effective_impressions, global_rate)
    return target_df["user_id"].map(rates).fillna(global_rate).to_numpy(dtype=np.float64)


def _in_sample_duration_group_rate(train_df) -> np.ndarray:
    """In-sample (`train_df is target_df`) leak-safe long_view rate by
    duration bucket.

    Bucket edges are fit on the full `train_df["duration_ms"]` distribution
    via `_duration_bucket_edges` -- safe even though that spans rows past a
    given row's own `time_ms`, because `duration_ms` is a static,
    pre-exposure video property carrying no outcome information (the same
    reasoning `duration_bucket` documents). The `long_view` RATE within
    each bucket is what must stay strictly historical, and that part uses
    `_prior_group_stats`/`_prior_global_rates` exactly like
    `_in_sample_group_rate`, keyed on the row's own bucket assignment.
    """
    edges = _duration_bucket_edges(train_df, DURATION_BUCKET_COUNT)
    buckets = _apply_duration_buckets(train_df, edges)
    labels = _numeric_labels(train_df)
    times = _event_times(train_df)
    global_rates = _prior_global_rates(labels, times)
    frame_with_bucket = train_df.assign(_duration_bucket=buckets)
    prior_sum, prior_count = _prior_group_stats(
        frame_with_bucket, labels, times, ["_duration_bucket"]
    )
    return np.divide(prior_sum, prior_count, out=global_rates, where=prior_count > 0)


@feature("long_view_rate_by_duration_group")
def long_view_rate_by_duration_group(train_df, target_df) -> np.ndarray:
    """EB-smoothed `long_view` rate within train-quantile duration bucket.

    D2Q-style target encoding (Zhan et al., KDD 2022): `long_view` is
    duration-derived, so a raw historical long-view rate mixes together
    videos of very different lengths. Bucketing by `_duration_bucket_
    edges`/`_apply_duration_buckets` -- the exact same helpers `duration_
    bucket` uses, so the two features are consistent by construction --
    isolates that confound before EB-smoothing (`eb_smooth`, alpha
    default) the within-bucket rate toward the train-wide `long_view`
    mean. Edges are fit on `train_df` only and reused unchanged for
    `target_df`, satisfying the same no-refit acceptance criterion as
    `duration_bucket`.
    """
    if train_df is target_df:
        return _in_sample_duration_group_rate(train_df)

    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_labels(train_df)
    global_rate = float(labels.mean())

    edges = _duration_bucket_edges(train_df, DURATION_BUCKET_COUNT)
    train_buckets = _apply_duration_buckets(train_df, edges)
    fitting_rows = pd.DataFrame({"_bucket": train_buckets, "_label": labels.to_numpy()})
    grouped = fitting_rows.groupby("_bucket", sort=False)["_label"].agg(["sum", "count"])
    rates = eb_smooth(grouped["sum"], grouped["count"], global_rate)

    target_buckets = _apply_duration_buckets(target_df, edges)
    return (
        pd.Series(target_buckets)
        .map(rates)
        .fillna(global_rate)
        .to_numpy(dtype=np.float64)
    )

# --- Auxiliary-signal historical rates (B11) --------------------------------
#
# Per-user and per-video historical rates of the OTHER feedback signals
# (is_click, is_like, is_follow), plus a video-level counterpart of the
# scored long_view rate. Time-decayed (decay_weights, B5) and EB-smoothed
# (eb_smooth, B5), fit on train_df only.
#
# Citation (Appendix D idea bank): ESMM-style multi-feedback usage — "11
# unscored signals carry information about the scored one." is_click,
# is_like, and is_follow are all in FORBIDDEN_SAME_ROW: illegal as same-row
# inputs read off target_df, but explicitly legal as historical aggregates
# over a user's/video's PAST rows read off train_df only (Section 6.8's
# historical-aggregate carve-out; Section 11 trap 2). Every function below
# reads its auxiliary column from train_df exclusively, never target_df —
# that is what makes the carve-out apply.
#
# Overlap note: `user_ctr` (B4) and `user_ctr_decayed` (B5) already cover
# the raw and decayed+smoothed user-level long_view rate. There was no
# video-level decayed+smoothed long_view counterpart before this task; see
# `video_long_view_rate_decayed` below, which fills that gap. See
# `user_long_view_rate_decayed`'s docstring for why it is *not* a plain
# delegate to `user_ctr_decayed` despite matching it exactly on the
# cross-frame path.


def _numeric_outcome(train_df, column: str) -> pd.Series:
    """Validate and normalise a train-only auxiliary outcome column.

    Generalises `_numeric_labels` to any FORBIDDEN_SAME_ROW auxiliary
    signal (is_click, is_like, is_follow) as well as LABEL itself, so the
    B11 helpers below can be parameterised by column name. Reading this
    column off train_df is the legal half of Section 6.8's historical-
    aggregate carve-out; it must never be read off target_df.
    """
    values = pd.to_numeric(train_df[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"training column {column!r} must contain only finite numbers")
    return values.astype(np.float64)


def _prior_group_decayed_stats(
    frame,
    labels: pd.Series,
    times: pd.Series,
    key: str,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decayed weighted label-sum / weighted impression-sum for `key`,
    strictly before each row's own event time.

    `weights` are precomputed per-row decay weights relative to a single
    frame-wide cutoff (matching the cross-frame convention `user_ctr_decayed`
    and `_decayed_smoothed_rate` below use), so — unlike the label values —
    they carry no future information and need no additional masking beyond
    the strictly-earlier-time grouping shared with `_prior_group_stats`.
    Rows sharing a timestamp are excluded from each other's prior sums,
    exactly as `_prior_group_stats` and `_event_times` document: two rows
    at the same time see the same prior state and never one another's
    label.
    """
    source = pd.DataFrame(
        {
            "_position": np.arange(len(frame)),
            "_time": times.to_numpy(),
            "_weighted_label": labels.to_numpy() * np.asarray(weights, dtype=np.float64),
            "_weight": np.asarray(weights, dtype=np.float64),
            key: frame[key].to_numpy(),
        }
    )
    group_keys = [key, "_time"]
    grouped = source.groupby(group_keys, sort=True, dropna=False)[
        ["_weighted_label", "_weight"]
    ].sum()
    grouped["_prior_label"] = grouped.groupby(level=0, sort=False)["_weighted_label"].cumsum()
    grouped["_prior_label"] -= grouped["_weighted_label"]
    grouped["_prior_weight"] = grouped.groupby(level=0, sort=False)["_weight"].cumsum()
    grouped["_prior_weight"] -= grouped["_weight"]
    states = source.merge(
        grouped[["_prior_label", "_prior_weight"]].reset_index(),
        on=group_keys,
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_position", kind="stable")
    return (
        states["_prior_label"].to_numpy(dtype=np.float64),
        states["_prior_weight"].to_numpy(dtype=np.float64),
    )


def _in_sample_decayed_smoothed_rate(
    train_df,
    key: str,
    column: str,
    alpha: float = 20.0,
    half_life_days: float = 7.0,
) -> np.ndarray:
    """In-sample counterpart of the cross-frame branch in
    `_decayed_smoothed_rate`, used when `train_df is target_df`
    (`pipeline/train.py::_matrix` builds the training matrix this way).

    `_assert_historical_cutoff` would reject same-frame input outright
    (train and target dates overlap by construction), so this instead
    mirrors `_in_sample_group_rate`'s pattern: per-row prior sums keyed by
    `time_ms` via `_event_times`/`_prior_group_decayed_stats`, which is why
    a row never sees its own or a same-timestamp row's outcome (see
    `_event_times`'s docstring — this is the literal mechanism behind the
    acceptance criterion "a row on date t uses only rows with date < t";
    `time_ms` is finer-grained than the date and strictly implies it).

    The decay cutoff is the whole frame's max training date — a fixed,
    non-label-derived scalar — matching the single-cutoff convention
    `user_ctr_decayed` uses on the cross-frame path (recency is measured
    against the end of the training window, not against each row's own
    date); reusing a fixed cutoff here introduces no leakage, it only
    changes which reference point recency is measured from. The EB prior is
    the row-specific *prior* global rate (`_prior_global_rates`, matching
    `_in_sample_group_rate`'s convention), not the whole-frame mean
    `user_ctr_decayed` uses on the cross-frame path — the whole-frame mean
    would leak later rows into an earlier row's smoothing prior here.
    """
    labels = _numeric_outcome(train_df, column)
    times = _event_times(train_df)
    global_rates = _prior_global_rates(labels, times)

    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError(f"{column} decayed rate requires valid training dates")
    weights = decay_weights(dates, dates.max(), half_life_days=half_life_days)

    prior_weighted_label, prior_weight = _prior_group_decayed_stats(
        train_df, labels, times, key, weights
    )
    # Same formula as eb_smooth (Appendix A.1: r = (c + alpha*g)/(n+alpha)),
    # written out directly because eb_smooth's signature takes one scalar
    # global_rate, while this path's EB prior (global_rates) is per-row.
    return (prior_weighted_label + alpha * global_rates) / (prior_weight + alpha)


def _decayed_smoothed_rate(train_df, target_df, key: str, column: str) -> np.ndarray:
    """Time-decayed, EB-smoothed historical rate of `column` per `key`.

    Shared implementation behind the eight B11 auxiliary-signal features.
    Mirrors `user_ctr_decayed`'s (B5) cross-frame computation, generalised
    to any train-only outcome column and grouping key, plus an in-sample
    branch (`_in_sample_decayed_smoothed_rate`) that `user_ctr_decayed`
    itself does not have (point 4 of this task requires both paths, since
    `pipeline/train.py::_matrix` calls features both cross-frame and as
    `(train_frame, train_frame)`).
    """
    if train_df is target_df:
        return _in_sample_decayed_smoothed_rate(train_df, key, column)

    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_outcome(train_df, column)
    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError(f"{column} decayed rate requires valid training dates")

    weights = decay_weights(dates, dates.max())
    fitting_rows = pd.DataFrame(
        {
            key: train_df[key].to_numpy(),
            "_weighted_positive": weights * labels.to_numpy(),
            "_weight": weights,
        }
    )
    grouped = fitting_rows.groupby(key, sort=False)
    weighted_positives = grouped["_weighted_positive"].sum()
    effective_impressions = grouped["_weight"].sum()
    global_rate = float(labels.mean())
    rates = eb_smooth(weighted_positives, effective_impressions, global_rate)
    return target_df[key].map(rates).fillna(global_rate).to_numpy(dtype=np.float64)


@feature("user_click_rate_decayed")
def user_click_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_click rate per user (B11).

    is_click is FORBIDDEN_SAME_ROW: illegal as a same-row input read off
    target_df, legal as a historical aggregate over train_df's past rows
    only (Section 6.8, Section 11 trap 2). ESMM-style multi-feedback usage:
    an unscored signal (click) carries information about the scored one
    (long_view).
    """
    return _decayed_smoothed_rate(train_df, target_df, "user_id", "is_click")


@feature("user_like_rate_decayed")
def user_like_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_like rate per user (B11).

    See `user_click_rate_decayed` for the leakage/citation rationale.
    """
    return _decayed_smoothed_rate(train_df, target_df, "user_id", "is_like")


@feature("user_follow_rate_decayed")
def user_follow_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_follow rate per user (B11).

    See `user_click_rate_decayed` for the leakage/citation rationale.
    """
    return _decayed_smoothed_rate(train_df, target_df, "user_id", "is_follow")


@feature("video_click_rate_decayed")
def video_click_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_click rate per video (B11).

    See `user_click_rate_decayed` for the leakage/citation rationale.
    """
    return _decayed_smoothed_rate(train_df, target_df, "video_id", "is_click")


@feature("video_like_rate_decayed")
def video_like_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_like rate per video (B11).

    See `user_click_rate_decayed` for the leakage/citation rationale.
    """
    return _decayed_smoothed_rate(train_df, target_df, "video_id", "is_like")


@feature("video_follow_rate_decayed")
def video_follow_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical is_follow rate per video (B11).

    See `user_click_rate_decayed` for the leakage/citation rationale.
    """
    return _decayed_smoothed_rate(train_df, target_df, "video_id", "is_follow")


@feature("video_long_view_rate_decayed")
def video_long_view_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical long_view rate per video (B11).

    Fills a real gap: B4/B5 built the raw and decayed+smoothed *user*-level
    long_view rate (`user_ctr`, `user_ctr_decayed`), but no video-level
    decayed+smoothed counterpart existed before this task (`video_ctr` is
    raw/undecayed only).
    """
    return _decayed_smoothed_rate(train_df, target_df, "video_id", LABEL)


@feature("user_long_view_rate_decayed")
def user_long_view_rate_decayed(train_df, target_df) -> np.ndarray:
    """Time-decayed, EB-smoothed historical long_view rate per user (B11).

    Naming-symmetry counterpart to the other seven B11 features (every
    signal x granularity pair is registered as
    `<granularity>_<signal>_rate_decayed`), so this signal x granularity
    pair is discoverable the same way by A10's ablation drawer and B12's
    within-user variance screen.

    NOT a plain delegate to `user_ctr_decayed` (B5), even though the two
    are numerically identical on the cross-frame path (see
    `tests/test_aux_rates.py`): `user_ctr_decayed` has no
    `train_df is target_df` branch, so calling it in-sample hits
    `_assert_historical_cutoff` and raises. Routing this feature through
    the same `_decayed_smoothed_rate` helper as the other seven gives it
    working in-sample support too, which `pipeline/train.py::_matrix`'s
    `(train_frame, train_frame)` call pattern requires of every registered
    feature — a real capability `user_ctr_decayed` alone does not have.
    """
    return _decayed_smoothed_rate(train_df, target_df, "user_id", LABEL)


# --- sim_to_history (B14) ----------------------------------------------------
#
# New task, not yet written into AGENT_PLAN.md when this was built -- handed
# down directly by the team, high priority because it paces Ethan's C10
# (the sequence-model rung, Section 6.7's rung 6, "skip unless ahead of
# schedule" -- the team is ahead of schedule, having already shipped C9,
# rung 5, multi-task DeepFM, also originally conditional on being ahead).
#
# Citation (Section 6.8's feature-policy table): the "Sequence summaries:
# last-N author/category affinity, time since prior event" row, controlled
# as "Use after audit / Chronological construction, no look-ahead". This is
# the first feature in this codebase implementing that row -- prior work
# (user_tag_affinity, B4) covers "Identifiers"/tag-based target encoding,
# not a recency-anchored sequence summary.
#
# Registered under the exact name `sim_to_history` (not, e.g.,
# `user_tag_history_similarity`) on explicit team instruction: C10 will
# reference it by this literal string.


def _prior_sum_by_time(source, group_cols: list[str], value_col: str) -> np.ndarray:
    """Strictly-prior cumulative sum of `value_col` within `group_cols`,
    keyed by `_time`.

    Generalises the cumsum-then-subtract-current-bucket trick
    `_prior_group_stats`/`_prior_group_decayed_stats` already use in this
    file to an arbitrary value column and an arbitrary number of grouping
    columns -- needed here for the composite (user_id, tag) key that
    `_in_sample_sim_to_history` groups by. Rows sharing a `_time` within the
    same group exclude each other from one another's prior sum, exactly as
    `_event_times`'s docstring establishes for every other in-sample helper
    in this file.

    `source` must carry a `_time` column plus every column named in
    `group_cols`. The returned array is aligned to `source`'s own row order
    (via an internal row counter, then a stable sort undoing whatever order
    the groupby/merge round trip left rows in) regardless of `source`'s own
    pandas index.
    """
    working = source.reset_index(drop=True)
    working = working.assign(_row=np.arange(len(working)))
    group_keys = [*group_cols, "_time"]
    bucket_sums = working.groupby(group_keys, sort=True, dropna=False)[value_col].sum()
    bucket_sums = bucket_sums.to_frame("_bucket_sum")
    level = list(range(len(group_cols)))
    bucket_sums["_prior_sum"] = bucket_sums.groupby(level=level, sort=False)["_bucket_sum"].cumsum()
    bucket_sums["_prior_sum"] -= bucket_sums["_bucket_sum"]
    merged = working.merge(
        bucket_sums[["_prior_sum"]].reset_index(),
        on=group_keys,
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_row", kind="stable")
    return merged["_prior_sum"].to_numpy(dtype=np.float64)


def _in_sample_sim_to_history(train_df) -> np.ndarray:
    """Strictly-prior in-sample counterpart of `sim_to_history`'s cross-frame
    branch, used when `train_df is target_df`
    (`pipeline/train.py::_matrix` builds the training design matrix this
    way; Section 8.4's dual-path requirement).

    Positivity is read from `train_df`'s own label column -- legal, since it
    is determined per-row from `train_df`'s own data, exactly the reasoning
    `_in_sample_pcr`'s docstring gives for reading `play_time_ms` off
    `train_df` only.

    Every row of `train_df` (positive or not) is exploded by its own tags
    into ONE combined table, tagging each exploded entry with a
    "positive decayed weight contribution": the row's own `decay_weights`
    value when its label is positive, else 0.0. Both roles this feature
    needs -- "history" (a positive row's weight feeding its own (user, tag)
    bucket) and "query" (every row, positive or not, needs an output value
    computed from strictly-prior positive history) -- are read off that same
    table. That is what lets a plain `_time`-keyed cumulative-sum-then-
    subtract (`_prior_sum_by_time`) work at all here: every (user, tag,
    time) bucket a query row needs is guaranteed already present to merge
    against, because a negative row still creates a bucket (contributing
    weight 0), it just never inflates any prior sum. Splitting "history"
    and "query" into two independently-exploded frames would not have this
    property -- a query row's own (user, tag, time) triple would not
    reliably exist in a history table built only from positive rows.

    Decay weighting IS applied here, unlike `_in_sample_pcr`'s deliberate
    no-decay choice (see that function's own docstring for its rationale,
    specific to that feature, not a blanket rule for this file) -- decay is
    the entire reason this feature is a *recency*-anchored signal rather
    than a plain historical rate, so dropping it on the in-sample path would
    silently make this a different, weaker feature than its cross-frame
    twin.
    """
    labels = _numeric_labels(train_df)
    times = _event_times(train_df)
    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError("sim_to_history requires valid training dates")
    weights = decay_weights(dates, dates.max())
    is_positive = labels.to_numpy() == 1.0

    exploded = pd.DataFrame(
        {
            "_position": np.arange(len(train_df)),
            "_time": times.to_numpy(),
            "user_id": train_df["user_id"].to_numpy(),
            "_tag": train_df["tag"].map(_split_tags).to_numpy(),
            "_pos_weight": np.where(is_positive, weights, 0.0),
        }
    ).explode("_tag", ignore_index=True)
    exploded = exploded.loc[exploded["_tag"].notna()].copy()

    result = np.zeros(len(train_df), dtype=np.float64)
    if exploded.empty:
        return result

    tag_prior = _prior_sum_by_time(exploded, ["user_id", "_tag"], "_pos_weight")
    user_prior_total = _prior_sum_by_time(exploded, ["user_id"], "_pos_weight")
    exploded["_share"] = np.divide(
        tag_prior,
        user_prior_total,
        out=np.zeros(len(exploded), dtype=np.float64),
        where=user_prior_total > 0,
    )
    per_row = exploded.groupby("_position", sort=False)["_share"].mean()
    result[per_row.index.to_numpy(dtype=int)] = per_row.to_numpy(dtype=np.float64)
    return result


@feature("sim_to_history")
def sim_to_history(train_df, target_df) -> np.ndarray:
    """Attention-SHARE overlap between a candidate video's tags and a
    user's own recent, positively-engaged watch history (B14).

    Formula. For each user u, build a decayed tag-exposure distribution
    from u's historical positive rows only (this file's scored label is
    positive when it equals 1, per the correction documented near `LABEL`
    above): for every (u, tag) pair, sum `decay_weights` (B5) over u's
    positive history rows containing that tag (a video can carry several
    tags; `_split_tags` explodes one row per tag, following the same
    "explode, ignore_index=True, drop nulls" pattern `user_tag_affinity`
    already uses for its own multi-tag column). Each (u, tag) weight is then
    normalised by u's TOTAL decayed positive weight across all their tags,
    giving a probability-like distribution over the tags u has recently,
    positively engaged with (sums to 1 per user with any positive history).
    For a target row belonging to user u with candidate tags T (read off
    `target_df`'s own tag column -- static, pre-exposure metadata, legal to
    read directly off the target frame, exactly as `user_tag_affinity`
    already does): `sim_to_history = mean over t in T of
    normalized_weight(u, t)`. A (user, tag) combination with no historical
    positive signal contributes 0.0.

    Fallback (0.0, not a global-mean fallback). A target row with no tags,
    or a user with no positive history at all, returns 0.0. This is
    deliberately different from the global-rate fallback `_group_rate` and
    the decayed/EB-smoothed rate features (`user_ctr_decayed`,
    `_decayed_smoothed_rate`) use for their own unseen groups: those are
    RATE-style features estimating "how likely is this outcome", where a
    population-level average rate is a meaningful prior for a sparse or
    unseen group. `sim_to_history` estimates an ATTENTION-SHARE / overlap
    quantity -- "what fraction of this user's own recent positive attention
    landed on tags this candidate shares" -- and there is no analogous
    population-level "average similarity" a user with zero positive history
    could sensibly be assigned; 0.0 ("no information, no measured overlap")
    is the only well-defined value in that case, not an approximation of
    one that was skipped for convenience.

    Why this is NOT redundant with `user_tag_affinity` (B4), mirroring how
    `user_long_view_rate_decayed`'s docstring documents its own
    non-redundancy with `user_ctr_decayed` rather than leaving the
    similarity unexplained. `user_tag_affinity` estimates an ENGAGEMENT
    RATE: among a user's rows carrying a given tag, what fraction were a
    positive outcome. That rate can be high from a single old row (n=1,
    positive) and never decays or restricts itself to positive-only
    history -- it mixes positive and negative rows together to estimate a
    conditional probability. `sim_to_history` instead estimates ATTENTION
    SHARE within positive-only, decayed-recency history: "of what this user
    has recently, positively watched, how much of it shares this
    candidate's tags." A user who watched one old video on tag X to
    completion (rate-wise, a perfect affinity for X) years before a burst of
    recent positive activity entirely on tag Y will show near-zero
    `sim_to_history` for an X-tagged candidate today (that mass now belongs
    almost entirely to Y) while `user_tag_affinity` for X stays high --
    they diverge by construction, not by coincidence. This is the
    recency-anchored "last-N tag affinity" signal Section 6.8's Sequence
    summaries row names, and the first feature in this codebase to
    implement that row (`user_tag_affinity`/B4 belongs to the Identifiers
    row's target-encoding control, not Sequence summaries).

    Dual path (Section 8.4, frozen). Cross-frame:
    `_assert_historical_cutoff` first (fails closed unless every train date
    precedes every target date), fit entirely on `train_df`'s positive rows,
    apply to `target_df`. In-sample (`train_df is target_df`):
    `_assert_historical_cutoff` would reject same-frame input outright
    (dates overlap by construction), so this delegates to
    `_in_sample_sim_to_history`, which uses `_event_times` plus a strictly-
    prior cumulative decayed sum (`_prior_sum_by_time`) so a row never sees
    its own or a same-`time_ms` row's positive tags -- see that function's
    own docstring for why decay weighting is applied there too, unlike
    `_in_sample_pcr`'s undecayed precedent.

    Leakage. Reads `target_df`'s user_id and tag columns only -- both
    static, pre-exposure, legal identifiers/metadata (the same target-side
    reads `user_tag_affinity` already makes) -- and never reads target_df's
    scored label or any FORBIDDEN_SAME_ROW column. Passes `leakage_check`
    both statically and dynamically.
    """
    if train_df is target_df:
        return _in_sample_sim_to_history(train_df)

    _assert_historical_cutoff(train_df, target_df)
    labels = _numeric_labels(train_df)
    dates = _parse_dates(train_df["date"])
    if dates.isna().any():
        raise ValueError("sim_to_history requires valid training dates")
    weights = decay_weights(dates, dates.max())

    result = np.zeros(len(target_df), dtype=np.float64)
    is_positive = labels.to_numpy() == 1.0
    if not is_positive.any() or target_df.empty:
        return result

    positive_train = pd.DataFrame(
        {
            "user_id": train_df["user_id"].to_numpy()[is_positive],
            "_weight": weights[is_positive],
            "_tag": train_df["tag"].map(_split_tags).to_numpy()[is_positive],
        }
    ).explode("_tag", ignore_index=True)
    positive_train = positive_train.loc[positive_train["_tag"].notna()]
    if positive_train.empty:
        return result

    tag_weight = positive_train.groupby(["user_id", "_tag"], sort=False)["_weight"].sum()
    user_total = positive_train.groupby("user_id", sort=False)["_weight"].sum()
    denominators = tag_weight.index.get_level_values("user_id").map(user_total).to_numpy(
        dtype=np.float64
    )
    normalized = tag_weight / denominators

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
    exploded_target = exploded_target.assign(
        _share=normalized.reindex(pairs).fillna(0.0).to_numpy()
    )
    row_share = exploded_target.groupby("_position", sort=False)["_share"].mean()
    result[row_share.index.to_numpy(dtype=int)] = row_share.to_numpy(dtype=np.float64)
    return result
