"""B10 acceptance: the duration-bias feature pack.

`video_duration`, `duration_bucket`, `pcr_hist`,
`long_view_rate_by_duration_group` -- each fits on train_df only, passes
the existing leakage guard, and (for the two quantile-bucketed features)
reuses train-fit bucket edges unchanged on validation/test rather than
refitting them on the target frame's own distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features as feature_module
from pipeline.features import (
    DURATION_BUCKET_COUNT,
    duration_bucket,
    eb_smooth,
    leakage_check,
    long_view_rate_by_duration_group,
    pcr_hist,
    video_duration,
)

NEW_FEATURE_NAMES = {
    "video_duration",
    "duration_bucket",
    "pcr_hist",
    "long_view_rate_by_duration_group",
}


def test_all_four_features_are_registered_and_callable_by_name():
    assert NEW_FEATURE_NAMES <= set(feature_module.FEATURES)
    for name in NEW_FEATURE_NAMES:
        assert callable(feature_module.get(name))


def test_duration_bucket_count_default_matches_d2q_spirit():
    assert DURATION_BUCKET_COUNT == 10


# --- video_duration -----------------------------------------------------


def test_video_duration_returns_raw_pre_exposure_duration():
    train_df = pd.DataFrame({"duration_ms": [1.0]})  # unused, must not raise
    target_df = pd.DataFrame({"duration_ms": [5000, 15000.0, 0]})

    result = video_duration(train_df, target_df)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [5000.0, 15000.0, 0.0])


def test_video_duration_ignores_train_df_entirely():
    target_df = pd.DataFrame({"duration_ms": [1000.0, 2000.0]})
    empty_train = pd.DataFrame()

    result = video_duration(empty_train, target_df)

    np.testing.assert_allclose(result, [1000.0, 2000.0])


def test_video_duration_rejects_non_finite_duration():
    target_df = pd.DataFrame({"duration_ms": [1000.0, np.nan]})
    with pytest.raises(ValueError, match="duration_ms"):
        video_duration(pd.DataFrame(), target_df)


# --- duration_bucket / _duration_bucket_edges ----------------------------


def test_duration_bucket_edges_match_the_documented_quantile_recipe():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"duration_ms": rng.integers(1_000, 200_000, size=500).astype(float)})

    edges = feature_module._duration_bucket_edges(train_df, n_buckets=10)
    expected = np.quantile(train_df["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    np.testing.assert_allclose(edges, expected)
    assert edges.shape == (9,)


def test_duration_bucket_edges_bucket_count_is_configurable():
    train_df = pd.DataFrame({"duration_ms": [1.0, 2.0, 3.0, 4.0]})

    assert feature_module._duration_bucket_edges(train_df, n_buckets=4).shape == (3,)
    assert feature_module._duration_bucket_edges(train_df, n_buckets=2).shape == (1,)


def test_duration_bucket_edges_reject_empty_or_non_positive_bucket_count():
    train_df = pd.DataFrame({"duration_ms": [1.0, 2.0]})
    with pytest.raises(ValueError, match="positive integer"):
        feature_module._duration_bucket_edges(train_df, n_buckets=0)
    with pytest.raises(ValueError, match="empty training frame"):
        feature_module._duration_bucket_edges(pd.DataFrame({"duration_ms": []}))


def test_duration_bucket_edges_never_refit_on_the_target_distribution():
    """Core B10 acceptance criterion: edges computed on train are reused
    unchanged for validation/test, regardless of the target's own spread."""
    train_df = pd.DataFrame({"duration_ms": np.linspace(1_000, 100_000, 50)})
    edges = feature_module._duration_bucket_edges(train_df)

    wide_target = pd.DataFrame({"duration_ms": [500.0, 50_000.0, 999_999.0]})
    narrow_target = pd.DataFrame({"duration_ms": [1.0, 2.0, 3.0]})  # far outside train's range

    result_wide = duration_bucket(train_df, wide_target)
    result_narrow = duration_bucket(train_df, narrow_target)

    np.testing.assert_array_equal(
        result_wide, np.searchsorted(edges, wide_target["duration_ms"].to_numpy())
    )
    np.testing.assert_array_equal(
        result_narrow, np.searchsorted(edges, narrow_target["duration_ms"].to_numpy())
    )
    # Edges recomputed from train_df alone are identical no matter which
    # (or whether any) target frame was previously applied to them.
    np.testing.assert_array_equal(edges, feature_module._duration_bucket_edges(train_df))


def test_duration_bucket_applies_unchanged_edges_even_in_sample():
    """train_df is target_df: still no refitting, since duration_ms carries
    no outcome information (unlike the label-dependent features below)."""
    train_df = pd.DataFrame({"duration_ms": [1000.0, 5000.0, 9000.0, 15000.0]})

    result = duration_bucket(train_df, train_df)
    edges = feature_module._duration_bucket_edges(train_df)
    expected = np.searchsorted(edges, train_df["duration_ms"].to_numpy())

    np.testing.assert_array_equal(result, expected)


def test_duration_bucket_rejects_non_finite_duration():
    train_df = pd.DataFrame({"duration_ms": [1000.0, 2000.0]})
    target_df = pd.DataFrame({"duration_ms": [1000.0, np.inf]})
    with pytest.raises(ValueError, match="duration_ms"):
        duration_bucket(train_df, target_df)


# --- shared fixtures for pcr_hist / long_view_rate_by_duration_group -----


@pytest.fixture
def duration_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u1"],
            "video_id": ["v1", "v2", "v1", "v3", "v4"],
            "date": [20220409, 20220410, 20220411, 20220412, 20220413],
            "duration_ms": [5000, 15000, 5000, 25000, 15000],
            "play_time_ms": [2500, 15000, 1000, 5000, 7500],
            "long_view": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def duration_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "video_id": ["v1", "v2", "v4"],
            "date": [20220422, 20220423, 20220424],
            "duration_ms": [5000.0, 15000.0, 999_999.0],
            "long_view": [0, 1, 0],
            "is_click": [1, 0, 1],
            "play_time_ms": [0, 0, 0],
        }
    )


# --- pcr_hist -------------------------------------------------------------


def test_completion_ratio_guards_zero_duration_and_clips_overwatch():
    train_df = pd.DataFrame(
        {
            "duration_ms": [0.0, 10000.0, 1000.0],
            "play_time_ms": [500.0, 8000.0, 5000.0],
        }
    )
    ratio = feature_module._completion_ratio(train_df)
    np.testing.assert_allclose(ratio.to_numpy(), [0.0, 0.8, 1.0])


def test_pcr_hist_matches_hand_computed_decay_and_eb_smoothing():
    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "date": [20220407, 20220414, 20220421, 20220407],
            "duration_ms": [10000, 10000, 10000, 10000],
            "play_time_ms": [5000, 10000, 2000, 0],
        }
    )
    target_df = pd.DataFrame(
        {"user_id": ["u1", "u2", "unseen"], "date": [20220422, 20220422, 20220422]},
        index=[9, 4, 71],
    )

    weights = feature_module.decay_weights(train_df["date"], train_df["date"].max())
    ratio = np.array([0.5, 1.0, 0.2, 0.0])
    global_rate = float(ratio.mean())
    expected_u1 = eb_smooth(weights[:3] @ ratio[:3], weights[:3].sum(), global_rate)
    expected_u2 = eb_smooth(weights[3] * ratio[3], weights[3], global_rate)

    result = pcr_hist(train_df, target_df)

    np.testing.assert_allclose(result, [expected_u1, expected_u2, global_rate])
    assert result.shape == (len(target_df),)
    assert np.isfinite(result).all()


def test_pcr_hist_ignores_target_outcomes_and_only_reads_train_play_time(
    duration_train_df, duration_target_df
):
    changed = duration_target_df.copy()
    changed["long_view"] = 1 - changed["long_view"]
    changed["is_click"] = 1 - changed["is_click"]
    changed["play_time_ms"] = 999_999  # forbidden same-row column, must be irrelevant

    original = pcr_hist(duration_train_df, duration_target_df)
    after = pcr_hist(duration_train_df, changed)

    np.testing.assert_array_equal(original, after)


def test_pcr_hist_rejects_overlapping_target_window(duration_train_df, duration_target_df):
    overlapping = duration_target_df.copy()
    overlapping["date"] = duration_train_df["date"].max()

    with pytest.raises(ValueError, match="strictly earlier"):
        pcr_hist(duration_train_df, overlapping)


def test_pcr_hist_in_sample_uses_only_strictly_prior_events_by_time_ms():
    frame = pd.DataFrame(
        {
            "time_ms": [300, 100, 200, 200, 400],
            "date": [20220408] * 5,
            "user_id": ["u1", "u1", "u1", "u2", "u1"],
            "duration_ms": [10000] * 5,
            "play_time_ms": [8000, 4000, 6000, 5000, 0],
        }
    )
    # ratio = [0.8, 0.4, 0.6, 0.5, 0.0]
    # row0 u1@300: prior u1 events (time100=0.4, time200=0.6) -> mean 0.5
    # row1 u1@100: no prior u1 events, no prior events at all -> INITIAL_RATE 0.5
    # row2 u1@200: prior u1 events (time100=0.4) -> 0.4
    # row3 u2@200: no prior u2 events; prior global (time100=0.4) -> 0.4
    # row4 u1@400: prior u1 events (0.4, 0.6, 0.8) -> mean 0.6
    expected = [0.5, 0.5, 0.4, 0.4, 0.6]

    result = pcr_hist(frame, frame)

    np.testing.assert_allclose(result, expected)


def test_pcr_hist_in_sample_requires_time_ms(duration_train_df):
    with pytest.raises(ValueError, match="time_ms"):
        pcr_hist(duration_train_df, duration_train_df)


# --- long_view_rate_by_duration_group -------------------------------------


def test_long_view_rate_by_duration_group_matches_hand_computed_eb_rates(monkeypatch):
    monkeypatch.setattr(feature_module, "DURATION_BUCKET_COUNT", 2)
    train_df = pd.DataFrame(
        {
            "duration_ms": [1000, 1000, 1000, 3000, 3000],
            "long_view": [1, 1, 0, 0, 1],
            "date": [20220409] * 5,
        }
    )
    target_df = pd.DataFrame(
        {"duration_ms": [1000.0, 3000.0, 5_000_000.0], "date": [20220422] * 3}
    )

    # edges = [1000.0] (median of [1000,1000,1000,3000,3000]).
    # bucket0 (duration 1000): sum=2, count=3 -> eb_smooth(2, 3, 0.6)
    # bucket1 (duration >=1000... i.e. 3000 and beyond): sum=1, count=2 -> eb_smooth(1, 2, 0.6)
    global_rate = 0.6
    expected_bucket0 = eb_smooth(2, 3, global_rate)
    expected_bucket1 = eb_smooth(1, 2, global_rate)

    result = long_view_rate_by_duration_group(train_df, target_df)

    np.testing.assert_allclose(result, [expected_bucket0, expected_bucket1, expected_bucket1])


def test_long_view_rate_by_duration_group_ignores_target_outcomes(
    duration_train_df, duration_target_df
):
    changed = duration_target_df.copy()
    changed["long_view"] = 1 - changed["long_view"]
    changed["is_click"] = 1 - changed["is_click"]

    original = long_view_rate_by_duration_group(duration_train_df, duration_target_df)
    after = long_view_rate_by_duration_group(duration_train_df, changed)

    np.testing.assert_array_equal(original, after)


def test_long_view_rate_by_duration_group_rejects_overlapping_target_window(
    duration_train_df, duration_target_df
):
    overlapping = duration_target_df.copy()
    overlapping["date"] = duration_train_df["date"].max()

    with pytest.raises(ValueError, match="strictly earlier"):
        long_view_rate_by_duration_group(duration_train_df, overlapping)


def test_long_view_rate_by_duration_group_in_sample_uses_prior_events_only(monkeypatch):
    monkeypatch.setattr(feature_module, "DURATION_BUCKET_COUNT", 2)
    frame = pd.DataFrame(
        {
            "time_ms": [100, 200, 300, 400],
            "date": [20220409] * 4,
            "duration_ms": [1000, 1000, 3000, 3000],
            "long_view": [1, 0, 1, 0],
        }
    )
    # edges = [2000.0]; bucket0 = durations<2000 (rows 0,1), bucket1 = rows 2,3.
    # row0 bucket0@100: no prior bucket0 events, none at all -> INITIAL_RATE 0.5
    # row1 bucket0@200: prior bucket0 (row0, label1) -> 1.0
    # row2 bucket1@300: no prior bucket1 events; prior global (row0=1,row1=0) -> 0.5
    # row3 bucket1@400: prior bucket1 (row2, label1) -> 1.0
    expected = [0.5, 1.0, 0.5, 1.0]

    result = long_view_rate_by_duration_group(frame, frame)

    np.testing.assert_allclose(result, expected)


def test_long_view_rate_by_duration_group_in_sample_requires_time_ms(duration_train_df):
    with pytest.raises(ValueError, match="time_ms"):
        long_view_rate_by_duration_group(duration_train_df, duration_train_df)


def test_duration_bucket_and_duration_group_rate_use_identical_edges(monkeypatch):
    """B10: the two quantile-bucketed features must not compute edges two
    different ways -- two target rows duration_bucket() puts in the same
    bucket must get the same long_view_rate_by_duration_group() value."""
    monkeypatch.setattr(feature_module, "DURATION_BUCKET_COUNT", 3)
    train_df = pd.DataFrame(
        {
            "duration_ms": [1000, 2000, 3000, 8000, 9000, 15000, 16000, 17000, 18000],
            "long_view": [1, 0, 1, 0, 1, 0, 1, 1, 0],
            "date": [20220409] * 9,
        }
    )
    target_df = pd.DataFrame(
        {"duration_ms": [1500.0, 2500.0, 17500.0], "date": [20220422] * 3}
    )  # first two share a bucket

    buckets = duration_bucket(train_df, target_df)
    rates = long_view_rate_by_duration_group(train_df, target_df)

    assert buckets[0] == buckets[1]
    assert rates[0] == pytest.approx(rates[1])
    assert buckets[2] != buckets[0]


# --- leakage guard (B6) ----------------------------------------------------


def test_video_duration_and_duration_bucket_pass_leakage_check(
    duration_train_df, duration_target_df
):
    assert leakage_check(video_duration, duration_train_df, duration_target_df) is True
    assert leakage_check(duration_bucket, duration_train_df, duration_target_df) is True


def test_pcr_hist_passes_leakage_check_cross_frame(duration_train_df, duration_target_df):
    assert leakage_check(pcr_hist, duration_train_df, duration_target_df) is True


def test_long_view_rate_by_duration_group_passes_leakage_check_cross_frame(
    duration_train_df, duration_target_df
):
    assert (
        leakage_check(long_view_rate_by_duration_group, duration_train_df, duration_target_df)
        is True
    )


@pytest.fixture
def duration_in_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "duration_ms": [5000, 5000, 15000],
            "play_time_ms": [2500, 5000, 7500],
            "long_view": [1, 0, 1],
            "time_ms": [100, 200, 300],
        }
    )


def test_pcr_hist_and_duration_group_rate_pass_leakage_check_in_sample(
    duration_in_sample_frame,
):
    assert leakage_check(pcr_hist, duration_in_sample_frame, duration_in_sample_frame) is True
    assert (
        leakage_check(
            long_view_rate_by_duration_group, duration_in_sample_frame, duration_in_sample_frame
        )
        is True
    )
