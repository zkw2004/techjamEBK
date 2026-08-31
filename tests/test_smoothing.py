"""B5 acceptance: empirical-Bayes smoothing and exponential time decay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import FEATURES, decay_weights, eb_smooth, user_ctr_decayed


def test_eb_smooth_matches_the_pinned_formula_and_sparse_groups_shrink():
    smoothed = eb_smooth(clicks=1, impressions=2, global_rate=0.1)

    assert smoothed == pytest.approx((1 + 20 * 0.1) / (2 + 20))
    assert abs(smoothed - 0.1) < 0.05
    assert abs(smoothed - 0.1) < abs(0.5 - 0.1)
    assert eb_smooth(0, 0, 0.42) == pytest.approx(0.42)


def test_eb_smooth_supports_custom_alpha_arrays_and_fractional_counts():
    clicks = np.array([0.0, 1.5, 80.0])
    impressions = np.array([0.0, 2.5, 100.0])

    result = eb_smooth(clicks, impressions, global_rate=0.25, alpha=4.0)
    expected = (clicks + 4.0 * 0.25) / (impressions + 4.0)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_eb_smooth_preserves_series_index_for_group_mapping():
    index = pd.Index(["u2", "u1"], name="user_id")
    clicks = pd.Series([1.0, 8.0], index=index, name="weighted_positives")
    impressions = pd.Series([2.0, 10.0], index=index)

    result = eb_smooth(clicks, impressions, global_rate=0.4)

    assert isinstance(result, pd.Series)
    assert result.index.equals(index)
    assert result.name == clicks.name


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_eb_smooth_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        eb_smooth(1, 2, 0.2, alpha=alpha)


@pytest.mark.parametrize("global_rate", [-0.01, 1.01, np.nan, np.inf])
def test_eb_smooth_rejects_invalid_global_rate(global_rate):
    with pytest.raises(ValueError, match="global_rate"):
        eb_smooth(1, 2, global_rate)


@pytest.mark.parametrize(
    ("clicks", "impressions"),
    [
        (-1, 2),
        (1, -2),
        (3, 2),
        (np.nan, 2),
        (1, np.inf),
        ([1, 2], [2]),
    ],
)
def test_eb_smooth_rejects_invalid_counts(clicks, impressions):
    with pytest.raises(ValueError):
        eb_smooth(clicks, impressions, 0.2)


def test_eb_smooth_rejects_misaligned_series():
    clicks = pd.Series([1.0], index=["u1"])
    impressions = pd.Series([2.0], index=["u2"])

    with pytest.raises(ValueError, match="identical indexes"):
        eb_smooth(clicks, impressions, 0.2)


def test_decay_weights_obey_half_life_identities_for_real_date_encodings():
    dates = [20220421, 20220414.0, "2022-04-07"]

    result = decay_weights(dates, cutoff=20220421, half_life_days=7)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 0.5, 0.25])


def test_decay_weights_are_configurable_and_preserve_input_order():
    dates = np.array(["2022-03-31", "2022-04-07"], dtype="datetime64[D]")

    seven_day = decay_weights(dates, cutoff=pd.Timestamp("2022-04-07"), half_life_days=7)
    fourteen_day = decay_weights(20220331, cutoff=20220407, half_life_days=14)

    np.testing.assert_allclose(seven_day, [0.5, 1.0])
    np.testing.assert_allclose(fourteen_day, [np.sqrt(0.5)])
    assert fourteen_day.shape == (1,)
    assert decay_weights([], cutoff=20220407).shape == (0,)


@pytest.mark.parametrize("half_life", [0, -1, np.nan, np.inf])
def test_decay_weights_reject_invalid_half_life(half_life):
    with pytest.raises(ValueError, match="half_life_days"):
        decay_weights([20220401], cutoff=20220407, half_life_days=half_life)


@pytest.mark.parametrize(
    ("dates", "cutoff", "message"),
    [
        ([20220431], 20220501, "valid calendar dates"),
        ([20220401, None], 20220407, "valid calendar dates"),
        ([20220408], 20220407, "later than cutoff"),
        ([20220401], 20220431, "valid calendar date"),
    ],
)
def test_decay_weights_reject_invalid_or_future_dates(dates, cutoff, message):
    with pytest.raises(ValueError, match=message):
        decay_weights(dates, cutoff)


@pytest.fixture
def decayed_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "date": [20220407, 20220414, 20220421, 20220407],
            "long_view": [1, 0, 1, 0],
        }
    )
    target_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "unseen", "u1"],
            "date": [20220422] * 4,
            "long_view": [0, 1, 1, 1],
        },
        index=[90, 4, 71, 90],
    )
    return train_df, target_df


def test_user_ctr_decayed_is_registered_and_maps_smoothed_train_history(decayed_frames):
    train_df, target_df = decayed_frames
    global_rate = 0.5
    expected_u1 = eb_smooth(1.25, 1.75, global_rate)
    expected_u2 = eb_smooth(0.0, 0.25, global_rate)

    assert FEATURES["user_ctr_decayed"] is user_ctr_decayed
    result = user_ctr_decayed(train_df, target_df)

    np.testing.assert_allclose(result, [expected_u1, expected_u2, global_rate, expected_u1])
    assert result.shape == (len(target_df),)
    assert np.isfinite(result).all()


def test_user_ctr_decayed_ignores_target_outcomes_and_accepts_string_labels(decayed_frames):
    train_df, target_df = decayed_frames
    changed_target = target_df.assign(long_view=1 - target_df["long_view"])
    string_train = train_df.assign(long_view=train_df["long_view"].astype(str))

    expected = user_ctr_decayed(train_df, target_df)
    np.testing.assert_array_equal(user_ctr_decayed(train_df, changed_target), expected)
    np.testing.assert_array_equal(user_ctr_decayed(string_train, target_df), expected)


def test_user_ctr_decayed_rejects_overlapping_windows_and_handles_empty_target(decayed_frames):
    train_df, target_df = decayed_frames
    overlapping = target_df.assign(date=20220421)

    with pytest.raises(ValueError, match="strictly earlier"):
        user_ctr_decayed(train_df, overlapping)

    assert user_ctr_decayed(train_df, target_df.iloc[0:0]).shape == (0,)


def test_user_ctr_decayed_builds_a_strictly_historical_training_matrix():
    """Its registered training-matrix path must not leak later row labels.

    ``pipeline.train._matrix`` calls every registered feature with the same
    frame twice when constructing the fit matrix. This regression protects
    the path that previously raised during the leakage audit and caused the
    feature sweep to quarantine ``user_ctr_decayed``.
    """
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "date": [20220407, 20220408, 20220408, 20220409],
            "time_ms": [100, 200, 200, 300],
            "long_view": [1, 0, 1, 0],
        }
    )

    result = user_ctr_decayed(frame, frame)
    changed_future = frame.assign(long_view=[1, 1, 0, 1])

    # The first event has no history; equal-time rows cannot see each other.
    np.testing.assert_allclose(result[:3], [0.5, 1.0, 1.0])
    # Changing outcomes at/after 200 cannot affect the first row's feature.
    assert user_ctr_decayed(changed_future, changed_future)[0] == pytest.approx(result[0])
    assert result.shape == (len(frame),)
    assert np.isfinite(result).all()
