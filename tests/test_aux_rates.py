"""B11 acceptance: auxiliary-signal (click/like/follow/long_view) historical
rates, per user and per video, time-decayed and EB-smoothed. A feature value
never changes when future-dated rows are edited; same-day rows are excluded;
every new feature is registered in the feature registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features as feature_module
from pipeline.features import (
    FEATURES,
    decay_weights,
    eb_smooth,
    leakage_check,
    user_click_rate_decayed,
    user_ctr_decayed,
    user_follow_rate_decayed,
    user_like_rate_decayed,
    user_long_view_rate_decayed,
    video_click_rate_decayed,
    video_follow_rate_decayed,
    video_like_rate_decayed,
    video_long_view_rate_decayed,
)

AUX_FEATURE_NAMES = {
    "user_click_rate_decayed",
    "user_like_rate_decayed",
    "user_follow_rate_decayed",
    "user_long_view_rate_decayed",
    "video_click_rate_decayed",
    "video_like_rate_decayed",
    "video_follow_rate_decayed",
    "video_long_view_rate_decayed",
}

AUX_FEATURE_FUNCS = {
    "user_click_rate_decayed": user_click_rate_decayed,
    "user_like_rate_decayed": user_like_rate_decayed,
    "user_follow_rate_decayed": user_follow_rate_decayed,
    "user_long_view_rate_decayed": user_long_view_rate_decayed,
    "video_click_rate_decayed": video_click_rate_decayed,
    "video_like_rate_decayed": video_like_rate_decayed,
    "video_follow_rate_decayed": video_follow_rate_decayed,
    "video_long_view_rate_decayed": video_long_view_rate_decayed,
}


def test_all_eight_aux_rate_features_are_registered():
    assert AUX_FEATURE_NAMES <= set(FEATURES)
    for name, fn in AUX_FEATURE_FUNCS.items():
        assert FEATURES[name] is fn


# --- Cross-frame numeric correctness ----------------------------------------


@pytest.fixture
def cross_frame_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "video_id": ["v1", "v1", "v2", "v1"],
            "date": [20220407, 20220414, 20220421, 20220407],
            "is_click": [1, 0, 1, 0],
            "is_like": [0, 1, 0, 1],
            "is_follow": [1, 0, 0, 0],
            "long_view": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def cross_frame_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "unseen"],
            "video_id": ["v1", "v2", "unseen"],
            "date": [20220422, 20220423, 20220424],
        }
    )


def test_user_click_rate_decayed_matches_hand_computed_eb_smoothed_value(
    cross_frame_train_df, cross_frame_target_df
):
    # cutoff = max train date = 20220421, half_life = 7 days (defaults).
    # u1 rows: ages 14, 7, 0 -> weights 0.25, 0.5, 1.0; is_click = 1, 0, 1
    # weighted_positive = 0.25*1 + 0.5*0 + 1*1 = 1.25; impressions = 1.75
    # u2 row: age 14 -> weight 0.25; is_click = 0 -> weighted_positive = 0
    global_rate = 0.5  # mean(is_click) over all 4 training rows
    expected_u1 = eb_smooth(1.25, 1.75, global_rate)
    expected_u2 = eb_smooth(0.0, 0.25, global_rate)

    result = user_click_rate_decayed(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result, [expected_u1, expected_u2, global_rate])


def test_video_click_rate_decayed_matches_hand_computed_eb_smoothed_value(
    cross_frame_train_df, cross_frame_target_df
):
    # v1 rows: (u1,0407,click1,w0.25), (u1,0414,click0,w0.5), (u2,0407,click0,w0.25)
    # weighted_positive = 0.25*1 + 0.5*0 + 0.25*0 = 0.25; impressions = 1.0
    # v2 row: (u1,0421,click1,w1.0) -> weighted_positive = 1.0; impressions = 1.0
    global_rate = 0.5
    expected_v1 = eb_smooth(0.25, 1.0, global_rate)
    expected_v2 = eb_smooth(1.0, 1.0, global_rate)

    result = video_click_rate_decayed(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result, [expected_v1, expected_v2, global_rate])


def test_user_follow_rate_decayed_matches_hand_computed_eb_smoothed_value(
    cross_frame_train_df, cross_frame_target_df
):
    # is_follow: u1 = [1, 0, 0] with weights [0.25, 0.5, 1.0]
    # weighted_positive = 0.25; impressions = 1.75
    # u2: is_follow = 0, weight 0.25 -> weighted_positive = 0, impressions = 0.25
    global_rate = 0.25  # mean(is_follow) = 1/4
    expected_u1 = eb_smooth(0.25, 1.75, global_rate)
    expected_u2 = eb_smooth(0.0, 0.25, global_rate)

    result = user_follow_rate_decayed(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result, [expected_u1, expected_u2, global_rate])


def test_video_long_view_rate_decayed_fills_the_video_level_gap(
    cross_frame_train_df, cross_frame_target_df
):
    """No video-level decayed+smoothed long_view feature existed before B11
    (video_ctr, B4, is raw/undecayed only)."""
    global_rate = 0.5
    expected_v1 = eb_smooth(0.25, 1.0, global_rate)  # same shape as click, long_view mirrors it
    expected_v2 = eb_smooth(1.0, 1.0, global_rate)

    result = video_long_view_rate_decayed(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result, [expected_v1, expected_v2, global_rate])


def test_user_long_view_rate_decayed_matches_user_ctr_decayed_on_cross_frame_path(
    cross_frame_train_df, cross_frame_target_df
):
    """Documents the naming-symmetry decision: numerically identical to the
    pre-existing B5 feature on the cross-frame path, but (unlike it) also
    supports the in-sample path -- see the two tests below."""
    expected = user_ctr_decayed(cross_frame_train_df, cross_frame_target_df)
    result = user_long_view_rate_decayed(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result, expected)


def test_user_ctr_decayed_has_no_in_sample_support_unlike_its_b11_counterpart():
    """Regression-documentation: this is *why* user_long_view_rate_decayed
    is not a plain delegate to user_ctr_decayed."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "date": [20220407, 20220408],
            "time_ms": [100, 200],
            "long_view": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="strictly earlier"):
        user_ctr_decayed(frame, frame)

    # The B11 feature, built on the shared in-sample-aware helper, handles it.
    result = user_long_view_rate_decayed(frame, frame)
    assert result.shape == (2,)
    assert np.isfinite(result).all()


# --- Leakage guard coverage (one per signal, both granularities) -----------


@pytest.fixture
def leakage_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "video_id": ["v1", "v2", "v1", "v3"],
            "date": [20220409, 20220410, 20220411, 20220412],
            "is_click": [1, 0, 1, 0],
            "is_like": [0, 1, 0, 1],
            "is_follow": [1, 0, 0, 1],
            "long_view": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def leakage_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "video_id": ["v1", "v2", "v4"],
            "date": [20220422, 20220423, 20220424],
            "long_view": [0, 1, 0],
            "is_click": [1, 0, 1],
        }
    )


@pytest.mark.parametrize("name", sorted(AUX_FEATURE_NAMES))
def test_every_aux_rate_feature_passes_the_leakage_guard_cross_frame(
    name, leakage_train_df, leakage_target_df
):
    fn = AUX_FEATURE_FUNCS[name]
    assert leakage_check(fn, leakage_train_df, leakage_target_df) is True


@pytest.mark.parametrize("name", sorted(AUX_FEATURE_NAMES))
def test_every_aux_rate_feature_passes_the_leakage_guard_in_sample(name):
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "video_id": ["v1", "v2", "v1"],
            "date": [20220409, 20220409, 20220409],
            "time_ms": [100, 200, 300],
            "is_click": [1, 0, 1],
            "is_like": [0, 1, 0],
            "is_follow": [1, 0, 0],
            "long_view": [1, 0, 1],
        }
    )
    fn = AUX_FEATURE_FUNCS[name]
    assert leakage_check(fn, frame, frame) is True


def test_aux_rate_features_ignore_target_side_outcomes(leakage_train_df, leakage_target_df):
    """Editing target_df's own outcome columns must never move the output --
    these are same-row FORBIDDEN_SAME_ROW / LABEL columns."""
    changed_target = leakage_target_df.copy()
    changed_target["long_view"] = 1 - changed_target["long_view"]
    changed_target["is_click"] = 1 - changed_target["is_click"]

    for name, fn in AUX_FEATURE_FUNCS.items():
        original = fn(leakage_train_df, leakage_target_df)
        changed = fn(leakage_train_df, changed_target)
        np.testing.assert_array_equal(original, changed, err_msg=name)


# --- Acceptance property 1: future-dated row edits never move an earlier
#     row's value ----------------------------------------------------------


@pytest.fixture
def in_sample_varying_dates_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "video_id": ["v1", "v1", "v1", "v1"],
            "date": [20220408, 20220409, 20220410, 20220411],
            "time_ms": [1, 2, 3, 4],
            "is_click": [1, 0, 1, 0],
            "is_like": [0, 1, 0, 1],
            "is_follow": [1, 0, 1, 0],
            "long_view": [1, 0, 1, 0],
        }
    )


@pytest.mark.parametrize("name", sorted(AUX_FEATURE_NAMES))
def test_editing_a_future_dated_row_never_changes_an_earlier_rows_value(
    name, in_sample_varying_dates_frame
):
    """Row 2 (date 20220410) is edited. Rows 0 and 1 -- strictly earlier --
    must be unaffected. Row 3 (strictly later than row 2) is allowed, and
    expected, to change: this proves the earlier-row invariance is a real
    property of the implementation, not a vacuous "nothing ever changes"."""
    fn = AUX_FEATURE_FUNCS[name]
    frame = in_sample_varying_dates_frame

    baseline = fn(frame, frame)

    mutated = frame.copy()
    for column in ("is_click", "is_like", "is_follow", "long_view"):
        mutated.loc[2, column] = 1 - mutated.loc[2, column]
    edited = fn(mutated, mutated)

    np.testing.assert_array_equal(baseline[:2], edited[:2], err_msg=name)
    assert baseline[3] != edited[3], f"{name}: test is vacuous, row 3 never moved"


def test_user_click_rate_decayed_in_sample_full_hand_computation(
    in_sample_varying_dates_frame,
):
    """Full hand-computed check of the in-sample path (weights, per-row
    prior global rate, and EB smoothing) for one representative feature."""
    frame = in_sample_varying_dates_frame
    # cutoff = max date = 20220411; half_life = 7 -> weights by age (days):
    # row0 age3 -> 0.5**(3/7); row1 age2 -> 0.5**(2/7);
    # row2 age1 -> 0.5**(1/7); row3 age0 -> 1.0
    # (all rows are the same user, so decay weight matters only insofar as
    # it scales each prior contribution -- verified against decay_weights
    # directly rather than re-deriving the exponents by hand.)
    dates = feature_module._parse_dates(frame["date"])
    weights = decay_weights(dates, dates.max())

    # click sequence is [1, 0, 1, 0]; prior global click rate (strictly
    # before each row's own time bucket, via _prior_global_rates):
    # row0: no prior rows -> INITIAL_RATE fallback (0.5)
    # row1: prior = {row0: click=1} -> rate 1/1 = 1.0
    # row2: prior = {row0, row1} -> rate (1+0)/2 = 0.5
    # row3: prior = {row0, row1, row2} -> rate (1+0+1)/3 = 2/3
    global_rates = [0.5, 1.0, 0.5, 2 / 3]

    alpha = 20.0
    expected = [global_rates[0]]  # row0: no prior group history at all
    # row1: prior group (user u1) history = {row0: click=1, weight=w0}
    w0, w1, w2, _w3 = weights
    prior_label = w0 * 1
    prior_weight = w0
    expected.append((prior_label + alpha * global_rates[1]) / (prior_weight + alpha))
    # row2: prior group history = {row0: click=1, row1: click=0}
    prior_label = w0 * 1 + w1 * 0
    prior_weight = w0 + w1
    expected.append((prior_label + alpha * global_rates[2]) / (prior_weight + alpha))
    # row3: prior group history = {row0: click=1, row1: click=0, row2: click=1}
    prior_label = w0 * 1 + w1 * 0 + w2 * 1
    prior_weight = w0 + w1 + w2
    expected.append((prior_label + alpha * global_rates[3]) / (prior_weight + alpha))

    result = user_click_rate_decayed(frame, frame)

    np.testing.assert_allclose(result, expected)


# --- Acceptance property 2: same-day (same time_ms) rows are excluded ------


def test_prior_group_decayed_stats_excludes_same_timestamp_rows():
    """Mirrors _event_times' docstring: rows sharing a timestamp see the
    same prior state and never one another's label -- tested directly
    against the in-sample helper, as the acceptance criteria for B11
    suggest."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "date": [20220408, 20220408, 20220408],
            "time_ms": [100, 200, 200],
            "is_click": [1, 1, 0],
        }
    )
    labels = feature_module._numeric_outcome(frame, "is_click")
    times = feature_module._event_times(frame)
    dates = feature_module._parse_dates(frame["date"])
    weights = decay_weights(dates, dates.max())

    prior_label, prior_weight = feature_module._prior_group_decayed_stats(
        frame, labels, times, "user_id", weights
    )

    # Row 0 (t=100) has no history at all.
    assert prior_label[0] == 0.0
    assert prior_weight[0] == 0.0
    # Rows 1 and 2 share t=200: both see only row 0 as prior state (sum=1,
    # weight=1, since all rows share the same date -> weight 1.0 each), and
    # -- crucially -- neither sees the other's is_click value even though
    # row1 = 1 and row2 = 0 differ.
    np.testing.assert_allclose(prior_label[[1, 2]], [1.0, 1.0])
    np.testing.assert_allclose(prior_weight[[1, 2]], [1.0, 1.0])


def test_same_day_rows_produce_identical_feature_output_regardless_of_each_others_label():
    """End-to-end version of the property above, through a registered
    feature: two same-timestamp rows for the same user get the identical
    smoothed rate, independent of what each other's own outcome was."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "video_id": ["v1", "v1", "v1"],
            "date": [20220408, 20220408, 20220408],
            "time_ms": [100, 200, 200],
            "is_click": [1, 1, 0],
            "is_like": [0, 0, 1],
            "is_follow": [0, 1, 1],
            "long_view": [1, 1, 0],
        }
    )

    for fn in AUX_FEATURE_FUNCS.values():
        result = fn(frame, frame)
        assert result[1] == pytest.approx(result[2])
