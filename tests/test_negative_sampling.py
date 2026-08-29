"""B7 acceptance: three negative-sampling strategies (all, in_session,
pop_weighted), selectable via Config.negative_sampling (Section 8.6),
implementing pipeline.data.sample_negatives per Section 6.5."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.data import (
    NEGATIVE_SAMPLING_STRATEGIES,
    sample_negatives,
)


def test_strategies_match_the_frozen_config_literal():
    """agent/schema.py pins Literal["all", "in_session", "pop_weighted"]."""
    assert NEGATIVE_SAMPLING_STRATEGIES == ("all", "in_session", "pop_weighted")


@pytest.fixture
def session_train_df() -> pd.DataFrame:
    """Two users with one positive each and mixed-session negatives, plus a
    third user with no positives at all."""
    rows = [
        # user, video, date, tab, long_view
        ("u1", "v1", 1, "A", 1),  # positive, session (u1,1,A)
        ("u1", "v2", 1, "A", 0),  # eligible: same session as u1's positive
        ("u1", "v3", 1, "A", 0),  # eligible: same session as u1's positive
        ("u1", "v4", 1, "B", 0),  # ineligible: different tab, no positive there
        ("u1", "v5", 2, "A", 0),  # ineligible: different date, no positive there
        ("u2", "v1", 1, "A", 1),  # positive, session (u2,1,A)
        ("u2", "v1", 1, "A", 0),  # eligible, duplicate (user, video) pair (trap 4)
        ("u2", "v6", 1, "A", 0),  # eligible
        ("u2", "v2", 1, "B", 0),  # ineligible: different tab
        ("u3", "v1", 1, "A", 0),  # ineligible: u3 has zero positives
        ("u3", "v7", 1, "A", 0),  # ineligible: u3 has zero positives
        ("u2", "v1", 1, "A", 0),  # eligible, second duplicate pair
    ]
    return pd.DataFrame(rows, columns=["user_id", "video_id", "date", "tab", "long_view"])


def test_unrecognised_strategy_is_rejected(session_train_df):
    with pytest.raises(ValueError, match="unknown negative_sampling strategy"):
        sample_negatives(session_train_df, strategy="bogus")


@pytest.mark.parametrize("bad_value", [0, -1, -0.5])
def test_non_positive_negatives_per_positive_is_rejected(session_train_df, bad_value):
    with pytest.raises(ValueError, match="negatives_per_positive"):
        sample_negatives(session_train_df, strategy="in_session", negatives_per_positive=bad_value)


def test_empty_frame_is_rejected():
    empty = pd.DataFrame(columns=["user_id", "video_id", "date", "tab", "long_view"])
    with pytest.raises(ValueError, match="empty"):
        sample_negatives(empty, strategy="all")


def test_missing_label_column_is_rejected():
    frame = pd.DataFrame({"user_id": ["u1"], "video_id": ["v1"]})
    with pytest.raises(ValueError, match="long_view"):
        sample_negatives(frame, strategy="all")


def test_missing_required_column_is_rejected(session_train_df):
    without_tab = session_train_df.drop(columns=["tab"])
    with pytest.raises(ValueError, match="'tab'"):
        sample_negatives(without_tab, strategy="in_session")

    without_video = session_train_df.drop(columns=["video_id"])
    with pytest.raises(ValueError, match="'video_id'"):
        sample_negatives(without_video, strategy="pop_weighted")


def test_all_strategy_returns_every_row_unchanged(session_train_df):
    result = sample_negatives(session_train_df, strategy="all")

    pd.testing.assert_frame_equal(result, session_train_df)
    assert result is not session_train_df  # never mutates / aliases the input


def test_in_session_always_keeps_every_positive(session_train_df):
    result = sample_negatives(session_train_df, strategy="in_session", seed=0)

    positive_index = session_train_df.index[session_train_df["long_view"] == 1]
    assert set(positive_index).issubset(set(result.index))


def test_in_session_excludes_negatives_outside_any_positive_session(session_train_df):
    """Rows 3, 4, 8, 9, 10 must never appear: wrong tab, wrong date, or a
    user (u3) with no positive at all."""
    result = sample_negatives(
        session_train_df, strategy="in_session", negatives_per_positive=10, seed=0
    )

    assert set(result.index) == {0, 1, 2, 5, 6, 7, 11}


def test_in_session_caps_negatives_per_positive_per_user(session_train_df):
    result = sample_negatives(
        session_train_df, strategy="in_session", negatives_per_positive=1, seed=0
    )

    negative_rows = result.loc[result["long_view"] == 0]
    assert (negative_rows["user_id"] == "u1").sum() == 1
    assert (negative_rows["user_id"] == "u2").sum() == 1
    assert set(negative_rows.index).issubset({1, 2, 6, 7, 11})


def test_in_session_preserves_original_row_order(session_train_df):
    result = sample_negatives(
        session_train_df, strategy="in_session", negatives_per_positive=10, seed=0
    )

    assert list(result.index) == sorted(result.index)


def test_in_session_is_deterministic_given_seed(session_train_df):
    kwargs = {"strategy": "in_session", "negatives_per_positive": 1, "seed": 7}
    first = sample_negatives(session_train_df, **kwargs)
    second = sample_negatives(session_train_df, **kwargs)

    pd.testing.assert_frame_equal(first, second)


@pytest.fixture
def popularity_train_df() -> pd.DataFrame:
    """One positive on a video that also appears three more times as a
    negative (impression count 4); four other negatives sit on distinct
    videos seen only once each (impression count 1)."""
    rows = [
        ("vPOP", 1),
        ("vPOP", 0),
        ("vPOP", 0),
        ("vPOP", 0),
        ("vRare1", 0),
        ("vRare2", 0),
        ("vRare3", 0),
        ("vRare4", 0),
    ]
    return pd.DataFrame(rows, columns=["video_id", "long_view"])


def test_pop_weighted_always_keeps_every_positive(popularity_train_df):
    result = sample_negatives(popularity_train_df, strategy="pop_weighted", seed=0)
    assert 0 in result.index  # the sole positive row


def test_pop_weighted_caps_the_budget_and_never_duplicates(popularity_train_df):
    result = sample_negatives(
        popularity_train_df, strategy="pop_weighted", negatives_per_positive=1, seed=0
    )

    assert (result["long_view"] == 0).sum() == 1  # budget = 1 * 1 positive
    assert result.index.is_unique
    assert set(result.index).issubset(set(popularity_train_df.index))


def test_pop_weighted_exceeding_the_pool_returns_everything(popularity_train_df):
    result = sample_negatives(
        popularity_train_df, strategy="pop_weighted", negatives_per_positive=100, seed=0
    )

    pd.testing.assert_frame_equal(result, popularity_train_df)


def test_pop_weighted_favours_the_more_popular_video():
    """Weighted by training-set impression count (Section 6.5): across many
    seeds, the single sampled negative should land on the popular video's
    row far more than chance (1/7) would predict."""
    trials = 400
    fixture = pd.DataFrame(
        [
            ("vPOP", 1),
            ("vPOP", 0),
            ("vPOP", 0),
            ("vPOP", 0),
            ("vRare1", 0),
            ("vRare2", 0),
            ("vRare3", 0),
            ("vRare4", 0),
        ],
        columns=["video_id", "long_view"],
    )

    popular_hits = 0
    for seed in range(trials):
        result = sample_negatives(
            fixture, strategy="pop_weighted", negatives_per_positive=1, seed=seed
        )
        selected_negative = result.loc[result["long_view"] == 0].iloc[0]
        if selected_negative["video_id"] == "vPOP":
            popular_hits += 1

    # Expected ~0.75 under the pinned weighting; chance alone would be ~0.43
    # (3 of 7 negative rows). A wide margin above chance keeps this stable.
    assert popular_hits / trials > 0.5


def test_pop_weighted_is_deterministic_given_seed(popularity_train_df):
    kwargs = {"strategy": "pop_weighted", "negatives_per_positive": 1, "seed": 3}
    first = sample_negatives(popularity_train_df, **kwargs)
    second = sample_negatives(popularity_train_df, **kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_pop_weighted_preserves_original_row_order(popularity_train_df):
    result = sample_negatives(
        popularity_train_df, strategy="pop_weighted", negatives_per_positive=2, seed=1
    )

    assert list(result.index) == sorted(result.index)


def test_no_positives_in_frame_yields_no_negatives_under_weighted_strategies():
    all_negative = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "video_id": ["v1", "v2"],
            "date": [1, 1],
            "tab": ["A", "A"],
            "long_view": [0, 0],
        }
    )

    in_session = sample_negatives(all_negative, strategy="in_session", seed=0)
    pop_weighted = sample_negatives(all_negative, strategy="pop_weighted", seed=0)

    assert in_session.empty
    assert pop_weighted.empty
