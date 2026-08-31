"""B14 acceptance: `sim_to_history` -- a per-row similarity between a
candidate video's tags and a user's own recent, positively-engaged (long_view
== 1) watch history, decayed by recency (B5) and normalised to a per-user
attention-share distribution over tags.

Covers: exact registration name, a hand-computed cross-frame fixture, the
recency/decay property, non-redundancy against `user_tag_affinity`, edge
cases (zero positive history, tagless target row), the leakage guard on
both paths, a realistic multi-user dual-path smoke run, in-sample
strictly-prior correctness (future-dated edits never move an earlier row,
same-timestamp rows are excluded from each other), and cross-frame /
in-sample numeric agreement on a fixture built so the two paths are
directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    FEATURES,
    leakage_check,
    sim_to_history,
    user_tag_affinity,
)


def test_registered_under_the_exact_name():
    """A teammate's downstream C10 work references this feature by the
    literal string 'sim_to_history' -- the name must match precisely."""
    assert "sim_to_history" in FEATURES
    assert FEATURES["sim_to_history"] is sim_to_history


# --- Hand-computed cross-frame fixture --------------------------------------


@pytest.fixture
def cross_frame_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "video_id": ["v1", "v2", "v3", "v4", "v5"],
            "tag": ["1,2", "1", "3", "2", "2"],
            "date": [20220408, 20220414, 20220421, 20220408, 20220421],
            "long_view": [1, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def cross_frame_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "video_id": ["t1", "t2", "t3", "t4"],
            "tag": ["1", "3", "2", ""],
            "date": [20220422, 20220422, 20220422, 20220422],
        }
    )


def test_hand_computed_high_similarity_for_shared_tags_and_zero_for_disjoint(
    cross_frame_train_df, cross_frame_target_df
):
    """u1's positive history (rows v1: tags {1,2}, v2: tag {1}; v3 is
    excluded, long_view=0) concentrates decayed weight on tag 1. A target
    sharing tag 1 gets a high share; a target on tag 3 (never positively
    watched by u1) gets exactly 0. u2's entire positive history is tag 2,
    so a target on tag 2 gets the maximum share of 1.0.

    cutoff = train max date = 20220421 (half_life=7, the decay_weights
    default): v1 age=13 -> w=0.5**(13/7); v2 age=7 -> w=0.5**(7/7)=0.5;
    v4 (u2) age=13 -> w=0.5**(13/7); v5 (u2) age=0 -> w=1.0.
    """
    w1 = 0.5 ** (13 / 7)  # v1 (u1, tags {1,2})
    w2 = 0.5 ** (7 / 7)  # v2 (u1, tag {1})
    u1_tag1 = w1 + w2
    u1_tag2 = w1
    u1_total = u1_tag1 + u1_tag2
    expected_u1_tag1_share = u1_tag1 / u1_total

    result = sim_to_history(cross_frame_train_df, cross_frame_target_df)

    np.testing.assert_allclose(result[0], expected_u1_tag1_share)  # u1, tag 1
    assert result[1] == 0.0  # u1, tag 3: no positive history on tag 3
    assert result[2] == pytest.approx(1.0)  # u2, tag 2: entire positive history
    assert result[3] == 0.0  # u3, no tags at all


# --- Recency / decay property ------------------------------------------------


def test_decay_makes_the_more_recent_tag_get_the_higher_normalized_share():
    """Two users, each with exactly one positive row on tag X and one on
    tag Y (equal RAW counts per tag), but with the recent/old role of X and
    Y swapped between the two users. Without decay, both tags would tie at
    a 0.5/0.5 share for every user (equal counts). With decay applied, the
    more-recent tag must win for whichever user has it recent -- proving
    decay, not just tag identity or row order, drives the normalized
    share. Cutoff = train max date = 20220421 (0 days old); the 'old' row
    sits 21 days back, well past one half-life (7 days default), so the
    effect is large and not a rounding artifact.
    """
    train_df = pd.DataFrame(
        {
            "user_id": ["u_a", "u_a", "u_b", "u_b"],
            "video_id": ["a_old", "a_recent", "b_old", "b_recent"],
            "tag": ["X", "Y", "Y", "X"],
            "date": [20220331, 20220421, 20220331, 20220421],
            "long_view": [1, 1, 1, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "user_id": ["u_a", "u_a", "u_b", "u_b"],
            "video_id": ["qx", "qy", "qx", "qy"],
            "tag": ["X", "Y", "X", "Y"],
            "date": [20220422, 20220422, 20220422, 20220422],
        }
    )

    result = sim_to_history(train_df, target_df)
    a_share_x, a_share_y, b_share_x, b_share_y = result

    # Without decay both would be exactly 0.5 (equal raw counts per tag).
    assert a_share_x < 0.5 < a_share_y  # u_a: X is old, Y is recent
    assert b_share_x > 0.5 > b_share_y  # u_b: X is recent, Y is old (flipped)
    # The flip is symmetric: same magnitude of effect either way round.
    np.testing.assert_allclose(a_share_y, b_share_x)
    np.testing.assert_allclose(a_share_x, b_share_y)


# --- Non-redundancy vs user_tag_affinity ------------------------------------


def test_diverges_from_user_tag_affinity_on_old_single_row_vs_recent_volume():
    """user_tag_affinity is an ENGAGEMENT RATE: one old positive row on a
    tag gives it a perfect 1.0 rate forever, no matter how stale. A burst
    of recent positive volume on a different tag doesn't touch that rate at
    all. sim_to_history is an ATTENTION SHARE within positive, decayed
    history: the same old lone row gets crowded out by the recent volume on
    the other tag. On the *same* target row (tag 'rare'), the two features
    must therefore disagree sharply -- direct proof this isn't a disguised
    duplicate."""
    old_row = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["old_v"],
            "tag": ["rare"],
            "date": [20220308],  # 44 days before cutoff
            "long_view": [1],
        }
    )
    recent_rows = pd.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "video_id": [f"r{i}" for i in range(5)],
            "tag": ["common"] * 5,
            "date": [20220421] * 5,
            "long_view": [1] * 5,
        }
    )
    train_df = pd.concat([old_row, recent_rows], ignore_index=True)
    target_df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["q0"],
            "tag": ["rare"],
            "date": [20220422],
        }
    )

    affinity = user_tag_affinity(train_df, target_df)[0]
    similarity = sim_to_history(train_df, target_df)[0]

    assert affinity == pytest.approx(1.0)  # single old row, perfect rate
    assert similarity < 0.05  # 44 days of decay vs. five same-day rivals
    assert abs(affinity - similarity) > 0.9  # clearly different numbers


# --- Edge cases --------------------------------------------------------------


def test_zero_positive_history_returns_zero():
    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "video_id": ["v1", "v2"],
            "tag": ["1", "1"],
            "date": [20220408, 20220414],
            "long_view": [0, 0],  # never positive
        }
    )
    target_df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["t1"],
            "tag": ["1"],
            "date": [20220422],
        }
    )
    result = sim_to_history(train_df, target_df)
    assert result[0] == 0.0


def test_no_tags_on_target_row_returns_zero(cross_frame_train_df):
    target_df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["t1"],
            "tag": [None],
            "date": [20220422],
        }
    )
    result = sim_to_history(cross_frame_train_df, target_df)
    assert result[0] == 0.0


def test_unseen_user_with_no_history_at_all_returns_zero(cross_frame_train_df):
    target_df = pd.DataFrame(
        {
            "user_id": ["brand_new_user"],
            "video_id": ["t1"],
            "tag": ["1"],
            "date": [20220422],
        }
    )
    result = sim_to_history(cross_frame_train_df, target_df)
    assert result[0] == 0.0


# --- Leakage guard -----------------------------------------------------------


def test_leakage_check_passes_cross_frame(cross_frame_train_df, cross_frame_target_df):
    assert leakage_check(sim_to_history, cross_frame_train_df, cross_frame_target_df) is True


def test_leakage_check_passes_in_sample():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "video_id": ["v1", "v2", "v1"],
            "tag": ["1", "2", "1"],
            "date": [20220409, 20220409, 20220409],
            "time_ms": [100, 200, 300],
            "long_view": [1, 0, 1],
        }
    )
    assert leakage_check(sim_to_history, frame, frame) is True


# --- Dual-path smoke run on a realistic multi-user, multi-tag fixture -------


@pytest.fixture
def multi_user_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3", "u3"],
            "video_id": ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
            "tag": ["1,2", "2", "3", "1", "1,3", "2", "2,4", "4", "1"],
            "date": [
                20220408,
                20220410,
                20220414,
                20220409,
                20220415,
                20220408,
                20220412,
                20220417,
                20220420,
            ],
            "time_ms": [100, 200, 300, 150, 350, 120, 260, 400, 450],
            "long_view": [1, 0, 1, 1, 1, 0, 1, 0, 1],
        }
    )


def test_cross_frame_path_runs_on_multi_user_fixture(multi_user_frame):
    train_df = multi_user_frame
    target_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "unseen"],
            "video_id": ["t1", "t2", "t3", "t4"],
            "tag": ["1", "1,3", "2,4", "9"],
            "date": [20220422, 20220422, 20220422, 20220422],
        }
    )
    result = sim_to_history(train_df, target_df)
    assert result.shape == (4,)
    assert np.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()


def test_in_sample_path_runs_on_multi_user_fixture(multi_user_frame):
    result = sim_to_history(multi_user_frame, multi_user_frame)
    assert result.shape == (len(multi_user_frame),)
    assert np.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()


# --- In-sample strictly-prior correctness -----------------------------------


def test_editing_a_future_dated_row_never_changes_an_earlier_rows_value():
    """Row 2 is edited (a later date, a different tag, and its own
    positivity flipped). Rows 0 and 1 -- strictly earlier -- must be
    unaffected. Row 3 -- strictly later, and sharing row 2's original tag --
    is allowed, and expected, to change: proof the earlier-row invariance
    is a real property, not a vacuous 'nothing ever changes'."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "video_id": ["v0", "v1", "v2", "v3"],
            "tag": ["A", "B", "C", "A"],
            "date": [20220408, 20220409, 20220410, 20220411],
            "time_ms": [1, 2, 3, 4],
            "long_view": [1, 0, 1, 0],
        }
    )
    baseline = sim_to_history(frame, frame)

    mutated = frame.copy()
    mutated.loc[2, "tag"] = "Z"
    mutated.loc[2, "long_view"] = 0
    edited = sim_to_history(mutated, mutated)

    np.testing.assert_array_equal(baseline[:2], edited[:2])
    assert baseline[3] != edited[3], "test is vacuous: row 3 never moved"


def test_same_timestamp_rows_are_excluded_from_each_others_history():
    """Rows 1 and 2 share time_ms=200 and the same query tag 'A', but
    differ in their own long_view label. Since same-timestamp rows must
    never see each other's positive signal, both must land on exactly the
    same strictly-prior history (row 0 only) and therefore produce an
    identical output -- independent of what the other row's own label
    was."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "video_id": ["v0", "v1", "v2"],
            "tag": ["A", "A", "A"],
            "date": [20220408, 20220408, 20220408],
            "time_ms": [100, 200, 200],
            "long_view": [1, 1, 0],
        }
    )
    result = sim_to_history(frame, frame)

    assert result[0] == 0.0  # no prior history at all
    assert result[1] == pytest.approx(result[2])
    assert result[1] == pytest.approx(1.0)  # only row 0's tag-A weight exists


def test_in_sample_never_reads_its_own_row_positivity():
    """A user's single row, positive, tag A: with no other history, the
    row must not see its own label -- output must be 0.0, not something
    derived from 'I am myself a 100% match for my own tag'."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["v0"],
            "tag": ["A"],
            "date": [20220408],
            "time_ms": [100],
            "long_view": [1],
        }
    )
    result = sim_to_history(frame, frame)
    assert result[0] == 0.0


# --- Cross-frame / in-sample numeric agreement ------------------------------


def test_cross_frame_and_in_sample_agree_on_a_comparable_fixture():
    """sim_to_history is a per-user RATIO of decayed weights (tag share
    over total share). A frame-wide decay cutoff scales every one of a
    user's rows by the same constant factor, which cancels out of the
    ratio -- so cross-frame (cutoff = train_df's own max date) and
    in-sample (cutoff = the full combined frame's max date) must agree
    exactly on the same underlying history, even though the two paths use
    numerically different cutoffs. This fixture puts the query row last (so
    it never contributes to its own strictly-prior history, matching
    both paths' history exactly) and gives the strongest evidence available
    that both branches implement the same formula."""
    history = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "video_id": ["h0", "h1", "h2"],
            "tag": ["1,2", "2", "1"],
            "date": [20220408, 20220410, 20220414],
            "time_ms": [100, 200, 300],
            "long_view": [1, 0, 1],
        }
    )
    query_row = pd.DataFrame(
        {
            "user_id": ["u1"],
            "video_id": ["q0"],
            "tag": ["1,3"],
            "date": [20220421],
            "time_ms": [400],
            "long_view": [1],
        }
    )
    full = pd.concat([history, query_row], ignore_index=True)

    cross_frame_value = sim_to_history(history, query_row)[0]
    in_sample_value = sim_to_history(full, full)[3]

    np.testing.assert_allclose(cross_frame_value, in_sample_value)
