"""B6 acceptance: a feature reading target_df["click"] is rejected; a
FORBIDDEN_SAME_ROW column of target_df is rejected; EXCLUDED_SOURCES are
rejected; legitimate historical aggregates pass."""

from __future__ import annotations

from pipeline.features import EXCLUDED_SOURCES, FORBIDDEN_SAME_ROW, LABEL
from tests.conftest import todo


def test_label_matches_the_shipped_starter_kit():
    """kuairand-starter-kit/data.py:5 — LABEL = 'long_view'.

    AGENT_PLAN.md says `click`. The shipped code wins (design commitment 1:
    read the label out of the code, never from prose).
    """
    assert LABEL == "long_view"


def test_forbidden_list_covers_post_exposure_signals():
    for col in ["is_click", "play_time_ms", "is_like", "profile_stay_time"]:
        assert col in FORBIDDEN_SAME_ROW


def test_label_is_not_in_its_own_forbidden_list():
    assert LABEL not in FORBIDDEN_SAME_ROW


def test_duration_ms_is_permitted():
    """A video property known before exposure, and one of the five official
    baseline fields (as dur_bucket). Forbidding it would drop a field the
    baseline we must beat already uses."""
    assert "duration_ms" not in FORBIDDEN_SAME_ROW


def test_monthly_statistics_excluded_by_default():
    assert "item_statistics_monthly" in EXCLUDED_SOURCES


@todo("B6")
def test_feature_reading_target_label_is_rejected():
    """def bad(train_df, target_df): return target_df["long_view"].values"""


@todo("B6")
def test_feature_reading_forbidden_same_row_column_is_rejected():
    """def bad(train_df, target_df): return target_df["is_click"].values"""


@todo("B6")
def test_excluded_source_is_rejected():
    """Any feature whose source references item_statistics_monthly."""


@todo("B6")
def test_legitimate_historical_aggregate_passes():
    """user_ctr_decayed fits on train_df and maps onto target_df — must pass."""


@todo("C1")
def test_leak_canary_fires_above_threshold():
    """run_experiment returns error_class="leak_suspected" when primary > 0.75."""
