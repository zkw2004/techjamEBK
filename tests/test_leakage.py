"""B6 acceptance: a feature reading target_df["click"] is rejected; a
FORBIDDEN_SAME_ROW column of target_df is rejected; EXCLUDED_SOURCES are
rejected; legitimate historical aggregates pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    EXCLUDED_SOURCES,
    FORBIDDEN_SAME_ROW,
    LABEL,
    leakage_check,
    user_ctr,
    user_ctr_decayed,
)


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


@pytest.fixture
def leakage_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "video_id": ["v1", "v2", "v1", "v3"],
            "tag": ["10", "20", "10", "30"],
            "date": [20220409, 20220410, 20220411, 20220412],
            "hourmin": [900, 1030, 1200, 1530],
            "long_view": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def leakage_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "video_id": ["v1", "v2", "v4"],
            "tag": ["10", "20", "30"],
            "date": [20220422, 20220423, 20220424],
            "hourmin": [900, 1030, 1600],
            "long_view": [0, 1, 0],
            "is_click": [1, 0, 1],
        }
    )


def test_feature_reading_target_label_is_rejected(leakage_train_df, leakage_target_df):
    def bad(train_df, target_df):
        del train_df
        return target_df["long_view"].to_numpy()

    assert leakage_check(bad, leakage_train_df, leakage_target_df) is False


def test_feature_reading_forbidden_same_row_column_is_rejected(
    leakage_train_df, leakage_target_df
):
    def bad(train_df, target_df):
        del train_df
        return target_df["is_click"].to_numpy()

    assert leakage_check(bad, leakage_train_df, leakage_target_df) is False


def test_excluded_source_is_rejected(leakage_train_df, leakage_target_df):
    def bad(train_df, target_df):
        del train_df, target_df
        # A feature reading the monthly aggregate file, whose window may
        # span the test period (Section 11, trap 3). Never executed here —
        # the static source scan must reject it before this line would run.
        return pd.read_parquet("data/item_statistics_monthly.parquet")["rate"].to_numpy()

    assert leakage_check(bad, leakage_train_df, leakage_target_df) is False


def test_legitimate_historical_aggregate_passes(leakage_train_df, leakage_target_df):
    """user_ctr and user_ctr_decayed fit on train_df and map onto target_df —
    neither reads a forbidden or label column off target_df, so both pass."""
    assert leakage_check(user_ctr, leakage_train_df, leakage_target_df) is True
    assert leakage_check(user_ctr_decayed, leakage_train_df, leakage_target_df) is True


def test_probe_only_gate_would_miss_a_direct_target_label_read(
    leakage_train_df, leakage_target_df
):
    """Regression for the flaw in Appendix A.2's pseudocode: a feature that
    reads target_df[LABEL] directly never looks at train_df's label column,
    so its output is identical whether or not train_df's labels are
    shuffled. A probe-only gate would read that as "label-independent,
    therefore safe" and never reach the static source check. The static
    check must run unconditionally, first — which is exactly what makes
    `leakage_check` still reject this."""

    def bad(train_df, target_df):
        del train_df
        return target_df["long_view"].to_numpy()

    shuffled_train = leakage_train_df.assign(
        long_view=leakage_train_df["long_view"].sample(frac=1, random_state=0).to_numpy()
    )
    assert np.array_equal(
        bad(leakage_train_df, leakage_target_df),
        bad(shuffled_train, leakage_target_df),
    )
    assert leakage_check(bad, leakage_train_df, leakage_target_df) is False


def _suspicious_tier(config, fidelity, seed):
    return {
        "status": "ok",
        "fidelity": fidelity,
        "gauc": 0.80,
        "ndcg": 0.76,
        "primary": 0.78,
        "fold_primaries": [0.77, 0.78, 0.79],
        "segments": {},
        "val_scores": np.array([0.1, 0.9]),
        "val_user_ids": np.array([1, 1]),
        "test_scores": np.array([0.2, 0.8]),
        "gpu_seconds": 0.0,
        "peak_rss_mb": 1.0,
    }


def test_leak_canary_fires_above_threshold(monkeypatch):
    """Removing the canary would promote an implausibly strong result."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _suspicious_tier, raising=False)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["stage"] == "leakage"
    assert result["error_class"] == "leak_suspected"
    assert result["primary"] == 0.78
