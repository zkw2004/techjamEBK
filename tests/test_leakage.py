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


def _leak_frames():
    train_df = pd.DataFrame(
        {
            "date": [20220408, 20220408, 20220409, 20220409, 20220410, 20220410],
            "user_id": [1, 2, 1, 2, 1, 2],
            "is_click": [1, 0, 0, 1, 1, 0],
            LABEL: [1, 0, 1, 0, 0, 1],
        }
    )
    target_df = pd.DataFrame(
        {
            "date": [20220422, 20220422, 20220423, 20220423],
            "user_id": [1, 2, 1, 3],
            "is_click": [0, 1, 1, 0],
            LABEL: [1, 0, 0, 1],
        }
    )
    return train_df, target_df


def test_feature_reading_target_label_is_rejected():
    def bad(train_df, target_df):
        return target_df["long_view"].to_numpy(dtype=float)

    train_df, target_df = _leak_frames()
    with pytest.raises(ValueError, match="post-exposure"):
        leakage_check(bad, train_df, target_df)


def test_feature_reading_forbidden_same_row_column_is_rejected():
    def bad(train_df, target_df):
        return target_df["is_click"].to_numpy(dtype=float)

    train_df, target_df = _leak_frames()
    with pytest.raises(ValueError, match="is_click"):
        leakage_check(bad, train_df, target_df)


def test_excluded_source_is_rejected():
    def bad(train_df, target_df):
        source = "item_statistics_monthly"
        del source
        return np.zeros(len(target_df))

    train_df, target_df = _leak_frames()
    with pytest.raises(ValueError, match="item_statistics_monthly"):
        leakage_check(bad, train_df, target_df)


def test_legitimate_historical_aggregate_passes():
    """user_ctr fits on train_df and maps onto target_df — must pass."""
    train_df, target_df = _leak_frames()
    assert leakage_check(user_ctr, train_df, target_df) is True


def test_dynamic_probe_catches_an_obfuscated_target_read():
    """A runtime-assembled column name defeats the static grep; the outcome
    probe (permute target-side outcomes, watch the output) still catches it."""

    def sneaky(train_df, target_df):
        column = "long" + "_view"
        return target_df[column].to_numpy(dtype=float)

    train_df, target_df = _leak_frames()
    with pytest.raises(ValueError, match="permuted"):
        leakage_check(sneaky, train_df, target_df)


def test_unauditable_feature_fails_closed():
    """No file-backed source and no __leak_source__ — refuse to trust it."""
    compiled = eval("lambda train_df, target_df: __import__('numpy').zeros(len(target_df))")
    train_df, target_df = _leak_frames()
    with pytest.raises(ValueError, match="source"):
        leakage_check(compiled, train_df, target_df)


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
