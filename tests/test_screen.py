"""B12 acceptance: within-user variance screen.

A pure user-level feature (e.g. ``user_activity``) is flagged
``metric_inert: true``; a video-varying feature (e.g. ``video_ctr``) is not.
Exercises the screen against small synthetic frames — no real dataset
archive required (see tools/screen.py's module docstring, design decision
5) — plus the frozen-registry features that are actually on ``main``
(``user_activity``, ``video_ctr``) for an end-to-end check of the literal
acceptance criterion.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import features as feature_module
from tools.screen import (
    DEFAULT_THRESHOLD,
    build_report,
    main,
    per_user_variance,
    render_report,
    screen_feature,
    screen_features,
    write_report,
)

# --- per_user_variance: the core statistic -----------------------------------


def test_per_user_variance_is_zero_for_a_within_user_constant_feature():
    values = np.array([5.0, 5.0, 5.0, 9.0, 9.0])
    user_ids = np.array(["u1", "u1", "u1", "u2", "u2"])

    mean_variance, n_considered, n_total = per_user_variance(values, user_ids)

    assert mean_variance == pytest.approx(0.0)
    assert n_considered == 2
    assert n_total == 2


def test_per_user_variance_is_positive_for_a_within_user_varying_feature():
    # u1 sees [0, 10] (variance 25 at ddof=0), u2 sees [0, 0] (variance 0).
    values = np.array([0.0, 10.0, 0.0, 0.0])
    user_ids = np.array(["u1", "u1", "u2", "u2"])

    mean_variance, n_considered, n_total = per_user_variance(values, user_ids)

    assert mean_variance == pytest.approx((25.0 + 0.0) / 2)
    assert n_considered == 2
    assert n_total == 2


def test_per_user_variance_uses_ddof_zero_by_default():
    # A single user with values [1, 2, 3]: ddof=0 population variance is
    # 2/3; ddof=1 sample variance is 1.0. Confirms the pinned default and
    # that ddof is threaded through correctly when overridden.
    values = np.array([1.0, 2.0, 3.0])
    user_ids = np.array(["u1", "u1", "u1"])

    default_variance, _, _ = per_user_variance(values, user_ids)
    sample_variance, _, _ = per_user_variance(values, user_ids, ddof=1)

    assert default_variance == pytest.approx(2.0 / 3.0)
    assert sample_variance == pytest.approx(1.0)


def test_single_impression_users_are_excluded_from_the_mean_not_zeroed_in():
    # u1 has one row (undefined/degenerate, excluded); u2 and u3 have two
    # rows each with nonzero variance. If the single-row user were wrongly
    # folded in as a variance-0 contributor, the mean would be pulled down
    # and n_users_considered would equal n_users_total (3), not 2.
    values = np.array([42.0, 0.0, 10.0, 0.0, 10.0])
    user_ids = np.array(["u1", "u2", "u2", "u3", "u3"])

    mean_variance, n_considered, n_total = per_user_variance(values, user_ids)

    assert n_total == 3
    assert n_considered == 2
    assert mean_variance == pytest.approx(25.0)  # mean of u2's and u3's variance (25 each)


def test_per_user_variance_returns_zero_and_zero_considered_when_all_users_single_row():
    values = np.array([1.0, 2.0, 3.0])
    user_ids = np.array(["u1", "u2", "u3"])

    mean_variance, n_considered, n_total = per_user_variance(values, user_ids)

    assert mean_variance == pytest.approx(0.0)
    assert n_considered == 0
    assert n_total == 3


def test_per_user_variance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matching length"):
        per_user_variance(np.array([1.0, 2.0]), np.array(["u1"]))


def test_per_user_variance_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        per_user_variance(np.array([1.0, np.nan]), np.array(["u1", "u1"]))


# --- screen_feature: per-feature orchestration, including failure handling --


def _val_df(user_ids, extra=None):
    frame = pd.DataFrame({"user_id": user_ids})
    if extra:
        for key, values in extra.items():
            frame[key] = values
    return frame


def test_screen_feature_flags_a_within_user_constant_feature_as_inert():
    val_df = _val_df(["u1", "u1", "u2", "u2"])

    def constant_per_user(train_df, target_df):
        return target_df["user_id"].map({"u1": 1.0, "u2": 2.0}).to_numpy()

    result = screen_feature("synthetic_user_level", constant_per_user, None, val_df)

    assert result["status"] == "ok"
    assert result["metric_inert"] is True
    assert result["mean_within_user_variance"] == pytest.approx(0.0)
    assert result["n_users_considered"] == 2
    assert result["n_users_total"] == 2
    assert result["error"] is None


def test_screen_feature_does_not_flag_a_within_user_varying_feature():
    val_df = _val_df(["u1", "u1", "u2", "u2"])

    def varying_within_user(train_df, target_df):
        return np.array([0.0, 1.0, 0.0, 1.0])

    result = screen_feature("synthetic_item_level", varying_within_user, None, val_df)

    assert result["status"] == "ok"
    assert result["metric_inert"] is False
    assert result["mean_within_user_variance"] > DEFAULT_THRESHOLD


def test_screen_feature_does_not_flag_video_completion_history_as_inert():
    """B10-fix is candidate-varying by construction, not a user scalar."""
    from pipeline.features import video_completion_ratio_hist

    train_df = pd.DataFrame(
        {
            "video_id": ["good", "good", "poor", "poor"],
            "date": [20220409, 20220410, 20220411, 20220412],
            "duration_ms": [10_000, 10_000, 10_000, 10_000],
            "play_time_ms": [9_000, 8_000, 1_000, 2_000],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "video_id": ["good", "poor"],
            "date": [20220422, 20220422],
        }
    )

    result = screen_feature(
        "video_completion_ratio_hist", video_completion_ratio_hist, train_df, val_df
    )

    assert result["status"] == "ok"
    assert result["metric_inert"] is False
    assert result["mean_within_user_variance"] > DEFAULT_THRESHOLD


def test_screen_feature_respects_a_custom_threshold():
    val_df = _val_df(["u1", "u1"])

    def tiny_variation(train_df, target_df):
        return np.array([1.0, 1.0 + 1e-6])

    # Variance is (1e-6)^2 / 4 ~= 2.5e-13, below the default threshold...
    default_result = screen_feature("tiny", tiny_variation, None, val_df)
    assert default_result["metric_inert"] is True

    # ...but not below a much smaller custom threshold.
    tight_result = screen_feature("tiny", tiny_variation, None, val_df, threshold=1e-20)
    assert tight_result["metric_inert"] is False


def test_screen_feature_catches_an_exception_without_raising():
    val_df = _val_df(["u1", "u2"])

    def broken(train_df, target_df):
        raise KeyError("video_duration")

    result = screen_feature("broken_feature", broken, None, val_df)

    assert result["status"] == "error"
    assert result["metric_inert"] is None
    assert result["mean_within_user_variance"] is None
    assert "video_duration" in result["error"]


def test_screen_feature_catches_a_wrong_length_return_as_an_error():
    val_df = _val_df(["u1", "u2", "u3"])

    def wrong_length(train_df, target_df):
        return np.array([1.0, 2.0])  # len 2, but val_df has 3 rows

    result = screen_feature("mismatched", wrong_length, None, val_df)

    assert result["status"] == "error"
    assert "shape" in result["error"] or "len" in result["error"]


def test_screen_feature_reports_a_missing_user_id_column_as_an_error():
    val_df = pd.DataFrame({"not_user_id": [1, 2, 3]})

    result = screen_feature("whatever", lambda t, v: np.zeros(3), None, val_df)

    assert result["status"] == "error"
    assert "user_id" in result["error"]


# --- screen_features: registry-agnostic iteration ----------------------------


def test_screen_features_is_registry_agnostic_and_order_preserving():
    """Works against any {name: fn} mapping, not a hardcoded feature list —
    this is what lets B10/B11's features be picked up automatically once
    merged, with no changes to screen.py."""
    val_df = _val_df(["u1", "u1", "u2", "u2"])

    features = {
        "z_constant": lambda t, v: v["user_id"].map({"u1": 1.0, "u2": 2.0}).to_numpy(),
        "a_varying": lambda t, v: np.array([0.0, 5.0, 0.0, 5.0]),
        "m_broken": lambda t, v: (_ for _ in ()).throw(RuntimeError("boom")),
    }

    results = screen_features(features, None, val_df)

    assert list(results) == list(features)  # order preserved
    assert results["z_constant"]["metric_inert"] is True
    assert results["a_varying"]["metric_inert"] is False
    assert results["m_broken"]["status"] == "error"
    # One bad feature must not have prevented the others from being screened.
    assert results["z_constant"]["status"] == "ok"
    assert results["a_varying"]["status"] == "ok"


def test_screen_features_works_on_an_empty_registry():
    assert screen_features({}, None, _val_df(["u1"])) == {}


# --- build_report / render_report / write_report: schema and I/O split ------


def test_build_report_has_the_documented_top_level_schema():
    results = {"f1": {"status": "ok", "mean_within_user_variance": 0.0, "metric_inert": True,
                       "n_users_considered": 1, "n_users_total": 1, "error": None}}

    report = build_report(results, threshold=1e-9, ddof=0, split="official_validation")

    assert report["schema_version"] == "1"
    assert report["threshold"] == 1e-9
    assert report["ddof"] == 0
    assert report["split"] == "official_validation"
    assert isinstance(report["timestamp"], str) and report["timestamp"]
    assert report["features"] == results


def test_build_report_is_json_serialisable():
    results = {"f1": {"status": "ok", "mean_within_user_variance": 0.1234, "metric_inert": False,
                       "n_users_considered": 5, "n_users_total": 5, "error": None},
               "f2": {"status": "error", "mean_within_user_variance": None, "metric_inert": None,
                      "n_users_considered": None, "n_users_total": None, "error": "boom"}}

    report = build_report(results)
    round_tripped = json.loads(json.dumps(report))

    assert round_tripped == report


def test_write_report_round_trips_through_a_tmp_path(tmp_path):
    report = build_report(
        {"f1": {"status": "ok", "mean_within_user_variance": 0.0, "metric_inert": True,
                "n_users_considered": 2, "n_users_total": 2, "error": None}}
    )
    out_path = tmp_path / "nested" / "screen_report.json"

    written = write_report(report, out_path)

    assert written == out_path
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == report


def test_render_report_names_inert_and_errored_features():
    report = build_report(
        {
            "inert_one": {"status": "ok", "mean_within_user_variance": 0.0, "metric_inert": True,
                          "n_users_considered": 3, "n_users_total": 3, "error": None},
            "varying_one": {"status": "ok", "mean_within_user_variance": 0.5,
                            "metric_inert": False, "n_users_considered": 3,
                            "n_users_total": 3, "error": None},
            "broken_one": {"status": "error", "mean_within_user_variance": None,
                           "metric_inert": None, "n_users_considered": None,
                           "n_users_total": None, "error": "RuntimeError: boom"},
        }
    )

    rendered = render_report(report)

    assert "inert_one" in rendered
    assert "varying_one" in rendered
    assert "broken_one" in rendered
    assert "metric_inert (1): inert_one" in rendered
    assert "errored (1): broken_one" in rendered


# --- The literal acceptance criterion, against the real registered features -


@pytest.fixture
def real_registry_train_and_val():
    """Small synthetic frames shaped like the real pipeline's, exercising
    the actually-registered ``user_activity`` and ``video_ctr`` (both on
    ``main`` since B4) end-to-end through the real feature functions."""
    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
            "video_id": ["vA", "vA", "vB", "vA", "vB", "vA", "vB"],
            "long_view": [1, 1, 0, 0, 1, 1, 0],
            "date": [20220408] * 7,
        }
    )
    # vA long_view rate: (1+1+0+1)/4 = 0.75; vB: (0+1+0)/3 = 1/3.
    val_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "video_id": ["vA", "vB", "vB", "vA", "vA", "vB"],
            "date": [20220422] * 6,
        }
    )
    return train_df, val_df


def test_user_activity_is_flagged_metric_inert_on_the_real_registry(
    real_registry_train_and_val,
):
    train_df, val_df = real_registry_train_and_val

    result = screen_feature(
        "user_activity", feature_module.get("user_activity"), train_df, val_df
    )

    assert result["status"] == "ok"
    assert result["metric_inert"] is True
    assert result["mean_within_user_variance"] == pytest.approx(0.0)


def test_video_ctr_is_not_flagged_metric_inert_on_the_real_registry(
    real_registry_train_and_val,
):
    train_df, val_df = real_registry_train_and_val

    result = screen_feature("video_ctr", feature_module.get("video_ctr"), train_df, val_df)

    assert result["status"] == "ok"
    assert result["metric_inert"] is False
    assert result["mean_within_user_variance"] > DEFAULT_THRESHOLD


def test_screen_features_over_the_real_registry_subset(real_registry_train_and_val):
    """End-to-end over a {name: fn} slice pulled straight from
    pipeline.features.FEATURES, exactly how the CLI entry point would call
    it against the real registry — just with a tiny synthetic split."""
    train_df, val_df = real_registry_train_and_val
    subset = {
        name: feature_module.get(name) for name in ("user_activity", "video_ctr", "user_ctr")
    }

    results = screen_features(subset, train_df, val_df)

    assert results["user_activity"]["metric_inert"] is True
    assert results["video_ctr"]["metric_inert"] is False
    # user_ctr varies by user_id only, and every user has a single rate
    # repeated across their own val rows -> within-user constant too.
    assert results["user_ctr"]["metric_inert"] is True


# --- main(): CLI wiring, exercised without the real dataset archive ---------


def test_main_writes_a_report_and_always_returns_zero(monkeypatch, tmp_path):
    train_df = pd.DataFrame(
        {"user_id": ["u1", "u1"], "video_id": ["vA", "vB"], "long_view": [1, 0],
         "date": [20220408, 20220408]}
    )
    val_df = pd.DataFrame({"user_id": ["u1", "u1"], "video_id": ["vA", "vB"],
                            "date": [20220422, 20220422]})

    import pipeline.data as real_pipeline_data
    monkeypatch.setattr(real_pipeline_data, "load", lambda: (train_df, val_df, val_df.iloc[0:0]))

    out_path = tmp_path / "report.json"
    exit_code = main(["--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["schema_version"] == "1"
    assert "user_activity" in written["features"]


def test_main_no_write_skips_the_file(monkeypatch, tmp_path):
    train_df = pd.DataFrame(
        {"user_id": ["u1", "u1"], "video_id": ["vA", "vB"], "long_view": [1, 0],
         "date": [20220408, 20220408]}
    )
    val_df = pd.DataFrame({"user_id": ["u1", "u1"], "video_id": ["vA", "vB"],
                            "date": [20220422, 20220422]})

    import pipeline.data as real_pipeline_data
    monkeypatch.setattr(real_pipeline_data, "load", lambda: (train_df, val_df, val_df.iloc[0:0]))

    out_path = tmp_path / "should_not_exist.json"
    exit_code = main(["--no-write", "--out", str(out_path)])

    assert exit_code == 0
    assert not out_path.exists()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
