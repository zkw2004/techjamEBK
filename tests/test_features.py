"""B3 acceptance: register, resolve, and invoke feature builders safely."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features as feature_module

BASELINE_FEATURE_NAMES = {
    "user_ctr",
    "video_ctr",
    "video_impressions",
    "user_activity",
    "user_tag_affinity",
    "hour_of_day",
    "day_of_week",
}
HISTORICAL_FEATURE_NAMES = BASELINE_FEATURE_NAMES - {"hour_of_day", "day_of_week"}


@pytest.fixture
def aggregate_train_df() -> pd.DataFrame:
    """Exactly 100 historical rows with hand-checkable group statistics."""
    blocks = [
        # user, video, tags, impressions, positives
        ("uA", "v1", "10", 20, 16),
        ("uA", "v2", "20", 10, 2),
        ("uA", "v3", "10,20", 10, 6),
        ("uB", "v1", "10", 10, 2),
        ("uB", "v2", "20", 20, 10),
        ("uB", "v4", "30", 10, 0),
        ("uC", "v3", "10,20", 5, 5),
        ("uC", "v4", "30", 5, 1),
        ("uC", "v5", pd.NA, 10, 0),
    ]
    rows = []
    row_number = 0
    for user_id, video_id, tag, impressions, positives in blocks:
        for offset in range(impressions):
            rows.append(
                {
                    "user_id": user_id,
                    "video_id": video_id,
                    "tag": tag,
                    "long_view": int(offset < positives),
                    "date": 20220409 + row_number % 13,
                    "hourmin": row_number % 24 * 100,
                }
            )
            row_number += 1

    frame = pd.DataFrame(rows)
    assert len(frame) == 100
    assert frame["long_view"].sum() == 42
    return frame


@pytest.fixture
def aggregate_target_df() -> pd.DataFrame:
    """Target rows cover unseen IDs, multi-tags, missing tags, and duplicates."""
    rows = [
        ("uA", "v1", "10", 20220422, 0),
        ("uA", "v2", "20", 20220423, 100),
        ("uA", "v3", "10,20", 20220424, 900),
        ("uB", "v4", "30", 20220425, 1200),
        ("uC", "v5", pd.NA, 20220426, 1800),
        ("uZ", "v9", "10", 20220427, 2300),
        ("uA", "v9", "10,99", 20220428, 500),
        ("uZ", "v1", "10", 20220429, 1400),
        ("uB", "v3", "20,10,20", 20220430, 700),
        ("uA", "v1", "10", 20220422, 0),
    ]
    return pd.DataFrame(
        rows,
        columns=["user_id", "video_id", "tag", "date", "hourmin"],
        index=[91, 7, 42, 5, 88, 13, 2, 73, 19, 91],
    ).assign(
        long_view=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        is_click=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Keep test registrations out of the process-wide feature registry."""
    registry = {}
    monkeypatch.setattr(feature_module, "FEATURES", registry)
    return registry


def test_register_then_call_returns_one_value_per_target_row(isolated_registry):
    def train_mean(train_df, target_df):
        return np.full(len(target_df), train_df["value"].mean())

    decorated = feature_module.feature("train_mean")(train_mean)
    train_df = pd.DataFrame({"value": [1.0, 3.0, 5.0]})
    target_df = pd.DataFrame({"row_id": [10, 11, 12, 13]})

    assert decorated is train_mean
    assert isolated_registry == {"train_mean": train_mean}
    assert feature_module.get("train_mean") is train_mean

    result = feature_module.get("train_mean")(train_df, target_df)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(target_df)
    np.testing.assert_array_equal(result, [3.0, 3.0, 3.0, 3.0])


def test_unknown_feature_raises_before_any_builder_runs(isolated_registry):
    calls = []

    @feature_module.feature("known")
    def known(train_df, target_df):
        calls.append((train_df, target_df))
        return np.zeros(len(target_df))

    with pytest.raises(KeyError, match="unknown feature 'missing'"):
        feature_module.get("missing")

    assert not calls
    assert isolated_registry == {"known": known}


def test_duplicate_feature_name_is_rejected_without_replacing_original(isolated_registry):
    @feature_module.feature("duplicate")
    def original(train_df, target_df):
        return np.zeros(len(target_df))

    with pytest.raises(ValueError, match="duplicate feature name: duplicate"):

        @feature_module.feature("duplicate")
        def replacement(train_df, target_df):
            return np.ones(len(target_df))

    assert isolated_registry == {"duplicate": original}
    assert feature_module.get("duplicate") is original


def test_all_seven_baseline_features_are_registered():
    assert BASELINE_FEATURE_NAMES <= set(feature_module.FEATURES)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("user_ctr", [0.6, 0.6, 0.6, 0.3, 0.3, 0.42, 0.6, 0.42, 0.3, 0.6]),
        (
            "video_ctr",
            [0.6, 0.4, 11 / 15, 1 / 15, 0.0, 0.42, 0.42, 0.6, 11 / 15, 0.6],
        ),
        ("video_impressions", [30, 30, 15, 15, 10, 0, 0, 30, 15, 30]),
        ("user_activity", [40, 40, 40, 40, 20, 0, 40, 0, 40, 40]),
        (
            "user_tag_affinity",
            [
                11 / 15,
                2 / 5,
                17 / 30,
                0.0,
                0.42,
                0.42,
                173 / 300,
                0.42,
                7 / 20,
                11 / 15,
            ],
        ),
        ("hour_of_day", [0, 1, 9, 12, 18, 23, 5, 14, 7, 0]),
        ("day_of_week", [4, 5, 6, 0, 1, 2, 3, 4, 5, 4]),
    ],
)
def test_baseline_feature_on_100_row_fixture(
    name,
    expected,
    aggregate_train_df,
    aggregate_target_df,
):
    result = feature_module.get(name)(aggregate_train_df, aggregate_target_df)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(aggregate_target_df)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected)


def test_baseline_features_do_not_read_target_outcomes(
    aggregate_train_df,
    aggregate_target_df,
):
    changed_target = aggregate_target_df.copy()
    changed_target["long_view"] = 1 - changed_target["long_view"]
    changed_target["is_click"] = 1 - changed_target["is_click"]

    for name in BASELINE_FEATURE_NAMES:
        original = feature_module.get(name)(aggregate_train_df, aggregate_target_df)
        changed = feature_module.get(name)(aggregate_train_df, changed_target)
        np.testing.assert_array_equal(original, changed, err_msg=name)


def test_historical_features_reject_overlapping_target_window(
    aggregate_train_df,
    aggregate_target_df,
):
    overlapping_target = aggregate_target_df.copy()
    overlapping_target["date"] = 20220421

    for name in HISTORICAL_FEATURE_NAMES:
        with pytest.raises(ValueError, match="strictly earlier"):
            feature_module.get(name)(aggregate_train_df, overlapping_target)


def test_historical_features_build_leak_safe_in_sample_history_by_timestamp():
    """Training features use only events strictly before each impression."""
    frame = pd.DataFrame(
        {
            # Deliberately unsorted: original CSV position is not time order.
            "time_ms": [300, 100, 200, 200, 400],
            "date": [20220408] * 5,
            "user_id": ["u1", "u1", "u1", "u2", "u1"],
            "video_id": ["v1", "v2", "v1", "v1", "v3"],
            "tag": ["10", "20", "10", "10", "20"],
            "long_view": [1, 0, 1, 1, 0],
        }
    )

    np.testing.assert_allclose(feature_module.user_ctr(frame, frame), [0.5, 0.5, 0, 0, 2 / 3])
    np.testing.assert_allclose(feature_module.video_ctr(frame, frame), [1, 0.5, 0, 0, 0.75])
    np.testing.assert_allclose(feature_module.video_impressions(frame, frame), [2, 0, 0, 0, 0])
    np.testing.assert_allclose(feature_module.user_activity(frame, frame), [2, 0, 1, 0, 3])
    np.testing.assert_allclose(
        feature_module.user_tag_affinity(frame, frame), [1, 0.5, 0, 0, 0]
    )


def test_in_sample_historical_features_require_an_event_timestamp(aggregate_train_df):
    with pytest.raises(ValueError, match="time_ms"):
        feature_module.user_ctr(aggregate_train_df, aggregate_train_df)


def test_rate_features_accept_numeric_string_training_labels(
    aggregate_train_df,
    aggregate_target_df,
):
    string_labels = aggregate_train_df.copy()
    string_labels["long_view"] = string_labels["long_view"].astype(str)

    for name in ("user_ctr", "video_ctr", "user_tag_affinity"):
        numeric = feature_module.get(name)(aggregate_train_df, aggregate_target_df)
        strings = feature_module.get(name)(string_labels, aggregate_target_df)
        np.testing.assert_array_equal(numeric, strings, err_msg=name)


def test_context_features_handle_mixed_missing_and_invalid_values():
    target_df = pd.DataFrame(
        {
            "date": [20220418, None, 20220431],
            "hourmin": [930, 2360, None],
        }
    )
    empty_train = pd.DataFrame()

    np.testing.assert_array_equal(feature_module.hour_of_day(empty_train, target_df), [9, -1, -1])
    np.testing.assert_array_equal(feature_module.day_of_week(empty_train, target_df), [0, -1, -1])
