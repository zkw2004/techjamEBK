"""B8 acceptance (stretch): the randomised-exposure slice loader is
separable from train/val/test; `randomised_exposure_pre_cutoff` exposes
only pre-cutoff rows -- which, for this data release, is a validated-empty
set (README correction 5); the IPS propensity/reweighting utilities are
documented with their assumptions and covered by synthetic unit tests that
do not require the ignored dataset archive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline.data as data_module
from pipeline.data import (
    DATA_DIR,
    LOG_FILES,
    RANDOM_LOG_FILE,
    TRAIN_END,
    VAL_START,
    estimate_item_propensity,
    load_randomised_exposure,
    randomised_exposure_pre_cutoff,
    self_normalized_ips_rate,
)

requires_random_log = pytest.mark.skipif(
    not (DATA_DIR / RANDOM_LOG_FILE).is_file(),
    reason="requires the ignored KuaiRand-Pure archive; run `make data` locally",
)
requires_kuairand_data = pytest.mark.skipif(
    not all((DATA_DIR / filename).is_file() for filename in (*LOG_FILES, RANDOM_LOG_FILE)),
    reason="requires the ignored KuaiRand-Pure archive; run `make data` locally",
)


# --- load_randomised_exposure / randomised_exposure_pre_cutoff, against the
# real archive when it is present locally -----------------------------------


@requires_random_log
def test_randomised_exposure_loads_as_a_separable_is_rand_slice():
    frame = load_randomised_exposure()
    assert (frame["is_rand"] == 1).all()
    assert len(frame) > 0


@requires_random_log
def test_randomised_exposure_is_entirely_after_the_training_cutoff():
    """README correction 5: log_random_4_22_to_5_08_pure.csv is dated
    entirely on or after VAL_START, so it cannot debias training under
    Section 6.5's own pre-cutoff-only rule."""
    frame = load_randomised_exposure()
    dates = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    assert dates.min() >= pd.Timestamp(VAL_START)
    assert dates.min() > pd.Timestamp(TRAIN_END)


@requires_random_log
def test_pre_cutoff_slice_is_validated_empty_against_the_real_archive():
    """The training-legal slice Section 6.5 asks for. Enforced in code, not
    left as a comment: currently zero rows, matching the reality above."""
    pre_cutoff = randomised_exposure_pre_cutoff()
    assert len(pre_cutoff) == 0
    assert list(pre_cutoff.columns) == list(load_randomised_exposure().columns)


@requires_kuairand_data
def test_randomised_exposure_shares_its_universe_with_the_standard_log():
    """Same users/items KuaiRand-Pure logs elsewhere -- not a different
    dataset accidentally wired in under RANDOM_LOG_FILE."""
    from pipeline.data import load

    random_frame = load_randomised_exposure()
    train, _, _ = load()
    assert random_frame["user_id"].isin(train["user_id"]).mean() > 0.9
    assert random_frame["video_id"].isin(train["video_id"]).mean() > 0.9


# --- load_randomised_exposure validation, against synthetic files ----------


def test_pre_cutoff_slice_accepts_a_given_frame_without_reloading():
    frame = pd.DataFrame({"date": [20220421, 20220422, 20220430]})

    result = randomised_exposure_pre_cutoff(frame)

    assert list(result["date"]) == [20220421]


def test_load_randomised_exposure_rejects_a_mixed_is_rand_file(tmp_path, monkeypatch):
    bad_file = tmp_path / "mixed_is_rand.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2],
            "video_id": [1, 2],
            "date": [20220422, 20220422],
            "is_rand": [1, 0],
        }
    ).to_csv(bad_file, index=False)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "RANDOM_LOG_FILE", bad_file.name)

    with pytest.raises(ValueError, match="is_rand"):
        data_module.load_randomised_exposure()


def test_load_randomised_exposure_rejects_a_row_dated_before_val_start(tmp_path, monkeypatch):
    bad_file = tmp_path / "too_early.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2],
            "video_id": [1, 2],
            "date": [20220420, 20220422],
            "is_rand": [1, 1],
        }
    ).to_csv(bad_file, index=False)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "RANDOM_LOG_FILE", bad_file.name)

    with pytest.raises(ValueError, match="before"):
        data_module.load_randomised_exposure()


# --- estimate_item_propensity ------------------------------------------------


def test_estimate_item_propensity_reflects_relative_over_and_under_exposure():
    random_frame = pd.DataFrame({"video_id": ["v1"] * 5 + ["v2"] * 5})  # 50/50 baseline
    standard_frame = pd.DataFrame({"video_id": ["v1"] * 8 + ["v2"] * 2})  # policy favours v1

    propensity = estimate_item_propensity(random_frame, standard_frame)

    assert propensity["v1"] == pytest.approx(0.8 / 0.5)
    assert propensity["v2"] == pytest.approx(0.2 / 0.5)
    assert propensity["v1"] > 1  # over-exposed relative to random
    assert propensity["v2"] < 1  # under-exposed relative to random


def test_estimate_item_propensity_excludes_videos_missing_from_the_random_slice():
    random_frame = pd.DataFrame({"video_id": ["v1", "v1"]})
    standard_frame = pd.DataFrame({"video_id": ["v1", "v2"]})  # v2 never randomly shown

    propensity = estimate_item_propensity(random_frame, standard_frame)

    assert list(propensity.index) == ["v1"]


@pytest.mark.parametrize(
    ("random_frame", "standard_frame"),
    [
        (pd.DataFrame({"video_id": []}), pd.DataFrame({"video_id": ["v1"]})),
        (pd.DataFrame({"video_id": ["v1"]}), pd.DataFrame({"video_id": []})),
    ],
)
def test_estimate_item_propensity_rejects_an_empty_frame(random_frame, standard_frame):
    with pytest.raises(ValueError, match="non-empty"):
        estimate_item_propensity(random_frame, standard_frame)


# --- self_normalized_ips_rate ------------------------------------------------


def test_self_normalized_ips_rate_matches_the_closed_form_on_a_small_case():
    propensity = pd.Series({"v1": 2.0, "v2": 0.5})
    labels = [1, 0, 1]
    video_ids = ["v1", "v1", "v2"]

    rate, coverage = self_normalized_ips_rate(labels, video_ids, propensity)

    weights = np.array([1 / 2.0, 1 / 2.0, 1 / 0.5])
    expected = np.sum(np.array([1.0, 0.0, 1.0]) * weights) / np.sum(weights)
    assert rate == pytest.approx(expected)
    assert coverage == 1.0


def test_self_normalized_ips_rate_reports_partial_coverage_and_drops_unweighted_rows():
    propensity = pd.Series({"v1": 1.0})
    labels = [1, 0, 1]
    video_ids = ["v1", "v2", "v2"]  # v2 has no propensity

    rate, coverage = self_normalized_ips_rate(labels, video_ids, propensity)

    assert coverage == pytest.approx(1 / 3)
    assert rate == pytest.approx(1.0)  # only the single v1 row (label=1) counted


def test_self_normalized_ips_rate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matching shapes"):
        self_normalized_ips_rate([1, 0], ["v1"], pd.Series({"v1": 1.0}))


def test_self_normalized_ips_rate_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one row"):
        self_normalized_ips_rate([], [], pd.Series(dtype=float))


def test_self_normalized_ips_rate_rejects_zero_coverage():
    with pytest.raises(ValueError, match="no row"):
        self_normalized_ips_rate([1], ["v9"], pd.Series({"v1": 1.0}))
