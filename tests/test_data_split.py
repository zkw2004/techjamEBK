"""B1 acceptance: exact row counts; assertion fires if splits swapped;
row order matches the starter kit."""

from __future__ import annotations

from pathlib import Path

import pytest
from pandas import DataFrame

from pipeline.data import DATA_DIR, LOG_FILES, N_TEST, N_TRAIN, N_VAL

DATA_FILES = ("video_features_basic_pure.csv", *LOG_FILES)
requires_kuairand_data = pytest.mark.skipif(
    not all((DATA_DIR / filename).is_file() for filename in DATA_FILES),
    reason="requires the ignored KuaiRand-Pure archive; run `make data` locally",
)


def test_split_constants_match_the_brief():
    assert (N_TRAIN, N_VAL, N_TEST) == (1_141_112, 124_909, 170_588)


def test_data_dir_matches_make_data_layout():
    expected = Path(__file__).resolve().parents[1] / "data" / "KuaiRand-Pure" / "data"
    assert DATA_DIR == expected


@requires_kuairand_data
def test_row_counts_exact():
    from pipeline.data import load
    train, val, test = load()
    assert (len(train), len(val), len(test)) == (N_TRAIN, N_VAL, N_TEST)


@requires_kuairand_data
def test_dates_are_strictly_ordered_across_splits():
    from pipeline.data import load
    train, val, test = load()
    assert train["date"].max() < val["date"].min() < test["date"].min()


def test_assertion_fires_if_splits_swapped():
    """Hand the order check a val/train swap and it must raise, not warn."""
    from pipeline.data import _assert_split_order

    train = DataFrame({"date": [20220408, 20220421]})
    val = DataFrame({"date": [20220422]})
    test = DataFrame({"date": [20220429]})

    with pytest.raises(AssertionError):
        _assert_split_order(val, train, test)


@requires_kuairand_data
def test_row_order_matches_starter_kit():
    """row_id is a positional index into the split; any reordering breaks
    submission alignment (trap 4)."""
    import importlib.util

    from pipeline.data import load

    spec = importlib.util.spec_from_file_location(
        "starter_data",
        "kuairand-starter-kit/data.py",
    )
    starter_data = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(starter_data)

    starter = starter_data.load(str(DATA_DIR))
    train, val, test = load()

    for ours, theirs in (
        (train, starter["train"]),
        (val, starter["valid"]),
        (test, starter["test"]),
    ):
        assert ours["date"].tolist() == [row[0] for row in theirs]
        assert ours["user_id"].astype(str).tolist() == [row[1] for row in theirs]
        assert ours["video_id"].astype(str).tolist() == [row[2] for row in theirs]


def test_load_splits_a_temporary_dataset_in_original_file_order(tmp_path, monkeypatch):
    """Keep the B1 loader logic under CI without downloading the 194 MB archive."""
    import pipeline.data as data

    DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "author_id": ["a1", "a2", "a3"],
            "tag": ["10", "20,30", "30"],
        }
    ).to_csv(tmp_path / "video_features_basic_pure.csv", index=False)
    DataFrame(
        {
            "date": [20220408, 20220421],
            "user_id": ["u1", "u2"],
            "video_id": ["v2", "v1"],
        }
    ).to_csv(tmp_path / LOG_FILES[0], index=False)
    DataFrame(
        {
            "date": [20220422, 20220429],
            "user_id": ["u3", "u4"],
            "video_id": ["v3", "missing"],
        }
    ).to_csv(tmp_path / LOG_FILES[1], index=False)

    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "N_TRAIN", 2)
    monkeypatch.setattr(data, "N_VAL", 1)
    monkeypatch.setattr(data, "N_TEST", 1)

    train, val, test = data.load()

    assert train["video_id"].tolist() == ["v2", "v1"]
    assert val["video_id"].tolist() == ["v3"]
    assert test["video_id"].tolist() == ["missing"]
    assert train["author_id"].tolist() == ["a2", "a1"]
    assert test["author_id"].tolist() == ["UNK"]
    assert train["tag"].tolist() == ["20,30", "10"]
    assert val["tag"].tolist() == ["30"]
    assert test["tag"].isna().all()
