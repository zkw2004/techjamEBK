"""B1 acceptance: exact row counts; assertion fires if splits swapped;
row order matches the starter kit."""

from __future__ import annotations
import pytest

from pipeline.data import N_TEST, N_TRAIN, N_VAL

from tests.conftest import todo


def test_split_constants_match_the_brief():
    assert (N_TRAIN, N_VAL, N_TEST) == (1_141_112, 124_909, 170_588)


def test_row_counts_exact():
    from pipeline.data import load
    train, val, test = load()
    assert (len(train), len(val), len(test)) == (N_TRAIN, N_VAL, N_TEST)


def test_dates_are_strictly_ordered_across_splits():
    from pipeline.data import load
    train, val, test = load()
    assert train["date"].max() < val["date"].min() < test["date"].min()


def test_assertion_fires_if_splits_swapped():
    """Hand the order check a val/train swap and it must raise, not warn."""
    from pipeline.data import _assert_split_order, load

    train, val, test = load()

    with pytest.raises(AssertionError):
        _assert_split_order(val, train, test)



def test_row_order_matches_starter_kit():
    """row_id is a positional index into the split; any reordering breaks
    submission alignment (trap 4)."""
    import importlib.util

    from pipeline.data import DATA_DIR, load

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
