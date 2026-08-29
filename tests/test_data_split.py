"""B1 acceptance: exact row counts; assertion fires if splits swapped;
row order matches the starter kit."""

from __future__ import annotations

from pipeline.data import N_TEST, N_TRAIN, N_VAL
from tests.conftest import todo


def test_split_constants_match_the_brief():
    assert (N_TRAIN, N_VAL, N_TEST) == (1_141_112, 124_909, 170_588)


@todo("B1")
def test_row_counts_exact():
    from pipeline.data import load
    train, val, test = load()
    assert (len(train), len(val), len(test)) == (N_TRAIN, N_VAL, N_TEST)


@todo("B1")
def test_dates_are_strictly_ordered_across_splits():
    from pipeline.data import load
    train, val, test = load()
    assert train["date"].max() < val["date"].min() < test["date"].min()


@todo("B1")
def test_assertion_fires_if_splits_swapped():
    """Hand load() a val/train swap and it must raise, not warn."""


@todo("B1")
def test_row_order_matches_starter_kit():
    """row_id is a positional index into the split; any reordering breaks
    submission alignment (trap 4)."""
