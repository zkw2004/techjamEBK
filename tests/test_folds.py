"""B2 acceptance: three folds; each asserts train.max < val.min; no fold
touches 22 April or later; expanding window verified."""

from __future__ import annotations

from pipeline.data import INTERNAL_FOLDS, VAL_START
from tests.conftest import todo


def test_three_folds_defined():
    assert len(INTERNAL_FOLDS) == 3


def test_no_fold_touches_the_official_validation_window():
    """The 22-28 April window is a budgeted resource, never screened against."""
    for _, _, _, fold_val_end in INTERNAL_FOLDS:
        assert fold_val_end < VAL_START


def test_folds_are_date_ordered_within_themselves():
    for tr_start, tr_end, va_start, va_end in INTERNAL_FOLDS:
        assert tr_start < tr_end < va_start <= va_end


def test_window_is_expanding_not_rolling():
    starts = [f[0] for f in INTERNAL_FOLDS]
    ends = [f[1] for f in INTERNAL_FOLDS]
    assert len(set(starts)) == 1, "train start must be fixed — expanding, not sliding"
    assert ends == sorted(ends) and len(set(ends)) == 3


@todo("B2")
def test_materialised_folds_respect_temporal_order():
    from pipeline.data import internal_folds
    for fold_train, fold_val in internal_folds():
        assert fold_train["date"].max() < fold_val["date"].min()
