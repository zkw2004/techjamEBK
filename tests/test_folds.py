"""B2 acceptance: three folds; each asserts train.max < val.min; no fold
touches 22 April or later; expanding window verified."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.data import INTERNAL_FOLDS, VAL_START


def _interleaved_training_rows() -> pd.DataFrame:
    """Small non-chronological fixture that exposes any accidental sorting."""
    dates = [
        20220416,
        20220408,
        20220420,
        20220409,
        20220417,
        20220410,
        20220418,
        20220411,
        20220421,
        20220412,
        20220419,
        20220413,
        20220414,
        20220415,
    ]
    return pd.DataFrame({"date": dates, "row_id": range(len(dates))})


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


def test_materialised_folds_respect_temporal_order(monkeypatch: pytest.MonkeyPatch):
    import pipeline.data as data_module

    source = _interleaved_training_rows()
    empty = source.iloc[0:0].copy()
    monkeypatch.setattr(data_module, "load", lambda: (source, empty, empty))

    folds = data_module.internal_folds()

    assert len(folds) == 3
    for fold_train, fold_val in folds:
        assert fold_train["date"].max() < fold_val["date"].min()
        assert fold_val["date"].max() < 20220422
        assert fold_train["row_id"].is_monotonic_increasing
        assert fold_val["row_id"].is_monotonic_increasing

    assert [fold_train["row_id"].tolist() for fold_train, _ in folds] == [
        [1, 3, 5, 7, 9, 11, 12, 13],
        [0, 1, 3, 4, 5, 7, 9, 11, 12, 13],
        [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13],
    ]
    assert [fold_val["row_id"].tolist() for _, fold_val in folds] == [
        [0, 4],
        [6, 10],
        [2, 8],
    ]
    assert [fold_train["date"].min() for fold_train, _ in folds] == [20220408] * 3
    assert [fold_train["date"].max() for fold_train, _ in folds] == [
        20220415,
        20220417,
        20220419,
    ]
    assert [fold_val["date"].min() for _, fold_val in folds] == [
        20220416,
        20220418,
        20220420,
    ]
    assert [fold_val["date"].max() for _, fold_val in folds] == [
        20220417,
        20220419,
        20220421,
    ]


def test_materialised_folds_assert_temporal_order(monkeypatch: pytest.MonkeyPatch):
    import pipeline.data as data_module

    source = _interleaved_training_rows()
    empty = source.iloc[0:0].copy()
    monkeypatch.setattr(data_module, "load", lambda: (source, empty, empty))
    monkeypatch.setattr(
        data_module,
        "INTERNAL_FOLDS",
        [("2022-04-08", "2022-04-17", "2022-04-16", "2022-04-18")],
    )

    with pytest.raises(AssertionError):
        data_module.internal_folds()
