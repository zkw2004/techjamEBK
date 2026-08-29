"""B3 acceptance: register, resolve, and invoke feature builders safely."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features as feature_module


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
