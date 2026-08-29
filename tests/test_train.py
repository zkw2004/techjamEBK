"""C1 fidelity-tier acceptance against small in-memory experiment fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TinySeedModel:
    """A deterministic model that also rejects an oversized smoke fit."""

    def __init__(self, seed=42, **hparams):
        self.seed = seed

    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        if len(X_train) > 1_000:
            raise AssertionError("smoke fit exceeded 1,000 rows")

    def predict(self, X):
        return np.full(len(X), float(self.seed))


class FullSeedModel(TinySeedModel):
    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        pass


class ScreenBudgetModel(FullSeedModel):
    def __init__(self, seed=42, **hparams):
        super().__init__(seed=seed)
        if hparams.get("max_epochs", 10_000) > 3:
            raise AssertionError("screen did not apply the reduced epoch budget")


class FirstColumnModel(FullSeedModel):
    def predict(self, X):
        return np.asarray(X[:, 0], dtype=float)


class FoldEpochModel(FullSeedModel):
    def __init__(self, seed=42, max_epochs=40, **hparams):
        super().__init__(seed=seed)
        self.max_epochs = max_epochs
        self.best_epoch = None

    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        if X_val is not None:
            self.best_epoch = self.seed

    def predict(self, X):
        return np.full(len(X), float(self.max_epochs))


def _frame(rows: int, offset: int = 0) -> dict[str, np.ndarray]:
    index = np.arange(rows) + offset
    return {
        "user_id": index // 2,
        "video_id": index,
        "long_view": index % 2,
    }


def _data():
    return _frame(1_200), _frame(6, 2_000), _frame(5, 3_000)


def _folds():
    return [
        (_frame(8, 10), _frame(4, 100)),
        (_frame(10, 20), _frame(4, 200)),
        (_frame(12, 30), _frame(4, 300)),
    ]


def _target_label_feature(train_frame, target_frame):
    return target_frame["long_view"]


def _pandas_frame(rows: int, offset: int = 0, label: str = "long_view") -> pd.DataFrame:
    index = np.arange(rows) + offset
    return pd.DataFrame(
        {
            "date": np.full(rows, 20220408),
            "user_id": index // 2,
            "video_id": index,
            "author_id": index % 3,
            "tab": index % 2,
            "duration_ms": np.full(rows, 1_000),
            "dur_bucket": index % 10,
            label: index % 2,
        }
    )


def _pandas_data(label: str = "long_view"):
    return (
        _pandas_frame(12, label=label),
        _pandas_frame(6, 2_000, label=label),
        _pandas_frame(5, 3_000, label=label),
    )


def _install_fixture_backend(monkeypatch, model_class):
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    monkeypatch.setattr(train, "_get_model_class", lambda name: model_class, raising=False)
    return train


def test_smoke_caps_training_at_1000_rows_and_skips_metrics(monkeypatch):
    train = _install_fixture_backend(monkeypatch, TinySeedModel)

    result = train.run_experiment({"model": "random"}, fidelity="smoke", seed=7)

    assert result["status"] == "ok"
    assert result["fidelity"] == "smoke"
    assert result["gauc"] is None
    assert result["ndcg"] is None
    assert result["primary"] is None
    assert result["fold_primaries"] == []


def test_screen_scores_exactly_three_internal_folds(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FullSeedModel)

    result = train.run_experiment({"model": "random"}, fidelity="screen", seed=8)

    assert result["status"] == "ok"
    assert result["fidelity"] == "screen"
    assert len(result["fold_primaries"]) == 3
    assert len(result["val_scores"]) == 12
    assert len(result["val_user_ids"]) == 12
    assert len(result["test_scores"]) == 0


def test_screen_applies_reduced_training_budget(monkeypatch):
    train = _install_fixture_backend(monkeypatch, ScreenBudgetModel)

    result = train.run_experiment(
        {"model": "random", "hparams": {"max_epochs": 40}},
        fidelity="screen",
    )

    assert result["status"] == "ok"


def test_full_scores_official_validation_and_test_rows(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FullSeedModel)

    result = train.run_experiment({"model": "random"}, fidelity="full", seed=9)

    assert result["status"] == "ok"
    assert result["fidelity"] == "full"
    assert len(result["fold_primaries"]) == 3
    assert len(result["val_scores"]) == 6
    assert len(result["val_user_ids"]) == 6
    assert len(result["test_scores"]) == 5
    assert result["segments"] == {}


def test_full_tier_is_deterministic_without_replacing_tier_execution(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FullSeedModel)

    first = train.run_experiment({"model": "random"}, fidelity="full", seed=19)
    second = train.run_experiment({"model": "random"}, fidelity="full", seed=19)

    assert first["primary"] == second["primary"]
    np.testing.assert_array_equal(first["val_scores"], second["val_scores"])
    np.testing.assert_array_equal(first["test_scores"], second["test_scores"])


def test_full_refit_uses_median_best_epoch_selected_on_internal_folds(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FoldEpochModel)

    result = train.run_experiment({"model": "fm"}, fidelity="full", seed=9)

    assert result["status"] == "ok"
    np.testing.assert_array_equal(result["val_scores"], np.full(6, 10.0))


def test_raw_label_cannot_be_selected_as_a_feature(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FullSeedModel)

    result = train.run_experiment(
        {"model": "random", "features": ["long_view"]},
        fidelity="smoke",
    )

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


def test_registered_feature_must_pass_leakage_guard(monkeypatch):
    import pipeline.features as features

    train = _install_fixture_backend(monkeypatch, FullSeedModel)
    monkeypatch.setitem(features.FEATURES, "target_label", _target_label_feature)
    monkeypatch.setattr(features, "leakage_check", lambda fn, train_df, target_df: False)

    result = train.run_experiment(
        {"model": "random", "features": ["target_label"]},
        fidelity="smoke",
    )

    assert result["status"] == "error"
    # leak_suspected, not schema: A5 gives schema errors a repair attempt,
    # and a leaky feature must be quarantined, never repaired into passing.
    assert result["error_class"] == "leak_suspected"


def test_runner_uses_b_workflow_label_with_pandas_frames(monkeypatch):
    import pipeline.data as data
    import pipeline.train as train

    monkeypatch.setattr(data, "LABEL", "target")
    monkeypatch.setattr(train, "_load_data", lambda: _pandas_data(label="target"))
    monkeypatch.setattr(train, "_get_model_class", lambda name: FullSeedModel)

    result = train.run_experiment({"model": "random"}, fidelity="smoke", seed=31)

    assert result["status"] == "ok"
    np.testing.assert_array_equal(result["val_user_ids"], [1000, 1000, 1001, 1001, 1002, 1002])


def test_runner_executes_b3_registered_feature_on_pandas_frames(monkeypatch):
    import pipeline.features as features
    import pipeline.train as train

    def row_number(train_frame, target_frame):
        return np.arange(len(target_frame), dtype=float)

    monkeypatch.setitem(features.FEATURES, "row_number", row_number)
    monkeypatch.setattr(features, "leakage_check", lambda fn, train_df, target_df: True)
    monkeypatch.setattr(train, "_load_data", _pandas_data)
    monkeypatch.setattr(train, "_get_model_class", lambda name: FirstColumnModel)

    result = train.run_experiment(
        {"model": "random", "features": ["row_number"]},
        fidelity="smoke",
        seed=32,
    )

    assert result["status"] == "ok"
    np.testing.assert_array_equal(result["val_scores"], [0, 1, 2, 3, 4, 5])


def test_matrix_derives_duration_bucket_from_training_quantiles():
    import pipeline.train as train

    training = pd.DataFrame({"duration_ms": np.arange(10, 110, 10)})
    target = pd.DataFrame({"duration_ms": [10, 25, 100]})

    matrix = train._matrix(training, target, ["dur_bucket"])

    np.testing.assert_array_equal(matrix[:, 0], [0, 1, 9])


def test_runner_uses_b3_public_feature_resolver(monkeypatch):
    import pipeline.features as features
    import pipeline.train as train

    def registry_value(train_frame, target_frame):
        return np.zeros(len(target_frame), dtype=float)

    def resolved_value(train_frame, target_frame):
        return np.arange(len(target_frame), dtype=float) + 10

    monkeypatch.setitem(features.FEATURES, "resolved", registry_value)
    monkeypatch.setattr(features, "get", lambda name: resolved_value)
    monkeypatch.setattr(features, "leakage_check", lambda fn, train_df, target_df: True)
    monkeypatch.setattr(train, "_load_data", _pandas_data)
    monkeypatch.setattr(train, "_get_model_class", lambda name: FirstColumnModel)

    result = train.run_experiment(
        {"model": "random", "features": ["resolved"]},
        fidelity="smoke",
        seed=33,
    )

    assert result["status"] == "ok"
    np.testing.assert_array_equal(result["val_scores"], [10, 11, 12, 13, 14, 15])


def test_confirm_averages_five_consecutive_seeds(monkeypatch):
    train = _install_fixture_backend(monkeypatch, FullSeedModel)

    result = train.run_experiment({"model": "random"}, fidelity="confirm", seed=10)

    assert result["status"] == "ok"
    assert result["fidelity"] == "confirm"
    np.testing.assert_array_equal(result["val_scores"], np.full(6, 12.0))
    np.testing.assert_array_equal(result["test_scores"], np.full(5, 12.0))
    assert len(result["fold_primaries"]) == 3
