"""C1b acceptance: the self-audit battery catches schema drift, nondeterminism,
label leakage (via the shuffle canary), and missing evaluation evidence —
each with an actionable finding."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import pipeline.audit as audit


def _pandas_frame(rows: int, dates: list[int], offset: int = 0) -> pd.DataFrame:
    index = np.arange(rows) + offset
    video_id = (index * 7) % 50
    return pd.DataFrame(
        {
            "date": np.asarray(dates)[index % len(dates)],
            "user_id": index % 10,
            "video_id": video_id,
            "author_id": video_id % 6,
            "tab": index % 2,
            "duration_ms": 1_000 + (index % 5) * 500,
            # label depends on video_id only: learnable from train labels,
            # gone the moment train labels are shuffled
            "long_view": (video_id % 7 == 0).astype(int),
        }
    )


def _data():
    return (
        _pandas_frame(1_200, list(range(20220408, 20220422))),
        _pandas_frame(300, list(range(20220422, 20220429)), offset=5_000),
        _pandas_frame(300, list(range(20220429, 20220509)), offset=9_000),
    )


def _folds():
    return [
        (
            _pandas_frame(400, [20220408 + i, 20220409 + i], offset=i * 100),
            _pandas_frame(300, [20220416 + i], offset=7_000 + i * 100),
        )
        for i in range(3)
    ]


class VideoRateModel:
    """Learns mean label per first feature column — honest, label-driven."""

    def __init__(self, seed=42, **hparams):
        self.rates = {}
        self.default = 0.5

    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        keys = np.asarray(X_train[:, 0])
        y = np.asarray(y_train, dtype=float)
        self.default = float(y.mean())
        for key in np.unique(keys):
            self.rates[float(key)] = float(y[keys == key].mean())

    def predict(self, X):
        return np.asarray(
            [self.rates.get(float(k), self.default) for k in np.asarray(X[:, 0])]
        )


class FirstColumnModel(VideoRateModel):
    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        pass

    def predict(self, X):
        return np.asarray(X[:, 0], dtype=float)


class NondeterministicModel(FirstColumnModel):
    def predict(self, X):
        noise = np.frombuffer(os.urandom(8 * len(X)), dtype=np.uint64).astype(float)
        return noise / np.iinfo(np.uint64).max


def _install(monkeypatch, model_class):
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    monkeypatch.setattr(train, "_get_model_class", lambda name: model_class, raising=False)
    return train


CONFIG = {"model": "random", "features": ["video_id", "user_id"]}


# --- result-schema audit -----------------------------------------------------


def test_schema_audit_passes_a_well_formed_result(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="smoke")
    finding = audit.audit_result_schema(result, "smoke")
    assert finding["passed"], finding["detail"]


def test_schema_audit_flags_missing_fields_with_an_actionable_message(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="smoke")
    del result["val_user_ids"]
    finding = audit.audit_result_schema(result, "smoke")
    assert not finding["passed"]
    assert "val_user_ids" in finding["detail"]


def test_schema_audit_flags_misaligned_score_vectors(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="smoke")
    result["val_scores"] = np.asarray(result["val_scores"])[:-1]
    finding = audit.audit_result_schema(result, "smoke")
    assert not finding["passed"]
    assert "aligned" in finding["detail"]


# --- determinism audit -------------------------------------------------------


def test_determinism_audit_passes_a_seeded_model(monkeypatch):
    _install(monkeypatch, FirstColumnModel)
    finding = audit.audit_determinism(CONFIG, fidelity="smoke", seed=11)
    assert finding["passed"], finding["detail"]


def test_determinism_audit_catches_unseeded_randomness(monkeypatch):
    _install(monkeypatch, NondeterministicModel)
    finding = audit.audit_determinism(CONFIG, fidelity="smoke", seed=11)
    assert not finding["passed"]
    assert "identical runs" in finding["detail"]


# --- label-shuffle canary ----------------------------------------------------


def test_label_shuffle_passes_an_honest_pipeline(monkeypatch):
    """A model that learns only from training labels collapses to chance
    when those labels are shuffled — no leak, finding passes."""
    _install(monkeypatch, VideoRateModel)
    finding = audit.audit_label_shuffle(CONFIG, seed=5)
    assert finding["passed"], finding["detail"]


def test_label_shuffle_flags_a_feature_reading_target_outcomes(monkeypatch):
    """A feature reading the target rows' own label keeps a perfect score no
    matter what happens to the training labels — the canary must fire."""
    import pipeline.features as features

    _install(monkeypatch, FirstColumnModel)
    monkeypatch.setitem(
        features.FEATURES, "oracle", lambda train_df, target_df: target_df["long_view"]
    )
    # Simulate a leak the source-level guard failed to see: the shuffle
    # canary is the score-level backstop.
    monkeypatch.setattr(features, "leakage_check", lambda fn, a, b: True)

    finding = audit.audit_label_shuffle(
        {"model": "random", "features": ["oracle"]}, seed=5
    )
    assert not finding["passed"]
    assert "canary" in finding["detail"] or "encodes" in finding["detail"]


# --- evidence audit ----------------------------------------------------------


def test_evidence_audit_passes_a_complete_full_result(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="full", seed=3)
    assert result["status"] == "ok"
    finding = audit.audit_evidence(result)
    assert finding["passed"], finding["detail"]


def test_evidence_audit_flags_empty_segments(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="full", seed=3)
    result["segments"] = {}
    finding = audit.audit_evidence(result)
    assert not finding["passed"]
    assert "segments" in finding["detail"]


def test_evidence_audit_rejects_pilot_fidelities(monkeypatch):
    train = _install(monkeypatch, FirstColumnModel)
    result = train.run_experiment(CONFIG, fidelity="screen", seed=3)
    finding = audit.audit_evidence(result)
    assert not finding["passed"]


# --- the battery -------------------------------------------------------------


def test_self_audit_composes_all_checks_and_passes_an_honest_config(monkeypatch):
    _install(monkeypatch, VideoRateModel)
    outcome = audit.self_audit(CONFIG, seed=7)
    assert outcome["passed"], outcome["checks"]
    assert [c["check"] for c in outcome["checks"]] == [
        "result_schema", "determinism", "label_shuffle",
    ]
