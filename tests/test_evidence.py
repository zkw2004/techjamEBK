"""C3b acceptance: full/confirm results package config, fidelity, seed,
per-fold metrics, segment metrics, hypothesis, disproof condition, and a
pass/fail decision with evidence — and the record is a pure function of
(config, fidelity, seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline.evidence as evidence
from pipeline.evidence import REQUIRED_RECORD_FIELDS, canonical, evidence_record, to_node


def _frame(rows: int, dates: list[int], offset: int = 0) -> pd.DataFrame:
    index = np.arange(rows) + offset
    video_id = (index * 7) % 60
    return pd.DataFrame(
        {
            "date": np.asarray(dates)[index % len(dates)],
            "user_id": index % 8,
            "video_id": video_id,
            "author_id": video_id % 6,
            "tab": index % 2,
            "duration_ms": 1_000 + (index % 5) * 500,
            "long_view": (video_id % 6 == 0).astype(int),
        }
    )


def _data():
    return (
        _frame(1_100, list(range(20220408, 20220422))),
        _frame(240, list(range(20220422, 20220429)), offset=4_000),
        _frame(240, list(range(20220429, 20220509)), offset=8_000),
    )


def _folds():
    return [
        (
            _frame(300, [20220408 + i], offset=i * 50),
            _frame(200, [20220416 + i], offset=6_000 + i * 50),
        )
        for i in range(3)
    ]


class FirstColumnModel:
    def __init__(self, seed=42, **hparams):
        pass

    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        pass

    def predict(self, X):
        return np.asarray(X[:, 0], dtype=float)


class ExplodingModel(FirstColumnModel):
    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        raise RuntimeError("synthetic training failure")


CONFIG = {"model": "random", "features": ["video_id", "user_id"]}
HYPOTHESIS = "video identity predicts long views on this fixture"
DISPROOF = "full primary fails to beat the stated baseline by the floor"


def _install(monkeypatch, model_class=FirstColumnModel):
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    monkeypatch.setattr(train, "_get_model_class", lambda name: model_class, raising=False)
    return train


def test_record_carries_every_required_field(monkeypatch):
    _install(monkeypatch)
    record = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=6, baseline_primary=0.0
    )
    for field in REQUIRED_RECORD_FIELDS:
        assert field in record, field
    assert record["fidelity"] == "full"
    assert record["seed"] == 6
    assert len(record["fold_primaries"]) == 3
    assert record["segments"], "full-fidelity segments must populate on framed fixtures"
    assert record["reproduce"] == {
        "config": record["config"], "fidelity": "full", "seed": 6,
    }


def test_decision_passes_and_fails_against_the_stated_baseline(monkeypatch):
    _install(monkeypatch)
    passing = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=6, baseline_primary=0.0
    )
    failing = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=6, baseline_primary=1.0
    )
    assert passing["decision"]["outcome"] == "pass" and passing["decision"]["passed"]
    assert failing["decision"]["outcome"] == "fail" and not failing["decision"]["passed"]
    assert failing["decision"]["delta"] < 0
    for record in (passing, failing):
        assert "primary" in record["decision"]["evidence"]
        assert record["decision"]["min_delta"] == pytest.approx(0.002)


def test_record_is_a_pure_function_of_config_fidelity_seed(monkeypatch):
    _install(monkeypatch)
    first = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=9, baseline_primary=0.0
    )
    second = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=9, baseline_primary=0.0
    )
    assert canonical(first) == canonical(second)


def test_failed_run_is_recorded_as_evidence_not_discarded(monkeypatch):
    _install(monkeypatch, ExplodingModel)
    record = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=6
    )
    assert record["decision"]["outcome"] == "error"
    assert not record["decision"]["passed"]
    assert "synthetic training failure" in record["decision"]["evidence"]
    assert record["hypothesis"] == HYPOTHESIS  # the hypothesis survives its failure


def test_pilot_fidelities_are_refused(monkeypatch):
    train = _install(monkeypatch)
    result = train.run_experiment(CONFIG, fidelity="screen", seed=6)
    with pytest.raises(ValueError, match="pilot"):
        evidence_record(HYPOTHESIS, DISPROOF, CONFIG, "screen", 6, result)


def test_hypothesis_and_disproof_condition_are_mandatory(monkeypatch):
    with pytest.raises(ValueError, match="hypothesis"):
        evidence_record("", DISPROOF, CONFIG, "full", 6, {"status": "ok"})
    with pytest.raises(ValueError, match="disproof"):
        evidence_record(HYPOTHESIS, "  ", CONFIG, "full", 6, {"status": "ok"})


def test_to_node_round_trips_through_the_append_only_store(monkeypatch, tmp_path):
    import agent.store as store

    _install(monkeypatch)
    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")

    record = evidence.run_with_evidence(
        HYPOTHESIS, DISPROOF, CONFIG, fidelity="full", seed=6, baseline_primary=0.0
    )
    node = to_node(
        record, parent="n000", family="feature", manifest_sha256="test-manifest"
    )
    path = store.write(node)
    loaded = store.read(path.stem)
    assert loaded["hypothesis"] == HYPOTHESIS
    assert loaded["disproof_condition"] == DISPROOF
    assert loaded["decision"]["outcome"] == "pass"
    assert loaded["seed"] == 6
    assert loaded["accepted"] is True
