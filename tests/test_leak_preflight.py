"""D13 leakage-canary acceptance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import leak_preflight


def _frames():
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "long_view": [0, 1, 0, 1, 0, 1],
        }
    )
    validation = pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "long_view": [0, 1, 1, 0]}
    )
    return train, validation


def test_permutation_preserves_each_users_label_multiset():
    train, _ = _frames()
    permuted = leak_preflight.permute_labels_within_user(train, seed=7)

    for user_id, group in train.groupby("user_id"):
        actual = permuted.loc[permuted["user_id"] == user_id, "long_view"]
        assert sorted(actual.tolist()) == sorted(group["long_view"].tolist())
    assert train["long_view"].tolist() == [0, 1, 0, 1, 0, 1]


def test_clean_pipeline_passes_and_injected_leak_trips_alarm():
    train, validation = _frames()

    def fit_predict(_config, _train, _validation, prediction_frames, _seed):
        return [np.linspace(0.1, 0.9, len(frame)) for frame in prediction_frames]

    def evaluate(_frame, _scores):
        return {"gauc": 0.507, "ndcg": 0.0, "primary": 0.0}

    def guard(fn, _train, _validation):
        return fn is not leak_preflight.deliberately_leaky_feature

    result = leak_preflight.run_preflight(
        frames=(train, validation),
        fit_predict=fit_predict,
        evaluate=evaluate,
        leakage_guard=guard,
    )

    assert result["status"] == "passed"
    assert result["clean_passed"] is True
    assert result["leaky_fixture_rejected"] is True


def test_preflight_fails_when_clean_pipeline_is_predictive():
    train, validation = _frames()

    with pytest.raises(leak_preflight.LeakPreflightError, match="outside"):
        leak_preflight.run_preflight(
            frames=(train, validation),
            fit_predict=lambda *_args: [np.zeros(len(validation))],
            evaluate=lambda *_args: {"gauc": 0.55},
            leakage_guard=lambda *_args: False,
        )


def test_preflight_fails_when_leaky_fixture_is_not_rejected():
    train, validation = _frames()

    with pytest.raises(leak_preflight.LeakPreflightError, match="was not rejected"):
        leak_preflight.run_preflight(
            frames=(train, validation),
            fit_predict=lambda *_args: [np.zeros(len(validation))],
            evaluate=lambda *_args: {"gauc": 0.5},
            leakage_guard=lambda *_args: True,
        )


def test_result_is_recorded_as_run_log_header(tmp_path, monkeypatch):
    from agent import store

    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    result = {
        "status": "passed",
        "clean_permuted_gauc": 0.5,
        "leaky_fixture_rejected": True,
    }

    leak_preflight.record_preflight(result)

    event = store.read_events()[0]
    assert event["event"] == "leakage_preflight"
    assert event["status"] == "passed"
    assert event["manual_intervention"] is False


def test_cli_preflight_mode_is_independent(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda **_kwargs: None))
    import cli

    result = {"status": "passed", "clean_permuted_gauc": 0.5}
    monkeypatch.setattr(leak_preflight, "run_preflight", lambda: result)
    monkeypatch.setattr(leak_preflight, "write_result", lambda payload: payload)
    monkeypatch.setattr(leak_preflight, "record_preflight", lambda payload: None)

    assert cli.main(["--preflight"]) == 0
    assert '"status": "passed"' in capsys.readouterr().out
