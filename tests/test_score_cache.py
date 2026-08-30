"""C7 deterministic parent-score cache behavior."""

from __future__ import annotations

import numpy as np
import pytest


def test_score_cache_key_is_canonical_and_seed_sensitive():
    from pipeline.train import _score_cache_key

    left = {"model": "fm", "hparams": {"lr": 0.1, "k": 8}}
    reordered = {"hparams": {"k": 8, "lr": 0.1}, "model": "fm"}

    assert _score_cache_key(left, "full", 4) == _score_cache_key(reordered, "full", 4)
    assert _score_cache_key(left, "full", 4) != _score_cache_key(left, "full", 5)


def test_score_cache_round_trips_without_changing_result_contract(monkeypatch, tmp_path):
    import pipeline.train as train

    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    result = {
        "val_scores": np.array([0.2, 0.8]),
        "val_user_ids": np.array([7, 7]),
        "test_scores": np.array([0.3]),
    }
    config = {"model": "fm"}

    path = train._write_score_cache(config, "full", 9, result)
    loaded = train._read_score_cache(config, "full", 9)

    assert path.is_file()
    np.testing.assert_array_equal(loaded["val_scores"], result["val_scores"])
    np.testing.assert_array_equal(loaded["val_user_ids"], result["val_user_ids"])
    np.testing.assert_array_equal(loaded["test_scores"], result["test_scores"])


def test_parent_resolution_rejects_unaccepted_nodes(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(
        "agent.store.read",
        lambda node_id: {"id": node_id, "status": "ok", "accepted": False, "config": {}},
    )

    with pytest.raises(ValueError, match="accepted"):
        train._parent_config("n007")


def test_cache_rejects_reordered_rows_even_when_user_ids_match(monkeypatch, tmp_path):
    import pipeline.train as train

    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    validation = {"user_id": np.array([7, 7]), "video_id": np.array([1, 2])}
    test = {"user_id": np.array([8]), "video_id": np.array([3])}
    result = {
        "val_scores": np.array([0.1, 0.8]),
        "val_user_ids": validation["user_id"],
        "test_scores": np.array([0.3]),
    }
    train._write_score_cache(
        {"model": "fm"}, "full", 9, result, validation_frame=validation, test_frame=test
    )
    reordered = {"user_id": np.array([7, 7]), "video_id": np.array([2, 1])}

    assert (
        train._read_score_cache(
            {"model": "fm"}, "full", 9, validation_frame=reordered, test_frame=test
        )
        is None
    )


def test_cache_key_changes_with_implementation_fingerprint(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_cache_code_fingerprint", lambda: "version-one")
    first = train._score_cache_key({"model": "fm"}, "full", 9)
    monkeypatch.setattr(train, "_cache_code_fingerprint", lambda: "version-two")
    second = train._score_cache_key({"model": "fm"}, "full", 9)

    assert first != second


@pytest.mark.parametrize("payload", [b"", b"PK\x03\x04truncated"])
def test_truncated_cache_is_a_miss_not_an_experiment_failure(monkeypatch, tmp_path, payload):
    import pipeline.train as train

    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    path = train._score_cache_path({"model": "fm"}, "full", 9)
    path.write_bytes(payload)

    assert train._read_score_cache({"model": "fm"}, "full", 9) is None


@pytest.mark.parametrize(
    ("fidelity", "action_type", "message"),
    [("confirm", "config", "full-fidelity"), ("full", "code", "source replay")],
)
def test_parent_resolution_refuses_fidelities_or_code_it_cannot_reproduce(
    monkeypatch,
    fidelity,
    action_type,
    message,
):
    import pipeline.train as train

    monkeypatch.setattr(
        "agent.store.read",
        lambda node_id: {
            "id": node_id,
            "status": "ok",
            "accepted": True,
            "fidelity": fidelity,
            "action_type": action_type,
            "config": {"model": "fm"},
        },
    )

    with pytest.raises(ValueError, match=message):
        train._parent_config("n007")
