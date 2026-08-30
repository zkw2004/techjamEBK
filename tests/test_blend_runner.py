"""C7 node-based ensembles through the frozen C1 entry point."""

import numpy as np
import pytest

from agent.schema import Config


class ParentScores:
    def __init__(self, seed=42, **kwargs):
        self.direction = kwargs.get("direction", 1)

    def fit(self, X_train, y_train, X_val, y_val, groups=None):
        pass

    def predict(self, X):
        return self.direction * np.asarray(X[:, 1], dtype=float)


def frame(offset=0):
    return {
        "user_id": np.repeat(np.arange(4) + offset, 6),
        "video_id": np.tile(np.arange(6), 4),
        "long_view": np.tile([0, 1, 0, 0, 0, 0], 4),
    }


@pytest.fixture
def backend(monkeypatch, tmp_path):
    import pipeline.train as train

    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(train, "_get_model_class", lambda name: ParentScores)
    monkeypatch.setattr(train, "_load_data", lambda: (frame(), frame(20), frame(40)))
    monkeypatch.setattr(train, "_load_folds", lambda: [(frame(), frame(i * 10)) for i in range(3)])
    monkeypatch.setattr(
        train,
        "_parent_config",
        lambda node_id: Config(
            model="random",
            seed=1 if node_id == "n001" else 100,
            hparams={"direction": 1 if node_id == "n001" else -1},
        ).model_dump(),
    )
    return train


def test_full_blend_runs_deterministically_and_rejects_failure_to_beat_both_parents(backend):
    config = {"model": "blend", "parents": ["n001", "n002"]}
    first = backend.run_experiment(config, fidelity="full")
    second = backend.run_experiment(config, fidelity="full")

    assert first["status"] == "ok", first
    assert first["parent_correlation"] < 0.5
    assert first["blend_accepted"] is False
    assert len(first["fold_primaries"]) == 3
    np.testing.assert_array_equal(first["val_scores"], second["val_scores"])
    np.testing.assert_array_equal(first["test_scores"], second["test_scores"])


def test_weighted_screen_never_loads_official_validation(backend, monkeypatch):
    def forbidden():
        raise AssertionError("official data was loaded during weight fitting")

    monkeypatch.setattr(backend, "_load_data", forbidden)
    result = backend.run_experiment(
        {"model": "blend", "parents": ["n001", "n002"], "blend_method": "weighted_rank"},
        fidelity="screen",
    )

    assert result["status"] == "ok", result
    assert 0.1 <= result["blend_weight"] <= 0.9
    assert result["weight_source"] == "internal_folds"


def test_full_model_persists_parent_score_cache(backend):
    config = Config(model="random", seed=1).model_dump()
    result = backend.run_experiment(config, fidelity="full", seed=1)

    assert result["status"] == "ok", result
    cached = backend._read_score_cache(config, "full", 1)
    assert cached is not None
    np.testing.assert_array_equal(cached["val_scores"], result["val_scores"])


def test_confirm_blend_refits_each_parent_across_five_seeds(backend, monkeypatch):
    from pipeline import blending

    seeds = []
    original = blending._full_parent_scores

    def record_seed(parent, validation, test=None):
        seeds.append(parent["seed"])
        return original(parent, validation, test)

    monkeypatch.setattr(blending, "_full_parent_scores", record_seed)
    config = Config(model="blend", parents=["n001", "n002"]).model_dump()
    result = blending.run_blend(config, "confirm", 42)

    assert result["status"] == "ok"
    assert result["fidelity"] == "confirm"
    assert seeds == [1, 100, 2, 101, 3, 102, 4, 103, 5, 104]
    assert result["blend_accepted"] is False


@pytest.mark.parametrize(
    ("weak_fold", "official_primary", "bootstrap", "accepted"),
    [
        (False, 0.64, True, True),
        (True, 0.64, True, False),
        (False, 0.621, True, False),
        (False, 0.64, False, False),
    ],
)
def test_acceptance_requires_every_fold_official_margin_and_bootstrap(
    backend,
    monkeypatch,
    weak_fold,
    official_primary,
    bootstrap,
    accepted,
):
    from pipeline import blending

    first = np.tile(np.arange(6, dtype=float), 4)
    second = first[::-1].copy()
    folds = [0.64, 0.61 if weak_fold else 0.64, 0.64]
    evidence = {
        "metrics": [{"primary": value} for value in folds],
        "parent_fold_primaries": [[0.60] * 3, [0.62] * 3],
        "parent_correlation": 0.8,
        "blend_weight": 0.5,
        "weight_source": "fixed",
        "blend_warning": None,
    }
    monkeypatch.setattr(blending, "_fold_evidence", lambda *args: evidence)

    def parent_scores(parent, validation, test):
        scores = first if parent["seed"] == 1 else second
        return {"val_scores": scores, "test_scores": scores, "val_user_ids": validation["user_id"]}

    def evaluate(validation, scores):
        value = (
            0.60
            if np.array_equal(scores, first)
            else (0.62 if np.array_equal(scores, second) else official_primary)
        )
        return {"primary": value, "gauc": value, "ndcg": value}

    monkeypatch.setattr(blending, "_full_parent_scores", parent_scores)
    monkeypatch.setattr(backend, "_evaluate", evaluate)
    monkeypatch.setattr(
        "agent.gate.accept",
        lambda cand, best, users, seed: (
            bootstrap and np.array_equal(best, second),
            (0.001, 0.02),
        ),
    )
    config = Config(model="blend", parents=["n001", "n002"]).model_dump()
    parents = [backend._parent_config(node) for node in config["parents"]]

    result = blending._run_with_parents(config, "full", 42, parents)

    assert result["blend_accepted"] is accepted
