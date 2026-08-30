"""C2/C3 acceptance: the reference rungs reproduce their published numbers.

C2 IS A HARD GATE — if random and popularity do not reproduce 0.4753 and
0.5715, nothing downstream means anything.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.data import DATA_DIR, FIELDS, LABEL
from pipeline.models import MODEL_REGISTRY
from pipeline.models.deepfm import DeepFMModel
from pipeline.models.fm import BASELINE_VALIDATION_PRIMARY, FM
from pipeline.models.lgbm import LGBM
from pipeline.models.popularity import (
    POPULARITY_REFERENCE,
    RANDOM_REFERENCE,
    PopularityModel,
    RandomModel,
)

EXPECTED_MODELS = {"random", "popularity", "fm", "lgbm", "deepfm", "deepfm_mtl", "blend"}


def test_registry_covers_every_model_in_the_config_schema():
    from agent.schema import Config
    schema_models = set(Config.model_fields["model"].annotation.__args__)
    assert schema_models == EXPECTED_MODELS
    assert EXPECTED_MODELS <= set(MODEL_REGISTRY)


def test_unknown_model_lookup_raises_before_training():
    import pytest

    from pipeline.models import get
    with pytest.raises(KeyError):
        get("wide_and_deep")


def test_random_model_matches_seeded_numpy_generator():
    model = RandomModel(seed=7)
    X = np.zeros((4, 2))

    model.fit(X, np.array([0, 1, 0, 1]), None, None)
    scores = model.predict(X)

    np.testing.assert_array_equal(scores, np.random.default_rng(7).random(4))


def test_popularity_model_applies_prior_and_global_fallback():
    model = PopularityModel(prior=2.0, item_col=1)
    X_train = np.array(
        [[0, "a"], [1, "a"], [2, "b"], [3, "c"]],
        dtype=object,
    )
    y_train = np.array([1, 0, 0, 1])
    X_target = np.array([[9, "a"], [9, "b"], [9, "c"], [9, "unseen"]], dtype=object)

    model.fit(X_train, y_train, None, None)
    scores = model.predict(X_target)

    np.testing.assert_allclose(scores, [0.5, 1 / 3, 2 / 3, 0.5])


@pytest.mark.parametrize("model_name", ["random", "popularity"])
def test_reference_model_runs_through_c1_smoke(monkeypatch, model_name):
    import pipeline.train as train

    def frame(offset):
        index = np.arange(8) + offset
        return {
            "user_id": index // 2,
            "video_id": index % 3,
            "long_view": index % 2,
        }

    monkeypatch.setattr(train, "_load_data", lambda: (frame(0), frame(20), frame(40)))

    result = train.run_experiment({"model": model_name}, fidelity="smoke", seed=5)

    assert result["status"] == "ok"
    assert np.isfinite(result["val_scores"]).all()
    assert np.isfinite(result["test_scores"]).all()


@lru_cache(maxsize=1)
def _reference_frames():
    from pipeline.data import load

    train, validation, _ = load()
    return train, validation


def _reference_matrix(frame):
    return frame[["user_id", "video_id"]].to_numpy()


requires_kuairand_data = pytest.mark.skipif(
    not (DATA_DIR / "video_features_basic_pure.csv").is_file(),
    reason="KuaiRand-Pure dataset is not installed",
)


@requires_kuairand_data
def test_random_reproduces_reference_primary():
    """Mean validation score over starter-kit seeds 0-4. GATE."""
    from pipeline.data import LABEL
    from pipeline.evaluate import evaluate

    _, validation = _reference_frames()
    X_validation = _reference_matrix(validation)
    labels = validation[LABEL].to_numpy()
    users = validation["user_id"].to_numpy()
    primaries = []
    for seed in range(5):
        scores = RandomModel(seed=seed).predict(X_validation)
        primaries.append(evaluate(users, labels, scores)["primary"])

    assert np.mean(primaries) == pytest.approx(RANDOM_REFERENCE["valid"], abs=0.0008)


@requires_kuairand_data
def test_popularity_reproduces_reference_primary():
    """Prior-20 item popularity on official validation. GATE."""
    from pipeline.data import LABEL
    from pipeline.evaluate import evaluate

    train, validation = _reference_frames()
    model = PopularityModel(prior=20.0, item_col=1)
    model.fit(_reference_matrix(train), train[LABEL].to_numpy(), None, None)
    scores = model.predict(_reference_matrix(validation))
    metrics = evaluate(validation["user_id"].to_numpy(), validation[LABEL].to_numpy(), scores)

    assert metrics["primary"] == pytest.approx(POPULARITY_REFERENCE["valid"], abs=0.0008)


def test_fm_learns_to_rank_repeated_positive_pattern_above_negative_pattern():
    positive = ["u1", "v1", "a1", "home", "1"]
    negative = ["u2", "v2", "a2", "search", "8"]
    X_train = np.asarray([positive] * 20 + [negative] * 20, dtype=object)
    y_train = np.asarray([1] * 20 + [0] * 20, dtype=float)
    model = FM(k=4, lr=0.05, max_epochs=20, batch_size=8, patience=4, seed=3)

    model.fit(X_train, y_train, None, None)
    scores = model.predict(np.asarray([positive, negative], dtype=object))

    assert scores[0] > scores[1]


def test_fm_training_is_deterministic_and_unseen_categories_are_supported():
    X_train = np.asarray(
        [
            ["u1", "v1", "a1", "home", "1"],
            ["u1", "v2", "a2", "home", "2"],
            ["u2", "v1", "a1", "search", "1"],
            ["u2", "v2", "a2", "search", "2"],
        ],
        dtype=object,
    )
    y_train = np.asarray([1, 0, 0, 1], dtype=float)
    target = np.asarray([["new-user", "new-video", "new-author", "new-tab", "9"]])
    models = [FM(k=2, lr=0.01, max_epochs=2, batch_size=2, seed=11) for _ in range(2)]

    for model in models:
        model.fit(X_train, y_train, None, None)

    first = models[0].predict(target)
    second = models[1].predict(target)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1,)
    assert np.isfinite(first).all()


def test_fm_accepts_runner_train_and_validation_group_tuple():
    X_train = np.asarray(
        [["u1", "v1"], ["u1", "v2"], ["u2", "v1"], ["u2", "v2"]],
        dtype=object,
    )
    y_train = np.asarray([1, 0, 0, 1], dtype=float)
    model = FM(k=2, max_epochs=1, batch_size=2, seed=5)

    model.fit(
        X_train,
        y_train,
        X_train,
        y_train,
        groups=(X_train[:, 0], X_train[:, 0]),
    )

    assert np.isfinite(model.predict(X_train)).all()


@pytest.mark.native_backend("lightgbm")
@pytest.mark.parametrize("loss", ["pointwise", "lambdarank"])
def test_lgbm_objectives_learn_with_categorical_features(loss):
    X_train = np.asarray(
        [["u1", "good"], ["u1", "bad"], ["u2", "good"], ["u2", "bad"]] * 8,
        dtype=object,
    )
    y_train = np.asarray([1, 0, 1, 0] * 8)
    groups = {
        "train": np.asarray(["u1", "u1", "u2", "u2"] * 8),
        "val": np.asarray(["u1", "u1", "u2", "u2"]),
    }
    X_val = np.asarray(
        [["u1", "good"], ["u1", "bad"], ["u2", "good"], ["u2", "bad"]],
        dtype=object,
    )
    y_val = np.asarray([1, 0, 1, 0])
    model = LGBM(loss=loss, n_estimators=30, min_data_in_leaf=1, seed=9)

    model.fit(X_train, y_train, X_val, y_val, groups=groups)
    scores = model.predict(X_val)

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    assert model.best_epoch >= 1
    assert np.isfinite(model.predict(np.asarray([["new-user", "unseen"]], dtype=object))).all()


def test_lambdarank_group_sizes_cover_each_row_once():
    sizes = LGBM._group_sizes(np.asarray(["u2", "u1", "u2", "u3", "u1"]))
    np.testing.assert_array_equal(sizes, [2, 2, 1])
    assert sizes.sum() == 5


@pytest.mark.native_backend("torch")
def test_deepfm_learns_a_repeated_pattern_and_handles_unknown_categories():
    positive = ["u1", "v-good", "home"]
    negative = ["u2", "v-bad", "search"]
    X_train = np.asarray([positive] * 24 + [negative] * 24, dtype=object)
    y_train = np.asarray([1] * 24 + [0] * 24, dtype=np.float32)
    model = DeepFMModel(
        emb_dim=4, mlp=(16, 8), dropout=0.0, seed=4,
        lr=0.02, max_epochs=8, batch_size=16, patience=1,
    )

    model.fit(X_train, y_train, X_train, y_train)
    scores = model.predict(np.asarray([positive, negative, ["new", "new", "new"]]))

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
    assert 1 <= model.best_epoch <= 8


@pytest.mark.native_backend("torch")
def test_deepfm_is_deterministic_for_a_fixed_seed():
    X = np.asarray([["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"]] * 3)
    y = np.asarray([1, 0, 0, 1] * 3, dtype=np.float32)
    models = [
        DeepFMModel(emb_dim=2, mlp=(4,), dropout=0.0, seed=12,
                    max_epochs=2, batch_size=4)
        for _ in range(2)
    ]
    for model in models:
        model.fit(X, y, None, None)
    np.testing.assert_array_equal(models[0].predict(X), models[1].predict(X))


@requires_kuairand_data
def test_fm_reproduces_baseline_validation_primary():
    """0.6016 within one seed-std (0.0008)."""
    from pipeline.evaluate import evaluate
    from pipeline.train import _matrix

    train, validation = _reference_frames()
    model = FM(seed=0)
    model.fit(
        _matrix(train, train, FIELDS),
        train[LABEL].to_numpy(),
        _matrix(train, validation, FIELDS),
        validation[LABEL].to_numpy(),
    )
    scores = model.predict(_matrix(train, validation, FIELDS))
    metrics = evaluate(validation["user_id"].to_numpy(), validation[LABEL].to_numpy(), scores)

    assert metrics["primary"] == pytest.approx(BASELINE_VALIDATION_PRIMARY, abs=0.0008)


def _successful_tier(config, fidelity, seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(4)
    metrics = (None, None, None) if fidelity == "smoke" else (0.60, 0.58, 0.59)
    fold_primaries = [] if fidelity == "smoke" else [0.57, 0.58, 0.59]
    return {
        "status": "ok",
        "fidelity": fidelity,
        "gauc": metrics[0],
        "ndcg": metrics[1],
        "primary": metrics[2],
        "fold_primaries": fold_primaries,
        "segments": {},
        "val_scores": scores,
        "val_user_ids": np.array([1, 1, 2, 2]),
        "test_scores": scores.copy(),
        "gpu_seconds": 0.0,
        "peak_rss_mb": 1.0,
    }


def _raising_tier(config, fidelity, seed):
    raise RuntimeError("controlled training failure")


def _hanging_tier(config, fidelity, seed):
    time.sleep(2)
    return _successful_tier(config, fidelity, seed)


def _syntax_tier(config, fidelity, seed):
    raise SyntaxError("invalid generated feature")


def _oom_tier(config, fidelity, seed):
    raise MemoryError("controlled allocation failure")


def _malformed_tier(config, fidelity, seed):
    return None


def _nonfinite_tier(config, fidelity, seed):
    result = _successful_tier(config, fidelity, seed)
    result["val_scores"] = np.array([0.1, np.nan, 0.3, 0.4])
    return result


def _nonfinite_telemetry_tier(config, fidelity, seed):
    result = _successful_tier(config, fidelity, seed)
    result["gpu_seconds"] = np.nan
    return result


def _wrong_fold_count_tier(config, fidelity, seed):
    result = _successful_tier(config, fidelity, seed)
    result["fold_primaries"] = [0.57, 0.58]
    return result


def _ignores_termination_tier(config, fidelity, seed):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(2)
    return _successful_tier(config, fidelity, seed)


def _killed_tier(config, fidelity, seed):
    os.kill(os.getpid(), signal.SIGKILL)


def _sometimes_leaky_full(config, seed):
    primary = 0.80 if seed == 11 else 0.50
    return {
        "status": "ok",
        "fidelity": "full",
        "gauc": primary,
        "ndcg": primary,
        "primary": primary,
        "fold_primaries": [primary, primary, primary],
        "segments": {},
        "val_scores": np.full(2, float(seed)),
        "val_user_ids": np.array([1, 1]),
        "test_scores": np.full(2, float(seed)),
        "gpu_seconds": 0.0,
        "peak_rss_mb": 1.0,
    }


def test_run_experiment_is_deterministic_given_seed(monkeypatch):
    """Removing seed propagation would make the score arrays differ."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _successful_tier, raising=False)

    first = train.run_experiment({"model": "random"}, fidelity="full", seed=17)
    second = train.run_experiment({"model": "random"}, fidelity="full", seed=17)

    assert first["status"] == second["status"] == "ok"
    assert first["primary"] == second["primary"]
    np.testing.assert_array_equal(first["val_scores"], second["val_scores"])
    np.testing.assert_array_equal(first["test_scores"], second["test_scores"])


def test_seed_setup_enables_torch_deterministic_execution(monkeypatch):
    import pipeline.train as train

    calls = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        use_deterministic_algorithms=lambda enabled: calls.append(("deterministic", enabled)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    train._seed_everything(23)

    assert ("cpu", 23) in calls
    assert ("cuda", 23) in calls
    assert ("deterministic", True) in calls
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_run_experiment_never_raises(monkeypatch):
    """An unexpected training exception must not escape into the agent loop."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _raising_tier, raising=False)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["stage"] == "full"
    assert result["error_class"] == "transient"
    assert "controlled training failure" in result["traceback"]


@pytest.mark.parametrize(
    ("tier", "expected_class"),
    [(_syntax_tier, "syntax"), (_oom_tier, "oom")],
)
def test_run_experiment_classifies_worker_failures(monkeypatch, tier, expected_class):
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", tier)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["error_class"] == expected_class


def test_malformed_worker_result_returns_schema_error(monkeypatch):
    """A broken tier must not make the public result handling raise."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _malformed_tier)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


def test_nonfinite_scores_return_schema_error(monkeypatch):
    """Removing the result-envelope check would allow NaN predictions through."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _nonfinite_tier)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


@pytest.mark.parametrize("tier", [_nonfinite_telemetry_tier, _wrong_fold_count_tier])
def test_invalid_success_metadata_returns_schema_error(monkeypatch, tier):
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", tier)

    result = train.run_experiment({"model": "random"}, fidelity="full")

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


def test_process_setup_failure_never_raises(monkeypatch):
    import pipeline.train as train

    def fail_to_create_context(*_args):
        raise MemoryError("cannot allocate process context")

    monkeypatch.setattr(train, "_process_context", fail_to_create_context)

    result = train.run_experiment({"model": "random"})

    assert result["status"] == "error"
    assert result["error_class"] == "oom"


def test_sigkill_worker_is_classified_as_oom(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _killed_tier)

    result = train.run_experiment({"model": "random"}, timeout_s=1)

    assert result["status"] == "error"
    assert result["error_class"] == "oom"


def test_invalid_config_returns_schema_error_instead_of_raising():
    from pipeline.train import run_experiment

    result = run_experiment({"model": "not-a-model"})

    assert result["status"] == "error"
    assert result["stage"] == "validation"
    assert result["error_class"] == "schema"


def test_run_experiment_enforces_timeout(monkeypatch):
    """Removing process termination would let this call run for two seconds."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _hanging_tier, raising=False)

    started = time.monotonic()
    result = train.run_experiment({"model": "random"}, fidelity="smoke", timeout_s=0.05)

    assert result["status"] == "error"
    assert result["stage"] == "smoke"
    assert result["error_class"] == "timeout"
    assert time.monotonic() - started < 1.0


def test_timeout_kills_worker_that_ignores_sigterm(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _ignores_termination_tier)

    started = time.monotonic()
    result = train.run_experiment({"model": "random"}, fidelity="smoke", timeout_s=0.05)

    assert result["status"] == "error"
    assert result["error_class"] == "timeout"
    assert time.monotonic() - started < 1.0


def test_confirm_rejects_any_individually_leaky_seed(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_run_full", _sometimes_leaky_full)

    result = train.run_experiment({"model": "random"}, fidelity="confirm", seed=10)

    assert result["status"] == "error"
    assert result["stage"] == "leakage"
    assert result["error_class"] == "leak_suspected"


def test_smoke_tier_completes_under_ten_seconds(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_execute_tier", _successful_tier, raising=False)

    started = time.monotonic()
    result = train.run_experiment({"model": "random"}, fidelity="smoke")

    assert result["status"] == "ok"
    assert result["fidelity"] == "smoke"
    assert time.monotonic() - started < 10.0
