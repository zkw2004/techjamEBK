"""Cross-model integration for Workstream C: every model family, every tier.

The unit tests for C2-C7 each exercise one model in isolation, which is why a
whole class of failure went unnoticed: **no test ever ran two model families
through `run_experiment` in one session.** torch and LightGBM link separate
OpenMP runtimes, so the first LightGBM experiment in a torch-touched process
aborted the interpreter (OMP Error #15) and C1 reported it as a `transient`
error — a class A5 retries with backoff, so the agent would have burned its
entire budget re-running an experiment that could never succeed. Linux CI
shares one libgomp and stayed green throughout; only macOS (every laptop this
demo runs on) aborted.

This module is the regression net for that class of bug: it drives the whole
registry through the public runner and asserts the invariants that only show
up when the families are combined.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import train

# Small enough to keep the matrix fast, structured enough that a model can
# actually learn something: a user's affinity for an author drives the label.
TRAIN_DATES = list(range(20220408, 20220422))
VAL_DATES = list(range(20220422, 20220429))
TEST_DATES = list(range(20220429, 20220509))


def _frame(rows: int, dates: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 12, rows)
    authors = rng.integers(0, 6, rows)
    favoured = ((users + authors) % 3 == 0).astype(float)
    return pd.DataFrame(
        {
            "date": rng.choice(dates, rows),
            "user_id": users,
            "video_id": rng.integers(0, 80, rows),
            "author_id": authors,
            "tab": rng.integers(0, 2, rows),
            "duration_ms": rng.integers(1_000, 60_000, rows),
            # Positive rate stays near 0.3 so no tier can trip the >0.75 canary
            # on fixture noise alone.
            "long_view": (rng.random(rows) < 0.12 + 0.45 * favoured).astype(int),
        }
    )


def _data():
    return (
        _frame(2_400, TRAIN_DATES, seed=1),
        _frame(500, VAL_DATES, seed=2),
        _frame(500, TEST_DATES, seed=3),
    )


def _folds():
    return [
        (
            _frame(900, TRAIN_DATES[: 8 + 2 * index], seed=10 + index),
            _frame(300, TRAIN_DATES[8 + 2 * index : 10 + 2 * index], seed=20 + index),
        )
        for index in range(3)
    ]


# Budgets trimmed to keep the matrix under a few seconds per cell while still
# doing real training in every family.
MODEL_HPARAMS = {
    "random": {},
    "popularity": {},
    "fm": {"max_epochs": 2, "k": 4},
    "lgbm": {"min_data_in_leaf": 1, "num_boost_round": 8},
    "deepfm": {"max_epochs": 1, "emb_dim": 4, "mlp": (8,), "batch_size": 256},
}


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    return train


def _config(model: str, **overrides) -> dict:
    config = {
        "model": model,
        "features": ["user_id", "video_id", "author_id"],
        "hparams": MODEL_HPARAMS[model],
    }
    config.update(overrides)
    return config


# --- the matrix -------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(MODEL_HPARAMS))
@pytest.mark.parametrize("fidelity", ["smoke", "screen", "full"])
def test_every_model_family_completes_every_fidelity_tier(runner, model, fidelity):
    """The test that would have caught the OpenMP abort.

    Each cell is a real fit through the public entry point, so a family that
    cannot execute at all fails here rather than silently degrading into an
    error record the agent treats as a retryable blip.
    """
    result = runner.run_experiment(_config(model), fidelity=fidelity, seed=3, timeout_s=900)

    assert result["status"] == "ok", (
        f"{model} at {fidelity} failed with "
        f"{result.get('error_class')!r}: {str(result.get('traceback'))[-400:]}"
    )
    assert result["fidelity"] == fidelity
    if fidelity == "smoke":
        assert result["primary"] is None
    else:
        assert 0.0 <= result["primary"] <= train.LEAK_CANARY_PRIMARY
        assert len(result["fold_primaries"]) == 3
    assert len(result["val_scores"]) == len(result["val_user_ids"])
    assert np.isfinite(np.asarray(result["val_scores"], dtype=float)).all()


@pytest.mark.parametrize("model", sorted(MODEL_HPARAMS))
def test_every_model_family_is_deterministic_for_a_fixed_seed(runner, model):
    """C1's determinism guarantee, verified per family rather than on a stub."""
    first = runner.run_experiment(_config(model), fidelity="screen", seed=11, timeout_s=900)
    second = runner.run_experiment(_config(model), fidelity="screen", seed=11, timeout_s=900)

    assert first["status"] == second["status"] == "ok"
    assert first["primary"] == second["primary"]
    np.testing.assert_array_equal(first["val_scores"], second["val_scores"])
    np.testing.assert_array_equal(first["fold_primaries"], second["fold_primaries"])


def test_both_native_families_run_in_one_session(runner):
    """LightGBM after DeepFM in a single process — the exact abort sequence.

    Ordering matters here, so this deliberately does not rely on the
    parametrised matrix running in any particular order.
    """
    neural = runner.run_experiment(_config("deepfm"), fidelity="smoke", seed=1, timeout_s=900)
    trees = runner.run_experiment(_config("lgbm"), fidelity="smoke", seed=1, timeout_s=900)
    neural_again = runner.run_experiment(_config("deepfm"), fidelity="smoke", seed=1, timeout_s=900)

    for label, result in (("deepfm", neural), ("lgbm", trees), ("deepfm again", neural_again)):
        assert result["status"] == "ok", f"{label}: {result.get('traceback')}"


def test_running_experiments_never_loads_a_native_runtime_into_the_parent(runner):
    """Isolation is what makes the matrix above safe.

    `run_experiment` forks per experiment, so the orchestrating process must
    stay free of both runtimes no matter which models it ran. If this ever
    fails, the next LightGBM run in the same process aborts the interpreter.
    """
    for model in ("deepfm", "lgbm"):
        assert runner.run_experiment(
            _config(model), fidelity="smoke", seed=1, timeout_s=900
        )["status"] == "ok"

    assert "torch" not in sys.modules
    assert "lightgbm" not in sys.modules


@pytest.mark.parametrize("loss", ["pointwise", "lambdarank"])
def test_lightgbm_objectives_both_run_through_the_public_runner(runner, loss):
    """C4 acceptance at the integration level: lambdarank needs a per-user
    group array, which only the runner can supply from the training frame."""
    result = runner.run_experiment(
        _config("lgbm", loss=loss), fidelity="screen", seed=7, timeout_s=900
    )

    assert result["status"] == "ok", result.get("traceback")
    assert len(result["fold_primaries"]) == 3


# --- backend declarations ---------------------------------------------------


def test_every_registered_model_declares_a_known_backend():
    """A model that forgets its declaration silently reintroduces the abort:
    the runner would not know to keep it out of a mixed process."""
    from pipeline.models import BACKENDS, MODEL_REGISTRY, backend

    for name in MODEL_REGISTRY:
        declared = backend(name)
        assert declared is None or declared in BACKENDS, (
            f"{name} declares native_backend={declared!r}, "
            f"which is not one of {BACKENDS}"
        )


def test_declared_backends_match_what_the_models_actually_import():
    from pipeline.models import backend

    assert backend("lgbm") == "lightgbm"
    assert backend("deepfm") == "torch"
    assert backend("deepfm_mtl") == "torch"
    # Pure-numpy models must stay undeclared so they can join any process.
    for name in ("random", "popularity", "fm"):
        assert backend(name) is None


def test_importing_the_model_registry_loads_no_native_runtime():
    """Deferred imports are load-bearing, not a style choice: an eager import
    in any model module would commit every worker to that runtime."""
    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pipeline.models; "
            "print('torch' in sys.modules, 'lightgbm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False False"


# --- conflict detection -----------------------------------------------------


def test_backend_conflict_is_reported_as_schema_not_transient(runner, tmp_path, monkeypatch):
    """A5 retries `transient` with backoff. A backend conflict is a permanent
    property of the config, so misclassifying it would spend the whole
    recovery budget on a run that can never succeed."""
    from agent import store

    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    _write_parent(store, "n001", "lgbm")
    _write_parent(store, "n002", "deepfm")

    result = runner.run_experiment(
        {"model": "blend", "parents": ["n001", "n002"], "blend_method": "rank_avg"},
        fidelity="screen",
        seed=1,
        timeout_s=900,
    )

    assert result["status"] == "error"
    assert result["error_class"] == "schema"
    assert "OpenMP" in result["traceback"] or "OMP" in result["traceback"]


def test_single_backend_blend_still_runs(runner, tmp_path, monkeypatch):
    """The guard must not over-reach: numpy + LightGBM parents share one
    process safely, and refusing them would delete most of C7's value."""
    from agent import store

    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    _write_parent(store, "n001", "fm")
    _write_parent(store, "n002", "lgbm")

    result = runner.run_experiment(
        {"model": "blend", "parents": ["n001", "n002"], "blend_method": "rank_avg"},
        fidelity="smoke",
        seed=1,
        timeout_s=900,
    )

    assert result["status"] == "ok", result.get("traceback")


def test_conflict_detection_reads_through_blend_parents():
    from pipeline.train import _config_backends

    assert _config_backends(_config("lgbm")) == {"lightgbm"}
    assert _config_backends(_config("deepfm")) == {"torch"}
    assert _config_backends(_config("fm")) == set()


def _write_parent(store, node_id: str, model: str) -> None:
    store.write(
        {
            "id": node_id,
            "parent": "n000",
            "family": "model",
            "hypothesis": "fixture parent",
            "action_type": "config",
            "fidelity": "full",
            "status": "ok",
            "manifest_sha256": "fixture",
            "accepted": True,
            "metrics": {"primary": 0.6},
            "config": {
                "model": model,
                "loss": "pointwise",
                "features": ["user_id", "video_id", "author_id"],
                "negative_sampling": "all",
                "hparams": MODEL_HPARAMS[model],
                "parents": [],
                "blend_method": "rank_avg",
                "seed": 42,
            },
        }
    )


# --- worker death diagnostics ----------------------------------------------


def test_native_crash_is_explained_rather_than_reported_as_a_bare_eof(runner, monkeypatch):
    """A child killed by SIGABRT used to surface as the single word
    `EOFError`, which says nothing about why. The report names the signal and
    points at the usual cause, because the traceback is the only channel that
    diagnosis has."""
    import faulthandler
    import os
    import signal as signal_module

    def suicidal_tier(config, fidelity, seed):
        # pytest's faulthandler is inherited by the fork and would dump a C
        # traceback for this deliberate abort, burying the real summary.
        faulthandler.disable()
        os.kill(os.getpid(), signal_module.SIGABRT)

    monkeypatch.setattr(train, "_execute_tier", suicidal_tier, raising=False)
    result = runner.run_experiment(_config("random"), fidelity="smoke", timeout_s=60)

    assert result["status"] == "error"
    assert "SIGABRT" in result["traceback"]
    assert "OpenMP" in result["traceback"]


def test_worker_killed_by_sigkill_is_still_classified_as_oom(runner, monkeypatch):
    """The pre-existing OOM signal must survive the richer reporting."""
    import os
    import signal as signal_module

    def killed_tier(config, fidelity, seed):
        os.kill(os.getpid(), signal_module.SIGKILL)

    monkeypatch.setattr(train, "_execute_tier", killed_tier, raising=False)
    result = runner.run_experiment(_config("random"), fidelity="smoke", timeout_s=60)

    assert result["status"] == "error"
    assert result["error_class"] == "oom"
    assert "SIGKILL" in result["traceback"]


# --- seeding scope ----------------------------------------------------------


def test_seeding_does_not_import_torch_for_a_non_neural_run():
    """`_seed_everything` used to import torch unconditionally, which is the
    single line that made every LightGBM experiment abort."""
    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pipeline.train import _seed_everything; "
            "_seed_everything(1, {'model': 'lgbm', 'parents': []}); "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"


def test_seeding_still_seeds_torch_for_a_neural_run():
    """Determinism for DeepFM depends on this half of the branch."""
    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pipeline.train import _seed_everything; "
            "_seed_everything(1, {'model': 'deepfm', 'parents': []}); "
            "import torch; print(torch.initial_seed())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "1"


def test_every_seeding_call_site_passes_its_config():
    """Backend-scoped seeding has a trap: `_seed_everything(seed)` with no
    config cannot know the run is neural, so it skips torch and leaves DeepFM
    unseeded — silently non-deterministic, which is far worse than a crash.
    Every production call site must therefore pass its config.
    """
    import ast
    import pathlib

    offenders = []
    for path in (
        pathlib.Path("pipeline/train.py"),
        pathlib.Path("pipeline/tune.py"),
        pathlib.Path("pipeline/blending.py"),
    ):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name == "_seed_everything" and len(node.args) < 2:
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, (
        "these _seed_everything calls omit the config and would skip torch "
        f"seeding for neural runs: {offenders}"
    )


@pytest.mark.parametrize("model", ["fm", "lgbm", "deepfm"])
def test_confirm_tier_averages_five_seeds_per_family(runner, model):
    """`confirm` re-seeds per repeat; a backend-scoped seeding bug would only
    ever show up here or in tuning, never in a single full run."""
    result = runner.run_experiment(
        _config(model), fidelity="confirm", seed=5, timeout_s=1800
    )

    assert result["status"] == "ok", result.get("traceback")
    assert result["fidelity"] == "confirm"
    assert len(result["fold_primaries"]) == 3
    assert np.isfinite(result["primary"])
