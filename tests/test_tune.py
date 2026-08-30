"""C6 Optuna TPE, pruning, and SQLite resume behavior."""

from __future__ import annotations

import numpy as np
import optuna
import pytest


def test_suggest_translates_supported_action_search_specs():
    from pipeline.tune import _suggest

    trial = optuna.trial.FixedTrial({"lr": 0.01, "emb_dim": 16, "num_leaves": 63, "dropout": 0.2})

    assert _suggest(trial, "lr", ["loguniform", 1e-4, 1e-1]) == 0.01
    assert _suggest(trial, "emb_dim", ["categorical", [8, 16, 32]]) == 16
    assert _suggest(trial, "num_leaves", ["int", 31, 255]) == 63
    assert _suggest(trial, "dropout", ["uniform", 0.0, 0.5]) == 0.2


def test_run_study_resumes_existing_sqlite_budget(monkeypatch, tmp_path):
    import pipeline.tune as tune

    def fold_primaries(config, seed, trial):
        value = float(config["hparams"]["num_leaves"]) / 20
        for step in range(3):
            trial.report(value, step)
        return [value, value, value]

    monkeypatch.setattr(tune, "_trial_fold_primaries", fold_primaries)
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    arguments = {
        "base_config": {"model": "lgbm"},
        "search_space": {"num_leaves": ["int", 3, 9]},
        "seed": 7,
        "storage": storage,
        "study_name": "resume-test",
    }

    first = tune.run_study(**arguments, budget=2)
    resumed = tune.run_study(**arguments, budget=4)

    assert len(first["trials"]) == 2
    assert len(resumed["trials"]) == 4
    assert resumed["best_params"]["num_leaves"] in range(3, 10)
    assert (tmp_path / "resume.db").is_file()


def test_run_study_records_pruning_and_contains_failed_trials(monkeypatch, tmp_path):
    import pipeline.tune as tune

    def fold_primaries(config, seed, trial):
        if trial.number == 0:
            raise RuntimeError("controlled failed trial")
        value = 0.65 if trial.number < 5 else 0.45
        for step in range(3):
            trial.report(value, step)
            if trial.should_prune():
                trial.set_user_attr("seconds_saved", float(3 - step - 1))
                raise optuna.TrialPruned()
        return [value, value, value]

    monkeypatch.setattr(tune, "_trial_fold_primaries", fold_primaries)

    result = tune.run_study(
        {"model": "lgbm"},
        {"num_leaves": ["int", 3, 9]},
        budget=8,
        seed=3,
        storage=f"sqlite:///{tmp_path / 'prune.db'}",
        study_name="prune-test",
    )

    assert len(result["trials"]) == 8
    assert result["n_pruned"] >= 1
    assert result["seconds_saved"] > 0
    assert any(trial["error"] == "controlled failed trial" for trial in result["trials"])


def test_invalid_search_spec_is_contained_as_trial_evidence(tmp_path):
    from pipeline.tune import run_study

    result = run_study(
        {"model": "lgbm"},
        {"num_leaves": ["unsupported", 3, 9]},
        budget=2,
        storage=f"sqlite:///{tmp_path / 'invalid.db'}",
    )

    assert len(result["trials"]) == 2
    assert all("unsupported search spec" in trial["error"] for trial in result["trials"])
    assert result["best_value"] is None


def test_resume_refuses_a_different_experiment_under_the_same_study_name(monkeypatch, tmp_path):
    from pipeline import tune

    monkeypatch.setattr(tune, "_trial_fold_primaries", lambda *args: [0.5] * 3)
    storage = f"sqlite:///{tmp_path / 'identity.db'}"
    tune.run_study({"model": "random"}, {}, budget=1, storage=storage)

    with pytest.raises(ValueError, match="different experiment"):
        tune.run_study({"model": "fm"}, {}, budget=2, storage=storage)


def test_pruning_accounting_excludes_interruption_downtime(tmp_path):
    from datetime import datetime, timedelta

    from pipeline.tune import run_study

    storage = f"sqlite:///{tmp_path / 'accounting.db'}"
    study = optuna.create_study(storage=storage, study_name="accounting")
    for state, duration, attrs in [
        (optuna.trial.TrialState.PRUNED, 2, {"seconds_saved": 2.0}),
        (optuna.trial.TrialState.FAIL, 1000, {"interrupted": True}),
    ]:
        trial = optuna.trial.create_trial(state=state, user_attrs=attrs)
        trial.datetime_start = datetime(2026, 8, 30)
        trial.datetime_complete = trial.datetime_start + timedelta(seconds=duration)
        study.add_trial(trial)

    result = run_study({"model": "random"}, {}, budget=1, storage=storage, study_name="accounting")

    assert result["pruning_savings_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("values", [[float("nan")] * 3, [0.9] * 3])
def test_invalid_or_leak_suspected_scores_cannot_win_a_study(monkeypatch, tmp_path, values):
    import pipeline.tune as tune

    monkeypatch.setattr(tune, "_trial_fold_primaries", lambda *args: values)
    result = tune.run_study(
        {"model": "lgbm"},
        {},
        budget=1,
        storage=f"sqlite:///{tmp_path / 'scores.db'}",
    )

    assert result["best_value"] is None
    assert result["trials"][0]["error"]


def test_resume_replaces_orphan_running_trial_without_consuming_budget(monkeypatch, tmp_path):
    import pipeline.tune as tune

    storage = f"sqlite:///{tmp_path / 'orphan.db'}"
    study = optuna.create_study(storage=storage, study_name="orphan", direction="maximize")
    completed = study.ask()
    study.tell(completed, 0.5)
    study.ask()  # models the RUNNING state left by a killed owner
    monkeypatch.setattr(tune, "_trial_fold_primaries", lambda *args: [0.6] * 3)

    result = tune.run_study({"model": "random"}, {}, budget=2, storage=storage, study_name="orphan")

    assert all(trial["state"] != "running" for trial in result["trials"])
    assert sum(trial["state"] == "complete" for trial in result["trials"]) == 2
    assert result["n_interrupted"] == 1


@pytest.mark.parametrize("failure", ["hang", "crash"])
def test_trial_native_crash_and_hang_are_contained(monkeypatch, tmp_path, failure):
    import os
    import time

    from pipeline import train, tune

    def broken_fit(*args):
        if failure == "crash":
            os._exit(13)
        time.sleep(5)

    monkeypatch.setattr(train, "_seed_everything", lambda seed, config=None: None)
    monkeypatch.setattr(train, "_load_folds", lambda: [(None, None)] * 3)
    monkeypatch.setattr(train, "_fit_and_predict", broken_fit)
    started = time.monotonic()
    result = tune.run_study(
        {"model": "random", "hparams": {"trial_timeout_s": 0.1}},
        {},
        budget=1,
        storage=f"sqlite:///{tmp_path / 'contain.db'}",
    )

    assert time.monotonic() - started < 3
    assert result["trials"][0]["error"]
    assert result["best_value"] is None


def test_real_fold_loop_prunes_before_second_fit_without_official_data(monkeypatch, tmp_path):
    from pipeline import train, tune

    marker = tmp_path / "fits.txt"
    frame = {"user_id": np.zeros(6), "long_view": np.array([0, 1, 0, 0, 0, 0])}

    def fit(*args):
        with marker.open("a") as handle:
            handle.write("fit\n")
        return [np.arange(6, dtype=float)]

    def forbidden():
        raise AssertionError("official data requested")

    monkeypatch.setattr(train, "_seed_everything", lambda seed, config=None: None)
    monkeypatch.setattr(train, "_load_folds", lambda: [(frame, frame)] * 3)
    monkeypatch.setattr(train, "_load_data", forbidden)
    monkeypatch.setattr(train, "_fit_and_predict", fit)
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(lower=0.4))
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        tune._trial_fold_primaries({"model": "random", "hparams": {}}, 42, trial)

    assert marker.read_text() == "fit\n"
    assert trial.user_attrs["completed_folds"] == 1


def test_killed_owner_releases_lock_and_terminates_its_worker(monkeypatch, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import time

    from pipeline import tune

    marker = tmp_path / "worker.pid"
    storage = f"sqlite:///{tmp_path / 'killed.db'}"
    script = textwrap.dedent(f"""
        import os, time
        from pathlib import Path
        from pipeline import train, tune
        train._seed_everything = lambda seed, config=None: None
        train._load_folds = lambda: [(None, None)] * 3
        def fitting(*args):
            Path({str(marker)!r}).write_text(str(os.getpid()))
            time.sleep(60)
        train._fit_and_predict = fitting
        tune.run_study({{"model": "random"}}, {{}}, budget=1,
                       storage={storage!r}, study_name="kill-test")
    """)
    owner = subprocess.Popen([sys.executable, "-c", script])
    worker_pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "worker failed to begin"
        worker_pid = int(marker.read_text())
        owner.kill()
        owner.wait(timeout=5)
        monkeypatch.setattr(tune, "_trial_fold_primaries", lambda *args: [0.5] * 3)
        result = tune.run_study(
            {"model": "random"}, {}, budget=1, storage=storage, study_name="kill-test"
        )
        assert result["n_interrupted"] == 1
        assert result["best_value"] == 0.5
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                worker_pid = None
                break
            time.sleep(0.02)
        assert worker_pid is None, "killed owner left a live worker"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
