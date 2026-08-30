"""Optuna tuning over the frozen internal temporal folds.

The official validation split is intentionally absent from this module.  A
trial is evaluated fold-by-fold so the pruner can stop weak configurations
without paying for all three expanding-window fits.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

from agent.schema import Config

_ACTIVE_LOCK_FD: ContextVar[int | None] = ContextVar("c6_lock_fd", default=None)


def _watch_owner(owner_pid: int) -> None:
    while os.getppid() == owner_pid:
        time.sleep(0.1)
    os._exit(1)  # only runs in the disposable trial child after its owner dies


def _suggest(trial: optuna.trial.BaseTrial, name: str, spec: list[Any]) -> Any:
    """Translate an Action search-space entry into an Optuna suggestion."""
    if not isinstance(spec, list) or not spec:
        raise ValueError(f"search spec for {name!r} must be a non-empty list")

    kind = spec[0]
    if kind == "categorical" and len(spec) == 2 and isinstance(spec[1], list):
        if not spec[1]:
            raise ValueError(f"categorical search spec for {name!r} cannot be empty")
        return trial.suggest_categorical(name, spec[1])
    if kind == "int" and len(spec) == 3:
        return trial.suggest_int(name, int(spec[1]), int(spec[2]))
    if kind in {"uniform", "loguniform"} and len(spec) == 3:
        return trial.suggest_float(
            name,
            float(spec[1]),
            float(spec[2]),
            log=kind == "loguniform",
        )
    raise ValueError(f"unsupported search spec for {name!r}: {spec!r}")


def _fold_worker(
    connection,
    owner_connection,
    config: dict,
    seed: int,
    inherited_lock_fd: int | None,
    owner_pid: int,
) -> None:
    """Stream fold metrics, waiting for the owner's pruning decision each time."""
    from pipeline import train

    owner_connection.close()
    if inherited_lock_fd is not None:
        os.close(inherited_lock_fd)
    threading.Thread(target=_watch_owner, args=(owner_pid,), daemon=True).start()
    try:
        train._seed_everything(seed)
        folds = train._load_folds()
        if len(folds) != 3:
            raise ValueError("tuning requires exactly three internal folds")
        for index, (fold_train, fold_validation) in enumerate(folds):
            started = time.monotonic()
            (scores,) = train._fit_and_predict(
                config,
                fold_train,
                fold_validation,
                [fold_validation],
                seed + index,
            )
            primary = train._evaluate(fold_validation, scores)["primary"]
            connection.send(("fold", primary, time.monotonic() - started))
            if index < 2 and connection.recv() != "continue":
                return
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _trial_fold_primaries(config: dict[str, Any], seed: int, trial: optuna.Trial) -> list[float]:
    """Supervise a bounded child so native crashes and hangs cannot kill a study."""
    from pipeline import train

    config = train._screen_config(config)
    timeout = float(config["hparams"].pop("trial_timeout_s", 1800))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("trial_timeout_s must be finite and positive")
    context = train._process_context()
    owner, worker = context.Pipe(duplex=True)
    inherited_lock = _ACTIVE_LOCK_FD.get() if context.get_start_method() == "fork" else None
    process = context.Process(
        target=_fold_worker,
        args=(worker, owner, config, seed, inherited_lock, os.getpid()),
        daemon=True,
    )
    primaries, fold_seconds = [], []
    deadline = time.monotonic() + timeout
    try:
        process.start()
        worker.close()
        for index in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not owner.poll(remaining):
                raise TimeoutError(f"trial exceeded trial_timeout_s={timeout}")
            try:
                message = owner.recv()
            except (EOFError, OSError) as exc:
                process.join(train.PROCESS_STOP_GRACE_S)
                raise RuntimeError(
                    f"trial worker exited without a result ({process.exitcode})"
                ) from exc
            if message[0] == "error":
                raise RuntimeError(message[1])
            _, primary, seconds = message
            if not np.isfinite(primary):
                raise ValueError("fold primary must be finite")
            primaries.append(float(primary))
            fold_seconds.append(float(seconds))
            trial.set_user_attr("completed_folds", len(primaries))
            trial.set_user_attr("fold_seconds", fold_seconds)
            trial.report(float(np.mean(primaries)), step=index)
            if trial.should_prune():
                seconds_saved = float(np.mean(fold_seconds) * (3 - len(primaries)))
                trial.set_user_attr("seconds_saved", seconds_saved)
                raise optuna.TrialPruned()
            if index < 2:
                owner.send("continue")
        return primaries
    finally:
        train._stop_process(process)
        owner.close()
        worker.close()


def _sqlite_parent(storage: str) -> Path | None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return None
    raw_path = storage[len(prefix) :]
    return Path(raw_path).expanduser().parent


def _serialise_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    duration = None if trial.duration is None else trial.duration.total_seconds()
    return {
        "number": trial.number,
        "state": trial.state.name.lower(),
        "value": trial.value,
        "params": dict(trial.params),
        "error": trial.user_attrs.get("error"),
        "seconds_saved": float(trial.user_attrs.get("seconds_saved", 0.0)),
        "duration_seconds": duration,
    }


def _run_locked_study(
    base_config: dict[str, Any],
    search_space: dict[str, list[Any]],
    budget: int = 20,
    seed: int = 42,
    storage: str = "sqlite:///logs/optuna/c6.db",
    study_name: str = "c6",
) -> dict[str, Any]:
    """Run or resume a seeded study up to a total trial budget.

    Trial errors are contained and recorded with a sentinel objective value so
    one bad generated configuration cannot terminate the study.
    """
    if budget < 1:
        raise ValueError("budget must be at least one trial")

    config = Config(**base_config).model_dump()
    parent = _sqlite_parent(storage)
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    identity = json.dumps(
        {"config": config, "search_space": search_space, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    )
    previous_identity = study.user_attrs.get("experiment_identity")
    if previous_identity is not None and previous_identity != identity:
        raise ValueError("study name is already bound to a different experiment")
    study.set_user_attr("experiment_identity", identity)
    # The outer single-owner lock proves no local owner is still running these.
    for orphan in study.get_trials(states=(TrialState.RUNNING,)):
        trial = optuna.Trial(study, orphan._trial_id)
        trial.set_user_attr("interrupted", True)
        trial.set_user_attr("error", "owner interrupted before completing trial")
        study.tell(orphan.number, state=TrialState.FAIL)

    def objective(trial: optuna.Trial) -> float:
        try:
            trial_config = {
                **config,
                "hparams": {
                    **config["hparams"],
                    **{name: _suggest(trial, name, spec) for name, spec in search_space.items()},
                },
            }
            primaries = _trial_fold_primaries(trial_config, seed, trial)
            if len(primaries) != 3 or not np.isfinite(primaries).all():
                raise ValueError("tuning requires three finite fold primaries")
            value = float(np.mean(primaries))
            from pipeline.train import LEAK_CANARY_PRIMARY, LeakSuspectedError

            if value > LEAK_CANARY_PRIMARY:
                raise LeakSuspectedError(value)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # an invalid trial must not kill the study
            trial.set_user_attr("error", str(exc))
            return 0.0
        return value

    consumed = sum(not trial.user_attrs.get("interrupted", False) for trial in study.trials)
    remaining = max(0, budget - consumed)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)

    trials = study.trials
    completed = [
        trial
        for trial in trials
        if trial.state == TrialState.COMPLETE and not trial.user_attrs.get("error")
    ]
    best = max(completed, key=lambda trial: float(trial.value)) if completed else None
    pruned = [trial for trial in trials if trial.state == TrialState.PRUNED]
    seconds_saved = sum(float(trial.user_attrs.get("seconds_saved", 0.0)) for trial in pruned)
    elapsed = sum(
        trial.duration.total_seconds()
        for trial in trials
        if trial.duration is not None and not trial.user_attrs.get("interrupted")
    )

    return {
        "best_params": {} if best is None else dict(best.params),
        "best_value": None if best is None else float(best.value),
        "n_pruned": len(pruned),
        "n_interrupted": sum(bool(trial.user_attrs.get("interrupted")) for trial in trials),
        "seconds_saved": seconds_saved,
        "measured_trial_seconds": elapsed,
        "pruning_savings_fraction": (
            seconds_saved / (elapsed + seconds_saved) if elapsed + seconds_saved else 0.0
        ),
        "trials": [_serialise_trial(trial) for trial in trials],
    }


def run_study(
    base_config: dict[str, Any],
    search_space: dict[str, list[Any]],
    budget: int = 20,
    seed: int = 42,
    storage: str = "sqlite:///logs/optuna/c6.db",
    study_name: str = "c6",
) -> dict[str, Any]:
    """Run/resume a single-owner SQLite study with crash-safe trial accounting.

    `budget` counts completed/pruned/error attempts, but excludes owner-interrupted
    attempts. Model execution is isolated with `hparams.trial_timeout_s` (1800s
    default). Lock acquisition fails fast if another local owner is active.
    """
    parent = _sqlite_parent(storage)
    if parent is None:
        raise ValueError("C6 currently requires local sqlite:/// storage")
    parent.mkdir(parents=True, exist_ok=True)
    database = Path(storage[len("sqlite:///") :]).name
    suffix = hashlib.sha256(study_name.encode()).hexdigest()[:16]
    with (parent / f".{database}.{suffix}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"study {study_name!r} already has an active owner") from exc
        token = _ACTIVE_LOCK_FD.set(lock.fileno())
        try:
            return _run_locked_study(base_config, search_space, budget, seed, storage, study_name)
        finally:
            _ACTIVE_LOCK_FD.reset(token)
