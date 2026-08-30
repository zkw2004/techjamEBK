"""run_experiment(): the single entry point the agent calls.

Contract: AGENT_PLAN.md Section 8.5 (FROZEN). Owner: Workstream C (Ethan). Task C1.

MUST NOT raise. MUST enforce timeout_s. MUST be deterministic given seed.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import signal
import sys
import tempfile
import time
import traceback as traceback_module
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from pydantic import ValidationError

from agent.schema import Config

FIDELITIES = ("smoke", "screen", "full", "confirm")

ERROR_CLASSES = ("syntax", "schema", "timeout", "oom", "transient", "leak_suspected")

LEAK_CANARY_PRIMARY = 0.75  # Section 3; realistic ceiling is nowhere near this
PROCESS_STOP_GRACE_S = 0.1

SCREEN_BUDGET_CAPS = {
    "max_epochs": 3,
    "epochs": 3,
    "n_estimators": 200,
    "num_boost_round": 200,
    "num_trials": 5,
}

SCORE_CACHE_DIR = Path("logs/scores")


class LeakSuspectedError(RuntimeError):
    def __init__(self, primary: float):
        self.primary = primary
        super().__init__(
            f"primary {primary:.6f} exceeds canary {LEAK_CANARY_PRIMARY:.2f}"
        )


class BackendConflictError(ValueError):
    """A run would co-load two OpenMP runtimes in one process.

    Classified `schema`, not `transient`: the conflict is a deterministic
    property of the config, so retrying it can never succeed and A5 must not
    spend the recovery budget on it.
    """


class FeatureLeakError(ValueError):
    """A feature failed the pre-training leakage guard (B6).

    Classified as `leak_suspected`, not `schema`: A5's recovery policy gives
    schema errors a repair attempt, and a leak must be quarantined rather
    than regenerated until it slips past the guard.
    """


def _cache_code_fingerprint() -> str:
    """Invalidate derived scores when implementation or source-file identity changes."""
    from pipeline.data import DATA_DIR, LOG_FILES

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = list(root.glob("*.py")) + list((root / "models").glob("*.py"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for filename in (*LOG_FILES, "video_features_basic_pure.csv"):
        path = DATA_DIR / filename
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _frame_fingerprint(frame) -> str:
    """Hash prediction row identity/context only; never inspect target outcomes."""
    digest = hashlib.sha256()
    for name in ("user_id", "video_id", "author_id", "date", "time_ms", "tab", "duration_ms"):
        if _has_column(frame, name):
            values = _column(frame, name).astype(str)
            digest.update(f"{name}:{values.shape}:{values.dtype}".encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _score_cache_key(config: dict, fidelity: str, seed: int) -> str:
    payload = json.dumps(
        {"config": config, "fidelity": fidelity, "seed": seed,
         "implementation": _cache_code_fingerprint()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _score_cache_path(config: dict, fidelity: str, seed: int) -> Path:
    return SCORE_CACHE_DIR / f"{_score_cache_key(config, fidelity, seed)}.npz"


def _write_score_cache(config: dict, fidelity: str, seed: int, result: dict,
                       validation_frame=None, test_frame=None) -> Path:
    """Atomically persist only the score arrays needed by a future blend."""
    path = _score_cache_path(config, fidelity, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            val_scores=np.asarray(result["val_scores"]),
            val_user_ids=np.asarray(result["val_user_ids"]),
            test_scores=np.asarray(result["test_scores"]),
            validation_fingerprint=(
                "" if validation_frame is None else _frame_fingerprint(validation_frame)
            ),
            test_fingerprint="" if test_frame is None else _frame_fingerprint(test_frame),
        )
    os.replace(temporary, path)
    return path


def _read_score_cache(config: dict, fidelity: str, seed: int,
                      validation_frame=None, test_frame=None) -> dict | None:
    path = _score_cache_path(config, fidelity, seed)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            for frame, name in ((validation_frame, "validation_fingerprint"),
                                (test_frame, "test_fingerprint")):
                if frame is not None and (
                    name not in archive or str(archive[name]) != _frame_fingerprint(frame)
                ):
                    return None
            result = {name: archive[name].copy()
                      for name in ("val_scores", "val_user_ids", "test_scores")}
        if any(array.ndim != 1 for array in result.values()):
            return None
        if len(result["val_scores"]) != len(result["val_user_ids"]):
            return None
        if not all(np.isfinite(result[key]).all() for key in ("val_scores", "test_scores")):
            return None
        return result
    except (ValueError, OSError, KeyError, EOFError, BadZipFile):
        return None


def _parent_config(node_id: str) -> dict:
    from agent import store

    node = store.read(node_id)
    if node.get("status") != "ok" or not node.get("accepted"):
        raise ValueError(f"blend parent {node_id!r} must be an accepted successful node")
    if node.get("fidelity") != "full":
        raise ValueError("C7 first-pass parents must be full-fidelity nodes, not seed averages")
    if node.get("action_type") == "code":
        raise ValueError("generated-code parents require source replay before they can be blended")
    config = Config(**node.get("config", {})).model_dump()
    if config["model"] == "blend":
        raise ValueError("nested blend parents are not supported in the first C7 pass")
    return config


def _error_record(
    stage: str,
    error_class: str,
    started: float,
    traceback_text: str,
    **extra,
) -> dict:
    return {
        "status": "error",
        "stage": stage,
        "error_class": error_class,
        "traceback": traceback_text,
        "seconds": time.monotonic() - started,
        **extra,
    }


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, (LeakSuspectedError, FeatureLeakError)):
        return "leak_suspected"
    if isinstance(exc, SyntaxError):
        return "syntax"
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (KeyError, TypeError, ValueError, ValidationError)):
        return "schema"
    return "transient"


def _config_backends(config: dict) -> set[str]:
    """Native runtimes this config will load, following blends to their parents."""
    from pipeline.models import backend

    if config["model"] != "blend":
        return {backend(config["model"])} - {None}
    backends = set()
    for node_id in config.get("parents", []):
        try:
            parent = _parent_config(node_id)
        except (KeyError, OSError, ValueError):
            continue  # resolution errors surface later with a better message
        backends |= _config_backends(parent)
    return backends


def _assert_single_backend(config: dict) -> None:
    """Refuse a run that would load two OpenMP runtimes into one process.

    torch and LightGBM each link their own libomp; co-loading them aborts the
    interpreter with OMP Error #15 (see pipeline/models/__init__.py). An abort
    is uncatchable, so the parent would only see a dead child and classify it
    `transient` — which A5 retries with backoff, burning the budget on a
    failure that can never succeed. Failing here instead is deterministic,
    explains itself, and classifies as a schema error that A5 does not retry.
    """
    backends = _config_backends(config)
    if len(backends) > 1:
        raise BackendConflictError(
            f"this run needs both {' and '.join(sorted(backends))} in one process; "
            "torch and LightGBM link separate OpenMP runtimes and co-loading them "
            "aborts the interpreter (OMP Error #15). Blend their cached full-tier "
            "scores instead of refitting mixed-backend parents together, or run the "
            "parents in separate experiments first so the score cache is warm."
        )


def _execute_tier(config: dict, fidelity: str, seed: int) -> dict:
    """Execute one fidelity tier inside the isolated child process."""
    _assert_single_backend(config)
    _seed_everything(seed, config)
    if config["model"] == "blend":
        from pipeline.blending import run_blend

        return run_blend(config, fidelity, seed)
    if fidelity == "smoke":
        return _run_smoke(config, seed)
    if fidelity == "screen":
        return _run_screen(config, seed)
    if fidelity == "full":
        return _run_full(config, seed)
    if fidelity == "confirm":
        return _run_confirm(config, seed)
    raise ValueError(f"unsupported fidelity {fidelity!r}")


def _seed_everything(seed: int, config: dict | None = None) -> None:
    """Seed every RNG this run can touch.

    Torch is seeded only when the run actually needs it. Importing torch
    unconditionally used to co-load its libomp alongside LightGBM's and abort
    the process, so every LightGBM experiment died before training (see
    `_assert_single_backend`). `config=None` keeps the historical behaviour
    for callers that seed without a config in hand.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config is not None and "torch" not in _config_backends(config):
        return
    if config is None and "torch" not in sys.modules:
        # Nothing has committed this process to torch; importing it now could
        # be the very co-load that aborts the interpreter.
        return
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data():
    from pipeline.data import load

    return load()


def _load_folds():
    from pipeline.data import internal_folds

    return internal_folds()


def _get_model_class(name: str):
    from pipeline.models import get

    return get(name)


def _head(frame, rows: int):
    if isinstance(frame, Mapping):
        return {name: np.asarray(values)[:rows] for name, values in frame.items()}
    if hasattr(frame, "head"):
        return frame.head(rows)
    return frame[:rows]


def _row_count(frame) -> int:
    if not isinstance(frame, Mapping):
        return len(frame)
    lengths = {len(values) for values in frame.values()}
    if len(lengths) != 1:
        raise ValueError("frame columns must have equal lengths")
    return lengths.pop() if lengths else 0


def _column(frame, name: str) -> np.ndarray:
    try:
        values = frame[name]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"required column {name!r} is unavailable") from exc
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


def _has_column(frame, name: str) -> bool:
    if isinstance(frame, Mapping):
        return name in frame
    columns = getattr(frame, "columns", ())
    return name in columns


def _matrix(train_frame, target_frame, feature_names: list[str]) -> np.ndarray:
    from pipeline.data import FIELDS
    from pipeline.features import get as get_feature
    from pipeline.features import leakage_check

    columns = []
    for name in feature_names:
        if name == "dur_bucket" and not _has_column(target_frame, name):
            train_durations = np.asarray(_column(train_frame, "duration_ms"), dtype=float)
            target_durations = np.asarray(_column(target_frame, "duration_ms"), dtype=float)
            if not np.isfinite(train_durations).all() or not np.isfinite(target_durations).all():
                raise ValueError("duration_ms must contain only finite numbers")
            edges = np.quantile(train_durations, np.linspace(0, 1, 11)[1:-1])
            values = np.searchsorted(edges, target_durations)
        elif name in FIELDS and _has_column(target_frame, name):
            values = _column(target_frame, name)
        else:
            builder = get_feature(name)
            try:
                safe = leakage_check(builder, train_frame, target_frame)
            except (AssertionError, ValueError) as exc:
                raise FeatureLeakError(f"feature {name!r} failed leakage checks") from exc
            if not safe:
                raise FeatureLeakError(f"feature {name!r} failed leakage checks")
            values = builder(train_frame, target_frame)
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != _row_count(target_frame):
            raise ValueError(f"feature {name!r} must return one value per target row")
        columns.append(values)
    if not columns:
        return np.empty((_row_count(target_frame), 0), dtype=float)
    return np.column_stack(columns)


def _new_model(config: dict, seed: int):
    model_class = _get_model_class(config["model"])
    kwargs = dict(config["hparams"])
    kwargs.update(
        seed=seed,
        loss=config["loss"],
        parents=config["parents"],
        blend_method=config["blend_method"],
        negative_sampling=config["negative_sampling"],
        feature_names=config["features"],
    )
    return model_class(**kwargs)


def _fit_and_predict(
    config: dict,
    train_frame,
    validation_frame,
    prediction_frames: list,
    seed: int,
    fit_summaries: list[dict] | None = None,
) -> list[np.ndarray]:
    from pipeline.data import LABEL

    features = config["features"]
    model = _new_model(config, seed)
    train_matrix = _matrix(train_frame, train_frame, features)
    train_labels = _column(train_frame, LABEL)
    train_users = _column(train_frame, "user_id")
    if config.get("negative_sampling", "all") != "all":
        from pipeline.data import sample_negatives

        # Build historical features BEFORE selecting examples. B5 uses frame
        # identity for strictly past-only training aggregates; using the sampled
        # frame as target would accidentally expose later training outcomes.
        selected = sample_negatives(
            train_frame.reset_index(drop=True),
            strategy=config["negative_sampling"],
            seed=seed,
        ).index.to_numpy()
        if not len(selected):
            raise ValueError("negative sampling selected no training rows")
        train_matrix = train_matrix[selected]
        train_labels = train_labels[selected]
        train_users = train_users[selected]

    if validation_frame is None:
        validation_matrix = validation_labels = None
    else:
        validation_matrix = _matrix(train_frame, validation_frame, features)
        validation_labels = _column(validation_frame, LABEL)

    validation_users = (
        None if validation_frame is None else _column(validation_frame, "user_id")
    )
    model.fit(
        train_matrix,
        train_labels,
        validation_matrix,
        validation_labels,
        groups=(train_users, validation_users),
    )
    if fit_summaries is not None:
        fit_summaries.append({"best_epoch": getattr(model, "best_epoch", None)})
    outputs = []
    for frame in prediction_frames:
        scores = np.asarray(model.predict(_matrix(train_frame, frame, features)), dtype=float)
        if scores.ndim != 1 or len(scores) != _row_count(frame):
            raise ValueError("model predict() must return one score per row")
        if not np.isfinite(scores).all():
            raise ValueError("model predict() returned non-finite scores")
        outputs.append(scores)
    return outputs


def _evaluate(frame, scores: np.ndarray) -> dict[str, float]:
    from pipeline.data import LABEL
    from pipeline.evaluate import evaluate

    raw = evaluate(_column(frame, "user_id"), _column(frame, LABEL), scores)
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }


def _peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _success(fidelity: str, **values) -> dict:
    return {
        "status": "ok",
        "fidelity": fidelity,
        "gauc": None,
        "ndcg": None,
        "primary": None,
        "fold_primaries": [],
        "segments": {},
        "val_scores": np.array([], dtype=float),
        "val_user_ids": np.array([], dtype=np.int64),
        "test_scores": np.array([], dtype=float),
        "gpu_seconds": 0.0,
        "peak_rss_mb": _peak_rss_mb(),
        **values,
    }


def _run_smoke(config: dict, seed: int) -> dict:
    train_frame, validation_frame, test_frame = _load_data()
    train_sample = _head(train_frame, 1_000)
    validation_sample = _head(validation_frame, 1_000)
    test_sample = _head(test_frame, 1_000)
    validation_scores, test_scores = _fit_and_predict(
        config,
        train_sample,
        None,
        [validation_sample, test_sample],
        seed,
    )
    return _success(
        "smoke",
        val_scores=validation_scores,
        val_user_ids=_column(validation_sample, "user_id"),
        test_scores=test_scores,
    )


def _score_folds(
    config: dict,
    seed: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, list[int]]:
    metrics = []
    all_scores = []
    all_user_ids = []
    fit_summaries = []
    folds = _load_folds()
    if len(folds) != 3:
        raise ValueError(f"screen requires exactly three internal folds, got {len(folds)}")
    for index, (fold_train, fold_validation) in enumerate(folds):
        (scores,) = _fit_and_predict(
            config,
            fold_train,
            fold_validation,
            [fold_validation],
            seed + index,
            fit_summaries,
        )
        metrics.append(_evaluate(fold_validation, scores))
        all_scores.append(scores)
        all_user_ids.append(_column(fold_validation, "user_id"))
    best_epochs = [
        summary["best_epoch"]
        for summary in fit_summaries
        if isinstance(summary.get("best_epoch"), int) and summary["best_epoch"] > 0
    ]
    return metrics, np.concatenate(all_scores), np.concatenate(all_user_ids), best_epochs


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([item[name] for item in metrics]))
        for name in ("gauc", "ndcg", "primary")
    }


def _screen_config(config: dict) -> dict:
    reduced = {**config, "hparams": dict(config["hparams"])}
    if config.get("model") == "lgbm" and "n_estimators" in reduced["hparams"]:
        reduced["hparams"].setdefault("num_boost_round", reduced["hparams"]["n_estimators"])
    for name, cap in SCREEN_BUDGET_CAPS.items():
        current = reduced["hparams"].get(name, cap)
        if isinstance(current, bool) or not isinstance(current, Real) or current <= 0:
            raise ValueError(f"{name} must be a positive number")
        reduced["hparams"][name] = min(current, cap)
    return reduced


def _run_screen(config: dict, seed: int) -> dict:
    fold_metrics, scores, user_ids, _ = _score_folds(_screen_config(config), seed)
    return _success(
        "screen",
        **_mean_metrics(fold_metrics),
        fold_primaries=[item["primary"] for item in fold_metrics],
        val_scores=scores,
        val_user_ids=user_ids,
    )


def _quartile_labels(counts_by_key, keys) -> np.ndarray:
    """1-based quartile of each row's key, by the train-side count distribution."""
    per_key_counts = np.asarray(list(counts_by_key.values()), dtype=float)
    edges = np.quantile(per_key_counts, [0.25, 0.5, 0.75])
    row_counts = np.asarray([counts_by_key.get(key, 0) for key in keys], dtype=float)
    return np.searchsorted(edges, row_counts, side="right") + 1


def _segment_metrics(train_frame, validation_frame, scores: np.ndarray) -> dict:
    """Primary by user-activity quartile, item-popularity quartile, and day (6.6).

    Returns {} when the frames do not carry the meta columns (synthetic test
    fixtures); on the real split all three breakdowns populate. Quartiles are
    fitted on the TRAIN side only, so segment membership is knowable before
    the impression.
    """
    from pipeline.data import LABEL

    needed = ("user_id", "video_id", "date", LABEL)
    if not all(_has_column(frame, name) for frame in (train_frame, validation_frame)
               for name in needed):
        return {}
    from collections import Counter

    from agent.gate import segments as gate_segments

    user_counts = Counter(_column(train_frame, "user_id").tolist())
    video_counts = Counter(_column(train_frame, "video_id").tolist())
    validation_users = _column(validation_frame, "user_id")
    validation_videos = _column(validation_frame, "video_id")
    dates = _column(validation_frame, "date")
    day_index = {value: i + 1 for i, value in enumerate(sorted(set(dates.tolist())))}
    meta = {
        "labels": _column(validation_frame, LABEL),
        "activity_q": _quartile_labels(user_counts, validation_users.tolist()),
        "pop_q": _quartile_labels(video_counts, validation_videos.tolist()),
        "day": np.asarray([day_index[value] for value in dates.tolist()]),
    }
    try:
        return gate_segments(scores, validation_users, meta)
    except NotImplementedError:  # D4 reverted — segments stay empty, runs stay alive
        return {}


def _run_full(config: dict, seed: int) -> dict:
    fold_metrics, _, _, best_epochs = _score_folds(config, seed)
    refit_config = config
    if best_epochs:
        budget_name = "num_boost_round" if config["model"] == "lgbm" else "max_epochs"
        refit_config = {
            **config,
            "hparams": {
                **config["hparams"],
                budget_name: int(np.median(best_epochs)),
            },
        }
    train_frame, validation_frame, test_frame = _load_data()
    validation_scores, test_scores = _fit_and_predict(
        refit_config,
        train_frame,
        None,
        [validation_frame, test_frame],
        seed,
    )
    metrics = _evaluate(validation_frame, validation_scores)
    result = _success(
        "full",
        **metrics,
        fold_primaries=[item["primary"] for item in fold_metrics],
        segments=_segment_metrics(train_frame, validation_frame, validation_scores),
        val_scores=validation_scores,
        val_user_ids=_column(validation_frame, "user_id"),
        test_scores=test_scores,
    )
    if metrics["primary"] > LEAK_CANARY_PRIMARY:
        raise LeakSuspectedError(metrics["primary"])
    _write_score_cache(config, "full", seed, result,
                       validation_frame=validation_frame, test_frame=test_frame)
    return result


def _run_confirm(config: dict, seed: int) -> dict:
    runs = []
    for offset in range(5):
        run_seed = seed + offset
        _seed_everything(run_seed, config)
        run = _run_full(config, run_seed)
        if run["primary"] > LEAK_CANARY_PRIMARY:
            raise LeakSuspectedError(run["primary"])
        runs.append(run)
    metrics = _mean_metrics(runs)
    fold_primaries = np.mean(
        np.asarray([run["fold_primaries"] for run in runs], dtype=float), axis=0
    ).tolist()
    segment_keys = sorted({key for run in runs for key in run["segments"]})
    mean_segments = {
        key: float(np.mean([run["segments"][key] for run in runs if key in run["segments"]]))
        for key in segment_keys
    }
    return _success(
        "confirm",
        **metrics,
        fold_primaries=fold_primaries,
        segments=mean_segments,
        val_scores=np.mean([run["val_scores"] for run in runs], axis=0),
        val_user_ids=runs[0]["val_user_ids"],
        test_scores=np.mean([run["test_scores"] for run in runs], axis=0),
    )


def _child_entry(send_conn, config: dict, fidelity: str, seed: int) -> None:
    started = time.monotonic()
    try:
        result = _execute_tier(config, fidelity, seed)
    except BaseException as exc:  # the public contract must contain every worker failure
        extra = {"primary": exc.primary} if isinstance(exc, LeakSuspectedError) else {}
        result = _error_record(
            "leakage" if isinstance(exc, LeakSuspectedError) else fidelity,
            _classify_exception(exc),
            started,
            traceback_module.format_exc(),
            **extra,
        )

    try:
        send_conn.send(result)
    finally:
        send_conn.close()


def _process_context():
    """Prefer fork so dynamically registered experiment components are inherited."""
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _stop_process(process) -> None:
    if process is None:
        return
    try:
        alive = process.is_alive()
    except (AssertionError, ValueError):
        return
    if not alive:
        process.join(PROCESS_STOP_GRACE_S)
        return
    process.terminate()
    process.join(PROCESS_STOP_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(PROCESS_STOP_GRACE_S)


def _child_death_report(process, exc: BaseException) -> str:
    """Explain a worker that died without sending a result.

    A bare `EOFError` says only that the pipe closed. When the child was
    killed by a signal the exit code carries the reason, and a native abort
    (SIGABRT) is nearly always two OpenMP runtimes co-loaded — worth naming,
    because the traceback is the only place that diagnosis can surface.
    """
    detail = "".join(traceback_module.format_exception_only(type(exc), exc)).strip()
    exitcode = process.exitcode if process is not None else None
    if exitcode is None or exitcode >= 0:
        return f"{detail}\nworker exited with code {exitcode} before sending a result"
    try:
        name = signal.Signals(-exitcode).name
    except ValueError:
        name = f"signal {-exitcode}"
    hint = ""
    if -exitcode in (signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS):
        hint = (
            "\nA native crash, not a transient fault. The usual cause is two "
            "OpenMP runtimes in one process (torch + LightGBM, OMP Error #15) — "
            "check the model's native_backend declaration."
        )
    return f"{detail}\nworker killed by {name} (exit {exitcode}){hint}"


def _result_schema_issue(result: dict, fidelity: str) -> str | None:
    status = result.get("status")
    if status == "error":
        required = ("stage", "error_class", "traceback", "seconds")
        missing = [name for name in required if name not in result]
        if missing:
            return f"error result missing fields: {missing}"
        if result["error_class"] not in ERROR_CLASSES:
            return f"unsupported error_class {result['error_class']!r}"
        return None
    if status != "ok":
        return f"worker status must be 'ok' or 'error', got {status!r}"

    required = (
        "fidelity", "gauc", "ndcg", "primary", "fold_primaries", "segments",
        "val_scores", "val_user_ids", "test_scores", "seconds", "gpu_seconds",
        "peak_rss_mb",
    )
    missing = [name for name in required if name not in result]
    if missing:
        return f"success result missing fields: {missing}"
    if result["fidelity"] != fidelity:
        return f"worker returned fidelity {result['fidelity']!r}, expected {fidelity!r}"

    metrics = (result["gauc"], result["ndcg"], result["primary"])
    if fidelity == "smoke":
        if any(value is not None for value in metrics):
            return "smoke must skip gauc, ndcg, and primary metrics"
    elif any(not isinstance(value, Real) or not np.isfinite(value) for value in metrics):
        return "gauc, ndcg, and primary must be finite numbers"

    fold_primaries = np.asarray(result["fold_primaries"], dtype=float)
    if fold_primaries.ndim != 1 or not np.isfinite(fold_primaries).all():
        return "fold_primaries must be a finite one-dimensional sequence"
    expected_folds = 0 if fidelity == "smoke" else 3
    if len(fold_primaries) != expected_folds:
        return f"{fidelity} must return exactly {expected_folds} fold primary scores"
    if not isinstance(result["segments"], dict):
        return "segments must be a dictionary"

    for name in ("seconds", "gpu_seconds", "peak_rss_mb"):
        value = result[name]
        if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            return f"{name} must be a finite non-negative number"

    val_scores = np.asarray(result["val_scores"])
    val_user_ids = np.asarray(result["val_user_ids"])
    test_scores = np.asarray(result["test_scores"])
    if val_scores.ndim != 1 or test_scores.ndim != 1 or val_user_ids.ndim != 1:
        return "score and user-id outputs must be one-dimensional"
    if len(val_scores) != len(val_user_ids):
        return "val_scores and val_user_ids must remain aligned"
    if not np.isfinite(val_scores).all() or not np.isfinite(test_scores).all():
        return "validation and test scores must be finite"
    return None


def run_experiment(
    config: dict,
    fidelity: str = "full",
    seed: int = 42,
    timeout_s: int = 1800,
) -> dict:
    """Run one experiment at the requested fidelity tier (Section 6.3).

      smoke   -> 1000-row fit, correctness checks only, no metrics
      screen  -> three internal folds, reduced budget
      full    -> full train prefix, all official validation rows
      confirm -> full, repeated across 5 seeds

    Success:
    {
      "status": "ok", "fidelity": str,
      "gauc": float, "ndcg": float, "primary": float,
      "fold_primaries": list[float],   # screen/full
      "segments": dict,                # full/confirm, see 6.6
      "val_scores": np.ndarray,        # aligned with val rows
      "val_user_ids": np.ndarray,
      "test_scores": np.ndarray,
      "seconds": float, "gpu_seconds": float, "peak_rss_mb": float,
    }

    Failure:
    {"status": "error", "stage": str, "error_class": str,
     "traceback": str, "seconds": float}

    MUST return status="error", error_class="leak_suspected" if primary > 0.75.
    """
    started = time.monotonic()

    try:
        if fidelity not in FIDELITIES:
            raise ValueError(f"unknown fidelity {fidelity!r}; expected one of {FIDELITIES}")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        parsed_config = Config.model_validate(config).model_dump()
    except (TypeError, ValueError, ValidationError) as exc:
        return _error_record(
            "validation",
            "schema",
            started,
            "".join(traceback_module.format_exception_only(type(exc), exc)),
        )

    receive_conn = send_conn = process = None
    try:
        context = _process_context()
        receive_conn, send_conn = context.Pipe(duplex=False)
        process = context.Process(
            target=_child_entry,
            args=(send_conn, parsed_config, fidelity, seed),
            daemon=True,
        )
        process.start()
        send_conn.close()
        send_conn = None

        if not receive_conn.poll(timeout_s):
            _stop_process(process)
            return _error_record(
                fidelity,
                "timeout",
                started,
                f"experiment exceeded timeout_s={timeout_s}",
            )

        try:
            result = receive_conn.recv()
        except (EOFError, OSError) as exc:
            process.join(PROCESS_STOP_GRACE_S)
            error_class = "oom" if process.exitcode == -signal.SIGKILL else "transient"
            return _error_record(
                fidelity,
                error_class,
                started,
                _child_death_report(process, exc),
            )
        finally:
            process.join(PROCESS_STOP_GRACE_S)
            if process.is_alive():
                _stop_process(process)
    except BaseException as exc:
        _stop_process(process)
        return _error_record(
            fidelity,
            _classify_exception(exc),
            started,
            traceback_module.format_exc(),
        )
    finally:
        if receive_conn is not None:
            receive_conn.close()
        if send_conn is not None:
            send_conn.close()

    if not isinstance(result, dict):
        return _error_record(
            fidelity,
            "schema",
            started,
            f"worker result must be a dict, got {type(result).__name__}",
        )
    result.setdefault("seconds", time.monotonic() - started)
    try:
        schema_issue = _result_schema_issue(result, fidelity)
    except (TypeError, ValueError) as exc:
        schema_issue = "".join(traceback_module.format_exception_only(type(exc), exc))
    if schema_issue is not None:
        return _error_record(fidelity, "schema", started, schema_issue)

    primary = result.get("primary")
    if result.get("status") == "ok" and primary is not None and primary > LEAK_CANARY_PRIMARY:
        return _error_record(
            "leakage",
            "leak_suspected",
            started,
            f"primary {result['primary']:.6f} exceeds canary {LEAK_CANARY_PRIMARY:.2f}",
            primary=result["primary"],
        )
    return result
