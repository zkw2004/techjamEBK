"""Refit a selected experiment on train+validation and create a submission.

The official validation window is no longer needed after a winner is chosen.
Using it as additional training data is temporally legal because every test
row is later; this module does not use its labels for any final-model choice.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent import store
from agent.schema import Config

N_SEEDS = 5
SUBMISSION_COLUMNS = ["row_id", "user_id", "video_id", "score"]
DEFAULT_OUTPUT = Path("submission.csv")
STARTER_SUBMIT = Path("kuairand-starter-kit/submit.py")


class FinaliseError(RuntimeError):
    """The chosen configuration cannot safely produce a final submission."""


@dataclass(frozen=True)
class FinalisationResult:
    output_path: Path
    seeds: tuple[int, ...]
    row_count: int
    mean_score: float


def _load_data():
    from pipeline.data import load

    return load()


def _concat_train_and_validation(train_frame: Any, validation_frame: Any):
    """Append validation in its original order without resetting split logic."""
    if hasattr(train_frame, "columns") and hasattr(validation_frame, "columns"):
        import pandas as pd

        return pd.concat([train_frame, validation_frame], ignore_index=True)
    if isinstance(train_frame, dict) and isinstance(validation_frame, dict):
        if set(train_frame) != set(validation_frame):
            raise FinaliseError("train and validation mappings have different columns")
        return {
            name: np.concatenate(
                [np.asarray(train_frame[name]), np.asarray(validation_frame[name])]
            )
            for name in train_frame
        }
    raise FinaliseError("train and validation frames must both be pandas frames or mappings")


def _column(frame: Any, name: str) -> np.ndarray:
    try:
        values = frame[name]
    except (KeyError, TypeError) as exc:
        raise FinaliseError(f"submission frame is missing required column {name!r}") from exc
    return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)


def selected_config(node_id: str | None = None) -> dict:
    """Load an accepted full-fidelity winner from the append-only ledger."""
    if node_id is None:
        node = store.best_node()
        if node is None:
            raise FinaliseError("no accepted full-fidelity node is available to finalise")
    else:
        try:
            node = store.read(node_id)
        except FileNotFoundError as exc:
            raise FinaliseError(f"selected node {node_id!r} does not exist") from exc
        if not node.get("accepted") or node.get("status") != "ok":
            raise FinaliseError(f"selected node {node_id!r} is not an accepted successful result")
    if node.get("fidelity") not in {"full", "confirm"}:
        raise FinaliseError("the final model must come from a full or confirm node")
    try:
        return Config.model_validate(node["config"]).model_dump()
    except (KeyError, ValueError) as exc:
        raise FinaliseError("selected node has an invalid model config") from exc


def _refit_predict(config: dict, train_frame: Any, test_frame: Any, seed: int) -> np.ndarray:
    """Train once on the final permitted period and predict exact test rows."""
    from pipeline import train

    train._assert_single_backend(config)  # noqa: SLF001 - central backend safety contract
    train._seed_everything(seed, config)  # noqa: SLF001 - runner's deterministic seed policy
    (scores,) = train._fit_and_predict(  # noqa: SLF001 - final fit intentionally has no validation
        config,
        train_frame,
        None,
        [test_frame],
        seed,
    )
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(test_frame):
        raise FinaliseError("final model did not return one score per test row")
    if not np.isfinite(scores).all():
        raise FinaliseError("final model returned NaN or infinite scores")
    return scores


def _final_predict(config: dict, train_frame: Any, test_frame: Any, seed: int) -> np.ndarray:
    """Return one seed's final scores, including a legal same-backend blend."""
    if config["model"] != "blend":
        return _refit_predict(config, train_frame, test_frame, seed)

    from pipeline import blending, train
    from pipeline.models import backend
    from pipeline.models.blend import blend_scores

    parents = blending._parents(config)  # noqa: SLF001 - C7 owns parent resolution
    native_backends = {backend(parent["model"]) for parent in parents} - {None}
    if len(native_backends) > 1:
        raise FinaliseError(
            "cannot refit a blend whose parents require different native backends"
        )
    evidence = blending._fold_evidence(config, parents, "full")  # noqa: SLF001
    parent_scores = [
        _refit_predict(parent, train_frame, test_frame, seed + index)
        for index, parent in enumerate(parents)
    ]
    return np.asarray(
        blend_scores(
            parent_scores,
            train._column(test_frame, "user_id"),  # noqa: SLF001
            config["blend_method"],
            weight=evidence["blend_weight"],
        ),
        dtype=float,
    )


def write_submission(path: Path | str, test_frame: Any, scores: Sequence[float]) -> Path:
    """Write organiser-required rows in the loader's unmodified test order."""
    destination = Path(path)
    values = np.asarray(scores, dtype=float)
    user_ids = _column(test_frame, "user_id")
    video_ids = _column(test_frame, "video_id")
    if not (
        len(values) == len(test_frame)
        and len(user_ids) == len(test_frame)
        and len(video_ids) == len(test_frame)
    ):
        raise FinaliseError("test ids and final scores must have identical row counts")
    if not np.isfinite(values).all():
        raise FinaliseError("submission scores must be finite")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUBMISSION_COLUMNS)
        rows = zip(user_ids, video_ids, values, strict=True)
        for row_id, (user_id, video_id, score) in enumerate(rows):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".17g")])
    return destination


def _run_starter_check(path: Path) -> None:
    """Delegate shape/alignment validation to the unmodified organiser checker."""
    completed = subprocess.run(
        [
            sys.executable,
            str(STARTER_SUBMIT),
            str(path),
            "--check",
            "--data_dir",
            "data/KuaiRand-Pure/data",
            "--split",
            "test",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise FinaliseError(f"starter submission check failed: {detail}")


def finalise(
    config: dict,
    *,
    output_path: Path | str = DEFAULT_OUTPUT,
    seeds: Sequence[int] = tuple(range(N_SEEDS)),
    frames: tuple[Any, Any, Any] | None = None,
    checker: Callable[[Path], None] | None = None,
) -> FinalisationResult:
    """Refit ``config`` on train+validation, average five seeds, and validate.

    ``frames`` and ``checker`` are injectable only for deterministic tests; the
    production path uses the frozen loader and the starter-kit checker.
    """
    parsed = Config.model_validate(config).model_dump()
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) != N_SEEDS or len(set(seed_values)) != N_SEEDS:
        raise FinaliseError(f"finalisation requires exactly {N_SEEDS} distinct seeds")
    train_frame, validation_frame, test_frame = _load_data() if frames is None else frames
    final_training = _concat_train_and_validation(train_frame, validation_frame)
    predictions = [
        _final_predict(parsed, final_training, test_frame, seed)
        for seed in seed_values
    ]
    mean_scores = np.mean(np.stack(predictions, axis=0), axis=0)
    destination = write_submission(output_path, test_frame, mean_scores)
    (checker or _run_starter_check)(destination)
    return FinalisationResult(
        output_path=destination,
        seeds=seed_values,
        row_count=len(test_frame),
        mean_score=float(np.mean(mean_scores)),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node", help="accepted full/confirm node id; defaults to best accepted node"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    result = finalise(selected_config(args.node), output_path=args.output)
    print(
        f"Wrote {result.output_path} with {result.row_count:,} aligned test rows "
        f"from {len(result.seeds)} seed refits."
    )


if __name__ == "__main__":
    main()
