"""Edge cases for the C runner: malformed input and degenerate data.

Two invariants are load-bearing for an unattended loop and are easy to break
without noticing:

1. `run_experiment` must never raise into the parent (Section 8.5), whatever
   it is handed.
2. A permanent failure must classify as something A5 does **not** retry.
   `transient` means "retry with backoff", so misclassifying a condition that
   can never change burns the recovery budget on a run that cannot succeed —
   the same failure shape as the OpenMP abort.

A third invariant is subtler. The starter-kit evaluator is total by design:
it returns GAUC 0.5 when no user is rankable and nDCG 0.0 for an empty set.
So a degenerate evaluation frame yields a *plausible* primary rather than an
error, and nothing downstream can tell a fabricated 0.25 from a measured one.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import train

TRAIN_DATES = list(range(20220408, 20220422))
VAL_DATES = list(range(20220422, 20220429))
TEST_DATES = list(range(20220429, 20220509))


def _frame(rows: int, dates: list[int], offset: int = 0, seed: int = 0, label=None):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, rows) if label is None else np.full(rows, label)
    return pd.DataFrame(
        {
            "date": rng.choice(dates, rows) if rows else np.array([], dtype=int),
            "user_id": rng.integers(0, 10, rows),
            "video_id": rng.integers(0, 50, rows),
            "author_id": rng.integers(0, 5, rows),
            "tab": rng.integers(0, 2, rows),
            "duration_ms": rng.integers(1_000, 60_000, rows),
            "long_view": labels,
        }
    )


def _folds():
    return [
        (
            _frame(300, TRAIN_DATES, offset=index * 50, seed=10 + index),
            _frame(150, [20220416 + index], offset=6_000 + index * 50, seed=20 + index),
        )
        for index in range(3)
    ]


def _install(monkeypatch, data):
    monkeypatch.setattr(train, "_load_data", data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    return train


@pytest.fixture
def runner(monkeypatch):
    return _install(
        monkeypatch,
        lambda: (
            _frame(600, TRAIN_DATES, seed=1),
            _frame(200, VAL_DATES, 4_000, seed=2),
            _frame(200, TEST_DATES, 8_000, seed=3),
        ),
    )


# --- malformed input: the runner must classify, never raise ------------------

MALFORMED = [
    ("config is None", None, "full", 42, 60),
    ("config is a list", [], "full", 42, 60),
    ("config has no model", {}, "full", 42, 60),
    ("unknown model name", {"model": "nope"}, "full", 42, 60),
    ("unknown fidelity", {"model": "random"}, "turbo", 42, 60),
    ("fidelity is None", {"model": "random"}, None, 42, 60),
    ("seed is a bool", {"model": "random"}, "smoke", True, 60),
    ("timeout is zero", {"model": "random"}, "smoke", 42, 0),
    ("timeout is negative", {"model": "random"}, "smoke", 42, -5),
    ("timeout is None", {"model": "random"}, "smoke", 42, None),
    ("unknown feature name", {"model": "random", "features": ["nope"]}, "smoke", 42, 60),
    ("label used as a feature", {"model": "random", "features": ["long_view"]}, "smoke", 42, 60),
    ("hparams is not a dict", {"model": "random", "hparams": [1, 2]}, "smoke", 42, 60),
    ("unknown config key", {"model": "random", "bogus": 1}, "smoke", 42, 60),
    ("blend with no parents", {"model": "blend", "parents": []}, "smoke", 42, 60),
    ("blend with one parent", {"model": "blend", "parents": ["n001"]}, "smoke", 42, 60),
    ("blend with duplicate parents", {"model": "blend", "parents": ["n1", "n1"]}, "smoke", 42, 60),
]


@pytest.mark.parametrize(
    "label, config, fidelity, seed, timeout",
    MALFORMED,
    ids=[case[0] for case in MALFORMED],
)
def test_malformed_input_returns_an_error_record_and_never_raises(
    runner, label, config, fidelity, seed, timeout
):
    result = runner.run_experiment(config, fidelity=fidelity, seed=seed, timeout_s=timeout)

    assert result["status"] == "error"
    assert result["error_class"] in train.ERROR_CLASSES
    for field in ("stage", "error_class", "traceback", "seconds"):
        assert field in result


def test_a_tiny_timeout_is_enforced_rather_than_hanging(runner):
    result = runner.run_experiment({"model": "fm"}, fidelity="full", timeout_s=0.001)

    assert result["status"] == "error"
    assert result["error_class"] == "timeout"


# --- permanent failures must not be classified `transient` ------------------

MODELS = [
    "random",
    "popularity",
    "fm",
    pytest.param(
        "lgbm",
        marks=pytest.mark.skipif(
            sys.platform == "darwin",
            reason="local LightGBM wheel requires Homebrew libomp; covered in Linux CI",
        ),
    ),
    pytest.param(
        "deepfm",
        marks=pytest.mark.skipif(
            sys.platform == "darwin",
            reason="fixture-backed native model tests use Linux CI; macOS production uses spawn",
        ),
    ),
]
HPARAMS = {
    "random": {},
    "popularity": {},
    "fm": {"max_epochs": 1, "k": 4},
    "lgbm": {"min_data_in_leaf": 1, "num_boost_round": 5},
    "deepfm": {"max_epochs": 1, "emb_dim": 4, "mlp": (8,), "batch_size": 128},
}


def _run(runner, model, fidelity="full"):
    return runner.run_experiment(
        {"model": model, "features": ["user_id", "video_id"], "hparams": HPARAMS[model]},
        fidelity=fidelity,
        seed=3,
        timeout_s=180,
    )


@pytest.mark.parametrize("model", MODELS)
def test_missing_blend_parent_is_permanent_not_transient(runner, tmp_path, monkeypatch, model):
    """A node absent from an append-only ledger will never appear, so retrying
    the blend can only waste budget. `store.read` raises FileNotFoundError,
    an OSError, which fell through to `transient`."""
    del model  # the classification is model-independent; parametrised for symmetry
    from agent import store

    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")

    result = runner.run_experiment(
        {"model": "blend", "parents": ["n404", "n405"], "blend_method": "rank_avg"},
        fidelity="smoke",
        seed=1,
        timeout_s=120,
    )

    assert result["status"] == "error"
    assert result["error_class"] == "schema", (
        "a missing parent node is permanent; `transient` would make A5 retry it"
    )


@pytest.mark.parametrize("model", MODELS)
def test_empty_training_frame_is_rejected_uniformly(monkeypatch, model):
    """RandomModel ignores its labels, so an unguarded empty frame produced a
    different class for it than for every other family."""
    runner = _install(
        monkeypatch,
        lambda: (
            _frame(0, TRAIN_DATES, seed=1),
            _frame(200, VAL_DATES, 4_000, seed=2),
            _frame(200, TEST_DATES, 8_000, seed=3),
        ),
    )

    result = _run(runner, model)

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


@pytest.mark.parametrize("model", MODELS)
def test_non_finite_training_labels_are_rejected_uniformly(monkeypatch, model):
    """The one model that cannot notice corrupt labels was the one reporting a
    score from them: `random` returned ok while every other family raised."""
    runner = _install(
        monkeypatch,
        lambda: (
            _frame(600, TRAIN_DATES, seed=1).assign(long_view=np.nan),
            _frame(200, VAL_DATES, 4_000, seed=2),
            _frame(200, TEST_DATES, 8_000, seed=3),
        ),
    )

    result = _run(runner, model)

    assert result["status"] == "error"
    assert result["error_class"] == "schema"


@pytest.mark.parametrize("model", MODELS)
def test_empty_evaluation_frame_never_reports_a_fabricated_score(monkeypatch, model):
    """The evaluator is total: an empty frame scores GAUC 0.5 / nDCG 0.0, i.e.
    a primary of exactly 0.25. Nothing was measured, so reporting it as `ok`
    hands the agent a number indistinguishable from a real one."""
    runner = _install(
        monkeypatch,
        lambda: (
            _frame(600, TRAIN_DATES, seed=1),
            _frame(0, VAL_DATES, 4_000, seed=2),
            _frame(200, TEST_DATES, 8_000, seed=3),
        ),
    )

    result = _run(runner, model)

    assert result["status"] == "error", (
        f"expected a refusal, got primary={result.get('primary')!r} from zero rows"
    )
    assert result["error_class"] == "schema"


def test_evaluate_refuses_empty_and_non_finite_frames():
    frame = _frame(4, VAL_DATES, seed=1)
    scores = np.arange(4, dtype=float)

    with pytest.raises(ValueError, match="empty"):
        train._evaluate(_frame(0, VAL_DATES, seed=1), np.array([], dtype=float))
    with pytest.raises(ValueError, match="finite"):
        train._evaluate(frame.assign(long_view=np.nan), scores)
    with pytest.raises(ValueError, match="aligned"):
        train._evaluate(frame, scores[:2])


# --- degenerate-but-legal data must still be handled ------------------------


@pytest.mark.parametrize("model", MODELS)
def test_a_single_user_across_every_split_still_scores(monkeypatch, model):
    """GAUC is per-user, so one user is the smallest legal evaluation. It must
    produce a real score rather than an error or a crash."""
    runner = _install(
        monkeypatch,
        lambda: (
            _frame(600, TRAIN_DATES, seed=1).assign(user_id=0),
            _frame(200, VAL_DATES, 4_000, seed=2).assign(user_id=0),
            _frame(200, TEST_DATES, 8_000, seed=3).assign(user_id=0),
        ),
    )

    result = _run(runner, model)

    assert result["status"] == "ok", result.get("traceback")
    assert np.isfinite(result["primary"])


@pytest.mark.parametrize("model", MODELS)
def test_validation_users_unseen_in_training_are_handled(monkeypatch, model):
    """Cold-start users are ordinary in a temporal split: the validation window
    contains people the training window never saw. Every encoder must map them
    to its unknown bucket rather than fail."""
    runner = _install(
        monkeypatch,
        lambda: (
            _frame(600, TRAIN_DATES, seed=1),
            _frame(200, VAL_DATES, 4_000, seed=2).assign(
                user_id=lambda frame: frame["user_id"] + 9_999
            ),
            _frame(200, TEST_DATES, 8_000, seed=3),
        ),
    )

    result = _run(runner, model)

    assert result["status"] == "ok", result.get("traceback")
    assert np.isfinite(result["primary"])
