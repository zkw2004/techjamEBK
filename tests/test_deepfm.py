"""C5 DeepFM behavior and factorization-machine math."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pipeline.models.deepfm import DeepFMModel

# NOT imported at module scope: a top-level `import torch` loads torch into
# the pytest *parent* at collection time, and every later forked child then
# inherits it — so a LightGBM test would still co-load both runtimes and
# abort (OMP Error #15). Import it inside the test bodies, which the
# conftest hook runs in isolated children.

# Every test here drives torch in-process. The conftest hook runs them
# in a forked child so the pytest process never co-loads torch and LightGBM
# (OMP Error #15 aborts the whole session).
pytestmark = pytest.mark.native_backend("torch")


def test_registry_import_does_not_initialize_torch_in_non_neural_workers():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import pipeline.models; "
         "assert 'torch' not in sys.modules"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_deprioritized_multitask_model_does_not_masquerade_as_single_task():
    from pipeline.models.deepfm import DeepFMMultiTask

    with pytest.raises(NotImplementedError, match="multi-task"):
        DeepFMMultiTask().fit(np.array([[1]]), np.array([0]), None, None)


def _learnable_rows(repeats: int = 20) -> tuple[np.ndarray, np.ndarray]:
    positive = ["u1", "v-positive", "a1", "home", "1"]
    negative = ["u2", "v-negative", "a2", "search", "8"]
    X = np.asarray([positive] * repeats + [negative] * repeats, dtype=object)
    y = np.asarray([1] * repeats + [0] * repeats, dtype=float)
    return X, y


def test_deepfm_second_order_matches_hand_computed_pairwise_dots():
    import torch

    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    interaction = DeepFMModel._second_order(embeddings)

    torch.testing.assert_close(interaction, torch.tensor([[67.0]]))


def test_deepfm_learns_positive_pattern_above_negative_pattern():
    X, y = _learnable_rows()
    model = DeepFMModel(
        emb_dim=4,
        mlp=(8,),
        dropout=0.0,
        lr=0.02,
        max_epochs=8,
        batch_size=8,
        patience=1,
        seed=6,
    )

    model.fit(X, y, X, y, groups=(X[:, 0], X[:, 0]))
    scores = model.predict(np.asarray([X[0], X[-1]], dtype=object))

    assert scores[0] > scores[1]
    assert model.best_epoch is not None


def test_deepfm_is_deterministic_and_handles_unseen_categories():
    X, y = _learnable_rows(6)
    target = np.asarray([["new-user", "new-video", "new-author", "new-tab", "9"]])
    models = [
        DeepFMModel(
            emb_dim=2,
            mlp=(4,),
            dropout=0.0,
            lr=0.01,
            max_epochs=2,
            batch_size=4,
            seed=12,
        )
        for _ in range(2)
    ]

    for model in models:
        model.fit(X, y, None, None, groups=(X[:, 0], None))

    first = models[0].predict(target)
    second = models[1].predict(target)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1,)
    assert np.isfinite(first).all()
