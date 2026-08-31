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


def test_multitask_model_rejects_the_bare_groups_tuple_other_models_accept():
    """C9: deepfm_mtl is implemented now, but it still cannot masquerade as a
    single-task model — it requires the aux-carrying groups dict that
    pipeline.train._fit_and_predict builds only when AUX_TARGETS is declared
    on the model class."""
    from pipeline.models.deepfm import DeepFMMultiTask

    with pytest.raises(ValueError, match="requires auxiliary training targets"):
        DeepFMMultiTask().fit(
            np.array([[1]]), np.array([0.0]), None, None, groups=("u1",)
        )


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


# --- C9: multi-task DeepFM ----------------------------------------------

_MTLRows = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _mtl_rows(n_users: int = 20) -> _MTLRows:
    """Rows where long_view, is_click and is_like each depend on a different
    field, so a head that isn't wired to its own signal is easy to catch."""
    rng = np.random.default_rng(3)
    rows, long_view, click, like, users = [], [], [], [], []
    for user in range(n_users):
        for item in range(6):
            rows.append([f"u{user}", f"v{item}", f"a{item % 2}", "tab0", str(item % 3)])
            long_view.append(1.0 if item % 3 == 0 else 0.0)
            click.append(1.0 if item % 2 == 0 else 0.0)
            like.append(float(rng.random() < 0.3))
            users.append(f"u{user}")
    return (
        np.asarray(rows, dtype=object),
        np.asarray(long_view, dtype=np.float32),
        np.asarray(click, dtype=np.float32),
        np.asarray(like, dtype=np.float32),
        np.asarray(users, dtype=object),
    )


def _mtl_groups(users, click=None, like=None, val_users=None):
    train_aux = {}
    if click is not None:
        train_aux["is_click"] = click
    if like is not None:
        train_aux["is_like"] = like
    return {"train": users, "val": val_users, "aux_train": train_aux, "aux_val": None}


_MTL_KWARGS = dict(emb_dim=8, mlp=(16,), dropout=0.2, lr=0.02, max_epochs=3,
                    batch_size=16, patience=1, seed=11)


def test_c9_aux_weights_zero_reproduces_single_task_deepfm_exactly():
    """The addendum's acceptance criterion is 'within 0.001'; this asserts the
    stronger property the implementation actually provides: bit-exact.

    That strength matters here specifically because auxiliary-head
    construction draws from torch's global RNG (nn.Linear's default init),
    the same stream nn.Dropout reads from during training — a naive
    implementation would only match the single-task model *approximately*,
    with the gap growing over epochs as dropout masks drift apart. Exact
    equality is only possible because MultiTaskNetwork's aux-head
    construction is quarantined (save/restore torch's RNG state around it),
    which this test cannot see directly but validates the effect of.
    """
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()

    single = DeepFMModel(**_MTL_KWARGS)
    single.fit(X, y, None, None, groups=(users, None))

    mtl = DeepFMMultiTask(aux_click_weight=0.0, aux_like_weight=0.0, **_MTL_KWARGS)
    mtl.fit(X, y, None, None, groups=_mtl_groups(users, click, like))

    np.testing.assert_array_equal(single.predict(X), mtl.predict(X))


def test_c9_nonzero_aux_weights_measurably_change_training():
    """The inverse check: if the heads never moved anything, weight=0 and
    weight=0.8 would look the same too, which would make the exact-match
    test above vacuous rather than meaningful."""
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()

    off = DeepFMMultiTask(aux_click_weight=0.0, aux_like_weight=0.0, **_MTL_KWARGS)
    off.fit(X, y, None, None, groups=_mtl_groups(users, click, like))

    on = DeepFMMultiTask(aux_click_weight=0.8, aux_like_weight=0.5, **_MTL_KWARGS)
    on.fit(X, y, None, None, groups=_mtl_groups(users, click, like))

    assert not np.allclose(off.predict(X), on.predict(X))


def test_c9_output_shape_is_unchanged_by_auxiliary_heads():
    """'Output shape/interface unchanged' — predict() returns one score per
    row regardless of how many auxiliary heads exist, matching the frozen
    BaseModel.predict() -> np.ndarray contract (8.9)."""
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()
    model = DeepFMMultiTask(aux_click_weight=0.3, aux_like_weight=0.3, **_MTL_KWARGS)

    model.fit(X, y, None, None, groups=_mtl_groups(users, click, like))
    scores = model.predict(X)

    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


def test_c9_early_stopping_uses_only_the_primary_head():
    """Auxiliary heads affect training loss only — they must never gate which
    epoch is selected as best. A weight so large it dominates the loss must
    not change best_epoch versus the same run with aux weights at 0, since
    checkpoint selection reads _primary_validation_loss alone."""
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()
    kwargs = {**_MTL_KWARGS, "max_epochs": 5}

    off = DeepFMMultiTask(aux_click_weight=0.0, aux_like_weight=0.0, **kwargs)
    off.fit(X, y, X, y, groups=_mtl_groups(users, click, like, val_users=users))

    dominant = DeepFMMultiTask(aux_click_weight=50.0, aux_like_weight=50.0, **kwargs)
    dominant.fit(X, y, X, y, groups=_mtl_groups(users, click, like, val_users=users))

    assert off.best_epoch == dominant.best_epoch


def test_c9_is_deterministic_for_a_seed():
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()

    def fit_once():
        model = DeepFMMultiTask(aux_click_weight=0.4, aux_like_weight=0.2, **_MTL_KWARGS)
        model.fit(X, y, None, None, groups=_mtl_groups(users, click, like))
        return model.predict(X)

    np.testing.assert_array_equal(fit_once(), fit_once())


def test_c9_rejects_negative_aux_weights():
    from pipeline.models.deepfm import DeepFMMultiTask

    with pytest.raises(ValueError, match="non-negative"):
        DeepFMMultiTask(aux_click_weight=-0.1)


def test_c9_rejects_aux_targets_misaligned_with_training_rows():
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()
    model = DeepFMMultiTask(aux_click_weight=0.5, **_MTL_KWARGS)

    with pytest.raises(ValueError, match="one value per training row"):
        model.fit(X, y, None, None, groups=_mtl_groups(users, click=click[:5]))


def test_c9_train_layer_builds_the_aux_groups_dict_only_for_declaring_models():
    """pipeline.train._fit_and_predict must extend `groups` only for a model
    that declares AUX_TARGETS — every other model must still receive the
    exact (train_users, val_users) tuple it always did (agent/loop.py and
    the FM/LGBM `groups[0]`/`groups[1]` unpacking both depend on that shape
    being unchanged for them)."""
    import pandas as pd

    from pipeline.train import _fit_and_predict

    rows = {
        "user_id": ["u1", "u1", "u2", "u2"],
        "video_id": ["v1", "v2", "v1", "v2"],
        "author_id": ["a1", "a1", "a2", "a2"],
        "tab": ["t0"] * 4,
        "date": pd.to_datetime(["2022-04-08"] * 4),
        "long_view": [1, 0, 1, 0],
        "is_click": [1, 0, 0, 1],
        "is_like": [0, 0, 1, 0],
    }
    frame = pd.DataFrame(rows)
    seen_groups = {}

    class _Recorder:
        AUX_TARGETS = ("is_click", "is_like")

        def fit(self, X_train, y_train, X_val, y_val, groups=None):
            seen_groups["value"] = groups

        def predict(self, X):
            return np.zeros(len(X))

    import pipeline.train as train_module
    original = train_module._new_model
    train_module._new_model = lambda config, seed: _Recorder()
    try:
        _fit_and_predict(
            {"model": "fake", "features": ["user_id", "video_id"],
             "hparams": {}, "negative_sampling": "all"},
            frame, None, [], seed=0,
        )
    finally:
        train_module._new_model = original

    groups = seen_groups["value"]
    assert isinstance(groups, dict)
    assert list(groups["aux_train"]["is_click"]) == [1, 0, 0, 1]
    assert list(groups["aux_train"]["is_like"]) == [0, 0, 1, 0]


def test_c9_train_layer_leaves_ordinary_models_groups_untouched():
    """The FM/LGBM regression guard for the above: a model with no
    AUX_TARGETS must still get the bare tuple, unchanged by this feature."""
    import pandas as pd

    from pipeline.train import _fit_and_predict

    frame = pd.DataFrame({
        "user_id": ["u1", "u2"], "video_id": ["v1", "v2"],
        "author_id": ["a1", "a2"], "tab": ["t0", "t0"],
        "date": pd.to_datetime(["2022-04-08", "2022-04-08"]),
        "long_view": [1, 0],
    })
    seen_groups = {}

    class _Recorder:
        def fit(self, X_train, y_train, X_val, y_val, groups=None):
            seen_groups["value"] = groups

        def predict(self, X):
            return np.zeros(len(X))

    import pipeline.train as train_module
    original = train_module._new_model
    train_module._new_model = lambda config, seed: _Recorder()
    try:
        _fit_and_predict(
            {"model": "fake", "features": ["user_id", "video_id"],
             "hparams": {}, "negative_sampling": "all"},
            frame, None, [], seed=0,
        )
    finally:
        train_module._new_model = original

    groups = seen_groups["value"]
    assert isinstance(groups, tuple)
    assert len(groups) == 2


def test_c9_config_schema_accepts_deepfm_mtl():
    """agent/schema.py's UNIMPLEMENTED_MODELS gate must no longer reject it —
    C9 landing is exactly the change that was supposed to lift this."""
    from agent.schema import Config

    config = Config.model_validate({"model": "deepfm_mtl", "hparams": {
        "aux_click_weight": 0.3, "aux_like_weight": 0.1,
    }})

    assert config.model == "deepfm_mtl"


def test_c9_real_run_experiment_smoke_completes_end_to_end():
    """The full pipeline.train.run_experiment integration, not just the model
    class in isolation — proves the AUX_TARGETS plumbing survives negative
    sampling, _matrix construction, and the smoke-tier correctness checks."""
    from pipeline.train import run_experiment

    result = run_experiment(
        {"model": "deepfm_mtl", "features": ["user_id", "video_id", "author_id", "tab"],
         "hparams": {"emb_dim": 4, "mlp": [8], "max_epochs": 1, "batch_size": 64,
                     "aux_click_weight": 0.2, "aux_like_weight": 0.2}},
        fidelity="smoke",
        seed=0,
    )

    assert result["status"] == "ok"
