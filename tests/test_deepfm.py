"""C5 DeepFM behavior and factorization-machine math."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pipeline.data import DATA_DIR
from pipeline.models.deepfm import DeepFMModel

# The C9/C11/C12 end-to-end tests below call pipeline.train.run_experiment(),
# which calls pipeline.data.load() for real -- unlike the rest of this file,
# which drives DeepFMModel directly against synthetic arrays. CI's `tests`
# job deliberately never fetches the (445MB, git-ignored) archive (see
# .github/workflows/ci.yml's own comment: "No dataset needed"), so these two
# must be gated the same way every other archive-dependent test in this repo
# is (tests/test_data_split.py, tests/test_models.py) or they hard-fail on
# every environment without the dataset -- which is exactly what turned
# `main` red for five consecutive merges after they landed ungated.
requires_kuairand_data = pytest.mark.skipif(
    not (DATA_DIR / "video_features_basic_pure.csv").is_file(),
    reason="requires the ignored KuaiRand-Pure archive; run `make data` locally",
)

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


# --- C11 LHUC / C12 SENet -----------------------------------------------

_MTL_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]

_BLOCK_KWARGS = dict(emb_dim=8, mlp=(16, 8), dropout=0.2, lr=0.02, max_epochs=3,
                     batch_size=16, patience=1, seed=11)


def _fit_deepfm(X, y, users, **overrides):
    model = DeepFMModel(feature_names=list(_MTL_FIELDS), **{**_BLOCK_KWARGS, **overrides})
    model.fit(X, y, None, None, groups=(users, None))
    return model


@pytest.mark.parametrize("block", ["lhuc", "senet"])
def test_c11_c12_disabled_block_reproduces_plain_deepfm_exactly(block):
    """A block that is off must leave the network bit-identical, not merely
    close. Constructing any nn.Linear draws from torch's global RNG -- the same
    stream nn.Dropout reads during training -- so an unguarded implementation
    would shift every dropout mask purely because the block's code existed,
    even with the block switched off. This is the C9 lesson applied to C11/C12;
    _build_network's RNG save/restore is what this validates the effect of."""
    X, y, _click, _like, users = _mtl_rows()

    plain = _fit_deepfm(X, y, users)
    off = _fit_deepfm(X, y, users, **{block: False})

    np.testing.assert_array_equal(plain.predict(X), off.predict(X))


@pytest.mark.parametrize("block", ["lhuc", "senet"])
def test_c11_c12_enabled_block_is_the_identity_before_training(block):
    """Both blocks initialise to an exact no-op -- LHUC's 2*sigmoid(0) and
    SENet's zero-initialised excitation both emit exactly 1.0. So before any
    training step, on and off must produce the same scores.

    This is what makes an A/B between them a measurement of the *mechanism*: if
    the blocks started at a random transform instead, an observed delta would be
    confounded with simply having perturbed the starting point. Asserted on
    freshly built networks rather than through fit(), because the constructor
    (rightly) rejects the lr=0 that freezing training would otherwise need."""
    import torch

    from pipeline.models.deepfm import _build_network

    field_dims = [6, 6, 4, 2, 4]
    common = dict(emb_dim=8, mlp_dims=(16, 8), dropout=0.0, user_field_index=0)
    batch = torch.zeros(4, len(field_dims), dtype=torch.long)
    batch[1:, 0] = torch.arange(1, 4)

    torch.manual_seed(11)
    off = _build_network(field_dims, **common)
    torch.manual_seed(11)
    on = _build_network(field_dims, **{**common, block: True})

    off.eval()
    on.eval()
    with torch.no_grad():
        torch.testing.assert_close(off(batch), on(batch), rtol=0, atol=1e-6)


@pytest.mark.parametrize("block", ["lhuc", "senet"])
def test_c11_c12_enabled_block_measurably_changes_training(block):
    """The inverse guard. Without it the two tests above would both pass on an
    implementation that wired the blocks up to nothing at all."""
    X, y, _click, _like, users = _mtl_rows()

    off = _fit_deepfm(X, y, users)
    on = _fit_deepfm(X, y, users, **{block: True})

    assert not np.allclose(off.predict(X), on.predict(X))


def test_c11_lhuc_gate_is_user_conditioned():
    """The mechanism's whole claim: two users with otherwise identical rows get
    different hidden-unit scaling. Asserted on the gate directly, because at the
    score level a difference could equally come from the user embedding already
    feeding the MLP."""
    import torch

    X, y, _click, _like, users = _mtl_rows()
    model = _fit_deepfm(X, y, users, lhuc=True)

    encoded = torch.as_tensor(model._encode(X), dtype=torch.long)
    with torch.no_grad():
        embedded = torch.stack(
            [embedding(encoded[:, i]) for i, embedding in enumerate(model.network.embeddings)],
            dim=1,
        )
        gates = model.network.lhuc(embedded[:, model.network.user_field_index, :])

    first_layer = gates[0]
    assert not torch.allclose(first_layer[0], first_layer[-1])
    # 2*sigmoid bounds every gate to (0, 2), centred on the identity.
    assert float(first_layer.min()) > 0.0 and float(first_layer.max()) < 2.0


def test_c11_lhuc_produces_one_gate_group_per_hidden_layer():
    X, y, _click, _like, users = _mtl_rows()
    model = _fit_deepfm(X, y, users, lhuc=True)

    import torch

    with torch.no_grad():
        gates = model.network.lhuc(torch.zeros(3, model.emb_dim))

    assert [tuple(gate.shape) for gate in gates] == [(3, 16), (3, 8)]


def test_c11_lhuc_does_not_backpropagate_into_the_shared_embedding():
    """PPNet's stop-gradient. The gate is conditioned on the user embedding but
    must not reshape it, or the gate and the embedding chase each other and the
    embedding stops being a clean user representation for the rest of the net.

    Asserted with a forward hook on the real network, capturing what
    Network.forward actually hands the gate -- detaching inside the test instead
    would only prove the test detaches."""
    import torch

    X, y, _click, _like, users = _mtl_rows()
    network = _fit_deepfm(X, y, users, lhuc=True).network
    captured = {}
    handle = network.lhuc.register_forward_pre_hook(
        lambda _module, args: captured.setdefault("input", args[0])
    )
    try:
        network.eval()
        network(torch.zeros(2, len(network.embeddings), dtype=torch.long))
    finally:
        handle.remove()

    assert captured["input"].requires_grad is False


def test_c11_lhuc_requires_a_user_id_field():
    """Silently disabling would produce a node whose config claims a mechanism
    its score never had."""
    X = np.asarray([["v1", "a1"], ["v2", "a2"]], dtype=object)
    y = np.asarray([1.0, 0.0])
    model = DeepFMModel(feature_names=["video_id", "author_id"], lhuc=True,
                        **{**_BLOCK_KWARGS, "batch_size": 2})

    with pytest.raises(ValueError, match="requires a 'user_id' field"):
        model.fit(X, y, None, None, groups=(np.array(["u1", "u2"]), None))


def test_c12_senet_reweights_fields_per_row():
    """SENet's weights must vary across rows, not just across fields: a
    per-field constant would be absorbed into the embedding table and could not
    help a within-user comparison (AGENT_PLAN 5.4)."""
    import torch

    X, y, _click, _like, users = _mtl_rows()
    model = _fit_deepfm(X, y, users, senet=True)

    encoded = torch.as_tensor(model._encode(X), dtype=torch.long)
    with torch.no_grad():
        embedded = torch.stack(
            [embedding(encoded[:, i]) for i, embedding in enumerate(model.network.embeddings)],
            dim=1,
        )
        reweighted = model.network.senet(embedded)
        weights = (reweighted / embedded)[:, :, 0]

    assert weights.shape == (len(X), len(model.field_dims))
    assert not torch.allclose(weights[0], weights[-1])


def test_c12_rejects_a_non_positive_senet_reduction():
    with pytest.raises(ValueError, match="senet_reduction must be a positive integer"):
        DeepFMModel(senet=True, senet_reduction=0)


def test_c11_c12_compose_and_stay_deterministic():
    """They are independent mechanisms and must be usable together; determinism
    under a fixed seed is the C1 same-seed guarantee every model owes."""
    X, y, _click, _like, users = _mtl_rows()

    def fit_once():
        return _fit_deepfm(X, y, users, lhuc=True, senet=True).predict(X)

    both = fit_once()
    np.testing.assert_array_equal(both, fit_once())
    assert np.isfinite(both).all()
    assert not np.allclose(both, _fit_deepfm(X, y, users).predict(X))


def test_c11_c12_reach_the_multitask_model_too():
    """deepfm_mtl inherits __init__, so the hparams are accepted there. If they
    were not also wired into its network they would be silently ignored -- a
    config claiming a mechanism it never had."""
    from pipeline.models.deepfm import DeepFMMultiTask

    X, y, click, like, users = _mtl_rows()

    def fit_once(**blocks):
        model = DeepFMMultiTask(aux_click_weight=0.3, aux_like_weight=0.2,
                                feature_names=list(_MTL_FIELDS),
                                **{**_BLOCK_KWARGS, **blocks})
        model.fit(X, y, None, None, groups=_mtl_groups(users, click, like))
        return model.predict(X)

    plain = fit_once()
    np.testing.assert_array_equal(plain, fit_once(lhuc=False, senet=False))
    assert not np.allclose(plain, fit_once(lhuc=True))
    assert not np.allclose(plain, fit_once(senet=True))


@requires_kuairand_data
def test_c11_c12_run_through_run_experiment_end_to_end():
    """The full pipeline.train.run_experiment path, not the model in isolation:
    proves the hparams survive Config validation and _matrix construction."""
    from pipeline.train import run_experiment

    result = run_experiment(
        {"model": "deepfm", "features": ["user_id", "video_id", "author_id", "tab"],
         "hparams": {"emb_dim": 4, "mlp": [8], "max_epochs": 1, "batch_size": 64,
                     "lhuc": True, "senet": True}},
        fidelity="smoke",
        seed=0,
    )

    assert result["status"] == "ok"


@requires_kuairand_data
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


# --- epoch budget --------------------------------------------------------

def test_deepfm_default_epoch_budget_allows_convergence():
    """The default must let the model reach its optimum, not stop at 7% of it.

    Pins the *effect*, not the constant: this fixture needs well over three
    epochs at the default learning rate, so the old `max_epochs=3` default
    fails it while any budget large enough to converge passes. Measured on the
    real split, folds pinned to the cap at 3/10/25 epochs and only stopped on
    their own at 35-41 -- costing ~0.019 primary per fold. See the measurement
    table in DeepFMModel.__init__ and trap 11.
    """
    rows, labels = [], []
    for user in range(60):
        for item in range(8):
            rows.append([f"u{user}", f"v{item}", f"a{item % 4}", f"t{user % 3}",
                         str(item % 5)])
            labels.append(float(item % 2))
    X = np.asarray(rows, dtype=object)
    y = np.asarray(labels, dtype=float)
    kwargs = dict(emb_dim=8, mlp=(16,), dropout=0.0, lr=0.001, batch_size=32, seed=3)

    starved = DeepFMModel(max_epochs=3, **kwargs)
    starved.fit(X, y, X, y, groups=(X[:, 0], X[:, 0]))
    default = DeepFMModel(**kwargs)
    default.fit(X, y, X, y, groups=(X[:, 0], X[:, 0]))

    def separation(model):
        scores = model.predict(X)
        return float(scores[y == 1].mean() - scores[y == 0].mean())

    assert starved.best_epoch == 3, "fixture must be capped at the old default"
    assert default.best_epoch > 3, "the default budget must allow more training"
    assert separation(default) > separation(starved)


def test_deepfm_accepts_a_patience_above_one_but_still_defaults_to_one():
    """The lock on patience==1 is lifted so C6/Optuna can explore it; the
    default is unchanged, since the measurement shows patience=1 is not what
    was cutting training short."""
    assert DeepFMModel().patience == 1
    assert DeepFMModel(patience=4).patience == 4
    with pytest.raises(ValueError, match="patience must be at least 1"):
        DeepFMModel(patience=0)
