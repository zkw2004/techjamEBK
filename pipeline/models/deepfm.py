"""CPU-first DeepFM with the Appendix A.4 linear-time interaction term."""

from __future__ import annotations

import numpy as np

from pipeline.models import register


def _build_network(
    field_dims,
    emb_dim: int,
    mlp_dims,
    dropout: float,
    *,
    aux_heads=(),
    lhuc: bool = False,
    senet: bool = False,
    senet_reduction: int = 4,
    user_field_index: int | None = None,
):
    """Build lazily so importing the registry does not initialise Torch.

    One class serves every combination of the optional blocks: auxiliary heads
    (C9), LHUC/PPNet user gating (C11) and SENet field weighting (C12). They are
    genuinely independent — each may be on or off without the others — so
    enumerating them as separate network classes would mean eight near-identical
    bodies, and the earlier two-factory split had already duplicated the primary
    path once.

    **Construction order is load-bearing.** Embeddings, linear, bias and the MLP
    are built first, in exactly the order the original single-task network built
    them, so a fresh model given the same seed initialises the primary path's
    parameters identically no matter which optional blocks exist. Matching
    *parameters* is not sufficient by itself, though: constructing any
    ``nn.Linear`` draws from torch's global RNG for its default init, which
    advances the same stream ``nn.Dropout`` reads from during training — so even
    with identical parameters, the first dropout mask drawn in training would
    differ purely because an optional block existed. Every optional block is
    therefore constructed inside an RNG save/restore, which undoes that draw's
    effect on the shared stream. Together with none of the blocks carrying
    dropout of their own (so *computing* them later costs no extra randomness
    either), this is what lets a disabled block reproduce the plain network
    exactly rather than approximately.

    ``forward`` returns a bare tensor when there are no auxiliary heads and a
    ``(primary, aux_dict)`` pair when there are — matching what each caller
    already expects.
    """
    import torch
    from torch import nn

    class SENetBlock(nn.Module):
        """FiBiNET squeeze-and-excitation over fields (C12).

        Squeeze each field's embedding to a scalar by mean-pooling, learn a
        per-field weight from the resulting field vector, and rescale. The point
        on this task is that the weights are computed *per row*, so the same
        field can matter for one user's candidates and not another's — which is
        the only kind of signal a within-user metric can see (5.4).

        Initialised to the identity: the final excitation layer has zero weight
        and zero bias, and ``2 * sigmoid(0) == 1``, so it emits exactly 1.0 for
        every field before any training step. That makes ``senet=True`` at
        initialisation indistinguishable from ``senet=False``, so an A/B between
        them measures the mechanism rather than a different random starting
        point. Zeroing the last layer costs nothing in trainability — its own
        gradient is non-zero, so it leaves zero after the first step, the same
        trick ResNet's zero-init residual and LoRA's zero-init B matrix rely on.

        Two deviations from the FiBiNET paper, both forced by this task's field
        count. The paper squeezes through a ReLU bottleneck of ``n_fields /
        r``; with the five official fields and the usual r that is a *single*
        unit, and a single ReLU that happens to sit negative emits zero for
        every row — collapsing the excitation to its bias, i.e. one constant
        weight per field. A per-field constant is absorbed into the embedding
        table and is invisible to a within-user metric (5.4), so the block would
        silently become a no-op that still reported itself as active. Hence a
        floor of two hidden units, and ``2 * sigmoid`` rather than ReLU on the
        output: sigmoid has no dead region, so a weight can shrink toward zero
        without the row losing its gradient, and the gate lands on the same
        (0, 2)-around-identity scale LHUC uses.
        """

        def __init__(self, n_fields: int, reduction: int):
            super().__init__()
            hidden = max(2, n_fields // reduction)
            self.squeeze = nn.Linear(n_fields, hidden)
            self.excite = nn.Linear(hidden, n_fields)
            nn.init.zeros_(self.excite.weight)
            nn.init.zeros_(self.excite.bias)

        def forward(self, embeddings):
            hidden = torch.relu(self.squeeze(embeddings.mean(dim=2)))
            weights = 2.0 * torch.sigmoid(self.excite(hidden))
            return embeddings * weights.unsqueeze(-1)

    class LHUCGate(nn.Module):
        """PPNet-style user-conditioned modulation of the MLP's hidden units (C11).

        Projects the user's own embedding to one multiplicative gate per hidden
        unit of every MLP layer, squashed through ``2 * sigmoid`` so a gate spans
        (0, 2) and is exactly 1.0 — the identity — when the projection outputs
        zero. Both weight and bias start at zero, so like SENet above this begins
        as an exact no-op and diverges only as it learns.

        The projection reads a *detached* user embedding, following PPNet: the
        gate is conditioned on who the user is, but gradients from the gate do
        not reshape the shared embedding table itself. Without the stop-gradient
        the gate and the embedding chase each other, and the embedding stops
        being a clean representation of the user for the rest of the network.

        Parameterising the gate on the embedding rather than on a per-user table
        also keeps this small (``emb_dim x sum(mlp_dims)`` parameters, not
        ``n_users x sum(mlp_dims)``) and lets it produce a sensible gate for a
        user held out of training.
        """

        def __init__(self, emb_dim: int, mlp_dims):
            super().__init__()
            self.splits = [int(size) for size in mlp_dims]
            self.project = nn.Linear(emb_dim, sum(self.splits))
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)

        def forward(self, user_embedding):
            gates = 2.0 * torch.sigmoid(self.project(user_embedding))
            return torch.split(gates, self.splits, dim=1)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in field_dims])
            self.linear = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
            self.bias = nn.Parameter(torch.zeros(1))

            layers = []
            input_dim = len(field_dims) * emb_dim
            for hidden_dim in mlp_dims:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)

            self.user_field_index = user_field_index
            # Everything below is optional and therefore quarantined -- see the
            # construction-order note in this function's docstring.
            embed_total = len(field_dims) * emb_dim
            rng_state = torch.get_rng_state()
            try:
                self.senet = SENetBlock(len(field_dims), senet_reduction) if senet else None
                self.lhuc = LHUCGate(emb_dim, mlp_dims) if lhuc else None
                self.aux_heads = nn.ModuleDict(
                    {name: nn.Linear(embed_total, 1) for name in aux_heads}
                )
            finally:
                torch.set_rng_state(rng_state)

        def _deep(self, flat, gates):
            """The MLP, optionally gated after each hidden block.

            Gates are applied to each hidden block's output -- that is, after its
            dropout -- so what the next layer receives is exactly what was scaled.
            """
            if gates is None:
                return self.mlp(flat)
            index = 0
            for layer in self.mlp:
                flat = layer(flat)
                if isinstance(layer, nn.Dropout):
                    flat = flat * gates[index]
                    index += 1
            return flat

        def forward(self, X):
            raw = torch.stack(
                [embedding(X[:, index]) for index, embedding in enumerate(self.embeddings)],
                dim=1,
            )
            order_one = (
                torch.stack(
                    [linear(X[:, index]) for index, linear in enumerate(self.linear)],
                    dim=0,
                ).sum(dim=0)
                + self.bias
            )
            # SENet reweights the embeddings feeding *both* the second-order term
            # and the MLP ("before the FM/DeepFM interaction layer"). The
            # first-order term reads its own separate weight table and is left
            # alone, as in FiBiNET.
            embeddings = raw if self.senet is None else self.senet(raw)
            order_two = DeepFMModel._second_order(embeddings)
            flat = embeddings.flatten(start_dim=1)
            gates = None
            if self.lhuc is not None:
                # Read the user embedding before SENet touched it, so the gate is
                # conditioned on the user rather than on the other block's output.
                gates = self.lhuc(raw[:, self.user_field_index, :].detach())
            primary = (order_one + order_two + self._deep(flat, gates)).squeeze(1)
            if not self.aux_heads:
                return primary
            return primary, {name: head(flat).squeeze(1) for name, head in self.aux_heads.items()}

    return Network()


def _network(field_dims, emb_dim: int, mlp_dims, dropout: float, **blocks):
    """The single-task network: one score per row, no auxiliary heads."""
    return _build_network(field_dims, emb_dim, mlp_dims, dropout, **blocks)


def _multitask_network(field_dims, emb_dim: int, mlp_dims, dropout: float, aux_heads, **blocks):
    """The single-task network plus a linear probe per auxiliary target."""
    return _build_network(
        field_dims, emb_dim, mlp_dims, dropout, aux_heads=aux_heads, **blocks
    )


@register("deepfm")
class DeepFMModel:
    native_backend = "torch"  # see pipeline/models/__init__.py BACKENDS

    def __init__(
        self,
        emb_dim: int = 16,
        mlp=(256, 128, 64),
        dropout: float = 0.2,
        lr: float = 0.001,
        l2: float = 1e-6,
        # 40, matching FM, because that is where this model actually converges
        # -- not a guess. Measured on the real split via `_score_folds`, which
        # is what chooses the refit budget in `_run_full`:
        #
        #   cap   3 -> best_epochs [ 3,  3,  3]  folds [0.5530, 0.5276, 0.5234]
        #   cap  10 -> best_epochs [10, 10, 10]  folds [0.5652, 0.5377, 0.5332]
        #   cap  25 -> best_epochs [25, 25, 25]  folds [0.5716, 0.5465, 0.5391]
        #   cap  50 -> best_epochs [41, 40, 35]  folds [0.5751, 0.5508, 0.5419]
        #   cap 100 -> best_epochs [41, 40, 35]  folds [0.5751, 0.5508, 0.5419]
        #
        # Every fold pinned to the cap until 50, and caps 50 and 100 agree
        # exactly, so 35-41 is real convergence rather than another ceiling.
        # The old default of 3 stopped training at ~7% of what the model needed
        # and cost roughly 0.019 primary per fold. It came from trap 11's "CTR
        # models at this scale peak after 1 to 3 epochs then degrade", which
        # this measurement contradicts for this model -- validation loss fell
        # monotonically for ~40 epochs and patience never fired. See trap 11.
        max_epochs: int = 40,
        batch_size: int = 4096,
        patience: int = 1,
        seed: int = 42,
        feature_names: list[str] | None = None,
        num_threads: int = 1,
        lhuc: bool = False,
        senet: bool = False,
        senet_reduction: int = 4,
        **hparams,
    ):
        if emb_dim <= 0 or lr <= 0 or l2 < 0:
            raise ValueError("emb_dim and lr must be positive and l2 must be non-negative")
        # patience was hard-locked to exactly 1 on trap 11's premise. The
        # measurement above shows early stopping firing at 35-41 epochs, well
        # short of the cap, so patience=1 is not stopping this model
        # prematurely and stays the default. The lock itself is lifted: it also
        # prevented C6/Optuna from ever exploring the parameter, and there is no
        # principled reason to forbid a longer wait on a noisier loss curve.
        if max_epochs <= 0 or batch_size <= 0 or patience < 1 or num_threads <= 0:
            raise ValueError(
                "epochs, batch size, and threads must be positive; "
                "patience must be at least 1"
            )
        if not mlp or any(int(size) <= 0 for size in mlp):
            raise ValueError("mlp must contain positive hidden dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if int(senet_reduction) < 1:
            raise ValueError("senet_reduction must be a positive integer")
        # C11/C12: both default off, so every existing config keeps the exact
        # network it had. See _build_network for why "off" is bit-exact and not
        # merely close.
        self.lhuc = bool(lhuc)
        self.senet = bool(senet)
        self.senet_reduction = int(senet_reduction)
        self.emb_dim = emb_dim
        self.mlp_dims = tuple(int(size) for size in mlp)
        self.dropout = dropout
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.feature_names = list(feature_names or [])
        self.num_threads = num_threads
        self.vocabs: list[dict[object, int] | None] = []
        self.numeric_edges: list[np.ndarray | None] = []
        self.field_dims: list[int] = []
        self.network = None
        self.best_epoch: int | None = None

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        import torch
        from torch import nn

        del groups
        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train, dtype=np.float32)
        self._validate_xy(X_train, y_train, "train")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be None")
        supplied_feature_names = bool(self.feature_names)
        if not self.feature_names:
            self.feature_names = [f"field_{index}" for index in range(X_train.shape[1])]
        if len(self.feature_names) != X_train.shape[1]:
            raise ValueError("feature_names must match the matrix field count")

        self._fit_encoder(X_train, all_categorical=not supplied_feature_names)
        encoded_train = self._encode(X_train)
        encoded_val = validation_labels = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=object)
            validation_labels = np.asarray(y_val, dtype=np.float32)
            self._validate_xy(X_val, validation_labels, "validation")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("train and validation matrices must have the same field count")
            encoded_val = self._encode(X_val)

        torch.set_num_threads(self.num_threads)
        torch.manual_seed(self.seed)
        self.network = _network(
            self.field_dims,
            self.emb_dim,
            self.mlp_dims,
            self.dropout,
            **self._block_kwargs(),
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.BCEWithLogitsLoss()
        train_tensor = torch.as_tensor(encoded_train, dtype=torch.long)
        label_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_state = None
        bad_epochs = 0
        self.best_epoch = None

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            indices = rng.permutation(len(train_tensor))
            for start in range(0, len(indices), self.batch_size):
                batch = torch.as_tensor(indices[start : start + self.batch_size], dtype=torch.long)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(self.network(train_tensor[batch]), label_tensor[batch])
                loss.backward()
                optimiser.step()

            if encoded_val is None:
                self.best_epoch = epoch
                continue
            validation_loss = self._validation_loss(encoded_val, validation_labels, loss_fn)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                bad_epochs = 0
                self.best_epoch = epoch
                best_state = {
                    name: parameter.detach().clone()
                    for name, parameter in self.network.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def predict(self, X) -> np.ndarray:
        import torch

        if self.network is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be two-dimensional with the fitted field count")
        encoded = torch.as_tensor(self._encode(X), dtype=torch.long)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                outputs.append(self.network(encoded[start : start + self.batch_size]).cpu())
        if not outputs:
            return np.array([], dtype=np.float32)
        return torch.cat(outputs).numpy()

    def _block_kwargs(self) -> dict:
        """Optional-block arguments for ``_build_network``, validated.

        LHUC gates on the user's own embedding, so it needs to know which field
        that is. Raising here rather than silently disabling matters: a
        `lhuc: true` hparam that quietly did nothing would produce a node whose
        config claims a mechanism its score never had -- the same silent-no-op
        failure that cost a run eleven iterations of identical FM variants
        (agent/schema.py, SUPPORTED_LOSSES).
        """
        user_field_index = None
        if self.lhuc:
            if "user_id" not in self.feature_names:
                raise ValueError(
                    "lhuc=True gates on the user embedding and so requires a "
                    f"'user_id' field; got features {self.feature_names!r}"
                )
            user_field_index = self.feature_names.index("user_id")
        return {
            "lhuc": self.lhuc,
            "senet": self.senet,
            "senet_reduction": self.senet_reduction,
            "user_field_index": user_field_index,
        }

    @staticmethod
    def _second_order(embeddings):
        summed = embeddings.sum(dim=1)
        return 0.5 * (summed.pow(2) - embeddings.pow(2).sum(dim=1)).sum(
            dim=1,
            keepdim=True,
        )

    @staticmethod
    def _validate_xy(X: np.ndarray, y: np.ndarray, name: str) -> None:
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"{name} matrix must be non-empty and two-dimensional")
        if y.ndim != 1 or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError(f"{name} labels must be finite with one value per row")

    def _fit_encoder(self, X: np.ndarray, all_categorical: bool) -> None:
        from pipeline.data import FIELDS

        self.vocabs = []
        self.numeric_edges = []
        self.field_dims = []
        for index, name in enumerate(self.feature_names):
            if all_categorical or name in FIELDS:
                vocab = {}
                for value in X[:, index]:
                    if value not in vocab:
                        vocab[value] = len(vocab)
                self.vocabs.append(vocab)
                self.numeric_edges.append(None)
                self.field_dims.append(len(vocab) + 1)
                continue
            values = np.asarray(X[:, index], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"numeric feature {name!r} must be finite")
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 33)[1:-1]))
            self.vocabs.append(None)
            self.numeric_edges.append(edges)
            self.field_dims.append(len(edges) + 1)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.int64)
        for index, vocab in enumerate(self.vocabs):
            if vocab is not None:
                unknown = len(vocab)
                encoded[:, index] = [vocab.get(value, unknown) for value in X[:, index]]
                continue
            values = np.asarray(X[:, index], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"numeric feature {self.feature_names[index]!r} must be finite")
            encoded[:, index] = np.searchsorted(self.numeric_edges[index], values)
        return encoded

    def _validation_loss(self, X: np.ndarray, y: np.ndarray, loss_fn) -> float:
        import torch

        encoded = torch.as_tensor(X, dtype=torch.long)
        labels = torch.as_tensor(y, dtype=torch.float32)
        total = 0.0
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                stop = start + self.batch_size
                loss = loss_fn(self.network(encoded[start:stop]), labels[start:stop])
                total += float(loss) * len(encoded[start:stop])
        return total / len(encoded)


@register("deepfm_mtl")
class DeepFMMultiTask(DeepFMModel):
    """Rung 5 (C9): shared embeddings, auxiliary sigmoid heads for is_click
    and is_like, `long_view` read at inference. Read only the long_view head;
    the other two exist to force the shared embeddings toward richer
    representations, trained but never scored.

    The auxiliary labels are same-row `is_click`/`is_like` — forbidden as
    input features (``FORBIDDEN_SAME_ROW``, pipeline/features.py) but legal as
    training targets. They therefore cannot travel through the frozen
    ``fit(X, y, X_val, y_val, groups=None)`` signature's ``X``/``y`` slots, and
    the ``BaseModel`` protocol has no sixth parameter to add one — the
    contract is frozen (Section 8.9). ``groups`` is the one parameter models
    already interpret per-model (LightGBM accepts a plain ``(train, val)``
    tuple *or* a ``{"train":..., "val":...}`` dict; FM unpacks
    ``groups[1]``), so this model extends that existing precedent rather than
    inventing a new side channel: ``pipeline.train._fit_and_predict`` detects
    ``AUX_TARGETS`` on the model class and passes
    ``{"train": ..., "val": ..., "aux_train": {...}, "aux_val": {...}}``
    instead of the bare tuple every other model still receives unchanged.
    """

    AUX_TARGETS = ("is_click", "is_like")

    def __init__(
        self,
        aux_click_weight: float = 0.0,
        aux_like_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if aux_click_weight < 0 or aux_like_weight < 0:
            raise ValueError("aux head weights must be non-negative")
        # Optuna-tunable: both flow through Config.hparams -> **kwargs like
        # every other hparam (pipeline/tune.py's `_suggest` is generic over
        # hparam names, so a search_space entry for either needs no new code
        # here or in the tuner).
        self.aux_click_weight = float(aux_click_weight)
        self.aux_like_weight = float(aux_like_weight)

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        import torch
        from torch import nn

        if not isinstance(groups, dict) or "aux_train" not in groups:
            raise ValueError(
                "DeepFMMultiTask requires auxiliary training targets: call it "
                "through pipeline.train.run_experiment (which builds "
                "groups={'aux_train': {...}, 'aux_val': {...}, ...} whenever "
                "AUX_TARGETS is declared on the model class), not with the "
                "bare (train_users, val_users) tuple other models accept"
            )
        # Only the training-side aux labels are needed: early stopping and
        # checkpoint selection use the primary head's validation loss alone
        # (see the note below), so there is nothing to do with aux_val even
        # when the caller supplies one.
        aux_train = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in groups["aux_train"].items()
        }

        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train, dtype=np.float32)
        self._validate_xy(X_train, y_train, "train")
        for name, values in aux_train.items():
            if len(values) != len(y_train) or not np.isfinite(values).all():
                raise ValueError(f"aux target {name!r} must be finite, one value per training row")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be None")
        supplied_feature_names = bool(self.feature_names)
        if not self.feature_names:
            self.feature_names = [f"field_{index}" for index in range(X_train.shape[1])]
        if len(self.feature_names) != X_train.shape[1]:
            raise ValueError("feature_names must match the matrix field count")

        self._fit_encoder(X_train, all_categorical=not supplied_feature_names)
        encoded_train = self._encode(X_train)
        encoded_val = validation_labels = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=object)
            validation_labels = np.asarray(y_val, dtype=np.float32)
            self._validate_xy(X_val, validation_labels, "validation")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("train and validation matrices must have the same field count")
            encoded_val = self._encode(X_val)

        torch.set_num_threads(self.num_threads)
        torch.manual_seed(self.seed)
        self.network = _multitask_network(
            self.field_dims, self.emb_dim, self.mlp_dims, self.dropout, self.AUX_TARGETS,
            **self._block_kwargs(),
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.BCEWithLogitsLoss()
        train_tensor = torch.as_tensor(encoded_train, dtype=torch.long)
        label_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        aux_tensors = {
            name: torch.as_tensor(values, dtype=torch.float32) for name, values in aux_train.items()
        }
        aux_weights = {"is_click": self.aux_click_weight, "is_like": self.aux_like_weight}
        rng = np.random.default_rng(self.seed)
        best_loss = np.inf
        best_state = None
        bad_epochs = 0
        self.best_epoch = None

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            indices = rng.permutation(len(train_tensor))
            for start in range(0, len(indices), self.batch_size):
                batch = torch.as_tensor(indices[start : start + self.batch_size], dtype=torch.long)
                optimiser.zero_grad(set_to_none=True)
                primary_logits, aux_logits = self.network(train_tensor[batch])
                loss = loss_fn(primary_logits, label_tensor[batch])
                # A weight of exactly 0 never enters the sum below, so with
                # both weights 0 `loss` is the identical expression the
                # single-task network computes on the same batch — not an
                # approximation of it (see the class docstring / test_deepfm).
                for name, weight in aux_weights.items():
                    if weight and name in aux_tensors:
                        loss = loss + weight * loss_fn(aux_logits[name], aux_tensors[name][batch])
                loss.backward()
                optimiser.step()

            if encoded_val is None:
                self.best_epoch = epoch
                continue
            # Early stopping and checkpoint selection use the primary head's
            # validation loss only ("auxiliary heads affect training loss
            # only; output shape/interface unchanged", C9's acceptance
            # criterion) -- the aux heads never gate which epoch wins.
            validation_loss = self._primary_validation_loss(encoded_val, validation_labels, loss_fn)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                bad_epochs = 0
                self.best_epoch = epoch
                best_state = {
                    name: parameter.detach().clone()
                    for name, parameter in self.network.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def predict(self, X) -> np.ndarray:
        """Long_view head only -- the frozen interface (8.9) returns one
        score per row, and the aux heads exist to shape training, not to be
        read."""
        import torch

        if self.network is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be two-dimensional with the fitted field count")
        encoded = torch.as_tensor(self._encode(X), dtype=torch.long)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                primary_logits, _aux = self.network(encoded[start : start + self.batch_size])
                outputs.append(primary_logits.cpu())
        if not outputs:
            return np.array([], dtype=np.float32)
        return torch.cat(outputs).numpy()

    def _primary_validation_loss(self, X: np.ndarray, y: np.ndarray, loss_fn) -> float:
        import torch

        encoded = torch.as_tensor(X, dtype=torch.long)
        labels = torch.as_tensor(y, dtype=torch.float32)
        total = 0.0
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                stop = start + self.batch_size
                primary_logits, _aux = self.network(encoded[start:stop])
                loss = loss_fn(primary_logits, labels[start:stop])
                total += float(loss) * len(encoded[start:stop])
        return total / len(encoded)
