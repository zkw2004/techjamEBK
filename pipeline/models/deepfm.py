"""CPU-first DeepFM with the Appendix A.4 linear-time interaction term."""

from __future__ import annotations

import numpy as np

from pipeline.models import register


def _network(field_dims, emb_dim: int, mlp_dims, dropout: float):
    """Build lazily so importing the registry does not initialise Torch."""
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self, field_dims, emb_dim: int, mlp_dims, dropout: float):
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

        def forward(self, X):
            embeddings = torch.stack(
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
            order_two = DeepFMModel._second_order(embeddings)
            deep = self.mlp(embeddings.flatten(start_dim=1))
            return (order_one + order_two + deep).squeeze(1)


    return Network(field_dims, emb_dim, mlp_dims, dropout)


def _multitask_network(field_dims, emb_dim: int, mlp_dims, dropout: float, aux_heads):
    """The single-task network plus a linear probe per auxiliary target.

    Construction order matters: embeddings, linear, bias and the MLP are built
    in exactly the order ``_network`` builds them, so a fresh model given the
    same seed initialises the primary path's parameters identically whether or
    not auxiliary heads exist. Matching *parameters* is not sufficient by
    itself, though: constructing the aux heads' ``nn.Linear`` layers draws
    from torch's global RNG for their default init, which advances the same
    stream ``nn.Dropout`` reads from during training — so even with identical
    parameters, the first dropout mask drawn in training would differ purely
    because the aux heads existed. The save/restore around their construction
    below undoes that draw's effect on the shared stream, so training sees
    the exact same RNG position it would in the single-task network. Together
    with the aux heads carrying no dropout of their own (so *computing* them
    later costs no extra randomness either), this is what lets aux weights of
    0 reproduce the single-task model exactly rather than approximately.
    """
    import torch
    from torch import nn

    class MultiTaskNetwork(nn.Module):
        def __init__(self, field_dims, emb_dim: int, mlp_dims, dropout: float, aux_heads):
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

            # Quarantined: nn.Linear.reset_parameters() always draws its
            # default init from torch's global generator, with no hook to
            # give it an isolated one. Snapshotting and restoring the global
            # RNG state around construction is what keeps that draw from
            # shifting the position every later nn.Dropout call reads from.
            embed_total = len(field_dims) * emb_dim
            rng_state = torch.get_rng_state()
            self.aux_heads = nn.ModuleDict(
                {name: nn.Linear(embed_total, 1) for name in aux_heads}
            )
            torch.set_rng_state(rng_state)

        def forward(self, X):
            embeddings = torch.stack(
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
            order_two = DeepFMModel._second_order(embeddings)
            flat = embeddings.flatten(start_dim=1)
            deep = self.mlp(flat)
            primary = (order_one + order_two + deep).squeeze(1)
            aux = {name: head(flat).squeeze(1) for name, head in self.aux_heads.items()}
            return primary, aux

    return MultiTaskNetwork(field_dims, emb_dim, mlp_dims, dropout, aux_heads)


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
        max_epochs: int = 3,
        batch_size: int = 4096,
        patience: int = 1,
        seed: int = 42,
        feature_names: list[str] | None = None,
        num_threads: int = 1,
        **hparams,
    ):
        if emb_dim <= 0 or lr <= 0 or l2 < 0:
            raise ValueError("emb_dim and lr must be positive and l2 must be non-negative")
        if max_epochs <= 0 or batch_size <= 0 or patience != 1 or num_threads <= 0:
            raise ValueError("epochs, batch size, and threads must be positive; patience must be 1")
        if not mlp or any(int(size) <= 0 for size in mlp):
            raise ValueError("mlp must contain positive hidden dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
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
