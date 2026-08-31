"""Factorization Machine, ported verbatim from the starter kit. Task C3.

k=16, lr=0.001, 5 categorical fields, numpy only, ~40s on one CPU core.
Acceptance: reproduces validation primary 0.6016 within one seed-std (0.0008).

Read the actual starter-kit baseline.py rather than reimplementing from memory.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register

BASELINE_VALIDATION_PRIMARY = 0.6016  # test: 0.5946. Config: k=16, lr=0.001, batch=8192,
# max_epochs=40, patience=4, fields = the five in pipeline.data.FIELDS.

# "pointwise" is the organiser baseline and must stay byte-identical: the
# 0.6016 reference gate depends on it. "pairwise" is BPR — the loss framing the
# organisers rank as the most promising untested direction, on the argument
# that GAUC/nDCG are ranking metrics while BCE optimises a global objective.
SUPPORTED_LOSSES = frozenset({"pointwise", "pairwise"})

# Quantile bins for a continuous feature, matching DeepFM's encoder so the two
# models see the same discretisation of the same column.
NUMERIC_BINS = 32


@register("fm")
class FM:
    def __init__(
        self,
        k: int = 16,
        lr: float = 0.001,
        l2: float = 1e-6,
        max_epochs: int = 40,
        batch_size: int = 8192,
        patience: int = 4,
        seed: int = 42,
        loss: str = "pointwise",
        feature_names: list[str] | None = None,
        **hparams,
    ):
        if k <= 0 or lr <= 0 or l2 < 0:
            raise ValueError("k and lr must be positive and l2 must be non-negative")
        if max_epochs <= 0 or batch_size <= 0 or patience <= 0:
            raise ValueError("max_epochs, batch_size, and patience must be positive")
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(f"FM loss must be one of {sorted(SUPPORTED_LOSSES)}, got {loss!r}")
        self.loss = loss
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        # Already supplied by pipeline.train._new_model for every model; FM
        # previously swallowed it into **hparams and ignored it, which is why
        # it had no way to tell an identifier column from a continuous one.
        self.feature_names = list(feature_names or [])
        self.vocabs: list[dict[object, int] | None] = []
        self.numeric_edges: list[np.ndarray | None] = []
        self.unknown_ids = np.array([], dtype=np.int32)
        self.offsets = np.array([], dtype=np.int32)
        self.V: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.b = np.float32(0.0)
        self.mV: np.ndarray | None = None
        self.vV: np.ndarray | None = None
        self.mW: np.ndarray | None = None
        self.vW: np.ndarray | None = None
        self.t = 0
        self.best_epoch: int | None = None

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train, dtype=np.float32)
        self._validate_xy(X_train, y_train, "train")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be None")

        self._fit_encoder(X_train)
        encoded_train = self._encode(X_train)
        encoded_val = None
        validation_users = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=object)
            y_val = np.asarray(y_val, dtype=np.float32)
            self._validate_xy(X_val, y_val, "validation")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("train and validation matrices must have the same field count")
            encoded_val = self._encode(X_val)
            if isinstance(groups, tuple):
                validation_users = groups[1]
            else:
                validation_users = groups
            if validation_users is None:
                validation_users = X_val[:, 0]
            validation_users = np.asarray(validation_users)
            if len(validation_users) != len(y_val):
                raise ValueError("groups must have one value per validation row")

        self._initialise_parameters(int(self.unknown_ids[-1] + self.offsets[-1] + 1))
        rng = np.random.default_rng(self.seed)
        pair_index = None
        if self.loss == "pairwise":
            train_users = groups[0] if isinstance(groups, tuple) else None
            if train_users is None:
                train_users = X_train[:, 0]  # FIELDS[0] is user_id, as for validation
            train_users = np.asarray(train_users)
            if len(train_users) != len(y_train):
                raise ValueError("groups must have one training user id per training row")
            pair_index = self._build_pair_index(train_users, y_train)

        best = -np.inf
        best_state = None
        bad_epochs = 0
        self.best_epoch = None
        for epoch in range(1, self.max_epochs + 1):
            if pair_index is not None:
                positive_rows, negative_rows = self._sample_pairs(pair_index, rng)
                shuffle = rng.permutation(len(positive_rows))
                positive_rows, negative_rows = positive_rows[shuffle], negative_rows[shuffle]
                for start in range(0, len(positive_rows), self.batch_size):
                    stop = start + self.batch_size
                    self._step_pairwise(
                        encoded_train[positive_rows[start:stop]],
                        encoded_train[negative_rows[start:stop]],
                    )
            else:
                indices = rng.permutation(len(y_train))
                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start : start + self.batch_size]
                    self._step(encoded_train[batch], y_train[batch])

            if encoded_val is None:
                continue
            from pipeline.evaluate import evaluate

            primary = evaluate(
                validation_users,
                y_val,
                self._predict_encoded(encoded_val),
            )["primary"]
            if primary > best + 1e-5:
                best = primary
                bad_epochs = 0
                self.best_epoch = epoch
                best_state = (self.V.copy(), self.W.copy(), np.float32(self.b))
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.V, self.W, self.b = best_state

    def predict(self, X) -> np.ndarray:
        if self.V is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != len(self.vocabs):
            raise ValueError("X must be a two-dimensional matrix with the fitted field count")
        return self._predict_encoded(self._encode(X))

    @staticmethod
    def _build_pair_index(user_ids: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Index positives and same-user negatives for BPR sampling.

        Only users with at least one positive *and* one negative can produce a
        pair: a user whose impressions are all one class carries no
        within-user ordering signal, which is the same reason the evaluator
        excludes them from GAUC.

        Returns flat arrays rather than per-user lists so an epoch's sampling
        stays vectorised over ~1.1M rows.
        """
        positives = np.flatnonzero(y > 0)
        negatives = np.flatnonzero(y <= 0)
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("pairwise loss needs both positive and negative training rows")

        negative_users = user_ids[negatives]
        order = np.argsort(negative_users, kind="stable")
        negatives_by_user = negatives[order]
        users, starts, counts = np.unique(
            negative_users[order], return_index=True, return_counts=True
        )

        # Keep only positives whose user also has a negative to compare against.
        slot = np.searchsorted(users, user_ids[positives])
        in_range = slot < len(users)
        matched = np.zeros(len(positives), dtype=bool)
        matched[in_range] = users[slot[in_range]] == user_ids[positives][in_range]
        if not matched.any():
            raise ValueError("no user has both a positive and a negative training row")

        return {
            "positive_rows": positives[matched],
            "negative_slot_start": starts[slot[matched]],
            "negative_slot_count": counts[slot[matched]],
            "negatives_by_user": negatives_by_user,
        }

    @staticmethod
    def _sample_pairs(index: dict[str, np.ndarray], rng) -> tuple[np.ndarray, np.ndarray]:
        """One negative drawn uniformly per positive, from that user's rows."""
        draws = (rng.random(len(index["positive_rows"])) * index["negative_slot_count"]).astype(
            np.int64
        )
        draws = np.minimum(draws, index["negative_slot_count"] - 1)
        negative_rows = index["negatives_by_user"][index["negative_slot_start"] + draws]
        return index["positive_rows"], negative_rows

    @staticmethod
    def _validate_xy(X: np.ndarray, y: np.ndarray, name: str) -> None:
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"{name} matrix must be non-empty and two-dimensional")
        if y.ndim != 1 or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError(f"{name} labels must be finite with one value per row")

    def _is_categorical(self, n_fields: int) -> list[bool]:
        """Which columns get an exact-value vocabulary rather than bins.

        Mirrors ``DeepFMModel._fit_encoder``'s rule exactly -- the five
        official ``FIELDS`` are identifiers and stay categorical; every other
        registered feature is a continuous statistic and gets quantile bins.

        When no ``feature_names`` were supplied there is nothing to decide
        from, so every column is categorical. That is the pre-existing
        behaviour and keeps every direct ``FM(...)`` caller (the baseline
        reproduction test included) byte-identical.
        """
        from pipeline.data import FIELDS

        names = self.feature_names
        if not names:
            return [True] * n_fields
        return [
            index >= len(names) or names[index] in FIELDS for index in range(n_fields)
        ]

    def _fit_encoder(self, X: np.ndarray) -> None:
        """Vocabularies for identifier columns, quantile edges for the rest.

        Without the numeric branch FM memorises raw floats as categories: a
        continuous feature such as ``pcr_hist`` produced ~1.0M distinct values
        over 1.14M training rows, so its embeddings averaged one observation
        each and **100%** of validation rows fell through to the single
        unknown bucket -- the feature contributed a constant at scoring time
        after polluting training with a million noise vectors that the
        second-order term mixes into every real field. Measured cost of that
        on `sim_to_history`: -0.0052 primary, a 95% CI entirely below zero.
        Thirteen of the ~20 registered features were affected, eight of them
        at 100% unseen. See ``tests/test_models.py::test_fm_bins_*``.
        """
        n_fields = X.shape[1]
        categorical = self._is_categorical(n_fields)
        self.vocabs = []
        self.numeric_edges = []
        field_dims = []
        for field in range(n_fields):
            if categorical[field]:
                vocab: dict[object, int] = {}
                for value in X[:, field]:
                    if value not in vocab:
                        vocab[value] = len(vocab)
                self.vocabs.append(vocab)
                self.numeric_edges.append(None)
                field_dims.append(len(vocab) + 1)  # +1 slot for unseen values
                continue
            values = self._numeric_column(X, field)
            # `unique` collapses repeated edges, so a column with fewer than
            # NUMERIC_BINS distinct values keeps one bin per value rather than
            # gaining empty ones.
            edges = np.unique(
                np.quantile(values, np.linspace(0, 1, NUMERIC_BINS + 1)[1:-1])
            )
            self.vocabs.append(None)
            self.numeric_edges.append(edges)
            # searchsorted yields 0..len(edges) inclusive, so len(edges)+1 bins
            # cover every possible value and no unknown slot is needed: a value
            # outside the training range lands in the nearest end bin, which is
            # the behaviour we want from an ordinal feature.
            field_dims.append(len(edges) + 1)
        dims = np.asarray(field_dims, dtype=np.int32)
        self.unknown_ids = dims - 1
        self.offsets = np.cumsum(np.r_[0, dims[:-1]]).astype(np.int32)

    def _numeric_column(self, X: np.ndarray, field: int) -> np.ndarray:
        name = (
            self.feature_names[field]
            if self.feature_names and field < len(self.feature_names)
            else f"field_{field}"
        )
        try:
            values = np.asarray(X[:, field], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature {name!r} is not one of the categorical FIELDS, so FM "
                "bins it as a continuous column, but its values are not numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError(f"numeric feature {name!r} must be finite")
        return values

    def _encode(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.int32)
        for field in range(X.shape[1]):
            vocab = self.vocabs[field]
            offset = int(self.offsets[field])
            if vocab is None:
                values = self._numeric_column(X, field)
                encoded[:, field] = (
                    np.searchsorted(self.numeric_edges[field], values) + offset
                )
                continue
            unknown = int(self.unknown_ids[field])
            encoded[:, field] = [vocab.get(value, unknown) + offset for value in X[:, field]]
        return encoded

    def _initialise_parameters(self, dimension: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.V = rng.normal(0, 0.01, (dimension, self.k)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def _logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * (
            (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
        )
        return self.b + self.W[X].sum(axis=1) + interactions, embeddings, summed

    def _step(self, X: np.ndarray, y: np.ndarray) -> None:
        logits, embeddings, summed = self._logits(X)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        gradient = ((probabilities - y) / len(y)).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, X, gradient[:, None])
        np.add.at(
            gradient_v,
            X,
            gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        self._apply_gradients(gradient_v, gradient_w, float(gradient.sum()))

    def _step_pairwise(self, X_positive: np.ndarray, X_negative: np.ndarray) -> None:
        """One BPR step over (positive, negative) impressions of the same user.

        Optimises ``-log sigmoid(score_pos - score_neg)``. Because the metric
        ranks strictly within a user, only the *difference* between two of that
        user's rows carries signal — which is also why the global bias drops
        out here: it appears in both logits and cancels.
        """
        positive_logits, positive_embeddings, positive_summed = self._logits(X_positive)
        negative_logits, negative_embeddings, negative_summed = self._logits(X_negative)
        difference = positive_logits - negative_logits
        # d/d(difference) of -log sigmoid(difference); negative, so the update
        # pushes the positive above the negative.
        coefficient = (
            (1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30))) - 1.0) / len(difference)
        ).astype(np.float32)

        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, X_positive, coefficient[:, None])
        np.add.at(gradient_w, X_negative, -coefficient[:, None])
        np.add.at(
            gradient_v,
            X_positive,
            coefficient[:, None, None] * (positive_summed[:, None, :] - positive_embeddings),
        )
        np.add.at(
            gradient_v,
            X_negative,
            -coefficient[:, None, None] * (negative_summed[:, None, :] - negative_embeddings),
        )
        self._apply_gradients(gradient_v, gradient_w, 0.0)

    def _apply_gradients(
        self, gradient_v: np.ndarray, gradient_w: np.ndarray, gradient_b: float
    ) -> None:
        """Shared Adam update, so both loss framings train identically."""
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W

        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, parameter_gradient, first, second in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            first *= beta1
            first += (1 - beta1) * parameter_gradient
            second *= beta2
            second += (1 - beta2) * (parameter_gradient * parameter_gradient)
            corrected_first = first / (1 - beta1**self.t)
            corrected_second = second / (1 - beta2**self.t)
            parameter -= self.lr * corrected_first / (np.sqrt(corrected_second) + epsilon)
        self.b -= self.lr * gradient_b

    def _predict_encoded(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(
            [
                self._logits(X[start : start + batch_size])[0]
                for start in range(0, len(X), batch_size)
            ]
        )
