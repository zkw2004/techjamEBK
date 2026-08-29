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
        **hparams,
    ):
        if k <= 0 or lr <= 0 or l2 < 0:
            raise ValueError("k and lr must be positive and l2 must be non-negative")
        if max_epochs <= 0 or batch_size <= 0 or patience <= 0:
            raise ValueError("max_epochs, batch_size, and patience must be positive")
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.vocabs: list[dict[object, int]] = []
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
            validation_users = np.asarray(groups) if groups is not None else X_val[:, 0]
            if len(validation_users) != len(y_val):
                raise ValueError("groups must have one value per validation row")

        self._initialise_parameters(int(self.unknown_ids[-1] + self.offsets[-1] + 1))
        rng = np.random.default_rng(self.seed)
        best = -np.inf
        best_state = None
        bad_epochs = 0
        self.best_epoch = None
        for epoch in range(1, self.max_epochs + 1):
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
    def _validate_xy(X: np.ndarray, y: np.ndarray, name: str) -> None:
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"{name} matrix must be non-empty and two-dimensional")
        if y.ndim != 1 or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError(f"{name} labels must be finite with one value per row")

    def _fit_encoder(self, X: np.ndarray) -> None:
        self.vocabs = [dict() for _ in range(X.shape[1])]
        for row in X:
            for field, value in enumerate(row):
                vocab = self.vocabs[field]
                if value not in vocab:
                    vocab[value] = len(vocab)
        field_dims = np.asarray([len(vocab) + 1 for vocab in self.vocabs], dtype=np.int32)
        self.unknown_ids = field_dims - 1
        self.offsets = np.cumsum(np.r_[0, field_dims[:-1]]).astype(np.int32)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.int32)
        for row_index, row in enumerate(X):
            for field, value in enumerate(row):
                encoded[row_index, field] = (
                    self.vocabs[field].get(value, int(self.unknown_ids[field]))
                    + self.offsets[field]
                )
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
        self.b -= self.lr * gradient.sum()

    def _predict_encoded(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(
            [
                self._logits(X[start : start + batch_size])[0]
                for start in range(0, len(X), batch_size)
            ]
        )
