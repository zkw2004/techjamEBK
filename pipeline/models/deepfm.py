"""CPU-first DeepFM with the Appendix A.4 linear-time interaction term."""

from __future__ import annotations

import copy

import numpy as np

from pipeline.models import register


def _network(field_dims, emb_dim: int, mlp_dims, dropout: float):
    """Build lazily so importing the registry does not initialise Torch."""
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size, emb_dim) for size in field_dims]
            )
            self.linear = nn.ModuleList([nn.Embedding(size, 1) for size in field_dims])
            self.bias = nn.Parameter(torch.zeros(1))
            layers = []
            width = len(field_dims) * emb_dim
            for next_width in mlp_dims:
                layers.extend([nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(dropout)])
                width = next_width
            layers.append(nn.Linear(width, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            vectors = torch.stack(
                [embedding(x[:, index]) for index, embedding in enumerate(self.embeddings)],
                dim=1,
            )
            order1 = sum(
                layer(x[:, index]) for index, layer in enumerate(self.linear)
            ) + self.bias
            order2 = 0.5 * (
                vectors.sum(dim=1).pow(2) - vectors.pow(2).sum(dim=1)
            ).sum(dim=1, keepdim=True)
            return (order1 + order2 + self.mlp(vectors.flatten(start_dim=1))).squeeze(1)

    return Network()


@register("deepfm")
class DeepFMModel:
    def __init__(
        self, emb_dim: int = 16, mlp=(256, 128, 64), dropout: float = 0.2,
        seed: int = 42, **hparams,
    ):
        self.emb_dim, self.mlp, self.dropout, self.seed = emb_dim, tuple(mlp), dropout, seed
        self.hparams = hparams
        self.encoders: list[dict[object, int]] = []
        self.network = None
        self.best_epoch: int | None = None

    @staticmethod
    def _as_2d(X, name: str) -> np.ndarray:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2:
            raise ValueError(f"{name} must be a 2-D feature matrix")
        return values

    def _fit_encode(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.int64)
        self.encoders = []
        for column in range(X.shape[1]):
            mapping = {
                value: index + 1 for index, value in enumerate(dict.fromkeys(X[:, column]))
            }
            self.encoders.append(mapping)
            encoded[:, column] = [mapping[value] for value in X[:, column]]
        return encoded

    def _encode(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.encoders):
            raise ValueError("prediction feature count differs from training")
        encoded = np.empty(X.shape, dtype=np.int64)
        for column, mapping in enumerate(self.encoders):
            encoded[:, column] = [mapping.get(value, 0) for value in X[:, column]]
        return encoded

    @staticmethod
    def _loss(network, X, y, criterion, batch_size: int) -> float:
        import torch

        network.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                stop = min(start + batch_size, len(X))
                total += float(criterion(network(X[start:stop]), y[start:stop])) * (stop - start)
        return total / len(X)

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        import torch

        raw_train = self._as_2d(X_train, "X_train")
        labels = np.asarray(y_train, dtype=np.float32)
        if len(raw_train) != len(labels) or len(raw_train) == 0:
            raise ValueError("X_train and y_train must be non-empty and aligned")
        raw_val = None if X_val is None else self._as_2d(X_val, "X_val")
        val_labels = None if y_val is None else np.asarray(y_val, dtype=np.float32)
        if raw_val is not None and (val_labels is None or len(raw_val) != len(val_labels)):
            raise ValueError("X_val and y_val must align")

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        encoded_train = torch.as_tensor(self._fit_encode(raw_train), dtype=torch.long)
        target_train = torch.as_tensor(labels, dtype=torch.float32)
        encoded_val = target_val = None
        if raw_val is not None:
            encoded_val = torch.as_tensor(self._encode(raw_val), dtype=torch.long)
            target_val = torch.as_tensor(val_labels, dtype=torch.float32)

        self.network = _network(
            [len(mapping) + 1 for mapping in self.encoders],
            self.emb_dim, self.mlp, self.dropout,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self.hparams.get("lr", 1e-3)),
            weight_decay=float(self.hparams.get("l2", 0.0)),
        )
        epochs = int(self.hparams.get("max_epochs", self.hparams.get("epochs", 3)))
        batch_size = int(self.hparams.get("batch_size", 4096))
        patience = int(self.hparams.get("patience", 1))
        if epochs <= 0 or batch_size <= 0 or patience < 0:
            raise ValueError("epochs and batch_size must be positive; patience cannot be negative")

        generator = torch.Generator().manual_seed(self.seed)
        best_loss, best_state, stale_epochs = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            self.network.train()
            order = torch.randperm(len(encoded_train), generator=generator)
            for start in range(0, len(order), batch_size):
                take = order[start : start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.network(encoded_train[take]), target_train[take])
                loss.backward()
                optimiser.step()
            monitored = self._loss(
                self.network,
                encoded_val if encoded_val is not None else encoded_train,
                target_val if target_val is not None else target_train,
                criterion,
                batch_size,
            )
            if monitored < best_loss - 1e-8:
                best_loss, best_state, self.best_epoch = (
                    monitored, copy.deepcopy(self.network.state_dict()), epoch,
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()

    def predict(self, X) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("fit() must be called before predict()")
        import torch

        encoded = torch.as_tensor(self._encode(self._as_2d(X, "X")), dtype=torch.long)
        batch_size = int(self.hparams.get("batch_size", 4096))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                outputs.append(self.network(encoded[start : start + batch_size]).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64, copy=False) if outputs else np.array([])


@register("deepfm_mtl")
class DeepFMMultiTask(DeepFMModel):
    """Deprioritised registry-compatible placeholder using the label-head architecture."""
