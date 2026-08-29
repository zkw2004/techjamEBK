"""LightGBM, pointwise and lambdarank. Task C4.

lambdarank needs a per-user `group` array (counts per user, in row order).
Early stopping on an internal fold — NEVER on the official validation set
(trap 6).
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register


@register("lgbm")
class LGBM:
    def __init__(self, loss: str = "pointwise", seed: int = 42, **hparams):
        if loss not in {"pointwise", "lambdarank"}:
            raise ValueError("LGBM loss must be 'pointwise' or 'lambdarank'")
        self.loss, self.seed, self.hparams = loss, seed, hparams
        self.model = None
        self.encoders: list[dict[object, int]] = []
        self.best_epoch: int | None = None

    @staticmethod
    def _as_2d(X, name: str) -> np.ndarray:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2:
            raise ValueError(f"{name} must be a 2-D feature matrix")
        return values

    def _fit_encoders(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.float64)
        self.encoders = []
        for column in range(X.shape[1]):
            values = X[:, column]
            try:
                numeric = values.astype(np.float64)
            except (TypeError, ValueError):
                categories = {value: index for index, value in enumerate(dict.fromkeys(values))}
                self.encoders.append(categories)
                encoded[:, column] = [categories[value] for value in values]
            else:
                self.encoders.append({})
                encoded[:, column] = numeric
        if not np.isfinite(encoded).all():
            raise ValueError("LightGBM features must be finite")
        return encoded

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if len(self.encoders) != X.shape[1]:
            raise ValueError("prediction feature count differs from training")
        encoded = np.empty(X.shape, dtype=np.float64)
        for column, categories in enumerate(self.encoders):
            values = X[:, column]
            if categories:
                encoded[:, column] = [categories.get(value, -1) for value in values]
            else:
                try:
                    encoded[:, column] = values.astype(np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"feature column {column} must remain numeric") from exc
        if not np.isfinite(encoded).all():
            raise ValueError("LightGBM features must be finite")
        return encoded

    @staticmethod
    def _group_sizes(user_ids: np.ndarray) -> np.ndarray:
        if user_ids.ndim != 1:
            raise ValueError("ranking group ids must be one-dimensional")
        _, counts = np.unique(user_ids, return_counts=True)
        return counts.astype(np.int32)

    @staticmethod
    def _group_ids(groups, X_train: np.ndarray, X_val: np.ndarray | None):
        """Resolve train/validation user ids without changing the frozen fit signature."""
        if isinstance(groups, dict):
            return np.asarray(groups["train"]), np.asarray(groups.get("val"))
        if isinstance(groups, (tuple, list)) and len(groups) == 2:
            return np.asarray(groups[0]), np.asarray(groups[1])
        # The frozen runner's default feature order starts with user_id. This
        # fallback keeps direct model use ergonomic; callers with another
        # ordering must pass explicit group ids.
        train_ids = np.asarray(X_train[:, 0])
        val_ids = None if X_val is None else np.asarray(X_val[:, 0])
        return train_ids, val_ids

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        import lightgbm as lgb

        raw_train = self._as_2d(X_train, "X_train")
        labels = np.asarray(y_train, dtype=np.float64)
        if len(raw_train) != len(labels):
            raise ValueError("X_train and y_train must align")
        raw_val = None if X_val is None else self._as_2d(X_val, "X_val")
        val_labels = None if y_val is None else np.asarray(y_val, dtype=np.float64)
        if raw_val is not None and (val_labels is None or len(raw_val) != len(val_labels)):
            raise ValueError("X_val and y_val must align")

        train = self._fit_encoders(raw_train)
        validation = None if raw_val is None else self._transform(raw_val)
        train_group = val_group = None
        if self.loss == "lambdarank":
            train_users, val_users = self._group_ids(groups, raw_train, raw_val)
            if len(train_users) != len(train):
                raise ValueError("training group ids must align with X_train")
            train_order = np.argsort(train_users, kind="stable")
            train, labels = train[train_order], labels[train_order]
            train_group = self._group_sizes(train_users[train_order])
            if validation is not None:
                if val_users is None or len(val_users) != len(validation):
                    raise ValueError("validation group ids must align with X_val")
                val_order = np.argsort(val_users, kind="stable")
                validation, val_labels = validation[val_order], val_labels[val_order]
                val_group = self._group_sizes(val_users[val_order])

        num_boost_round = int(self.hparams.get("n_estimators", 300))
        params = {
            "learning_rate": float(self.hparams.get("learning_rate", 0.05)),
            "num_leaves": int(self.hparams.get("num_leaves", 63)),
            "min_data_in_leaf": int(self.hparams.get("min_data_in_leaf", 20)),
            "bagging_fraction": float(self.hparams.get("bagging_fraction", 1.0)),
            "feature_fraction": float(self.hparams.get("feature_fraction", 1.0)),
            "lambda_l2": float(self.hparams.get("lambda_l2", 0.0)),
            "seed": self.seed,
            "num_threads": int(self.hparams.get("n_jobs", 1)),
            "verbosity": -1,
            "objective": "lambdarank" if self.loss == "lambdarank" else "binary",
            "metric": "ndcg" if self.loss == "lambdarank" else "binary_logloss",
            "deterministic": True,
            "force_col_wise": True,
        }
        train_set = lgb.Dataset(train, label=labels, group=train_group, free_raw_data=False)
        valid_sets = None
        callbacks = []
        if validation is not None:
            valid_set = lgb.Dataset(
                validation,
                label=val_labels,
                group=val_group,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets = [valid_set]
            patience = int(self.hparams.get("early_stopping_rounds", 20))
            if patience > 0:
                callbacks.append(lgb.early_stopping(patience, verbose=False))
        if self.loss == "lambdarank":
            params["eval_at"] = [5]
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.best_epoch = int(self.model.best_iteration or num_boost_round)

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        matrix = self._transform(self._as_2d(X, "X"))
        scores = self.model.predict(matrix, num_iteration=self.best_epoch)
        return np.asarray(scores, dtype=np.float64)
