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
    native_backend = "lightgbm"  # see pipeline/models/__init__.py BACKENDS

    def __init__(
        self,
        loss: str = "pointwise",
        seed: int = 42,
        feature_names: list[str] | None = None,
        num_boost_round: int | None = None,
        early_stopping_rounds: int = 20,
        **hparams,
    ):
        if loss not in {"pointwise", "lambdarank"}:
            raise ValueError("LightGBM loss must be 'pointwise' or 'lambdarank'")
        # C1's fold-selected refit budget wins over the sklearn-style alias.
        if num_boost_round is None:
            num_boost_round = int(hparams.get("n_estimators", 500))
        if num_boost_round <= 0 or early_stopping_rounds < 0:
            raise ValueError("num_boost_round must be positive and early stopping non-negative")
        self.loss = loss
        self.seed = seed
        self.feature_names = list(feature_names or [])
        self._infer_types = not self.feature_names
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.hparams = {
            name: value
            for name, value in hparams.items()
            if name
            not in {
                "epochs",
                "max_epochs",
                "n_estimators",
                "num_trials",
            }
        }
        self.vocabs: list[dict[object, int] | None] = []
        self.unknown_ids: list[int | None] = []
        self.categorical_features: list[int] = []
        self.booster = None
        self.best_epoch: int | None = None

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        import lightgbm as lgb

        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train, dtype=float)
        self._validate_xy(X_train, y_train, "train")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be None")
        if not self.feature_names:
            self.feature_names = [f"field_{index}" for index in range(X_train.shape[1])]
        if len(self.feature_names) != X_train.shape[1]:
            raise ValueError("feature_names must match the matrix field count")

        self._fit_encoder(X_train)
        encoded_train = self._encode(X_train)
        encoded_val = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=object)
            y_val = np.asarray(y_val, dtype=float)
            self._validate_xy(X_val, y_val, "validation")
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError("train and validation matrices must have the same field count")
            encoded_val = self._encode(X_val)

        if groups is None and self.loss == "lambdarank":
            if not self._infer_types and self.feature_names[0] != "user_id":
                raise ValueError("lambdarank requires explicit user ids in groups")
            groups = (X_train[:, 0], None if X_val is None else X_val[:, 0])
        train_users, validation_users = self._split_groups(groups, len(X_train), X_val)
        train_group = validation_group = None
        if self.loss == "lambdarank":
            if train_users is None:
                raise ValueError("lambdarank requires training user ids in groups")
            order, train_group = self._group_order(train_users)
            encoded_train = encoded_train[order]
            y_train = y_train[order]
            self._assert_group_boundaries(train_users[order], train_group)
            if encoded_val is not None:
                if validation_users is None:
                    raise ValueError("lambdarank validation requires validation user ids")
                validation_order, validation_group = self._group_order(validation_users)
                encoded_val = encoded_val[validation_order]
                y_val = y_val[validation_order]
                self._assert_group_boundaries(validation_users[validation_order], validation_group)

        train_set = lgb.Dataset(
            encoded_train,
            label=y_train,
            group=train_group,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        valid_sets = []
        if encoded_val is not None:
            valid_sets.append(
                lgb.Dataset(
                    encoded_val,
                    label=y_val,
                    group=validation_group,
                    categorical_feature=self.categorical_features,
                    reference=train_set,
                    free_raw_data=False,
                )
            )

        params = {
            "objective": "binary" if self.loss == "pointwise" else "lambdarank",
            "metric": "binary_logloss" if self.loss == "pointwise" else "ndcg",
            "verbosity": -1,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            **self.hparams,
        }
        if self.loss == "lambdarank":
            params.update(label_gain=[0, 1], eval_at=[5])
        callbacks = [lgb.log_evaluation(0)]
        if valid_sets and self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.best_epoch = self.booster.best_iteration or self.num_boost_round

    def predict(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be two-dimensional with the fitted field count")
        return np.asarray(
            self.booster.predict(self._encode(X), num_iteration=self.best_epoch),
            dtype=float,
        )

    @staticmethod
    def _validate_xy(X: np.ndarray, y: np.ndarray, name: str) -> None:
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"{name} matrix must be non-empty and two-dimensional")
        if y.ndim != 1 or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError(f"{name} labels must be finite with one value per row")

    def _fit_encoder(self, X: np.ndarray) -> None:
        from pipeline.data import FIELDS

        self.vocabs = []
        self.unknown_ids = []
        self.categorical_features = []
        for index, name in enumerate(self.feature_names):
            categorical = name in FIELDS
            if self._infer_types:
                try:
                    np.asarray(X[:, index], dtype=float)
                except (TypeError, ValueError):
                    categorical = True
            if not categorical:
                self.vocabs.append(None)
                self.unknown_ids.append(None)
                continue
            vocab = {}
            for value in X[:, index]:
                if value not in vocab:
                    vocab[value] = len(vocab)
            self.vocabs.append(vocab)
            self.unknown_ids.append(len(vocab))
            self.categorical_features.append(index)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        encoded = np.empty(X.shape, dtype=np.float64)
        for index, vocab in enumerate(self.vocabs):
            if vocab is None:
                values = np.asarray(X[:, index], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"numeric feature {self.feature_names[index]!r} must be finite"
                    )
                encoded[:, index] = values
            else:
                unknown = self.unknown_ids[index]
                encoded[:, index] = [vocab.get(value, unknown) for value in X[:, index]]
        return encoded

    @staticmethod
    def _split_groups(groups, train_rows: int, X_val):
        if groups is None:
            return None, None
        if isinstance(groups, dict):
            groups = (groups["train"], groups.get("val"))
        if not isinstance(groups, (tuple, list)) or len(groups) != 2:
            raise ValueError("groups must contain train and validation user ids")
        train_users, validation_users = groups
        train_users = np.asarray(train_users)
        if train_users.ndim != 1 or len(train_users) != train_rows:
            raise ValueError("training groups must have one user id per row")
        if X_val is None:
            if validation_users is not None:
                raise ValueError("validation groups require validation data")
            return train_users, None
        validation_users = np.asarray(validation_users)
        if validation_users.ndim != 1 or len(validation_users) != len(X_val):
            raise ValueError("validation groups must have one user id per row")
        return train_users, validation_users

    @staticmethod
    def _assert_group_boundaries(users_in_row_order: np.ndarray, counts) -> None:
        """LambdaRank group boundaries must be exactly the user boundaries.

        This is the structural match between the loss and the metric: GAUC and
        nDCG@5 are computed per user, so lambdarank only optimises the scored
        quantity if each of its groups is one user's impression list. LightGBM
        takes `group` positionally — the first `counts[0]` rows are group 0,
        the next `counts[1]` are group 1 — and validates nothing beyond the
        total summing to the row count. A group array that disagrees with the
        row order is therefore accepted silently: training proceeds, ranks are
        optimised across user boundaries, and the only symptom is a quietly
        worse score.

        Checked here rather than trusted because `_group_order` deriving both
        halves consistently is a property of today's implementation, not of
        the interface: `fit` also accepts caller-supplied `groups`, and any
        future change that reorders rows without recomputing counts (or vice
        versa) reintroduces exactly this silent failure.

        Honest limit: this validates the *user ids* against the counts, so it
        catches inconsistent or non-contiguous grouping. It cannot detect a
        reordering applied to the user ids but not to the feature matrix —
        both would still look internally consistent from here.
        """
        counts = np.asarray(list(counts), dtype=np.int64)
        if counts.size and counts.min() <= 0:
            raise ValueError("lambdarank group sizes must all be positive")
        if int(counts.sum()) != len(users_in_row_order):
            raise ValueError(
                f"lambdarank group sizes sum to {int(counts.sum())} but there are "
                f"{len(users_in_row_order)} rows; every row must belong to exactly one group"
            )
        boundaries = np.cumsum(counts)[:-1]
        blocks = np.split(np.asarray(users_in_row_order).astype(str), boundaries)
        seen: set[str] = set()
        for index, block in enumerate(blocks):
            distinct = set(block.tolist())
            if len(distinct) != 1:
                raise ValueError(
                    f"lambdarank group {index} spans {len(distinct)} users; each group "
                    "must be exactly one user's impressions"
                )
            user = distinct.pop()
            if user in seen:
                raise ValueError(
                    f"user {user!r} appears in more than one lambdarank group; a user's "
                    "rows must be contiguous so its group is its whole impression list"
                )
            seen.add(user)

    @staticmethod
    def _group_sizes(user_ids: np.ndarray) -> np.ndarray:
        if user_ids.ndim != 1:
            raise ValueError("ranking group ids must be one-dimensional")
        return np.asarray(LGBM._group_order(user_ids)[1], dtype=np.int32)

    @staticmethod
    def _group_order(user_ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
        order = np.argsort(user_ids.astype(str), kind="stable")
        sorted_users = user_ids[order].astype(str)
        boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
        counts = np.diff(np.r_[0, boundaries, len(sorted_users)]).astype(int).tolist()
        return order, counts
