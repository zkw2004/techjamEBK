"""Reference rungs: random and popularity. Task C2.

GATE: nothing downstream proceeds until these reproduce the reference
scores within noise (seed std 0.0008).

NOTE vs AGENT_PLAN.md: the plan quotes 0.4753 / 0.5715, which are the TEST
figures. Test labels ARE present in the local archive and baseline.py prints
test scores, so both columns are computable. But Section 4.7 forbids using
test during development, so the gate reads `valid`; `test` is recorded here
only to reconcile with the plan's Section 3 numbers.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register

# From kuairand-starter-kit/baseline_scores.json. Gate against `valid`.
RANDOM_REFERENCE = {"valid": 0.4834, "test": 0.4753}
POPULARITY_REFERENCE = {"valid": 0.5807, "test": 0.5715}


@register("random")
class RandomModel:
    def __init__(self, seed: int = 42, **hparams):
        self.rng = np.random.default_rng(seed)

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        return None

    def predict(self, X) -> np.ndarray:
        return self.rng.random(len(X))


@register("popularity")
class PopularityModel:
    """Score = item click rate on train, EB-smoothed. No personalisation."""

    def __init__(self, seed: int = 42, prior: float = 20.0, item_col: int = 1, **hparams):
        if prior < 0:
            raise ValueError("prior must be non-negative")
        self.prior = float(prior)
        self.item_col = item_col
        self.global_rate: float | None = None
        self.item_scores: dict[object, float] = {}

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        X_train = np.asarray(X_train)
        labels = np.asarray(y_train, dtype=float)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a two-dimensional matrix")
        if len(X_train) != len(labels) or len(labels) == 0:
            raise ValueError("X_train and y_train must have the same non-zero length")

        items = X_train[:, self.item_col]
        unique_items, inverse = np.unique(items, return_inverse=True)
        impressions = np.bincount(inverse)
        positives = np.bincount(inverse, weights=labels)
        self.global_rate = float(labels.mean())
        smoothed = (positives + self.prior * self.global_rate) / (impressions + self.prior)
        self.item_scores = dict(zip(unique_items.tolist(), smoothed.tolist(), strict=True))

    def predict(self, X) -> np.ndarray:
        if self.global_rate is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        return np.fromiter(
            (self.item_scores.get(item, self.global_rate) for item in X[:, self.item_col]),
            dtype=float,
            count=len(X),
        )
