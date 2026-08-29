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
        raise NotImplementedError("C2")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C2")


@register("popularity")
class PopularityModel:
    """Score = item click rate on train, EB-smoothed. No personalisation."""

    def __init__(self, seed: int = 42, **hparams):
        pass

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        raise NotImplementedError("C2")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C2")
