"""Blending accepted nodes. Task C7, spec in Section 6.9.

Four methods; rank_avg is the default. Trap 10: never majority-vote labels —
ties destroy NDCG. Blend weights are fitted on internal folds only (trap 6).
A blend is rejected unless it beats BOTH parents on folds and on the
official metric.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register

METHODS = ("rank_avg", "logit_avg", "weighted_rank", "rrf")


@register("blend")
class Blend:
    def __init__(self, parents: list[str] | None = None, blend_method: str = "rank_avg",
                 seed: int = 42, **hparams):
        self.parents, self.blend_method = parents or [], blend_method

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        raise NotImplementedError("C7")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C7")


def per_user_spearman(a: np.ndarray, b: np.ndarray, user_ids: np.ndarray) -> float:
    """Mean within-user rank correlation between two score vectors.

    Reported alongside every blend: low correlation is the evidence that
    the parents' errors differ enough to be worth combining.
    """
    raise NotImplementedError("C7")
