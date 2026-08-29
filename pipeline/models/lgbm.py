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
        self.loss, self.seed, self.hparams = loss, seed, hparams

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        raise NotImplementedError("C4")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C4")
