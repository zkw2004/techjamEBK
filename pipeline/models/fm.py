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
    def __init__(self, k: int = 16, lr: float = 0.001, seed: int = 42, **hparams):
        self.k, self.lr, self.seed = k, lr, seed

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        raise NotImplementedError("C3")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C3")
