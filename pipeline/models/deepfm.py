"""DeepFM in PyTorch. Task C5. Skeleton: Appendix A.4.

CPU in under 10 min, early stopping patience 1 (trap 11: CTR models peak
after 1-3 epochs then degrade). The order-2 term MUST use the O(n) identity
0.5 * ((sum v)^2 - sum v^2), not a naive quadratic loop.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register


@register("deepfm")
class DeepFMModel:
    def __init__(self, emb_dim: int = 16, mlp=(256, 128, 64), dropout: float = 0.2,
                 seed: int = 42, **hparams):
        self.emb_dim, self.mlp, self.dropout, self.seed = emb_dim, mlp, dropout, seed

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        raise NotImplementedError("C5")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("C5")


@register("deepfm_mtl")
class DeepFMMultiTask(DeepFMModel):
    """Rung 5, deprioritised: 12 heads, one per feedback signal.

    Read only the click head at inference; the other 11 exist to force the
    shared embeddings to learn richer representations.
    """
