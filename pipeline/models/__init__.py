"""Model registry.

Contract: AGENT_PLAN.md Section 8.9 (FROZEN). Owner: Workstream C (Ethan).

Note: `BaseModel` here is the model Protocol, NOT pydantic's BaseModel.
The name is fixed by the frozen contract; keep the imports separate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseModel(Protocol):
    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None: ...
    def predict(self, X) -> np.ndarray: ...  # raw scores, higher = more likely


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register(name: str):
    """Decorator adding a model class to MODEL_REGISTRY."""

    def wrap(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"duplicate model name: {name}")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrap


def get(name: str) -> type[BaseModel]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


# Import side effects populate MODEL_REGISTRY. Keep at the bottom.
from pipeline.models import blend, deepfm, fm, lgbm, popularity  # noqa: E402,F401
