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

# --- Native backend declaration (OpenMP isolation) --------------------------
#
# torch ships its own `libomp.dylib`; LightGBM links the system one. Loading
# both into one process aborts the interpreter outright:
#
#     OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
#     already initialized.
#
# It is a hard SIGABRT in either import order, not an exception, so nothing
# in Python can catch it — the process simply dies and the C1 parent sees a
# closed pipe. Every model therefore declares which native runtime it pulls
# in, and the runner keeps a process committed to at most one of them.
# macOS-specific in practice (Linux CI shares a single libgomp), which is
# exactly why CI stays green while a developer or demo laptop crashes.
BACKENDS = ("torch", "lightgbm")


def backend(name: str) -> str | None:
    """The native runtime a model loads, or None when it is pure numpy."""
    return getattr(MODEL_REGISTRY.get(name), "native_backend", None)


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
