"""Pydantic Config and Action.

Contract: AGENT_PLAN.md Section 8.6 (FROZEN). Owner: Workstream A (Kaiwen). Task A1.
Examples of well-formed Actions: Appendix B.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Which loss framings each model class actually implements. The Literal below
# is frozen (Section 8.6) and lists every value the *schema* allows, but a
# model that does not read `loss` silently ignores it: FM and DeepFM optimise
# a fixed objective regardless. A run once spent 11 "objective family"
# iterations proposing pairwise and lambdarank variants of FM that produced
# byte-identical fold scores, because nothing rejected the combination.
SUPPORTED_LOSSES: dict[str, frozenset[str]] = {
    "fm": frozenset({"pointwise", "pairwise"}),
    "lgbm": frozenset({"pointwise", "lambdarank"}),
    "deepfm": frozenset({"pointwise"}),
    "deepfm_mtl": frozenset({"pointwise"}),
    "random": frozenset({"pointwise"}),
    "popularity": frozenset({"pointwise"}),
    "blend": frozenset({"pointwise"}),
}

# Registered in MODEL_REGISTRY but its constructor raises NotImplementedError.
# Rejecting one here turns a wasted smoke run into a cheap proposal re-ask.
# deepfm_mtl (C9) moved out of this set once its multi-task heads landed
# (pipeline/models/deepfm.py::DeepFMMultiTask) -- kept as an empty set rather
# than deleted so a future model stub has an obvious place to register.
UNIMPLEMENTED_MODELS: frozenset[str] = frozenset()


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["random", "popularity", "fm", "lgbm", "deepfm",
                   "deepfm_mtl", "blend"]
    loss: Literal["pointwise", "pairwise", "lambdarank"] = "pointwise"
    features: list[str] = ["user_id", "video_id"]
    negative_sampling: Literal["all", "in_session", "pop_weighted"] = "all"
    hparams: dict = {}
    parents: list[str] = []                 # blend only
    blend_method: Literal["rank_avg", "logit_avg",
                          "weighted_rank", "rrf"] = "rank_avg"
    seed: int = 42

    @model_validator(mode="after")
    def _reject_combinations_that_cannot_run(self) -> Config:
        """Fail combinations the runner can only ever reject at execution time.

        Catching them during proposal costs one cheap re-ask; letting them
        through costs a smoke run, an A5 repair attempt, and an iteration.
        """
        if self.model in UNIMPLEMENTED_MODELS:
            raise ValueError(
                f"model {self.model!r} is registered but not implemented; "
                "use model='deepfm' for the single-task network"
            )
        supported = SUPPORTED_LOSSES.get(self.model, frozenset({"pointwise"}))
        if self.loss not in supported:
            raise ValueError(
                f"model {self.model!r} does not implement loss {self.loss!r}; "
                f"it supports {sorted(supported)}"
            )
        if self.model == "blend" and len(set(self.parents)) != 2:
            raise ValueError(
                "a blend requires exactly two distinct accepted full-tier "
                f"parent node ids, got {self.parents!r}"
            )
        return self


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(..., min_length=1)  # required, graded — copied
    reasoning: str                              # verbatim into the run log
    type: Literal["config", "tune", "code", "blend"]
    family: Literal["feature", "model", "objective", "training", "ensemble"]
    parent: str                              # node id to branch from
    config: Config | None = None
    search_space: dict | None = None         # tune only
    budget: int = 20                         # tune only, trial count
    code: str | None = None                  # code only, Python source


FAMILIES = ("feature", "model", "objective", "training", "ensemble")
