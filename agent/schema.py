"""Pydantic Config and Action.

Contract: AGENT_PLAN.md Section 8.6 (FROZEN). Owner: Workstream A (Kaiwen). Task A1.
Examples of well-formed Actions: Appendix B.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
