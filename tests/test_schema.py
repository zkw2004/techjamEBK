"""A1 acceptance: valid JSON parses; unknown model rejected readably;
missing hypothesis or family rejected."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schema import Action, Config

VALID_ACTION = {
    "hypothesis": "GAUC is per-user, so a listwise objective may align better than pointwise BCE",
    "reasoning": "lambdarank optimises NDCG within each group directly",
    "type": "config",
    "family": "objective",
    "parent": "n009",
    "config": {
        "model": "lgbm",
        "loss": "lambdarank",
        "features": ["user_id", "video_id", "user_ctr_decayed"],
    },
}


def test_valid_action_parses():
    action = Action.model_validate(VALID_ACTION)
    assert action.config.model == "lgbm"
    assert action.config.loss == "lambdarank"
    assert action.config.seed == 42  # default applied


def test_config_defaults():
    cfg = Config(model="fm")
    assert cfg.loss == "pointwise"
    assert cfg.negative_sampling == "all"
    assert cfg.blend_method == "rank_avg"
    assert cfg.features == ["user_id", "video_id"]


def test_unknown_model_rejected_readably():
    with pytest.raises(ValidationError) as exc:
        Config(model="transformer4rec")
    assert "transformer4rec" in str(exc.value)


@pytest.mark.parametrize("missing", ["hypothesis", "family"])
def test_required_fields_rejected_when_missing(missing):
    payload = {k: v for k, v in VALID_ACTION.items() if k != missing}
    with pytest.raises(ValidationError) as exc:
        Action.model_validate(payload)
    assert missing in str(exc.value)


def test_empty_hypothesis_rejected():
    """hypothesis is graded and copied verbatim into the run log — never blank."""
    with pytest.raises(ValidationError):
        Action.model_validate({**VALID_ACTION, "hypothesis": ""})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Action.model_validate({**VALID_ACTION, "tempreature": 0.7})


def test_blend_action_parses():
    action = Action.model_validate({
        "hypothesis": "GBDT and DeepFM rank-correlate at 0.81 per user",
        "reasoning": "low enough that their errors differ",
        "type": "blend",
        "family": "ensemble",
        "parent": "n022",
        "config": {"model": "blend", "parents": ["n014", "n022"],
                   "blend_method": "rank_avg"},
    })
    assert action.config.parents == ["n014", "n022"]
