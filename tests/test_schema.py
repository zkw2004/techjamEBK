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


# --- combinations the runner can only ever reject at execution time -----------

def test_a_loss_the_model_ignores_is_rejected_at_proposal_time():
    """Regression: FM and DeepFM silently discarded `loss`, so 'objective
    family' experiments trained an identical model and reported an identical
    score. Rejecting here costs one cheap re-ask instead of a wasted
    smoke run, an A5 repair attempt, and an iteration."""
    with pytest.raises(ValidationError, match="does not implement loss"):
        Config.model_validate({"model": "fm", "loss": "lambdarank"})
    with pytest.raises(ValidationError, match="does not implement loss"):
        Config.model_validate({"model": "deepfm", "loss": "pairwise"})
    with pytest.raises(ValidationError, match="does not implement loss"):
        Config.model_validate({"model": "lgbm", "loss": "pairwise"})


def test_losses_each_model_really_implements_are_accepted():
    assert Config.model_validate({"model": "fm", "loss": "pointwise"}).loss == "pointwise"
    assert Config.model_validate({"model": "fm", "loss": "pairwise"}).loss == "pairwise"
    assert Config.model_validate({"model": "lgbm", "loss": "lambdarank"}).loss == "lambdarank"


def test_unimplemented_models_set_is_empty_now_that_c9_landed():
    """deepfm_mtl was the one entry in UNIMPLEMENTED_MODELS (a run wasted
    iterations rediscovering its NotImplementedError twice). C9 implemented
    it (pipeline/models/deepfm.py::DeepFMMultiTask), so the gate must accept
    it now — an empty set is kept, not deleted, as the obvious place to add
    a future stub."""
    from agent.schema import UNIMPLEMENTED_MODELS

    assert UNIMPLEMENTED_MODELS == frozenset()
    Config.model_validate({"model": "deepfm_mtl"})  # must not raise


def test_blend_requires_exactly_two_distinct_parents():
    """pipeline/blending.py requires exactly two; the proposer was only told
    'at least two', so three-parent blends passed proposal and died in the
    runner."""
    assert len(Config.model_validate({"model": "blend", "parents": ["n1", "n2"]}).parents) == 2
    for parents in ([], ["n1"], ["n1", "n2", "n3"], ["n1", "n1"]):
        with pytest.raises(ValidationError, match="exactly two distinct"):
            Config.model_validate({"model": "blend", "parents": parents})
