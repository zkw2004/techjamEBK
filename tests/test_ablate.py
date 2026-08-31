"""A10 acceptance: ablation-driven drawer targeting."""

from __future__ import annotations

import pytest

from agent import ablate
from agent.schema import Action

SIGNAL_FEATURE = "video_ctr"
INERT_FEATURE = "day_of_week"

BASE_CONFIG = {
    "model": "fm",
    "loss": "pointwise",
    "features": ["user_id", "video_id", SIGNAL_FEATURE, INERT_FEATURE],
    "hparams": {},
}


def _node(config: dict | None = None, **overrides) -> dict:
    node = {
        "id": "n007",
        "family": "model",
        "fidelity": "full",
        "status": "ok",
        "accepted": True,
        "config": config if config is not None else dict(BASE_CONFIG),
        "metrics": {"primary": 0.61},
    }
    node.update(overrides)
    return node


def _fixture_executor(scores: dict[frozenset, float], calls: list | None = None):
    """Score a run purely from which features its config carries.

    Lets a test state 'this feature group carries all the signal' directly,
    without training anything.
    """

    def execute_fn(action: Action, fidelity: str) -> dict:
        if calls is not None:
            calls.append((sorted(action.config.features), fidelity))
        key = frozenset(action.config.features)
        return {
            "status": "ok",
            "fidelity": fidelity,
            "metrics": {"primary": scores.get(key, 0.50)},
        }

    return execute_fn


def test_the_feature_carrying_the_signal_ranks_first_by_sensitivity():
    """The A10 acceptance criterion. One feature group carries the signal;
    removing it must move the score more than removing anything else."""
    full = frozenset(BASE_CONFIG["features"])
    scores = {
        full: 0.61,
        full - {SIGNAL_FEATURE}: 0.50,  # signal removed: large drop
        full - {INERT_FEATURE}: 0.6098,  # inert removed: noise
        full - {"user_id"}: 0.6090,
        full - {"video_id"}: 0.6085,
    }

    table = ablate.ablate(_node(), execute_fn=_fixture_executor(scores))

    assert table["base_primary"] == 0.61
    assert table["components"][0]["component"] == f"feature:{SIGNAL_FEATURE}"
    assert table["components"][0]["sensitivity"] == pytest.approx(0.11)
    # ...and the table is ranked, not merely correct at the top.
    sensitivities = [item["sensitivity"] for item in table["components"]]
    assert sensitivities == sorted(sensitivities, reverse=True)


def test_sensitivity_keeps_the_sign_separately_from_the_ranking():
    """A component whose removal *improves* the score is as interesting as one
    that hurts, so ranking uses magnitude while `delta` keeps the direction."""
    full = frozenset(BASE_CONFIG["features"])
    scores = {full: 0.60, full - {SIGNAL_FEATURE}: 0.65}

    table = ablate.ablate(_node(), execute_fn=_fixture_executor(scores))
    harmful = next(
        item for item in table["components"] if item["component"] == f"feature:{SIGNAL_FEATURE}"
    )

    assert harmful["delta"] == pytest.approx(0.05)
    assert harmful["sensitivity"] == pytest.approx(0.05)


def test_variants_cover_features_and_a_loss_the_model_implements():
    variants = dict(ablate.ablation_variants(BASE_CONFIG))

    for name in BASE_CONFIG["features"]:
        assert f"feature:{name}" in variants
        assert name not in variants[f"feature:{name}"]["features"]
    # FM implements pairwise, so the loss drawer is probeable; it does not
    # implement lambdarank, which must never be offered as a variant.
    assert variants["loss:pairwise"]["loss"] == "pairwise"
    assert "loss:lambdarank" not in variants


def test_a_single_feature_config_yields_no_feature_variants():
    """Dropping the last field leaves nothing to train, so it would measure
    the executor's error path rather than the component's contribution."""
    variants = ablate.ablation_variants({**BASE_CONFIG, "features": ["user_id"]})

    assert not any(name.startswith("feature:") for name, _ in variants)


def test_ablation_probes_run_at_screen_fidelity_and_stay_within_budget():
    """A10 budgets one round at <= 5 min. The guard that keeps that true is the
    variant cap plus screen fidelity — assert both, since a round that
    silently escalated to full tier would blow the budget by ~10x."""
    calls: list = []
    ablate.ablate(_node(), execute_fn=_fixture_executor({}, calls))

    assert {fidelity for _features, fidelity in calls} == {"screen"}
    assert len(calls) <= ablate.MAX_VARIANTS + 1  # + the base run


def test_max_variants_caps_a_wide_config():
    wide = {**BASE_CONFIG, "features": [f"f{i}" for i in range(40)]}

    table = ablate.ablate(_node(wide), execute_fn=_fixture_executor({}), max_variants=3)

    assert len(table["components"]) == 3


def test_no_table_rather_than_a_partial_one_when_the_base_run_fails():
    """A component missing from the table reads as an insensitive component.
    With no trustworthy base there is nothing to compare against, so the
    honest answer is no table at all."""

    def failing(action: Action, fidelity: str) -> dict:
        return {"status": "error", "fidelity": fidelity, "metrics": {}}

    assert ablate.ablate(_node(), execute_fn=failing) is None


def test_no_table_for_a_node_without_a_config():
    assert ablate.ablate(_node(config={}), execute_fn=_fixture_executor({})) is None


def test_rendered_table_is_markdown_and_names_every_component():
    table = ablate.ablate(_node(), execute_fn=_fixture_executor({frozenset(): 0.0}))

    rendered = ablate.render_sensitivity_table(table)

    assert "| component |" in rendered
    for item in table["components"]:
        assert item["component"] in rendered


def test_rendering_an_absent_table_is_empty_not_a_placeholder():
    """An empty string keeps the section out of the prompt entirely; a
    placeholder would spend tokens telling the model nothing."""
    assert ablate.render_sensitivity_table(None) == ""
    assert ablate.render_sensitivity_table({"base_primary": 0.6, "components": []}) == ""


def test_unused_features_excludes_what_the_incumbent_already_has():
    from pipeline.features import FEATURES

    unused = ablate.unused_features(None, BASE_CONFIG)

    assert SIGNAL_FEATURE not in unused
    assert INERT_FEATURE not in unused
    assert set(unused) == set(FEATURES) - set(BASE_CONFIG["features"])


def test_unused_features_does_not_re_suggest_a_measured_inert_feature():
    table = {
        "base_primary": 0.61,
        "components": [{"component": "feature:user_ctr", "sensitivity": 0.0, "delta": 0.0}],
    }

    assert "user_ctr" not in ablate.unused_features(table, {"features": []})
