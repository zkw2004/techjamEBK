"""A11 acceptance: convergence-aware scheduling."""

from __future__ import annotations

import pytest

from agent import schedule
from agent.schema import Action

BASELINE = 0.6016

INCUMBENT = {
    "id": "n004",
    "family": "model",
    "fidelity": "full",
    "status": "ok",
    "accepted": True,
    "config": {"model": "fm", "loss": "pointwise", "features": ["user_id", "video_id"]},
    "metrics": {"primary": BASELINE},
}


def test_a_flat_result_is_a_strike_and_a_real_gain_resets_it():
    scheduler = schedule.Scheduler(baseline=BASELINE)

    assert scheduler.observe(BASELINE + 0.0005) is False  # inside epsilon
    assert scheduler.strikes == 1
    assert scheduler.observe(BASELINE + 0.0300) is True  # a real improvement
    assert scheduler.strikes == 0


def test_an_iteration_that_never_scored_is_also_a_strike():
    """A candidate that died at smoke or screen produced no validation primary.
    It consumed an iteration and improved nothing, which is what the counter
    exists to count — treating it as neutral would hide the most common way a
    run burns its budget."""
    scheduler = schedule.Scheduler(baseline=BASELINE)

    scheduler.observe(None)
    scheduler.observe(None)

    assert scheduler.strikes == 2


def test_hedging_starts_one_strike_before_the_convergence_rule_can_fire():
    """N=3 ends the run, so the intervention has to land at 2."""
    scheduler = schedule.Scheduler(baseline=BASELINE)

    scheduler.observe(None)
    assert scheduler.should_hedge() is False
    scheduler.observe(None)
    assert scheduler.should_hedge() is True


def test_six_consecutive_exploration_failures_never_reach_three_strikes():
    """The A11 acceptance criterion, as a simulation of the loop's own
    sequencing: check for a hedge, fire it if one is due, then score whatever
    the iteration produced — hedges included."""
    scheduler = schedule.Scheduler(baseline=BASELINE)
    observed_strikes = []
    forced_iterations = 0

    for _ in range(6):
        hedge = scheduler.next_hedge(INCUMBENT) if scheduler.should_hedge() else None
        if hedge is not None:
            forced_iterations += 1
            scheduler.note_hedge_fired()
        scheduler.observe(None)  # everything fails, hedges included
        observed_strikes.append(scheduler.strikes)

    assert max(observed_strikes) < 3
    assert forced_iterations > 0


def test_a_failing_hedge_is_not_excused_from_the_accounting():
    """Firing a hedge resets the streak it interrupted, but the hedge then
    answers for its own result. Otherwise a hedge queue would suppress the
    counter indefinitely and quietly disable the convergence rule."""
    scheduler = schedule.Scheduler(baseline=BASELINE)
    scheduler.observe(None)
    scheduler.observe(None)

    scheduler.note_hedge_fired()
    assert scheduler.strikes == 0

    scheduler.observe(None)
    assert scheduler.strikes == 1


def _first_feature_hedge(scheduler) -> Action:
    """Skip past the tune hedge, which the queue offers first."""
    for _ in range(5):
        hedge = scheduler.next_hedge(INCUMBENT)
        if hedge is not None and hedge.type == "config":
            return hedge
    raise AssertionError("no feature hedge was offered")


def test_the_tune_hedge_is_offered_first_and_only_once():
    """Ordered by probability of clearing the bar, not novelty: refining a
    configuration already measured as good beats adding an untested signal.
    Offered once — a second study over the same space would re-search ground
    the first already covered."""
    scheduler = schedule.Scheduler(baseline=BASELINE)

    first = scheduler.next_hedge(INCUMBENT)
    second = scheduler.next_hedge(INCUMBENT)

    assert first.type == "tune"
    assert first.search_space
    assert first.budget == schedule.TUNE_BUDGET
    assert first.parent == INCUMBENT["id"]
    assert second.type == "config"


def test_the_tune_hedge_is_enabled_now_that_execute_dispatches_it():
    """Regression: this was deliberately off while execute() ignored
    search_space for type='tune' and ran the config unchanged, which would
    have written a node claiming to be a tuning run that silently re-ran its
    parent. agent/execute.py::_run_tune_action is what lifted the block."""
    assert schedule.TUNE_HEDGE_ENABLED is True

    scheduler = schedule.Scheduler(baseline=BASELINE)
    hedge = scheduler.next_hedge(INCUMBENT)

    assert hedge.type == "tune"
    # A tune Action with no search_space is exactly the no-op this guards.
    assert hedge.search_space, "a tune hedge without a search space is a no-op experiment"


def test_the_hedge_adds_one_untried_registered_feature_to_the_incumbent():
    scheduler = schedule.Scheduler(baseline=BASELINE)

    hedge = _first_feature_hedge(scheduler)

    assert isinstance(hedge, Action)
    assert hedge.type == "config"
    assert hedge.parent == INCUMBENT["id"]
    added = set(hedge.config.features) - set(INCUMBENT["config"]["features"])
    assert len(added) == 1
    # Model, loss and seed are held fixed so any delta is attributable.
    assert hedge.config.model == INCUMBENT["config"]["model"]
    assert hedge.config.loss == INCUMBENT["config"]["loss"]


def test_successive_hedges_do_not_re_offer_the_same_feature():
    scheduler = schedule.Scheduler(baseline=BASELINE)

    first = _first_feature_hedge(scheduler)
    second = _first_feature_hedge(scheduler)

    assert set(first.config.features) != set(second.config.features)


def test_the_hedge_queue_runs_dry_rather_than_repeating_itself():
    """An exhausted queue must return None so the loop falls back to free
    exploration, instead of re-running a configuration already measured."""
    from pipeline.features import FEATURES

    scheduler = schedule.Scheduler(baseline=BASELINE)
    # One tune hedge, then one per registered feature the incumbent lacks.
    unused = len(FEATURES) - len(
        set(FEATURES) & set(INCUMBENT["config"]["features"])
    )
    for _ in range(unused + 1):
        assert scheduler.next_hedge(INCUMBENT) is not None

    assert scheduler.next_hedge(INCUMBENT) is None


def test_no_hedge_without_an_incumbent_to_build_on():
    scheduler = schedule.Scheduler(baseline=BASELINE)

    assert scheduler.next_hedge(None) is None
    assert scheduler.next_hedge({"id": "n000", "config": {}}) is None


def test_the_hedge_cites_the_ablation_evidence_that_motivated_it():
    scheduler = schedule.Scheduler(baseline=BASELINE)
    ablation = {
        "base_primary": 0.61,
        "components": [{"component": "feature:video_ctr", "sensitivity": 0.11, "delta": -0.11}],
    }

    scheduler.next_hedge(INCUMBENT, ablation)  # the tune hedge, offered first
    hedge = scheduler.next_hedge(INCUMBENT, ablation)

    assert "feature:video_ctr" in hedge.hypothesis
    assert INCUMBENT["id"] in hedge.hypothesis


def test_iteration_event_carries_the_strike_counter():
    """A11 requires the counter in every iteration's log line."""
    scheduler = schedule.Scheduler(baseline=BASELINE)
    scheduler.observe(None)

    event = schedule.iteration_event(scheduler, forced=True, node_id="n009")

    assert event["event"] == "iteration"
    assert event["strikes"] == 1
    assert event["scheduler_forced"] is True
    assert event["manual_intervention"] is False


@pytest.mark.parametrize("baseline", [None, BASELINE])
def test_the_first_result_is_measured_against_the_baseline_when_given(baseline):
    """With a baseline the first candidate must actually beat it; without one
    there is nothing to compare against and the first score sets the mark."""
    scheduler = schedule.Scheduler(baseline=baseline)

    improved = scheduler.observe(BASELINE + 0.0005)

    assert improved is (baseline is None)
