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


def test_the_hedge_adds_one_untried_registered_feature_to_the_incumbent():
    scheduler = schedule.Scheduler(baseline=BASELINE)

    hedge = scheduler.next_hedge(INCUMBENT)

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

    first = scheduler.next_hedge(INCUMBENT)
    second = scheduler.next_hedge(INCUMBENT)

    assert set(first.config.features) != set(second.config.features)


def test_the_hedge_queue_runs_dry_rather_than_repeating_itself():
    """An exhausted queue must return None so the loop falls back to free
    exploration, instead of re-running a configuration already measured."""
    from pipeline.features import FEATURES

    scheduler = schedule.Scheduler(baseline=BASELINE)
    for _ in range(len(FEATURES)):
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

    hedge = scheduler.next_hedge(INCUMBENT, ablation)

    assert "feature:video_ctr" in hedge.hypothesis
    assert INCUMBENT["id"] in hedge.hypothesis


def test_tune_hedges_stay_off_until_execute_dispatches_them():
    """execute() ignores search_space for type='tune' and runs the config as
    is. Emitting a tune hedge before that dispatch lands would write a node
    claiming to be a tuning run that had silently re-run its parent — the
    no-op-experiment failure this project has already been bitten by once."""
    assert schedule.TUNE_HEDGE_ENABLED is False

    scheduler = schedule.Scheduler(baseline=BASELINE)
    hedges = [scheduler.next_hedge(INCUMBENT) for _ in range(3)]

    assert all(hedge.type == "config" for hedge in hedges if hedge is not None)


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
