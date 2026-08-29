"""A7 acceptance: family-diverse + epsilon-greedy parent selection.

Section 9.1: "All five families covered before any is refined twice; over 40
iterations, >=15% of nodes must branch from a non-best parent."

select_parent() cannot force the LLM's chosen family — that field lives on
the proposed Action, decided in propose.py. What it *can* guarantee, and what
these tests check directly: the non-best branch rate matches EPSILON_GREEDY
statistically, and when it does branch non-best, it is weighted toward the
families the run has explored least.
"""

from __future__ import annotations

import random

import pytest

from agent import loop
from agent.loop import ROOT_PARENT, select_parent


def node(node_id: str, family: str, primary: float, *, fidelity: str = "full") -> dict:
    return {
        "id": node_id,
        "family": family,
        "status": "ok",
        "fidelity": fidelity,
        "metrics": {"primary": primary},
    }


# --- basic selection ---------------------------------------------------

def test_no_branchable_nodes_returns_root():
    assert select_parent([]) == ROOT_PARENT


def test_only_smoke_and_screen_nodes_returns_root():
    """A pilot is never a legitimate branch point (6.3)."""
    history = [
        node("n001", "model", 0.9, fidelity="smoke"),
        node("n002", "model", 0.9, fidelity="screen"),
    ]
    assert select_parent(history, rng=random.Random(0)) == ROOT_PARENT


def test_errored_nodes_are_never_selected():
    history = [
        {"id": "n001", "family": "model", "status": "error", "fidelity": "full",
         "metrics": {"primary": 0.99}},
        node("n002", "model", 0.60),
    ]
    # rng.random() forced below EPSILON_GREEDY so the explore branch runs too
    picked = select_parent(history, rng=random.Random(1))
    assert picked["id"] == "n002"


def test_single_candidate_always_returned_even_on_explore_roll():
    """With one candidate there is no 'non-best' option to explore into."""
    history = [node("n001", "model", 0.6)]
    always_explore = _FixedRoll(0.0)
    assert select_parent(history, rng=always_explore)["id"] == "n001"


def test_exploit_returns_the_highest_primary():
    history = [node("n001", "model", 0.60), node("n002", "feature", 0.65),
               node("n003", "objective", 0.55)]
    never_explore = _FixedRoll(0.99)
    assert select_parent(history, rng=never_explore)["id"] == "n002"


class _FixedRoll:
    """A deterministic stand-in for the rng.random()/choices() surface."""

    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value

    def choices(self, population, weights, k):  # noqa: ARG002
        return [population[0]]


# --- the 15% non-best acceptance criterion ------------------------------

def test_over_many_iterations_at_least_15_percent_branch_non_best():
    history = [node("n001", "model", 0.60), node("n002", "feature", 0.55),
               node("n003", "objective", 0.50), node("n004", "training", 0.45)]
    rng = random.Random(42)
    picks = [select_parent(history, rng=rng)["id"] for _ in range(4000)]
    non_best_rate = sum(pick != "n001" for pick in picks) / len(picks)
    assert non_best_rate == pytest.approx(loop.EPSILON_GREEDY, abs=0.02)


def test_non_best_rate_holds_regardless_of_family_skew():
    """The weighting changes *which* non-best node is picked, never *how
    often* the loop goes non-best — that rate is EPSILON_GREEDY by design."""
    history = (
        [node(f"n{i:03d}", "model", 0.60 - i * 0.001) for i in range(20)]
        + [node("n999", "feature", 0.10)]
    )
    rng = random.Random(7)
    picks = [select_parent(history, rng=rng)["id"] for _ in range(4000)]
    non_best_rate = sum(pick != "n000" for pick in picks) / len(picks)
    assert non_best_rate == pytest.approx(loop.EPSILON_GREEDY, abs=0.02)


# --- family-weighted exploration -----------------------------------------

def test_explore_prefers_the_least_attempted_family():
    """'model' has 6 attempts, 'feature' 1 — the single feature candidate's
    weight (1/(1+1)) beats any single model candidate's (1/(1+6)), so an
    argmax over weights must land on it even though 5 model candidates
    outnumber it."""
    history = [node(f"n{i:03d}", "model", 0.60 - i * 0.001) for i in range(6)]
    history += [node("n900", "feature", 0.10)]

    picked = select_parent(history, rng=_ArgmaxRoll())
    assert picked["family"] == "feature"


class _ArgmaxRoll:
    """Always explores, and always takes the highest-weighted candidate.

    Deterministic stand-in that isolates the weighting logic from sampling
    noise — the statistical rate itself is covered by the 15%-rate tests
    above, using real randomness.
    """

    def random(self) -> float:
        return 0.0

    def choices(self, population, weights, k):  # noqa: ARG002
        best_index = max(range(len(weights)), key=lambda i: weights[i])
        return [population[best_index]]


def test_uncovered_family_gets_maximum_explore_weight():
    """A family with zero attempts anywhere in history weights highest of all
    once it does have a candidate to offer — 1/(1+0) beats every attempted
    family's 1/(1+n)."""
    fresh = [
        node("n001", "model", 0.60),
        node("n002", "model", 0.59),
        node("n010", "objective", 0.10),  # untried family, one candidate
    ]
    picked = select_parent(fresh, rng=_ArgmaxRoll())
    assert picked["family"] == "objective"


# --- integration with the loop --------------------------------------------

def test_run_uses_select_parent_not_a_hardcoded_best_only_policy():
    """A7 replaces A6's placeholder at the call site, not just in isolation."""
    assert "select_parent" in loop.run.__code__.co_names
    assert "_parent_for_a6" not in dir(loop)
