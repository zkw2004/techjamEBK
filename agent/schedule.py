"""Convergence-aware scheduling. Task A11.

The convergence rule (Section 4.5: epsilon 0.002, N=3) reads only full/confirm
nodes, so it can fire on the 4th one — right after the 3rd, if none improved on
the 1st. That is not a hypothetical: a real run converged after exactly 4 full
evaluations, having tried one idea per family and nothing twice (trap 13).

Three exploratory duds in a row therefore end a run that still has 46 of its 50
iterations left. This scheduler counts those duds and, one strike before the
rule can bite, forces a *hedge* — an action chosen for probability of clearing
the bar rather than for novelty — instead of another free exploration.

**What this does not do.** It does not defeat the convergence rule and is not
meant to. ``loop.converged()`` reads the node ledger independently; if the
hedges also fail to improve, the run still converges, correctly. The scheduler
changes which action is spent, not whether the run is allowed to stop. Claiming
otherwise would be the same kind of overreach as a gate that accepts noise.
"""

from __future__ import annotations

from typing import Any

from agent.schema import FAMILIES, Action

EPSILON = 0.002  # Section 4.5, same threshold the convergence rule uses
HEDGE_AT = 2  # fire one strike before N=3 can end the run

# execute() does not dispatch type="tune" to pipeline.tune.run_study: it runs
# the config as-is and ignores search_space (agent/execute.py). Emitting an
# Optuna-refinement hedge before that dispatch exists would produce a node
# claiming to be a tuning run that had silently re-run the parent config
# unchanged — a no-op experiment, which is the failure mode the accept gate and
# the schema validator were both hardened against. Flip this on in the same
# change that wires tune dispatch, not before.
TUNE_HEDGE_ENABLED = False


class Scheduler:
    """Strike counter over iteration outcomes, plus a hedge queue.

    An iteration is a *strike* when it did not improve the best validation
    primary by more than ``epsilon`` — including an iteration that produced no
    full-tier score at all, since a candidate that died at smoke or screen is
    exactly as unproductive as one that scored flat.
    """

    def __init__(
        self,
        *,
        epsilon: float = EPSILON,
        hedge_at: int = HEDGE_AT,
        baseline: float | None = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.hedge_at = int(hedge_at)
        self.strikes = 0
        self.hedges_fired = 0
        self._best = float(baseline) if baseline is not None else None
        self._offered_features: set[str] = set()

    def observe(self, primary: float | None) -> bool:
        """Record one iteration's best full-tier primary. True if it improved."""
        improved = (
            primary is not None
            and (self._best is None or float(primary) - self._best > self.epsilon)
        )
        if improved:
            value = float(primary)
            self._best = value if self._best is None else max(self._best, value)
            self.strikes = 0
        else:
            self.strikes += 1
        return improved

    def should_hedge(self) -> bool:
        return self.strikes >= self.hedge_at

    def note_hedge_fired(self) -> None:
        """A forced hedge breaks the streak it was fired to interrupt.

        Without this the counter keeps climbing through the hedge itself, the
        run hits three strikes anyway, and the intervention buys nothing. The
        hedge's own outcome is still observed on the next ``observe`` call, so
        a hedge that fails simply starts a fresh streak rather than being
        excused from the accounting.
        """
        self.hedges_fired += 1
        self.strikes = 0

    def next_hedge(self, best_node: dict | None, ablation: dict | None = None) -> Action | None:
        """The highest-probability action available, or None if the queue is dry.

        Returning None is a real outcome: with no incumbent, no untried
        feature, and no tune dispatch, there is no hedge worth forcing and the
        loop should fall back to free exploration rather than re-run something
        it has already measured.
        """
        config = (best_node or {}).get("config") or {}
        if not config.get("model"):
            return None

        feature = self._next_feature(config, ablation)
        if feature is not None:
            self._offered_features.add(feature)
            return self._feature_action(best_node or {}, config, feature, ablation)
        return None

    def _next_feature(self, config: dict, ablation: dict | None) -> str | None:
        from agent.ablate import unused_features

        for name in unused_features(ablation, config):
            if name not in self._offered_features:
                return name
        return None

    @staticmethod
    def _feature_action(
        best_node: dict, config: dict, feature: str, ablation: dict | None
    ) -> Action:
        parent_id = best_node.get("id") or "n000"
        evidence = ""
        if ablation and ablation.get("components"):
            top = ablation["components"][0]
            evidence = (
                f" The last ablation of {parent_id} ranked {top['component']} most "
                f"sensitive (|delta| {top['sensitivity']:.6f}), so the incumbent's "
                "score is not saturated in its feature block."
            )
        family = best_node.get("family")
        return Action(
            hypothesis=(
                f"Adding the registered feature {feature!r} to the incumbent "
                f"({parent_id}) supplies signal its current feature set does not "
                "carry, and lifts validation primary by more than the 0.002 "
                "minimum meaningful delta. If it lands inside that margin, this "
                "feature is inert for this model and the feature drawer is not "
                f"where the remaining headroom is.{evidence}"
            ),
            reasoning=(
                "Scheduler-forced hedge, not a free proposal: two consecutive "
                "iterations failed to improve the incumbent by more than epsilon, "
                "and a third would let the convergence rule end the run. A single "
                "registered feature added to a known-good config is the "
                "highest-probability action available, and it holds model, loss "
                "and seed fixed so any delta is attributable to the feature alone."
            ),
            type="config",
            family=family if family in FAMILIES else "feature",
            parent=parent_id,
            config={**config, "features": [*(config.get("features") or []), feature]},
        )


def iteration_event(scheduler: Scheduler, *, forced: bool, node_id: Any = None) -> dict:
    """The per-iteration log line A11 requires the strike counter to appear in."""
    return {
        "event": "iteration",
        "strikes": scheduler.strikes,
        "hedges_fired": scheduler.hedges_fired,
        "scheduler_forced": bool(forced),
        "node": node_id,
        "manual_intervention": False,
    }
