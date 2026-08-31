"""A6 acceptance: autonomous fidelity escalation and recovery orchestration."""

from __future__ import annotations

import pytest

from agent import execute as executor
from agent import loop, store
from agent.propose import ProposeError
from agent.schema import Action

MANIFEST = "b" * 64


@pytest.fixture
def isolated_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    preflights = []

    def fake_preflight():
        preflights.append(True)
        store.set_manifest_provider(lambda: MANIFEST)
        return {"metric_profile": {}}

    monkeypatch.setattr(loop.manifest, "preflight", fake_preflight)
    yield preflights
    store.set_manifest_provider(None)


def _action(index: int) -> Action:
    return Action(
        hypothesis=f"candidate {index}",
        reasoning="fixture",
        type="config",
        family="model",
        parent="n000",
        config={"model": "random", "hparams": {"candidate": index}},
    )


def _success(action: Action, fidelity: str, *, folds: list[float] | None = None) -> dict:
    metrics = {} if fidelity == "smoke" else {"gauc": 0.6, "ndcg": 0.58, "primary": 0.59}
    return executor._node(  # noqa: SLF001 - fixture uses A4's node contract directly
        action,
        fidelity,
        {
            "status": "ok",
            "gauc": metrics.get("gauc"),
            "ndcg": metrics.get("ndcg"),
            "primary": metrics.get("primary"),
            "fold_primaries": [] if fidelity == "smoke" else (folds or [0.58, 0.59, 0.60]),
            "segments": {},
            "seconds": 0.01,
            "gpu_seconds": 0.0,
        },
    )


def test_ten_candidates_run_unattended_and_cheap_tiers_filter_majority(isolated_loop, monkeypatch):
    actions = iter(_action(index) for index in range(10))
    calls: list[tuple[int, str]] = []

    def fake_propose(*_args):
        return next(actions), {"in": 1, "out": 1, "model": "fake"}

    def fake_execute(action: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        del timeout_s
        candidate = action.config.hparams["candidate"]
        calls.append((candidate, fidelity))
        if fidelity == "screen" and candidate < 7:
            # A malformed internal-fold result is a red flag and cannot spend
            # the official validation window.
            return _success(action, fidelity, folds=[0.59, 0.60])
        if fidelity == "screen":
            return _success(action, fidelity, folds=[0.65, 0.66, 0.67])
        return _success(action, fidelity)

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.execute, "execute", fake_execute)

    loop.run(max_iterations=10, knowledge="fixture", sleep_fn=lambda _seconds: None,
             seed_baseline=False)

    assert len(isolated_loop) == 1
    assert sum(fidelity == "smoke" for _, fidelity in calls) == 10
    assert sum(fidelity == "screen" for _, fidelity in calls) == 10
    assert sum(fidelity == "full" for _, fidelity in calls) == 3
    assert all(node["manual_intervention"] is False for node in store.list_nodes())
    assert all(node["manifest_sha256"] == MANIFEST for node in store.list_nodes())


def test_transient_failure_uses_a5_retry_without_human_input(isolated_loop, monkeypatch):
    action = _action(0)
    attempts = 0
    sleeps: list[float] = []

    def fake_propose(*_args):
        return action, {"in": 1, "out": 1, "model": "fake"}

    def fake_execute(current: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        nonlocal attempts
        del timeout_s
        if fidelity == "smoke" and attempts == 0:
            attempts += 1
            return executor._node(
                current,
                fidelity,
                {
                    "status": "error",
                    "stage": "smoke",
                    "error_class": "transient",
                    "traceback": "network hiccup",
                    "seconds": 0.01,
                },
            )
        return _success(current, fidelity)

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.execute, "execute", fake_execute)

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=sleeps.append,
             seed_baseline=False)

    assert sleeps == [1.0]
    assert [node["status"] for node in store.list_nodes()] == ["error", "ok", "ok", "ok"]
    assert all(node["manual_intervention"] is False for node in store.list_nodes())
    assert store.read_events()[0]["event"] == "recovery"


def test_proposal_failure_is_logged_and_the_loop_keeps_going(isolated_loop, monkeypatch):
    calls = 0

    def fake_propose(*_args):
        nonlocal calls
        calls += 1
        raise ProposeError("service unavailable", usage={"in": 3, "out": 0, "model": "fake"})

    monkeypatch.setattr(loop.propose, "propose", fake_propose)

    assert loop.run(max_iterations=2, knowledge="fixture", seed_baseline=False) == []

    assert calls == 2
    events = store.read_events()
    assert len(events) == 2
    assert all(event["manual_intervention"] is False for event in events)


def test_convergence_uses_only_full_evaluations():
    history = [
        {"status": "ok", "fidelity": "full", "metrics": {"primary": 0.60}},
        {"status": "ok", "fidelity": "smoke", "metrics": {"primary": 0.99}},
        {"status": "ok", "fidelity": "full", "metrics": {"primary": 0.601}},
        {"status": "ok", "fidelity": "full", "metrics": {"primary": 0.6005}},
        {"status": "ok", "fidelity": "confirm", "metrics": {"primary": 0.6015}},
    ]

    assert loop.converged(history)
    history[-1]["metrics"]["primary"] = 0.605
    assert not loop.converged(history)


@pytest.mark.parametrize("value", [0, -1, True, "10"])
def test_run_rejects_invalid_iteration_counts(value, isolated_loop):
    with pytest.raises(ValueError, match="max_iterations"):
        loop.run(max_iterations=value, knowledge="fixture")


@pytest.mark.parametrize("value", [0, -1, True])
def test_run_rejects_invalid_max_hours(value, isolated_loop):
    with pytest.raises(ValueError, match="max_hours"):
        loop.run(max_iterations=1, max_hours=value, knowledge="fixture")


def test_max_hours_stops_before_the_iteration_cap(isolated_loop, monkeypatch):
    """K24: a wall-clock ceiling, checked before each new candidate — never
    mid-candidate, so a running full evaluation is never killed partway."""

    def fake_propose(*_args):
        return _action(0), {"in": 1, "out": 1, "model": "fake"}

    def fake_execute(action: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        del timeout_s
        return _success(action, fidelity)

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.execute, "execute", fake_execute)

    # First check (before iteration 1) is under budget; every check after is
    # over it — so exactly one candidate completes before the deadline fires.
    clock = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    written = loop.run(
        max_iterations=50,
        max_hours=1 / 3600,  # 1 second
        knowledge="fixture",
        sleep_fn=lambda _seconds: None,
        time_fn=lambda: next(clock, 100.0),
        seed_baseline=False,
    )

    assert loop.LAST_STOP_REASON == "time_cap"
    assert len(written) > 0  # the in-flight candidate was allowed to finish


def test_last_stop_reason_reports_the_iteration_cap(isolated_loop, monkeypatch):
    def fake_propose(*_args):
        return _action(0), {"in": 1, "out": 1, "model": "fake"}

    def fake_execute(action: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        del timeout_s
        return _success(action, fidelity)

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.execute, "execute", fake_execute)

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _seconds: None,
             seed_baseline=False)

    assert loop.LAST_STOP_REASON == "iteration_cap"


def _full_result(primary: float) -> dict:
    """A raw C1-shaped result: has val_scores/val_user_ids, unlike _success()."""
    return {
        "status": "ok", "gauc": primary, "ndcg": primary, "primary": primary,
        "fold_primaries": [primary, primary, primary], "segments": {},
        "val_scores": [primary, primary], "val_user_ids": [0, 1],
        "seconds": 0.01, "gpu_seconds": 0.0,
    }


def _accept_aware_execute(primaries_by_call):
    """A fake execute() that genuinely applies accept_fn, like agent/execute.py
    does — needed to exercise run()'s _accept_full wiring, since _success()
    bypasses execute() entirely and always leaves accepted=False.

    It must normalise the verdict exactly as execute() does: accept_fn returns
    a verdict dict carrying the gate's evidence, and a dict is truthy, so a
    fake that passed it straight through as `accepted` would mark every node
    accepted regardless of the decision.
    """
    calls = iter(primaries_by_call)

    def fake_execute(action: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        del timeout_s
        if fidelity == "screen":
            # The incumbent's reference screen must score below the candidate,
            # or the screen gate correctly refuses to spend a full evaluation.
            reference = action.hypothesis == loop.SCREEN_REFERENCE_HYPOTHESIS
            folds = [0.50, 0.51, 0.52] if reference else [0.58, 0.59, 0.60]
            return _success(action, fidelity, folds=folds)
        if fidelity != "full":
            return _success(action, fidelity)
        raw = _full_result(next(calls))
        verdict = executor._normalise_verdict(accept_fn(raw)) if accept_fn is not None else None
        accepted = bool(verdict["accepted"]) if verdict is not None else False
        return executor._node(action, fidelity, raw, accepted=accepted, verdict=verdict)

    return fake_execute


def test_first_full_success_below_the_baseline_is_rejected(isolated_loop, monkeypatch):
    """Regression: a run once accepted its first full node unconditionally.

    That made whatever happened to run first the permanent bar. A real run
    anchored on a node scoring 0.5837 — below the shipped 0.6016 baseline —
    and then reported a later 0.6026 as a win. The first candidate is now
    gated on the official baseline like every other.
    """
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.5837]))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None,
             seed_baseline=False)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert len(full_nodes) == 1
    assert full_nodes[0]["accepted"] is False
    assert full_nodes[0]["gates"]["min_delta"] is False


def test_a_gain_inside_the_noise_floor_is_rejected(isolated_loop, monkeypatch):
    """0.6026 over a 0.6016 baseline is +0.0010 — below MIN_DELTA_FLOOR and
    barely over one seed std. The previous run promoted exactly this."""
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.6026]))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None,
             seed_baseline=False)

    node = [n for n in store.list_nodes() if n["fidelity"] == "full"][0]
    assert node["accepted"] is False
    assert node["delta_vs_best"] == pytest.approx(0.6026 - loop.BASELINE_VALIDATION_PRIMARY)


def test_a_clear_win_is_accepted_and_records_its_evidence(isolated_loop, monkeypatch):
    """Above baseline + MIN_DELTA_FLOOR, and the node carries the evidence.

    gates/ci_95/delta_vs_best were `{}`/None on all 69 nodes of the previous
    run because execute() never wrote them — acceptance was unfalsifiable.
    """
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.65]))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None,
             seed_baseline=False)

    node = [n for n in store.list_nodes() if n["fidelity"] == "full"][0]
    assert node["accepted"] is True
    assert node["gates"] == {"statistical": True, "min_delta": True}
    assert node["delta_vs_best"] == pytest.approx(0.65 - loop.BASELINE_VALIDATION_PRIMARY)


def test_second_full_candidate_is_gated_against_the_first(isolated_loop, monkeypatch):
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    # Both clear the baseline margin. The second must additionally clear the
    # bootstrap CI against the first — a real gate.accept call would need
    # per-row alignment against the official validation set, so it is mocked;
    # run()'s wiring is under test, not gate.accept's bootstrap correctness
    # (covered in test_gate.py).
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (True, (0.01, 0.05)))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.65, 0.90]))

    loop.run(max_iterations=2, knowledge="fixture", sleep_fn=lambda _s: None,
             seed_baseline=False)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert [n["accepted"] for n in full_nodes] == [True, True]
    assert [n["metrics"]["primary"] for n in full_nodes] == [0.65, 0.90]
    assert full_nodes[1]["ci_95"] == [0.01, 0.05]


def test_rejected_full_candidate_is_never_selected_as_the_next_parent(isolated_loop, monkeypatch):
    parents_seen = []

    def fake_propose(history, _knowledge, parent):
        parents_seen.append(parent["id"])
        return _action(len(parents_seen)), {"in": 1, "out": 1, "model": "fake"}

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (False, (-0.02, 0.01)))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.65, 0.30]))

    loop.run(max_iterations=2, knowledge="fixture", sleep_fn=lambda _s: None,
             seed_baseline=False)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert [n["accepted"] for n in full_nodes] == [True, False]
    # the rejected node (n002-equivalent) must never become a parent
    assert parents_seen[-1] != full_nodes[1]["id"]


def test_screen_gate_compares_like_budgets_not_capped_against_uncapped():
    """Regression: the screen gate compared a budget-capped candidate against
    the parent's *uncapped* full-tier folds (pipeline.train.SCREEN_BUDGET_CAPS
    clamps screen runs but _run_full does not). A candidate that is genuinely
    better under an equal budget still loses that comparison, which is why a
    50-iteration run promoted only 2 candidates to full."""
    capped_candidate = {"status": "ok", "fold_primaries": [0.59, 0.56, 0.55]}
    parent_full_budget = [0.60, 0.57, 0.56]
    parent_same_budget = [0.58, 0.55, 0.54]

    assert loop._screen_survives(capped_candidate, parent_full_budget) is False
    assert loop._screen_survives(capped_candidate, parent_same_budget) is True


def test_screen_gate_promotes_when_there_is_no_usable_reference():
    candidate = {"status": "ok", "fold_primaries": [0.59, 0.56, 0.55]}

    assert loop._screen_survives(candidate, None) is True
    assert loop._screen_survives(candidate, [0.1, float("nan"), 0.3]) is True


def test_baseline_anchor_runs_the_official_five_fields():
    """Config.features defaults to two fields, so an anchor that omitted them
    would train user_id x video_id and land near the popularity baseline
    instead of reproducing 0.6016 — the exact defect that made the previous
    run's headline comparison invalid."""
    from pipeline.data import FIELDS

    action = loop._baseline_anchor_action()

    assert action.config.features == list(FIELDS)
    assert "author_id" in action.config.features
    assert action.config.model == "fm"
    assert action.config.hparams == {"k": 16, "lr": 0.001}


def test_baseline_anchor_is_seeded_once_and_becomes_the_incumbent(isolated_loop, monkeypatch):
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.6016, 0.90]))
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (True, (0.01, 0.05)))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    anchor = full_nodes[0]
    assert anchor["gates"]["baseline_anchor"] is True
    assert anchor["accepted"] is True  # the reference by construction, not by gate
    assert loop._has_baseline_anchor(store.list_nodes()) is True
    assert store.read_events()[0]["event"] == "baseline_anchor"


def test_baseline_anchor_is_not_repeated_on_a_resumed_run(isolated_loop, monkeypatch):
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (True, (0.01, 0.05)))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.6016, 0.90, 0.91]))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None)
    anchors_after_first = sum(
        n.get("gates", {}).get("baseline_anchor") is True for n in store.list_nodes()
    )
    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None)
    anchors_after_second = sum(
        n.get("gates", {}).get("baseline_anchor") is True for n in store.list_nodes()
    )

    assert anchors_after_first == 1
    assert anchors_after_second == 1
