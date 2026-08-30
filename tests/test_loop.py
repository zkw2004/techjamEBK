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

    loop.run(max_iterations=10, knowledge="fixture", sleep_fn=lambda _seconds: None)

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

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=sleeps.append)

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

    assert loop.run(max_iterations=2, knowledge="fixture") == []

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

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _seconds: None)

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
    bypasses execute() entirely and always leaves accepted=False."""
    calls = iter(primaries_by_call)

    def fake_execute(action: Action, fidelity: str, timeout_s: int, accept_fn=None) -> dict:
        del timeout_s
        if fidelity != "full":
            return _success(action, fidelity)
        raw = _full_result(next(calls))
        accepted = accept_fn(raw) if accept_fn is not None else False
        return executor._node(action, fidelity, raw, accepted=accepted)

    return fake_execute


def test_first_full_success_is_accepted_unconditionally(isolated_loop, monkeypatch):
    """Nothing to compare against yet — the first full result establishes the
    reference, matching _screen_survives' identical convention one tier down."""
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.55]))

    loop.run(max_iterations=1, knowledge="fixture", sleep_fn=lambda _s: None)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert len(full_nodes) == 1
    assert full_nodes[0]["accepted"] is True


def test_second_full_candidate_is_gated_against_the_first(isolated_loop, monkeypatch):
    monkeypatch.setattr(loop.propose, "propose", lambda *_a: (
        _action(0), {"in": 1, "out": 1, "model": "fake"}
    ))
    # First candidate: 0.55, auto-accepted. Second: 0.90 — a real gate.accept
    # call would need real per-row alignment against the official validation
    # set, so it's mocked here; run()'s wiring is what's under test, not
    # gate.accept's own bootstrap correctness (covered in test_gate.py).
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (True, (0.01, 0.05)))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.55, 0.90]))

    loop.run(max_iterations=2, knowledge="fixture", sleep_fn=lambda _s: None)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert [n["accepted"] for n in full_nodes] == [True, True]
    assert [n["metrics"]["primary"] for n in full_nodes] == [0.55, 0.90]


def test_rejected_full_candidate_is_never_selected_as_the_next_parent(isolated_loop, monkeypatch):
    parents_seen = []

    def fake_propose(history, _knowledge, parent):
        parents_seen.append(parent["id"])
        return _action(len(parents_seen)), {"in": 1, "out": 1, "model": "fake"}

    monkeypatch.setattr(loop.propose, "propose", fake_propose)
    monkeypatch.setattr(loop.gate, "accept", lambda *_a, **_k: (False, (-0.02, 0.01)))
    monkeypatch.setattr(loop.execute, "execute", _accept_aware_execute([0.55, 0.30]))

    loop.run(max_iterations=2, knowledge="fixture", sleep_fn=lambda _s: None)

    full_nodes = [n for n in store.list_nodes() if n["fidelity"] == "full"]
    assert [n["accepted"] for n in full_nodes] == [True, False]
    # the rejected node (n002-equivalent) must never become a parent
    assert parents_seen[-1] != full_nodes[1]["id"]
