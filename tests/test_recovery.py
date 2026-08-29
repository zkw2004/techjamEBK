"""A5 acceptance: bounded, recorded per-error-class recovery policies."""

from __future__ import annotations

import pytest

from agent import recovery, store


@pytest.fixture(autouse=True)
def isolated_event_log(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")


def _node(error_class: str, *, fidelity: str = "full", batch_size: int = 4096) -> dict:
    return {
        "id": "n017",
        "status": "error",
        "fidelity": fidelity,
        "config": {"model": "deepfm", "hparams": {"batch_size": batch_size}},
        "errors": [
            {
                "stage": fidelity,
                "error_class": error_class,
                "traceback": f"{error_class} failure",
            }
        ],
    }


@pytest.mark.parametrize("error_class", ["syntax", "schema"])
def test_syntax_and_schema_get_exactly_one_haiku_repair(error_class):
    node = _node(error_class)

    plan = recovery.recover(node, attempt=0)

    assert plan["operation"] == "repair"
    assert plan["repair_attempted"] is True
    assert plan["error"] == node["errors"][0]
    assert recovery.recover(node, attempt=1) is None


def test_oom_retries_once_with_a_halved_batch_size_and_keeps_node_unchanged():
    node = _node("oom", batch_size=4096)

    plan = recovery.recover(node, attempt=0)

    assert plan["operation"] == "retry"
    assert plan["config"]["hparams"]["batch_size"] == 2048
    assert node["config"]["hparams"]["batch_size"] == 4096
    assert recovery.recover(node, attempt=1) is None


@pytest.mark.parametrize(
    ("fidelity", "expected"),
    [("confirm", "full"), ("full", "screen"), ("screen", "smoke")],
)
def test_timeout_retries_once_at_lower_fidelity(fidelity, expected):
    plan = recovery.recover(_node("timeout", fidelity=fidelity), attempt=0)

    assert plan["operation"] == "retry"
    assert plan["fidelity"] == expected


def test_smoke_timeout_has_no_lower_fidelity():
    assert recovery.recover(_node("timeout", fidelity="smoke"), attempt=0) is None


def test_transient_errors_back_off_exponentially_for_three_attempts():
    node = _node("transient")

    assert [recovery.recover(node, attempt=i)["backoff_s"] for i in range(3)] == [1.0, 2.0, 4.0]
    assert recovery.recover(node, attempt=3) is None


def test_leak_suspected_is_quarantined_without_retry():
    node = _node("leak_suspected")

    assert recovery.recover(node, attempt=0) is None
    event = store.read_events()[0]
    assert event["decision"] == "quarantine"
    assert event["manual_intervention"] is False


def test_every_recovery_decision_is_logged():
    node = _node("oom")

    recovery.recover(node, attempt=0)
    recovery.recover(node, attempt=1)

    events = store.read_events()
    assert [(event["decision"], event["attempt"]) for event in events] == [
        ("retry", 0),
        ("abandon", 1),
    ]
    assert all(event["node"] == "n017" for event in events)


def test_three_consecutive_dead_nodes_force_a_branch():
    failures = [{"status": "error"}] * recovery.MAX_CONSECUTIVE_DEAD_NODES

    assert recovery.should_force_branch(failures)
    assert not recovery.should_force_branch(failures[:2])
    assert not recovery.should_force_branch(failures + [{"status": "ok"}])
