"""A2 acceptance: write-read round-trips identically; best_node() returns the
highest accepted primary; the tree survives a process restart."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from agent import store

MANIFEST = "a" * 64


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Redirect the ledger into tmp so tests never touch the real deliverable."""
    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    store.set_manifest_provider(lambda: MANIFEST)
    yield tmp_path
    store.set_manifest_provider(None)


def make_node(**overrides):
    node = {
        "parent": "n009",
        "family": "objective",
        "hypothesis": "lambdarank should beat pointwise BCE since GAUC is per-user",
        "action_type": "config",
        "fidelity": "full",
        "status": "ok",
    }
    node.update(overrides)
    return node


def test_write_read_round_trips_identically():
    node = store.normalise(make_node(metrics={"gauc": 0.681, "primary": 0.614}))
    store.write(node)
    assert store.read(node["id"]) == node


def test_write_fills_defaults_and_stamps_timestamp():
    path = store.write(make_node())
    record = json.loads(path.read_text())
    assert record["accepted"] is False
    assert record["errors"] == []
    assert record["manual_intervention"] is False
    assert record["timestamp"].endswith("Z")


def test_ids_are_sequential_and_zero_padded():
    assert store.write(make_node()).stem == "n001"
    assert store.write(make_node()).stem == "n002"
    assert store.next_id() == "n003"


def test_explicit_id_is_respected():
    store.write(make_node(id="n017"))
    assert store.read("n017")["hypothesis"].startswith("lambdarank")
    assert store.next_id() == "n018"


# --- the integrity properties ----------------------------------------------

def test_every_node_carries_the_manifest_hash():
    """Section 8.3: a mid-run evaluator change must not silently invalidate
    earlier results, so the hash is stamped on every record."""
    assert store.read(store.write(make_node()).stem)["manifest_sha256"] == MANIFEST


def test_node_without_manifest_hash_is_rejected():
    store.set_manifest_provider(None)
    with pytest.raises(store.NodeValidationError, match="manifest_sha256"):
        store.write(make_node())


def test_ledger_is_append_only():
    """Section 7.2: the agent cannot delete a failed experiment from the
    ledger, so an existing node is never overwritten."""
    store.write(make_node(id="n001"))
    with pytest.raises(store.NodeValidationError, match="append-only"):
        store.write(make_node(id="n001", status="error"))


def test_node_missing_a_required_field_is_rejected():
    node = make_node()
    del node["hypothesis"]
    with pytest.raises(store.NodeValidationError, match="hypothesis"):
        store.write(node)


def test_failed_nodes_are_recorded_not_discarded():
    """Failures are graded output (design commitment 5)."""
    store.write(make_node(status="error", errors=[{"error_class": "oom"}]))
    assert store.list_nodes()[0]["status"] == "error"


# --- best_node -------------------------------------------------------------

def test_best_node_is_none_before_any_acceptance():
    store.write(make_node(accepted=False, metrics={"primary": 0.9}))
    assert store.best_node() is None


def test_best_node_returns_highest_accepted_primary():
    store.write(make_node(accepted=True, metrics={"primary": 0.601}))
    store.write(make_node(accepted=True, metrics={"primary": 0.614}))
    store.write(make_node(accepted=True, metrics={"primary": 0.608}))
    assert store.best_node()["metrics"]["primary"] == 0.614


def test_best_node_ignores_unaccepted_nodes_that_scored_higher():
    """A node that beat the incumbent on the raw number but failed the
    statistical or segment gate (6.6) must not become the branch parent."""
    store.write(make_node(accepted=True, metrics={"primary": 0.610}))
    store.write(make_node(accepted=False, metrics={"primary": 0.700}))
    assert store.best_node()["metrics"]["primary"] == 0.610


def test_best_node_ignores_errored_nodes():
    store.write(make_node(accepted=True, metrics={"primary": 0.610}))
    store.write(make_node(accepted=True, status="error", metrics={"primary": 0.9}))
    assert store.best_node()["metrics"]["primary"] == 0.610


def test_best_node_breaks_ties_deterministically():
    store.write(make_node(id="n004", accepted=True, metrics={"primary": 0.61}))
    store.write(make_node(id="n002", accepted=True, metrics={"primary": 0.61}))
    assert store.best_node()["id"] == "n002"


# --- numpy, ordering, durability -------------------------------------------

def test_numpy_values_survive_serialisation():
    """run_experiment returns numpy floats; a node that fails to serialise
    mid-run is a lost iteration."""
    store.write(make_node(
        id="n001",
        metrics={"primary": np.float32(0.614), "gauc": np.float64(0.681)},
        fold_primaries=np.array([0.609, 0.612, 0.617]),
        accepted=np.bool_(True),
        seconds=np.float64(94.2),
    ))
    record = store.read("n001")
    assert record["metrics"]["primary"] == pytest.approx(0.614, abs=1e-6)
    assert record["fold_primaries"] == pytest.approx([0.609, 0.612, 0.617])
    assert record["accepted"] is True
    json.dumps(record)  # must be plain JSON, no numpy left behind


def test_list_nodes_is_ordered_numerically_not_lexically():
    for node_id in ("n002", "n010", "n001"):
        store.write(make_node(id=node_id))
    assert [n["id"] for n in store.list_nodes()] == ["n001", "n002", "n010"]


def test_list_nodes_empty_before_any_write():
    assert store.list_nodes() == []


def test_partial_write_is_invisible_to_list_nodes():
    """An interrupted write leaves a temp file, never an nNNN.json."""
    store.write(make_node(id="n001"))
    (store.NODES_DIR / ".n002.garbage.tmp").write_text("{not json")
    assert [n["id"] for n in store.list_nodes()] == ["n001"]


def test_tree_survives_process_restart(isolated_logs):
    """Acceptance criterion, verified across a real process boundary rather
    than by trusting that the in-process read hit the disk."""
    store.write(make_node(id="n001", accepted=True, metrics={"primary": 0.614}))
    script = (
        "import sys; from pathlib import Path; from agent import store;"
        f"store.NODES_DIR = Path({str(store.NODES_DIR)!r});"
        "print(store.best_node()['metrics']['primary'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "0.614"


# --- event stream ----------------------------------------------------------

def test_append_event_writes_one_json_object_per_line():
    store.append_event({"event": "propose", "node": "n001"})
    store.append_event({"event": "smoke_failed", "error_class": "syntax"})
    events = store.read_events()
    assert [e["event"] for e in events] == ["propose", "smoke_failed"]
    assert all(e["timestamp"].endswith("Z") for e in events)


def test_pilots_are_logged_even_though_they_are_not_iterations():
    """A pilot is still an attempt (6.3) — cheap failures are not hidden."""
    store.append_event({"event": "pilot", "fidelity": "smoke", "status": "error"})
    assert store.read_events()[0]["fidelity"] == "smoke"


def test_read_events_tolerates_a_truncated_final_line():
    store.append_event({"event": "propose"})
    with open(store.EVENT_LOG, "a") as fh:
        fh.write('{"event": "killed mid-wri')
    assert [e["event"] for e in store.read_events()] == ["propose"]


def test_read_events_empty_before_any_append():
    assert store.read_events() == []


def test_rendered_run_log_keeps_citation_next_to_hypothesis():
    cited = make_node(
        id="n001",
        hypothesis=(
            "lambdarank should improve per-user ranking "
            "[ref: BPR - Rendle 2009; LambdaRank]"
        ),
    )
    text = store.render_run_log([store.normalise(cited)])

    assert "lambdarank should improve per-user ranking [ref: BPR - Rendle 2009;" in text
    assert "n001 ok full cited:" in text
