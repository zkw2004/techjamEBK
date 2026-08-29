"""Node tree persistence. One JSON per node in logs/nodes/.

Contract: AGENT_PLAN.md Section 8.7 (FROZEN). Owner: Workstream A. Task A2.

`logs/nodes/<id>.json` IS the run-log deliverable. Build the schema on Day 1
or the data will not exist on Day 3 (trap 12).
"""

from __future__ import annotations

from pathlib import Path

NODES_DIR = Path("logs/nodes")
EVENT_LOG = Path("logs/run.jsonl")


def write(node: dict) -> Path:
    """Persist a completed node. Must round-trip identically with read()."""
    raise NotImplementedError("A2")


def read(node_id: str) -> dict:
    raise NotImplementedError("A2")


def list_nodes() -> list[dict]:
    """All nodes on disk. The tree must survive a process restart."""
    raise NotImplementedError("A2")


def best_node() -> dict | None:
    """Highest accepted primary. None before the first acceptance."""
    raise NotImplementedError("A2")


def append_event(event: dict) -> None:
    """Append-only event stream: proposals, pilots, errors, recoveries.

    A pilot is still an attempt — smoke and screen runs are logged here even
    though they are not full iterations for convergence purposes (6.3).
    """
    raise NotImplementedError("A2")
