"""Node tree persistence. One JSON per node in logs/nodes/.

Contract: AGENT_PLAN.md Section 8.7 (FROZEN). Owner: Workstream A. Task A2.

`logs/nodes/<id>.json` IS the run-log deliverable. Build the schema on Day 1
or the data will not exist on Day 3 (trap 12).

Three properties this module has to hold, because the ledger is graded:

* **Append-only.** Nodes are never rewritten and never deleted. The agent
  cannot erase a failed experiment from its own record (Section 7.2).
* **Crash-safe.** Nodes are written atomically, so a kill mid-write leaves
  either the whole node or nothing — never a truncated JSON that poisons
  `list_nodes()` on resume.
* **Contract-stamped.** Every node carries the run manifest hash, so a
  mid-run evaluator change cannot silently invalidate earlier results
  (Section 8.3).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Resolved at call time, not import time, so tests can redirect them.
NODES_DIR = Path("logs/nodes")
EVENT_LOG = Path("logs/run.jsonl")

ID_PATTERN = re.compile(r"^n(\d{3,})$")
CITATION_PATTERN = re.compile(r"\[ref:\s*[^\]]+\]")

# Section 8.7. A node missing any of these is not a usable log entry: the
# report generator (D7) reads these keys directly and does no hand-assembly.
REQUIRED_FIELDS = (
    "id", "parent", "family", "hypothesis", "action_type", "fidelity",
    "status", "manifest_sha256",
)

# Filled in on write when the caller omits them, so every file on disk has
# the same shape and `logs/nodes/*.json` stays uniformly parseable.
DEFAULTS: dict[str, Any] = {
    "reasoning": "",
    "config": {},
    "diff": "",
    "metrics": {},
    "fold_primaries": [],
    "segments": {},
    "delta_vs_best": None,
    "ci_95": None,
    "gates": {},
    "accepted": False,
    "errors": [],
    "repair_attempted": False,
    "tokens": {},
    "seconds": 0.0,
    "gpu_seconds": 0.0,
    "manual_intervention": False,
}

# Key order on disk, so nodes diff cleanly in review and in the deliverable.
FIELD_ORDER = (
    "id", "parent", "family", "hypothesis", "reasoning", "action_type",
    "fidelity", "config", "diff", "manifest_sha256", "metrics",
    "fold_primaries", "segments", "delta_vs_best", "ci_95", "gates",
    "accepted", "status", "errors", "repair_attempted", "tokens",
    "seconds", "gpu_seconds", "manual_intervention", "timestamp",
)

# D2 owns build_manifest(). Wire it in through this hook rather than importing
# agent.manifest here — it keeps the dependency one-way and lets D2 land later
# without reworking the store.
_manifest_provider: Callable[[], str] | None = None


def set_manifest_provider(fn: Callable[[], str] | None) -> None:
    """Register the callable that returns the current run's manifest SHA-256."""
    global _manifest_provider
    _manifest_provider = fn


class NodeValidationError(ValueError):
    """A node was rejected before it reached disk."""


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars and arrays to plain Python.

    run_experiment returns numpy floats; json.dump refuses them, and a node
    that fails to serialise mid-run is a lost iteration.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat().replace("+00:00", "Z")
    return obj


def _node_path(node_id: str) -> Path:
    if not ID_PATTERN.match(node_id):
        raise NodeValidationError(
            f"bad node id {node_id!r}; expected the Section 8.7 form 'n017'"
        )
    return NODES_DIR / f"{node_id}.json"


def next_id() -> str:
    """Lowest unused id, zero-padded to three digits (Section 8.7: 'n017')."""
    highest = 0
    if NODES_DIR.exists():
        for path in NODES_DIR.glob("n*.json"):
            match = ID_PATTERN.match(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"n{highest + 1:03d}"


def normalise(node: dict) -> dict:
    """Fill defaults, stamp the manifest hash and timestamp, coerce numpy.

    Pure: returns a new dict and does not touch disk. `read(write(node))`
    equals `normalise(node)`, which is what makes the round-trip testable.
    """
    out = {**DEFAULTS, **node}

    if not out.get("id"):
        out["id"] = next_id()
    if not out.get("timestamp"):
        out["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    if not out.get("manifest_sha256") and _manifest_provider is not None:
        out["manifest_sha256"] = _manifest_provider()

    missing = [f for f in REQUIRED_FIELDS if out.get(f) in (None, "")]
    if missing:
        hint = ""
        if "manifest_sha256" in missing:
            hint = (
                " — call set_manifest_provider() with D2's build_manifest, or"
                " pass manifest_sha256 explicitly; every node must carry it"
                " (Section 8.3)"
            )
        raise NodeValidationError(f"node missing required field(s): {missing}{hint}")

    ordered = {k: out[k] for k in FIELD_ORDER if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return _jsonable(ordered)


def write(node: dict) -> Path:
    """Persist a completed node. Must round-trip identically with read().

    Atomic: the node is written to a temp file in the same directory and
    renamed into place, so a crash cannot leave a half-written record.
    Refuses to overwrite an existing node — the ledger is append-only.
    """
    record = normalise(node)
    path = _node_path(record["id"])
    NODES_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise NodeValidationError(
            f"node {record['id']} already exists; the ledger is append-only "
            "(Section 7.2 — a failed experiment cannot be edited away)"
        )

    fd, tmp = tempfile.mkstemp(dir=NODES_DIR, prefix=f".{record['id']}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read(node_id: str) -> dict:
    """Load one node by id."""
    return json.loads(_node_path(node_id).read_text())


def list_nodes() -> list[dict]:
    """All nodes on disk, ordered by id. The tree survives a process restart.

    Skips the temp files an interrupted write may have left behind; they are
    never named `nNNN.json`, so a partial write is invisible here.
    """
    if not NODES_DIR.exists():
        return []
    nodes = []
    for path in sorted(NODES_DIR.glob("n*.json")):
        if ID_PATTERN.match(path.stem):
            nodes.append(json.loads(path.read_text()))
    return sorted(nodes, key=lambda n: int(ID_PATTERN.match(n["id"]).group(1)))


def best_node() -> dict | None:
    """Highest accepted primary. None before the first acceptance.

    Accepted, not merely highest-scoring: a node that beat the incumbent on
    the raw number but failed the statistical or segment gate (Section 6.6)
    is not the best node, and must never become the branch parent.

    Ties break on the lower id, so the choice is deterministic across runs.
    """
    accepted = [
        n for n in list_nodes()
        if n.get("accepted") and n.get("status") == "ok"
        and isinstance(n.get("metrics", {}).get("primary"), (int, float))
    ]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda n: (n["metrics"]["primary"], -int(ID_PATTERN.match(n["id"]).group(1))),
    )


def append_event(event: dict) -> None:
    """Append-only event stream: proposals, pilots, errors, recoveries.

    A pilot is still an attempt — smoke and screen runs are logged here even
    though they are not full iterations for convergence purposes (6.3).
    """
    record = _jsonable(dict(event))
    record.setdefault(
        "timestamp",
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENT_LOG, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_events() -> list[dict]:
    """The event stream in order. Tolerates a truncated final line, which is
    what an interrupted run leaves behind."""
    if not EVENT_LOG.exists():
        return []
    events = []
    for line in EVENT_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def render_hypothesis(node: dict) -> str:
    """Judge-visible hypothesis text, including any embedded A12 citation."""
    return str(node.get("hypothesis", ""))


def render_run_log(nodes: list[dict] | None = None) -> str:
    """Render a compact node log without inventing fields outside the ledger."""
    rows = list_nodes() if nodes is None else nodes
    lines = []
    for node in rows:
        citation = " cited" if CITATION_PATTERN.search(render_hypothesis(node)) else " uncited"
        lines.append(
            "{id} {status} {fidelity}{citation}: {hypothesis}".format(
                id=node.get("id", ""),
                status=node.get("status", ""),
                fidelity=node.get("fidelity", ""),
                citation=citation,
                hypothesis=render_hypothesis(node),
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")
