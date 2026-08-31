"""Deterministic replay of OOM classification, recovery, and continuation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from agent import recovery, store


def run_demo(event_log: Path) -> tuple[list[dict], list[dict]]:
    """Inject an OOM node, apply the real A5 policy, then record a successful retry."""
    original_event_log = store.EVENT_LOG
    store.EVENT_LOG = event_log
    try:
        failed = {
            "id": "n900",
            "parent": "n000",
            "family": "training",
            "fidelity": "full",
            "status": "error",
            "accepted": False,
            "metrics": {},
            "config": {"model": "deepfm", "hparams": {"batch_size": 8192}},
            "errors": [{
                "stage": "fit", "error_class": "oom",
                "traceback": "MemoryError: deterministic demo injection",
            }],
            "manual_intervention": False,
        }
        plan = recovery.recover(failed, attempt=0)
        if plan is None:
            raise RuntimeError("OOM recovery policy unexpectedly abandoned the demo node")
        recovered_batch = plan["config"]["hparams"]["batch_size"]
        succeeded = {
            "id": "n901",
            "parent": "n900",
            "family": "training",
            "fidelity": plan["fidelity"],
            "status": "ok",
            "accepted": True,
            "metrics": {"primary": 0.61},
            "config": plan["config"],
            "errors": [],
            "manual_intervention": False,
            "diff": f"batch_size: 8192 -> {recovered_batch}",
        }
        return [failed, succeeded], store.read_events()
    finally:
        store.EVENT_LOG = original_event_log


def _demo_tree(nodes: list[dict]) -> Tree:
    root = Tree("n000  research root", style="bold magenta")
    failed, succeeded = nodes
    failure = root.add(
        f"{failed['id']}  error:oom  batch_size=8192", style="bold red"
    )
    failure.add(
        f"{succeeded['id']}  ok  batch_size=4096  ACCEPTED", style="bold green"
    )
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-log", type=Path,
        help="optional path for the replay event JSONL; defaults to a temporary file",
    )
    args = parser.parse_args(argv)
    console = Console()

    if args.event_log is not None:
        nodes, events = run_demo(args.event_log)
    else:
        with tempfile.TemporaryDirectory(prefix="techjam-oom-demo-") as directory:
            nodes, events = run_demo(Path(directory) / "run.jsonl")

    event = events[-1]
    console.print(
        Panel(
            "Injected batch_size=8192\n"
            f"Classifier: {event['error_class']}\n"
            f"Policy: {event['decision']} at batch_size={event['batch_size']}\n"
            "Manual intervention: false",
            title="Deterministic failure and recovery",
            border_style="yellow",
        )
    )
    console.print(Panel(_demo_tree(nodes), title="Linked recovery nodes", border_style="magenta"))


if __name__ == "__main__":
    main()
