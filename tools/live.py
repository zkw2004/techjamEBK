"""Live Rich view of the append-only experiment node tree."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from tools.report import load_nodes


def _node_label(node: dict[str, Any]) -> Text:
    status = str(node.get("status", "unknown"))
    accepted = bool(node.get("accepted", False))
    style = "bold green" if accepted else ("red" if status == "error" else "cyan")
    metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
    primary = metrics.get("primary")
    score = "—" if not isinstance(primary, (int, float)) else f"{float(primary):.6f}"
    label = (
        f"{node.get('id', '?')}  {node.get('family', '?')}/{node.get('fidelity', '?')}  "
        f"{status}  primary={score}"
    )
    if accepted:
        label += "  ACCEPTED"
    if node.get("repair_attempted"):
        label += "  repaired"
    return Text(label, style=style)


def build_tree(nodes: list[dict[str, Any]]) -> Tree:
    """Build hierarchy from parent ids while retaining orphaned evidence."""
    root = Tree(Text("n000  research root", style="bold magenta"))
    rich_nodes: dict[str, Tree] = {"n000": root}
    waiting = list(nodes)
    while waiting:
        progressed = False
        for node in waiting[:]:
            parent_id = str(node.get("parent", "n000"))
            if parent_id not in rich_nodes:
                continue
            node_id = str(node.get("id", "?"))
            rich_nodes[node_id] = rich_nodes[parent_id].add(_node_label(node))
            waiting.remove(node)
            progressed = True
        if not progressed:
            orphan_root = root.add(Text("unresolved parents", style="bold yellow"))
            for node in waiting:
                node_id = str(node.get("id", "?"))
                rich_nodes[node_id] = orphan_root.add(_node_label(node))
            break
    return root


def summary_table(nodes: list[dict[str, Any]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    measured = [
        node for node in nodes
        if isinstance(node.get("metrics"), dict)
        and isinstance(node["metrics"].get("primary"), (int, float))
    ]
    accepted = [node for node in nodes if node.get("accepted")]
    failures = [node for node in nodes if node.get("status") == "error"]
    best = max(measured, key=lambda node: node["metrics"]["primary"], default=None)
    table.add_row("Nodes", str(len(nodes)))
    table.add_row("Accepted", str(len(accepted)))
    table.add_row("Failures", str(len(failures)))
    table.add_row(
        "Best measured",
        "—" if best is None else f"{best.get('id')} ({best['metrics']['primary']:.6f})",
    )
    table.add_row(
        "Manual interventions",
        str(sum(bool(node.get("manual_intervention", False)) for node in nodes)),
    )
    return table


def renderable(nodes: list[dict[str, Any]]):
    return Group(
        Panel(summary_table(nodes), title="Run summary", border_style="blue"),
        Panel(build_tree(nodes), title="Experiment tree", border_style="magenta"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-dir", type=Path, default=Path("logs/nodes"))
    parser.add_argument("--watch", action="store_true", help="refresh until interrupted")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")

    if not args.watch:
        Console().print(renderable(load_nodes(args.nodes_dir)))
        return
    with Live(
        renderable(load_nodes(args.nodes_dir)), refresh_per_second=4, screen=False
    ) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(renderable(load_nodes(args.nodes_dir)), refresh=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
