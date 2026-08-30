"""Deliverable tables, generated from logs/nodes/*.json. Task D7.

Zero hand-assembly. Emits:
  - results table (validation-best component metrics)
  - absolute delta vs the official baseline (0.6016 val / 0.5946 test)
  - total input + output tokens across all LLM calls
  - total GPU-hours
  - manual intervention count, with the explicit definition used
  - pilot (smoke/screen) vs full iteration breakdown
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

BASELINE_VALIDATION = 0.6016
BASELINE_TEST = 0.5946
PILOT_FIDELITIES = {"smoke", "screen"}
FULL_FIDELITIES = {"full", "confirm"}


def load_nodes(nodes_dir: Path) -> list[dict[str, Any]]:
    """Load completed node records in stable node-id order."""
    if not nodes_dir.is_dir():
        return []
    nodes: list[dict[str, Any]] = []
    for path in sorted(nodes_dir.glob("*.json")):
        try:
            node = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read node record {path}: {exc}") from exc
        if not isinstance(node, dict):
            raise ValueError(f"node record must be a JSON object: {path}")
        nodes.append(node)
    return sorted(nodes, key=lambda node: str(node.get("id", "")))


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_report(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build every D7 deliverable from node JSON alone."""
    results = []
    for node in nodes:
        metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
        primary = metrics.get("primary")
        baseline = BASELINE_TEST if node.get("split") == "test" else BASELINE_VALIDATION
        results.append(
            {
                "id": node.get("id", ""),
                "fidelity": node.get("fidelity", ""),
                "status": node.get("status", ""),
                "accepted": bool(node.get("accepted", False)),
                "gauc": metrics.get("gauc"),
                "ndcg": metrics.get("ndcg"),
                "primary": primary,
                "delta_vs_baseline": (
                    float(primary) - baseline if isinstance(primary, (int, float)) else None
                ),
            }
        )

    token_in = token_out = 0
    for node in nodes:
        tokens = node.get("tokens") if isinstance(node.get("tokens"), dict) else {}
        token_in += int(_number(tokens.get("in")))
        token_out += int(_number(tokens.get("out")))

    fidelity_counts = Counter(str(node.get("fidelity", "unknown")) for node in nodes)
    pilot = sum(fidelity_counts[name] for name in PILOT_FIDELITIES)
    full = sum(fidelity_counts[name] for name in FULL_FIDELITIES)
    return {
        "results": results,
        "totals": {
            "nodes": len(nodes),
            "tokens_in": token_in,
            "tokens_out": token_out,
            "tokens": token_in + token_out,
            "gpu_hours": sum(_number(node.get("gpu_seconds")) for node in nodes) / 3600.0,
            "manual_interventions": sum(
                bool(node.get("manual_intervention", False)) for node in nodes
            ),
        },
        "iterations": {
            "pilot": pilot,
            "full": full,
            "other": len(nodes) - pilot - full,
            "by_fidelity": dict(sorted(fidelity_counts.items())),
        },
        "manual_intervention_definition": (
            "A completed node whose manual_intervention field is true; automated "
            "proposals, bounded retries, and recovery actions are not interventions."
        ),
    }


def _display(value: object, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a copy-ready report for README and Devpost use."""
    lines = [
        "# Experiment report",
        "",
        "| Node | Fidelity | Status | Accepted | GAUC | nDCG | Primary | Δ baseline |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            "| {id} | {fidelity} | {status} | {accepted} | {gauc} | {ndcg} | "
            "{primary} | {delta} |".format(
                id=row["id"],
                fidelity=row["fidelity"],
                status=row["status"],
                accepted="yes" if row["accepted"] else "no",
                gauc=_display(row["gauc"]),
                ndcg=_display(row["ndcg"]),
                primary=_display(row["primary"]),
                delta=_display(row["delta_vs_baseline"]),
            )
        )

    totals, iterations = report["totals"], report["iterations"]
    lines.extend(
        [
            "",
            "## Run totals",
            "",
            f"- Nodes: {totals['nodes']}",
            f"- Tokens: {totals['tokens']} ({totals['tokens_in']} in, {totals['tokens_out']} out)",
            f"- GPU-hours: {totals['gpu_hours']:.6f}",
            f"- Manual interventions: {totals['manual_interventions']}",
            f"- Pilot iterations: {iterations['pilot']}",
            f"- Full/confirm iterations: {iterations['full']}",
            f"- Other iterations: {iterations['other']}",
            "",
            f"Manual intervention means: {report['manual_intervention_definition']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-dir", type=Path, default=Path("logs/nodes"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = build_report(load_nodes(args.nodes_dir))
    print(json.dumps(report, indent=2) if args.json else render_markdown(report), end="")


if __name__ == "__main__":
    main()
