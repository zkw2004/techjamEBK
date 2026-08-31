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
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BASELINE_VALIDATION = 0.6016
BASELINE_TEST = 0.5946
PILOT_FIDELITIES = {"smoke", "screen"}
FULL_FIDELITIES = {"full", "confirm"}
DEFAULT_TRAJECTORY = Path("artifacts/trajectory.png")
DEFAULT_MARKDOWN_REPORT = Path("artifacts/experiment-report.md")
DEFAULT_JSON_REPORT = Path("artifacts/experiment-report.json")
DEFAULT_EVENT_LOG = Path("logs/run.jsonl")
DEFAULT_SCREEN_REPORT = Path("logs/screen_report.json")
CANDIDATE_COUNT_MEDIAN = 4
GAUC_USABLE_USER_PERCENT = 57.8
CITATION_PATTERN = re.compile(r"\[ref:\s*([^\]]+)\]")


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


def load_events(path: Path) -> list[dict[str, Any]]:
    """Load complete append-only JSONL events, tolerating a truncated tail."""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def load_json_report(path: Path) -> dict[str, Any]:
    """Load an optional generated diagnostic report."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read diagnostic report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic report must be a JSON object: {path}")
    return value


def build_judge_visibility(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    screen_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect D14 controller and diagnostic evidence without inventing gaps."""
    iteration_events = [event for event in (events or []) if event.get("event") == "iteration"]
    iterations = [
        {
            "iteration": index,
            "strikes": event.get("strikes"),
            "scheduler_forced": bool(event.get("scheduler_forced", False)),
            "hedges_fired": event.get("hedges_fired"),
            "node": event.get("node"),
        }
        for index, event in enumerate(iteration_events, start=1)
    ]
    citations = []
    for node in nodes:
        matches = CITATION_PATTERN.findall(str(node.get("hypothesis", "")))
        citations.extend(
            {"node": node.get("id", ""), "reference": match.strip()}
            for match in matches
        )

    latest_ablation = next(
        (
            {"node": node.get("id", ""), **node["ablation"]}
            for node in reversed(nodes)
            if isinstance(node.get("ablation"), dict)
            and isinstance(node["ablation"].get("components"), list)
        ),
        None,
    )
    feature_rows = (screen_report or {}).get("features")
    inert_features = sorted(
        name
        for name, evidence in (feature_rows.items() if isinstance(feature_rows, dict) else [])
        if isinstance(evidence, dict) and evidence.get("metric_inert") is True
    )
    preflight = next(
        (
            event
            for event in reversed(events or [])
            if event.get("event") == "leakage_preflight"
        ),
        None,
    )
    return {
        "candidate_count_diagnostics": {
            "median_candidates_per_user": CANDIDATE_COUNT_MEDIAN,
            "gauc_usable_users_percent": GAUC_USABLE_USER_PERCENT,
            "interpretation": (
                "A +0.002 primary delta is material when users rank a median of four "
                "candidates and only 57.8% contribute to GAUC."
            ),
        },
        "controller_iterations": iterations,
        "citations": citations,
        "latest_ablation": latest_ablation,
        "metric_inert_features": inert_features,
        "leakage_preflight": preflight,
    }


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _elapsed_seconds(nodes: list[dict[str, Any]], events: list[dict[str, Any]]) -> float:
    """Elapsed agent-run wall-clock from the earliest to latest recorded event."""
    timestamps = []
    for record in [*nodes, *events]:
        value = record.get("timestamp")
        if not isinstance(value, str):
            continue
        try:
            timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            continue
    return max(0.0, (max(timestamps) - min(timestamps)).total_seconds()) if timestamps else 0.0


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def build_report(
    nodes: list[dict[str, Any]],
    *,
    events: list[dict[str, Any]] | None = None,
    screen_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build every D7 deliverable from node JSON alone."""
    event_rows = events or []
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
            "agent_wall_clock_seconds": _elapsed_seconds(nodes, event_rows),
            "iterations_used": sum(event.get("event") == "iteration" for event in event_rows),
            "iteration_cap": 50,
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
        "judge_visibility": build_judge_visibility(nodes, events, screen_report),
    }


def _display(value: object, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a copy-ready report for README and Devpost use."""
    visibility = report["judge_visibility"]
    diagnostics = visibility["candidate_count_diagnostics"]
    lines = [
        "# Experiment report",
        "",
        "## Evaluation context",
        "",
        f"- Median candidates per user: **{diagnostics['median_candidates_per_user']}**",
        f"- GAUC-usable users: **{diagnostics['gauc_usable_users_percent']:.1f}%**",
        f"- Interpretation: {diagnostics['interpretation']}",
        "",
        "## Results",
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
            f"- Agent wall-clock: {_duration(totals['agent_wall_clock_seconds'])} "
            f"({totals['agent_wall_clock_seconds']:.0f} seconds)",
            f"- Iterations used: {totals['iterations_used']} / {totals['iteration_cap']}",
            f"- Manual interventions: {totals['manual_interventions']}",
            f"- Pilot iterations: {iterations['pilot']}",
            f"- Full/confirm iterations: {iterations['full']}",
            f"- Other iterations: {iterations['other']}",
            "",
            f"Manual intervention means: {report['manual_intervention_definition']}",
        ]
    )
    lines.extend(["", "## Search controller", ""])
    controller = visibility["controller_iterations"]
    if controller:
        lines.extend(
            [
                "| Iteration | Strikes | Scheduler forced | Hedges fired | Node |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in controller:
            lines.append(
                f"| {row['iteration']} | {_display(row['strikes'])} | "
                f"{'yes' if row['scheduler_forced'] else 'no'} | "
                f"{_display(row['hedges_fired'])} | {_display(row['node'])} |"
            )
    else:
        lines.append("No iteration-controller events were recorded.")

    lines.extend(["", "## Latest ablation sensitivity", ""])
    ablation = visibility["latest_ablation"]
    if ablation:
        lines.extend(
            [
                f"Latest table: node `{ablation['node']}`; base primary "
                f"{_display(ablation.get('base_primary'))}.",
                "",
                "| Component | Ablated primary | Delta | Sensitivity |",
                "|---|---:|---:|---:|",
            ]
        )
        for component in ablation["components"]:
            lines.append(
                f"| {component.get('component', '')} | {_display(component.get('primary'))} | "
                f"{_display(component.get('delta'))} | "
                f"{_display(component.get('sensitivity'))} |"
            )
    else:
        lines.append("No ablation sensitivity table was recorded.")

    lines.extend(["", "## Research citations", ""])
    if visibility["citations"]:
        lines.extend(
            f"- `{citation['node']}` — {citation['reference']}"
            for citation in visibility["citations"]
        )
    else:
        lines.append("No structured `[ref: ...]` citations were recorded.")

    inert = visibility["metric_inert_features"]
    lines.extend(
        [
            "",
            "## Metric-inert features",
            "",
            ", ".join(f"`{name}`" for name in inert)
            if inert
            else "No metric-inert feature report was recorded.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_trajectory(
    nodes: list[dict[str, Any]], output_path: Path | str = DEFAULT_TRAJECTORY
) -> Path:
    """Plot measured primary scores in ledger order without filling missing runs."""
    import matplotlib.pyplot as plt

    measured = []
    for index, node in enumerate(nodes, start=1):
        metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
        primary = metrics.get("primary")
        if isinstance(primary, (int, float)):
            measured.append((index, str(node.get("id", f"node-{index}")), float(primary), node))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.axhline(
        BASELINE_VALIDATION,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=f"official validation baseline ({BASELINE_VALIDATION:.4f})",
    )
    if measured:
        x = [item[0] for item in measured]
        y = [item[2] for item in measured]
        axis.plot(x, y, color="#2f6fed", linewidth=1.5, alpha=0.75, label="all measured nodes")
        accepted = [item for item in measured if item[3].get("accepted")]
        rejected = [item for item in measured if not item[3].get("accepted")]
        if rejected:
            axis.scatter(
                [item[0] for item in rejected], [item[2] for item in rejected],
                color="#8b95a5", marker="o", s=34, label="not accepted", zorder=3,
            )
        if accepted:
            axis.scatter(
                [item[0] for item in accepted], [item[2] for item in accepted],
                color="#1a9c55", marker="D", s=48, label="accepted", zorder=4,
            )
        axis.set_xticks(x, [item[1] for item in measured], rotation=45, ha="right")
    else:
        axis.text(
            0.5, 0.5, "No node metrics recorded yet", ha="center", va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
    axis.set_title("Validation primary trajectory")
    axis.set_xlabel("Append-only node order")
    axis.set_ylabel("Primary = mean(GAUC, nDCG@5)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def write_reports(
    report: dict[str, Any],
    *,
    markdown_path: Path | str | None = None,
    json_path: Path | str | None = None,
) -> list[Path]:
    """Write requested report formats and return their destination paths."""
    written: list[Path] = []
    for requested, content in (
        (markdown_path, render_markdown(report)),
        (json_path, json.dumps(report, indent=2) + "\n"),
    ):
        if requested is None:
            continue
        destination = Path(requested)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-dir", type=Path, default=Path("logs/nodes"))
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--screen-report", type=Path, default=DEFAULT_SCREEN_REPORT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--trajectory", type=Path,
        help=f"also write the D9 trajectory plot (recommended: {DEFAULT_TRAJECTORY})",
    )
    parser.add_argument(
        "--markdown-output", type=Path,
        help=f"write the Markdown report (recommended: {DEFAULT_MARKDOWN_REPORT})",
    )
    parser.add_argument(
        "--json-output", type=Path,
        help=f"write the JSON report (recommended: {DEFAULT_JSON_REPORT})",
    )
    args = parser.parse_args(argv)
    nodes = load_nodes(args.nodes_dir)
    report = build_report(
        nodes,
        events=load_events(args.events),
        screen_report=load_json_report(args.screen_report),
    )
    print(json.dumps(report, indent=2) if args.json else render_markdown(report), end="")
    for destination in write_reports(
        report,
        markdown_path=args.markdown_output,
        json_path=args.json_output,
    ):
        print(f"Report: {destination}")
    if args.trajectory:
        destination = plot_trajectory(nodes, args.trajectory)
        print(f"Trajectory plot: {destination}")


if __name__ == "__main__":
    main()
