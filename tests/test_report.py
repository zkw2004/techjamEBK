"""D7 acceptance: all deliverable tables are generated from node JSON."""

import json

import pytest

from tools.report import build_report, load_nodes, plot_trajectory, render_markdown


def _write_node(directory, node):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{node['id']}.json").write_text(json.dumps(node), encoding="utf-8")


def test_report_aggregates_results_resources_and_iteration_tiers(tmp_path):
    nodes_dir = tmp_path / "nodes"
    _write_node(
        nodes_dir,
        {
            "id": "n002", "fidelity": "full", "status": "ok", "accepted": True,
            "metrics": {"gauc": 0.68, "ndcg": 0.55, "primary": 0.615},
            "tokens": {"in": 120, "out": 30}, "gpu_seconds": 1800,
            "manual_intervention": True,
        },
    )
    _write_node(
        nodes_dir,
        {
            "id": "n001", "fidelity": "screen", "status": "ok", "accepted": False,
            "metrics": {"gauc": 0.60, "ndcg": 0.50, "primary": 0.55},
            "tokens": {"in": 80, "out": 20}, "gpu_seconds": 0,
            "manual_intervention": False,
        },
    )

    report = build_report(load_nodes(nodes_dir))

    assert [row["id"] for row in report["results"]] == ["n001", "n002"]
    assert report["results"][1]["delta_vs_baseline"] == pytest.approx(0.0134)
    assert report["totals"] == {
        "nodes": 2, "tokens_in": 200, "tokens_out": 50, "tokens": 250,
        "gpu_hours": 0.5, "manual_interventions": 1,
    }
    assert report["iterations"] == {
        "pilot": 1, "full": 1, "other": 0,
        "by_fidelity": {"full": 1, "screen": 1},
    }

    rendered = render_markdown(report)
    assert "| n002 | full | ok | yes |" in rendered
    assert "Manual interventions: 1" in rendered
    assert "Pilot iterations: 1" in rendered


def test_report_handles_an_empty_log_directory(tmp_path):
    report = build_report(load_nodes(tmp_path / "missing"))
    assert report["results"] == []
    assert report["totals"]["nodes"] == 0
    assert report["iterations"]["pilot"] == 0


def test_report_rejects_a_malformed_node(tmp_path):
    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    (nodes_dir / "n001.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read node record"):
        load_nodes(nodes_dir)


def test_trajectory_plot_is_generated_from_measured_nodes(tmp_path):
    nodes = [
        {"id": "n001", "metrics": {"primary": 0.59}, "accepted": False},
        {"id": "n002", "metrics": {}, "accepted": False},
        {"id": "n003", "metrics": {"primary": 0.61}, "accepted": True},
    ]
    output = plot_trajectory(nodes, tmp_path / "trajectory.png")
    assert output.is_file()
    assert output.stat().st_size > 1_000
