"""Acceptance tests for Pinxin's live Rich experiment-tree view."""

from rich.console import Console

from tools.live import renderable


def test_live_view_shows_tree_state_metrics_and_run_summary():
    nodes = [
        {
            "id": "n001", "parent": "n000", "family": "model", "fidelity": "screen",
            "status": "ok", "accepted": False, "metrics": {"primary": 0.59},
            "manual_intervention": False,
        },
        {
            "id": "n002", "parent": "n001", "family": "feature", "fidelity": "full",
            "status": "ok", "accepted": True, "metrics": {"primary": 0.61},
            "repair_attempted": True, "manual_intervention": False,
        },
        {
            "id": "n003", "parent": "n001", "family": "training", "fidelity": "smoke",
            "status": "error", "accepted": False, "metrics": {},
            "manual_intervention": False,
        },
    ]
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable(nodes))
    text = console.export_text()

    assert "n000  research root" in text
    assert "n001" in text and "primary=0.590000" in text
    assert "n002" in text and "ACCEPTED" in text and "repaired" in text
    assert "n003" in text and "error" in text
    assert "Accepted" in text and "1" in text
    assert "Manual interventions" in text and "0" in text


def test_live_view_keeps_nodes_with_missing_parents_visible():
    console = Console(record=True, width=100, color_system=None)
    console.print(renderable([{"id": "n099", "parent": "missing", "status": "error"}]))
    text = console.export_text()
    assert "unresolved parents" in text
    assert "n099" in text
