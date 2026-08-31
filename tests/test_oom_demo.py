"""D10 replay must use the real bounded OOM recovery policy."""

from tools.oom_demo import run_demo


def test_oom_demo_reduces_batch_size_logs_recovery_and_continues(tmp_path):
    nodes, events = run_demo(tmp_path / "run.jsonl")
    failed, succeeded = nodes

    assert failed["errors"][0]["error_class"] == "oom"
    assert failed["config"]["hparams"]["batch_size"] == 8192
    assert succeeded["config"]["hparams"]["batch_size"] == 4096
    assert succeeded["status"] == "ok"
    assert succeeded["parent"] == failed["id"]
    assert succeeded["manual_intervention"] is False
    assert events[-1]["event"] == "recovery"
    assert events[-1]["decision"] == "retry"
    assert events[-1]["batch_size"] == 4096
