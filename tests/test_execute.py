"""A4 acceptance: execute() turns Actions into resilient node records."""

from __future__ import annotations

import time

import pytest

from agent import execute as E
from agent.schema import Action


def _action(**overrides) -> Action:
    values = {
        "hypothesis": "A deterministic reference model validates the runner path",
        "reasoning": "It establishes a reliable smoke-test branch.",
        "type": "config",
        "family": "model",
        "parent": "n001",
        "config": {"model": "random", "seed": 7},
    }
    values.update(overrides)
    return Action(**values)


def _success(fidelity: str) -> dict:
    metrics = (None, None, None) if fidelity == "smoke" else (0.61, 0.58, 0.595)
    return {
        "status": "ok",
        "fidelity": fidelity,
        "gauc": metrics[0],
        "ndcg": metrics[1],
        "primary": metrics[2],
        "fold_primaries": [] if fidelity == "smoke" else [0.59, 0.60, 0.61],
        "segments": {},
        "seconds": 0.25,
        "gpu_seconds": 0.0,
    }


def test_config_action_delegates_to_c1_and_returns_a_node(monkeypatch):
    calls = []

    def fake_runner(config, fidelity, seed, timeout_s):
        calls.append((config, fidelity, seed, timeout_s))
        return _success(fidelity)

    monkeypatch.setattr(E, "_run_experiment", fake_runner)
    node = E.execute(_action(), fidelity="full", timeout_s=12)

    assert calls == [
        (
            {"model": "random", "loss": "pointwise", "features": ["user_id", "video_id"],
             "negative_sampling": "all", "hparams": {}, "parents": [],
             "blend_method": "rank_avg", "seed": 7},
            "full",
            7,
            12,
        )
    ]
    assert node["status"] == "ok"
    assert node["metrics"] == {"gauc": 0.61, "ndcg": 0.58, "primary": 0.595}
    assert node["family"] == "model"
    assert node["manual_intervention"] is False


def test_runner_failure_becomes_a_logged_node(monkeypatch):
    monkeypatch.setattr(
        E,
        "_run_experiment",
        lambda *_: {
            "status": "error",
            "stage": "smoke",
            "error_class": "oom",
            "traceback": "MemoryError: exhausted",
            "seconds": 0.1,
        },
    )

    node = E.execute(_action())

    assert node["status"] == "error"
    assert node["errors"] == [
        {"stage": "smoke", "error_class": "oom", "traceback": "MemoryError: exhausted"}
    ]


def test_generated_code_exception_cannot_crash_the_parent():
    action = _action(type="code", code="raise RuntimeError('controlled failure')")

    node = E.execute(action, timeout_s=1)

    assert node["status"] == "error"
    assert node["errors"][0]["stage"] == "code"
    assert node["errors"][0]["error_class"] == "transient"
    assert "controlled failure" in node["errors"][0]["traceback"]


def test_generated_code_syntax_error_is_classified():
    action = _action(type="code", code="def broken(:\n    pass")

    node = E.execute(action, timeout_s=1)

    assert node["status"] == "error"
    assert node["errors"][0]["error_class"] == "syntax"


@pytest.mark.skipif(
    "fork" not in E.mp.get_all_start_methods(),
    reason="the injected C1 runner is inherited by fork-based workers only",
)
def test_generated_code_runs_before_delegating_to_c1(monkeypatch):
    monkeypatch.setattr(E, "_run_experiment", lambda *_: _success("smoke"))
    action = _action(type="code", code="answer = 42")

    node = E.execute(action, timeout_s=1)

    assert node["status"] == "ok"
    assert node["metrics"] == {}


def test_generated_code_hang_is_killed_at_timeout():
    action = _action(type="code", code="import time\ntime.sleep(2)")
    started = time.monotonic()

    node = E.execute(action, timeout_s=0.05)

    assert time.monotonic() - started < 1
    assert node["status"] == "error"
    assert node["errors"][0]["error_class"] == "timeout"


def test_action_without_executable_config_is_a_schema_error():
    action = _action(type="tune", config=None, search_space={"lr": ["loguniform", 1e-4, 1e-2]})

    node = E.execute(action)

    assert node["status"] == "error"
    assert node["errors"][0]["error_class"] == "schema"


# --- C4b wiring: type="code", family="feature" routes through codegen --------


def _feature_fixture_frames():
    import numpy as np
    import pandas as pd

    def frame(rows, dates, offset=0):
        index = np.arange(rows) + offset
        return pd.DataFrame(
            {
                "date": np.asarray(dates)[index % len(dates)],
                "user_id": index % 6,
                "video_id": index % 40,
                "author_id": index % 5,
                "tab": index % 2,
                "duration_ms": 1_000 + (index % 4) * 500,
                "long_view": (index % 5 == 0).astype(int),
            }
        )

    return (
        frame(1_100, list(range(20220408, 20220422))),
        frame(200, [20220422, 20220423], offset=4_000),
        frame(200, [20220429, 20220430], offset=8_000),
    )


@pytest.mark.skipif(
    "fork" not in E.mp.get_all_start_methods(),
    reason="fixture data is inherited by fork-based workers only",
)
def test_feature_code_action_registers_and_uses_the_generated_feature(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _feature_fixture_frames, raising=False)
    action = _action(
        type="code",
        family="feature",
        code=(
            "def constant_feature(train_df, target_df):\n"
            "    import numpy as np\n"
            "    return np.zeros(len(target_df))\n"
        ),
    )

    node = E.execute(action, fidelity="smoke", timeout_s=60)

    assert node["status"] == "ok", node.get("errors")
    # registration is contained in the worker: nothing leaks into this process
    from pipeline.features import FEATURES

    assert "gen_constant_feature" not in FEATURES


@pytest.mark.skipif(
    "fork" not in E.mp.get_all_start_methods(),
    reason="fixture data is inherited by fork-based workers only",
)
def test_leaky_feature_code_action_is_classified_leak_suspected(monkeypatch):
    """B6 rejects the read before training; A5 will quarantine, never repair."""
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _feature_fixture_frames, raising=False)
    action = _action(
        type="code",
        family="feature",
        code=(
            "def oracle(train_df, target_df):\n"
            "    return target_df[\"long_view\"].to_numpy(dtype=float)\n"
        ),
    )

    node = E.execute(action, fidelity="smoke", timeout_s=60)

    assert node["status"] == "error"
    assert node["errors"][0]["error_class"] == "leak_suspected"
