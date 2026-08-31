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


def test_accept_fn_is_called_on_success_and_sets_accepted(monkeypatch):
    """Section 8.7: accepted is decided on the raw C1 result (which still
    carries val_scores/val_user_ids) before conversion to the node shape
    drops them — this is the only point in the system with access to both."""
    seen = []

    def fake_accept(result: dict) -> bool:
        seen.append(result)
        return result["primary"] > 0.5

    def fake_runner(config, fidelity, seed, timeout_s):
        result = _success(fidelity)
        result["val_scores"] = [0.1, 0.2]
        result["val_user_ids"] = [0, 1]
        return result

    monkeypatch.setattr(E, "_run_experiment", fake_runner)
    node = E.execute(_action(), fidelity="full", timeout_s=12, accept_fn=fake_accept)

    assert node["accepted"] is True
    assert len(seen) == 1
    assert "val_scores" in seen[0]  # accept_fn saw the raw result, not the node


def test_accept_fn_can_reject(monkeypatch):
    def fake_runner(config, fidelity, seed, timeout_s):
        return _success(fidelity)

    monkeypatch.setattr(E, "_run_experiment", fake_runner)
    node = E.execute(_action(), fidelity="full", timeout_s=12, accept_fn=lambda _r: False)

    assert node["accepted"] is False


def test_accept_fn_is_never_called_on_a_failed_run(monkeypatch):
    calls = []

    def failing_runner(config, fidelity, seed, timeout_s):
        return {"status": "error", "stage": fidelity, "error_class": "transient",
                "traceback": "boom", "seconds": 0.1}

    monkeypatch.setattr(E, "_run_experiment", failing_runner)
    node = E.execute(
        _action(), fidelity="full", timeout_s=12,
        accept_fn=lambda r: calls.append(r) or True,
    )

    assert node["accepted"] is False
    assert calls == []


def test_accepted_defaults_to_false_without_an_accept_fn(monkeypatch):
    """smoke/screen never pass accept_fn (agent/loop.py) — must stay False,
    not silently accepted, when nothing decided otherwise."""
    monkeypatch.setattr(E, "_run_experiment", lambda *a: _success("smoke"))
    node = E.execute(_action(), fidelity="smoke", timeout_s=12)

    assert node["accepted"] is False


# --- type="tune" dispatch to the Optuna harness ------------------------------

def _tune_action(search_space=None, budget=4):
    return Action(
        hypothesis="the incumbent is undertuned rather than at its ceiling",
        reasoning="fixture",
        type="tune",
        family="training",
        parent="n004",
        config={"model": "fm", "features": ["user_id", "video_id"],
                "hparams": {"lr": 0.005, "k": 8}},
        search_space={"lr": ["loguniform", 1e-4, 1e-2]} if search_space is None else search_space,
        budget=budget,
    )


def _fake_study(best_params, calls=None):
    def run_study(base_config, search_space, budget, seed, storage, study_name):
        if calls is not None:
            calls.append({"base_config": base_config, "search_space": search_space,
                          "budget": budget, "seed": seed, "study_name": study_name})
        return {"best_params": best_params, "best_value": 0.61, "n_pruned": 2,
                "seconds_saved": 9.0, "measured_trial_seconds": 30.0,
                "pruning_savings_fraction": 0.23, "trials": [{}, {}, {}, {}]}
    return run_study


def test_tune_runs_a_study_and_evaluates_the_tuned_config(monkeypatch):
    """Regression: type='tune' used to fall through to the ordinary
    run_experiment call with search_space ignored entirely — a tuning node
    that had silently trained its parent's untouched config and reported that
    as the tuning result."""
    import pipeline.tune as tune_module

    calls, ran = [], []
    monkeypatch.setattr(tune_module, "run_study", _fake_study({"lr": 0.009}, calls))

    def fake_run_experiment(config, fidelity, seed, timeout_s):
        ran.append(config)
        return _success(fidelity)

    monkeypatch.setattr(E, "_run_experiment", fake_run_experiment)

    node = E.execute(_tune_action(), fidelity="screen", timeout_s=60)

    assert node["status"] == "ok"
    assert len(calls) == 1, "the study must actually run"
    # The winning params reach the evaluated config, merged over the base.
    assert ran[0]["hparams"]["lr"] == 0.009
    assert ran[0]["hparams"]["k"] == 8, "untouched base hparams must survive"


def test_the_node_records_the_tuned_config_not_the_action_it_arrived_with(monkeypatch):
    """A node showing the untuned hparams would describe an experiment that
    never ran, which is the whole failure this dispatch exists to prevent."""
    import pipeline.tune as tune_module

    monkeypatch.setattr(tune_module, "run_study", _fake_study({"lr": 0.009}))
    monkeypatch.setattr(E, "_run_experiment", lambda c, f, *_a, **_k: _success(f))

    node = E.execute(_tune_action(), fidelity="screen", timeout_s=60)

    assert node["config"]["hparams"]["lr"] == 0.009
    assert node["tuning"]["best_params"] == {"lr": 0.009}
    assert node["tuning"]["n_pruned"] == 2


def test_smoke_validates_the_search_space_without_paying_for_a_study(monkeypatch):
    """A 20-trial study is minutes of folds; smoke's contract is a seconds-long
    correctness check, and the loop runs every candidate through smoke first."""
    import pipeline.tune as tune_module

    calls = []
    monkeypatch.setattr(tune_module, "run_study", _fake_study({"lr": 0.009}, calls))
    monkeypatch.setattr(E, "_run_experiment", lambda c, f, *_a, **_k: _success(f))

    node = E.execute(_tune_action(), fidelity="smoke", timeout_s=60)

    assert node["status"] == "ok"
    assert calls == [], "smoke must not start a study"
    assert node.get("tuning") is None


@pytest.mark.parametrize(
    "search_space",
    [
        {},
        {"lr": ["unsupported-kind", 1e-4, 1e-2]},
        {"lr": []},
        {"k": ["categorical", []]},
    ],
)
def test_a_malformed_search_space_fails_cheaply_at_smoke(search_space, monkeypatch):
    monkeypatch.setattr(E, "_run_experiment", lambda c, f, *_a, **_k: _success(f))

    node = E.execute(_tune_action(search_space=search_space), fidelity="smoke", timeout_s=60)

    assert node["status"] == "error"
    assert node["errors"][0]["error_class"] == "schema"


def test_the_study_name_is_stable_across_tiers_but_unique_per_action(monkeypatch):
    """The loop runs one candidate through smoke -> screen -> full, so a tune
    action reaches execute() three times. Optuna studies are resumable and
    budget-capped, so a stable name makes later tiers reopen the finished
    study instead of paying for the search again."""
    import pipeline.tune as tune_module

    calls = []
    monkeypatch.setattr(tune_module, "run_study", _fake_study({"lr": 0.009}, calls))
    monkeypatch.setattr(E, "_run_experiment", lambda c, f, *_a, **_k: _success(f))

    E.execute(_tune_action(), fidelity="screen", timeout_s=60)
    E.execute(_tune_action(), fidelity="full", timeout_s=60)
    E.execute(_tune_action(budget=99), fidelity="screen", timeout_s=60)

    assert calls[0]["study_name"] == calls[1]["study_name"], "same action, same study"
    assert calls[2]["study_name"] != calls[0]["study_name"], "different budget, different study"


def test_a_config_action_is_untouched_by_the_tune_path(monkeypatch):
    """Regression guard: only type='tune' may take the study path."""
    import pipeline.tune as tune_module

    calls = []
    monkeypatch.setattr(tune_module, "run_study", _fake_study({"lr": 0.009}, calls))
    monkeypatch.setattr(E, "_run_experiment", lambda c, f, *_a, **_k: _success(f))

    node = E.execute(_action(), fidelity="screen", timeout_s=60)

    assert calls == []
    assert node.get("tuning") is None
