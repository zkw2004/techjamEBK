"""C4b acceptance: one generated feature traverses syntax -> schema -> leakage
-> smoke -> screen -> full; the safe temporal feature is accepted, the
deliberately leaky twin is quarantined with a clear reason."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline.codegen as codegen
from pipeline.features import FEATURES


def _frame(rows: int, dates: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 10, rows)
    authors = rng.integers(0, 5, rows)
    favoured = ((users + authors) % 3 == 0).astype(float)
    return pd.DataFrame(
        {
            "date": rng.choice(dates, rows),
            "user_id": users,
            "video_id": rng.integers(0, 200, rows),
            "author_id": authors,
            "tab": rng.integers(0, 2, rows),
            "duration_ms": rng.integers(1_000, 60_000, rows),
            "long_view": (rng.random(rows) < 0.10 + 0.4 * favoured).astype(int),
        }
    )


def _data():
    return (
        _frame(1_500, list(range(20220408, 20220422)), seed=1),
        _frame(400, list(range(20220422, 20220429)), seed=2),
        _frame(400, list(range(20220429, 20220509)), seed=3),
    )


def _folds():
    return [
        (
            _frame(700, list(range(20220408, 20220416 + 2 * i)), seed=10 + i),
            _frame(250, [20220416 + 2 * i, 20220417 + 2 * i], seed=20 + i),
        )
        for i in range(3)
    ]


@pytest.fixture
def runner(monkeypatch):
    import pipeline.train as train

    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    yield train
    codegen.unregister_generated("user_author_affinity")
    codegen.unregister_generated("user_author_affinity_leaky")
    codegen.unregister_generated("broken")


# --- code validation ---------------------------------------------------------


def test_syntax_error_is_rejected_with_line_context(runner):
    report = codegen.vet_generated_feature("broken", "def f(:\n    pass", log_events=False)
    assert report["status"] == "rejected"
    assert report["stages"][0]["stage"] == "syntax"
    assert "line" in report["reason"]
    assert "gen_broken" not in FEATURES


def test_wrong_signature_is_rejected_at_schema(runner):
    report = codegen.vet_generated_feature(
        "broken", "def f(only_one):\n    return only_one", log_events=False
    )
    assert report["status"] == "rejected"
    assert report["stages"][-1]["stage"] == "schema"
    assert "(train_df, target_df)" in report["reason"]


def test_load_feature_requires_a_named_or_single_function():
    with pytest.raises(codegen.GeneratedFeatureError):
        codegen.load_feature("def a(x, y):\n    return x\ndef b(x, y):\n    return y")
    fn = codegen.load_feature(
        "def a(x, y):\n    return x\ndef b(x, y):\n    return y", name="b"
    )
    assert fn.__name__ == "b"
    assert fn.__leak_source__.startswith("def a")


def test_generated_code_cannot_shadow_a_human_feature(monkeypatch):
    def human(train_df, target_df):
        return np.zeros(len(target_df))

    monkeypatch.setitem(FEATURES, "gen_taken", human)
    with pytest.raises(codegen.GeneratedFeatureError):
        codegen.register_generated("taken", codegen.load_feature(
            "def f(train_df, target_df):\n    return train_df", name="f"
        ))


# --- the vertical slice ------------------------------------------------------


def test_safe_user_author_affinity_is_accepted_end_to_end(runner):
    report = codegen.vet_generated_feature(
        "user_author_affinity", codegen.USER_AUTHOR_AFFINITY_SOURCE, log_events=False
    )
    assert report["status"] == "accepted", report["reason"]
    assert [s["stage"] for s in report["stages"]] == [
        "syntax", "schema", "leakage", "smoke", "screen", "full",
    ]
    assert all(s["passed"] for s in report["stages"])
    assert report["registered_name"] == "gen_user_author_affinity"
    assert "gen_user_author_affinity" in FEATURES
    assert report["results"]["full"]["status"] == "ok"


def test_leaky_twin_is_quarantined_by_the_dynamic_probe(runner):
    report = codegen.vet_generated_feature(
        "user_author_affinity_leaky", codegen.LEAKY_TWIN_SOURCE, log_events=False
    )
    assert report["status"] == "quarantined"
    failed = [s for s in report["stages"] if not s["passed"]]
    assert failed and failed[0]["stage"] == "leakage"
    assert "post-exposure" in report["reason"] or "outcome" in report["reason"]
    assert "gen_user_author_affinity_leaky" not in FEATURES
    # never reached a training run, let alone a score
    assert report["results"] == {}


def test_accepted_feature_is_usable_from_an_experiment_config(runner):
    codegen.vet_generated_feature(
        "user_author_affinity", codegen.USER_AUTHOR_AFFINITY_SOURCE, log_events=False
    )
    result = runner.run_experiment(
        {
            "model": "random",
            "features": ["user_id", "video_id", "gen_user_author_affinity"],
        },
        fidelity="smoke",
        seed=4,
    )
    assert result["status"] == "ok"


def test_affinity_respects_the_per_row_temporal_cutoff():
    """A target row dated before any history gets the smoothed global rate;
    later rows see strictly earlier interactions only."""
    fn = codegen.load_feature(codegen.USER_AUTHOR_AFFINITY_SOURCE)
    train_df = pd.DataFrame(
        {
            "date": [20220408, 20220409, 20220410],
            "user_id": [1, 1, 2],
            "author_id": [7, 7, 3],
            "long_view": [1, 1, 0],
        }
    )
    target_df = pd.DataFrame(
        {
            "date": [20220408, 20220411],
            "user_id": [1, 1],
            "author_id": [7, 7],
        }
    )
    values = fn(train_df, target_df)
    global_rate = 2.0 / 3.0
    # first row: dated on/before all history -> falls back to the smoothed prior
    assert values[0] == pytest.approx(global_rate)
    # second row: both (user 1, author 7) interactions are strictly earlier
    assert values[1] == pytest.approx((2 + 20 * global_rate) / (2 + 20))


def test_vet_is_deterministic(runner):
    first = codegen.vet_generated_feature(
        "user_author_affinity", codegen.USER_AUTHOR_AFFINITY_SOURCE, log_events=False
    )
    second = codegen.vet_generated_feature(
        "user_author_affinity", codegen.USER_AUTHOR_AFFINITY_SOURCE, log_events=False
    )

    def metrics(report):
        return {
            fidelity: {k: v for k, v in trimmed.items() if k != "seconds"}
            for fidelity, trimmed in report["results"].items()
        }

    assert metrics(first) == metrics(second)
