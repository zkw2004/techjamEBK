"""D5 acceptance: one call runs all five probes and prints comparable results."""

from tools.probes import PROBES, render_table, run_probes


def test_study_lock_failure_keeps_all_probe_results_visible(monkeypatch, tmp_path):
    from pipeline import tune

    def locked_study(*_args, **_kwargs):
        raise ValueError("study already has an active owner")

    monkeypatch.setattr(tune, "run_study", locked_study)
    rows = run_probes(lambda *_args, **_kwargs: {"status": "ok", "primary": 0.5},
                      study_storage=f"sqlite:///{tmp_path}/locked.db")
    assert len(rows) == 5
    assert [row["status"] for row in rows] == ["ok"] * 4 + ["error"]
    assert "active owner" in rows[-1]["traceback"]
    assert "P5" in render_table(rows)
    assert "error:transient" in render_table(rows)


def test_probe_configs_cover_the_five_pinned_questions(monkeypatch, tmp_path):
    calls, fold_calls = [], []

    def runner(config, fidelity, seed):
        calls.append((config, fidelity, seed))
        return {
            "status": "ok", "gauc": 0.6, "ndcg": 0.5,
            "primary": 0.55 + seed / 1_000_000, "seconds": 1.25,
        }

    def fold_runner(config, seed, trial):
        fold_calls.append((config, seed))
        return [0.5, 0.51, 0.52]

    from pipeline import tune
    monkeypatch.setattr(tune, "_trial_fold_primaries", fold_runner)
    rows = run_probes(runner, fidelity="full", fm_trials=2, seed=10,
                      study_storage=f"sqlite:///{tmp_path}/probes.db")

    assert [row["probe"] for row in rows] == list(PROBES)
    assert len(calls) == 5  # Official validation only once per probe, never in tuning.
    assert all(fidelity == "full" for _config, fidelity, _seed in calls)
    assert len(fold_calls) == 2
    assert [seed for _config, seed in fold_calls] == [14, 14]
    assert rows[-1]["tuning"]["best_value"] == 0.51
    assert calls[1][0]["model"] == "lgbm" and calls[1][0]["loss"] == "pointwise"
    assert calls[2][0]["model"] == "lgbm" and calls[2][0]["loss"] == "lambdarank"
    assert calls[3][0]["model"] == "deepfm"

    table = render_table(rows)
    assert "Probe" in table and "Primary" in table
    assert all(probe_id in table for probe_id in PROBES)


def test_failed_tuning_never_evaluates_a_fake_best_configuration(monkeypatch, tmp_path):
    from pipeline import tune

    def broken_fold(*_args):
        raise ValueError("invalid trial")

    calls = []

    def runner(config, **_kwargs):
        calls.append(config)
        return {"status": "ok", "primary": 0.5}

    monkeypatch.setattr(tune, "_trial_fold_primaries", broken_fold)
    rows = run_probes(runner, fm_trials=1,
                      study_storage=f"sqlite:///{tmp_path}/failed.db")
    assert len(calls) == 4
    assert rows[-1]["status"] == "error"
    assert rows[-1]["primary"] is None


def test_probe_table_keeps_failures_visible():
    rows = [{
        "probe": "P1", "question": "question", "status": "error",
        "gauc": None, "ndcg": None, "primary": None, "seconds": 0,
        "error_class": "timeout",
    }]
    table = render_table(rows)
    assert "error:timeout" in table
    assert "—" in table
