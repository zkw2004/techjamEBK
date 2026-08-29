"""D5 acceptance: one call runs all five probes and prints comparable results."""

from tools.probes import PROBES, render_table, run_probes


def test_probe_configs_cover_the_five_pinned_questions(monkeypatch):
    calls = []

    def runner(config, fidelity, seed):
        calls.append((config, fidelity, seed))
        return {
            "status": "ok", "gauc": 0.6, "ndcg": 0.5,
            "primary": 0.55 + seed / 1_000_000, "seconds": 1.25,
        }

    class FakeTrial:
        number = 0

        @staticmethod
        def suggest_categorical(_name, values):
            return values[0]

        @staticmethod
        def suggest_float(_name, low, _high, log=False):
            return low

        @staticmethod
        def suggest_int(_name, low, _high):
            return low

    class FakeStudy:
        best_trials = [object()]
        best_params = {"k": 8, "lr": 1e-4, "l2": 1e-6, "max_epochs": 1}

        def optimize(self, objective, n_trials):
            for _ in range(n_trials):
                objective(FakeTrial())

    import optuna
    monkeypatch.setattr(optuna, "create_study", lambda **_kwargs: FakeStudy())
    rows = run_probes(runner, fidelity="screen", fm_trials=2, seed=10)

    assert [row["probe"] for row in rows] == list(PROBES)
    assert len(calls) == 7  # P1-P4, two P5 trials, final best P5
    assert all(fidelity == "screen" for _config, fidelity, _seed in calls)
    assert calls[1][0]["model"] == "lgbm" and calls[1][0]["loss"] == "pointwise"
    assert calls[2][0]["model"] == "lgbm" and calls[2][0]["loss"] == "lambdarank"
    assert calls[3][0]["model"] == "deepfm"

    table = render_table(rows)
    assert "Probe" in table and "Primary" in table
    assert all(probe_id in table for probe_id in PROBES)


def test_probe_table_keeps_failures_visible():
    rows = [{
        "probe": "P1", "question": "question", "status": "error",
        "gauc": None, "ndcg": None, "primary": None, "seconds": 0,
        "error_class": "timeout",
    }]
    table = render_table(rows)
    assert "error:timeout" in table
    assert "—" in table
