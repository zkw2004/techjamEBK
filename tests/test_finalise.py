"""D8 acceptance: legal refit, five-seed averaging, and aligned submission."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from agent import store
from tools import finalise as F


def _frames():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "video_id": ["v1", "v2"],
            "long_view": [1, 0],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": ["u3"],
            "video_id": ["v3"],
            "long_view": [1],
        }
    )
    # Deliberately repeat a (user, video) pair: row_id must preserve both rows.
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u4"],
            "video_id": ["v9", "v9", "v8"],
            "long_view": [0, 0, 0],
        }
    )
    return train, validation, test


def test_finalise_refits_on_train_plus_validation_averages_five_seeds_and_checks_alignment(
    tmp_path, monkeypatch
):
    seen_training_sizes: list[int] = []
    checked = []

    def fake_refit(config, training, test, seed):
        assert config["model"] == "random"
        assert len(test) == 3
        seen_training_sizes.append(len(training))
        return np.asarray([seed, seed + 1, seed + 2], dtype=float)

    def fake_check(path):
        checked.append(path)
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows == [
            F.SUBMISSION_COLUMNS,
            ["0", "u1", "v9", "2"],
            ["1", "u1", "v9", "3"],
            ["2", "u4", "v8", "4"],
        ]

    monkeypatch.setattr(F, "_refit_predict", fake_refit)

    result = F.finalise(
        {"model": "random"},
        output_path=tmp_path / "submission.csv",
        frames=_frames(),
        checker=fake_check,
    )

    assert seen_training_sizes == [3] * F.N_SEEDS
    assert result.seeds == (0, 1, 2, 3, 4)
    assert result.row_count == 3
    assert result.mean_score == 3.0
    assert checked == [tmp_path / "submission.csv"]


def test_finalise_requires_exactly_five_distinct_seeds(tmp_path):
    with pytest.raises(F.FinaliseError, match="exactly 5 distinct"):
        F.finalise(
            {"model": "random"},
            output_path=tmp_path / "submission.csv",
            frames=_frames(),
            seeds=(1, 1, 2, 3, 4),
            checker=lambda _path: None,
        )


def test_write_submission_rejects_non_finite_scores(tmp_path):
    with pytest.raises(F.FinaliseError, match="finite"):
        F.write_submission(tmp_path / "submission.csv", _frames()[2], [0.0, np.nan, 1.0])


def test_selected_config_requires_an_accepted_full_node(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "run.jsonl")
    store.set_manifest_provider(lambda: "a" * 64)
    try:
        store.write(
            {
                "parent": "n000",
                "family": "model",
                "hypothesis": "winner",
                "action_type": "config",
                "fidelity": "full",
                "status": "ok",
                "accepted": True,
                "config": {"model": "random"},
                "metrics": {"primary": 0.6},
            }
        )
        assert F.selected_config() == {
            "model": "random",
            "loss": "pointwise",
            "features": ["user_id", "video_id"],
            "negative_sampling": "all",
            "hparams": {},
            "parents": [],
            "blend_method": "rank_avg",
            "seed": 42,
        }
    finally:
        store.set_manifest_provider(None)
