"""D1/D2 acceptance tests for the immutable scoring contract."""

from __future__ import annotations

import shutil

import pytest

from agent import manifest, store


@pytest.fixture
def isolated_contract(tmp_path, monkeypatch):
    """Use small copied contract files so CI never needs the ignored archive."""
    project_root = manifest.EVALUATOR.parents[1]
    evaluator = tmp_path / "evaluate.py"
    submit_checker = tmp_path / "submit.py"
    starter_evaluator = tmp_path / "starter_evaluate.py"
    starter_submit_checker = tmp_path / "starter_submit.py"
    starter_data = tmp_path / "starter_data.py"
    archive = tmp_path / "KuaiRand-Pure.tar.gz"

    for source, destination in (
        (project_root / manifest.EVALUATOR, evaluator),
        (project_root / manifest.SUBMIT_CHECKER, submit_checker),
        (project_root / manifest.STARTER_EVALUATOR, starter_evaluator),
        (project_root / manifest.STARTER_SUBMIT_CHECKER, starter_submit_checker),
        (project_root / manifest.STARTER_DATA, starter_data),
    ):
        shutil.copyfile(source, destination)
    archive.write_bytes(b"fixture archive")

    monkeypatch.setattr(manifest, "EVALUATOR", evaluator)
    monkeypatch.setattr(manifest, "SUBMIT_CHECKER", submit_checker)
    monkeypatch.setattr(manifest, "STARTER_EVALUATOR", starter_evaluator)
    monkeypatch.setattr(manifest, "STARTER_SUBMIT_CHECKER", starter_submit_checker)
    monkeypatch.setattr(manifest, "STARTER_DATA", starter_data)
    monkeypatch.setattr(manifest, "DATA_ARCHIVE", archive)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "logs" / "manifest.json")
    monkeypatch.setattr(store, "NODES_DIR", tmp_path / "logs" / "nodes")
    monkeypatch.setattr(store, "EVENT_LOG", tmp_path / "logs" / "run.jsonl")
    store.set_manifest_provider(None)
    yield evaluator
    store.set_manifest_provider(None)


def test_frozen_constants():
    assert manifest.BASELINE_VALIDATION == 0.6016
    assert manifest.BASELINE_SEED_STD == 0.0008
    assert manifest.CONVERGENCE == {"epsilon": 0.002, "no_improvement_iterations": 3}
    assert manifest.SUBMISSION["columns"] == ["row_id", "user_id", "video_id", "score"]
    assert manifest.SUBMISSION["preserve_repeated_pairs"] is True


def test_frozen_pipeline_copies_match_starter_kit_byte_for_byte():
    hashes = manifest.verify_starter_kit()
    assert hashes["evaluator_sha256"] == manifest.sha256(manifest.STARTER_EVALUATOR)
    assert hashes["submit_checker_sha256"] == manifest.sha256(manifest.STARTER_SUBMIT_CHECKER)


def test_typed_wrapper_reproduces_starter_kit_fixture_numbers():
    users = ["u1", "u1", "u2", "u2"]
    labels = [0, 1, 1, 0]
    scores = [0.1, 0.9, 0.8, 0.2]

    from pipeline.evaluate import evaluate

    expected = evaluate(users, labels, scores)
    actual = manifest.evaluate_scores(users, labels, scores)

    assert actual.gauc == expected["GAUC"]
    assert actual.ndcg == expected["nDCG@5"]
    assert actual.primary == expected["primary"]
    assert (actual.users, actual.rows) == (expected["users"], expected["rows"])


def test_typed_wrapper_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        manifest.evaluate_scores(["u"], [1], [0.1, 0.2])


def test_preflight_computes_all_three_hashes(isolated_contract):
    recorded = manifest.preflight()
    profile = recorded["metric_profile"]

    assert profile["evaluator_sha256"] == manifest.sha256(manifest.EVALUATOR)
    assert profile["submit_checker_sha256"] == manifest.sha256(manifest.SUBMIT_CHECKER)
    assert profile["data_sha256"] == manifest.sha256(manifest.DATA_ARCHIVE)
    assert manifest.MANIFEST_PATH.is_file()


def test_modified_evaluator_fails_preflight_closed(isolated_contract):
    manifest.preflight()
    isolated_contract.write_text(isolated_contract.read_text(encoding="utf-8") + "\n# changed\n")

    with pytest.raises(manifest.ManifestError, match="differs"):
        manifest.preflight()


def test_metric_profile_is_read_from_the_shipped_evaluator(isolated_contract):
    profile = manifest.build_manifest()["metric_profile"]

    assert profile["target_label"] == "long_view"
    assert profile["group_key"] == "user_id"
    assert profile["metrics"] == ["GAUC", "nDCG@5"]
    assert profile["cutoffs"] == {"nDCG": 5}
    assert profile["aggregation"] == "mean(GAUC, nDCG@5)"


def test_preflight_stamps_every_subsequent_node(isolated_contract):
    manifest.preflight()
    path = store.write(
        {
            "parent": "n000",
            "family": "baseline",
            "hypothesis": "contract fixture",
            "action_type": "config",
            "fidelity": "smoke",
            "status": "success",
        }
    )

    assert store.read(path.stem)["manifest_sha256"] == manifest.manifest_sha256()
