"""D2 acceptance: hashes computed at preflight; a modified evaluator fails
preflight closed; every node carries the manifest hash."""

from __future__ import annotations

from agent.manifest import BASELINE_SEED_STD, BASELINE_VALIDATION, CONVERGENCE, SUBMISSION
from tests.conftest import todo


def test_frozen_constants():
    assert BASELINE_VALIDATION == 0.6016
    assert BASELINE_SEED_STD == 0.0008
    assert CONVERGENCE == {"epsilon": 0.002, "no_improvement_iterations": 3}
    assert SUBMISSION["columns"] == ["row_id", "user_id", "video_id", "score"]
    assert SUBMISSION["preserve_repeated_pairs"] is True


@todo("D2")
def test_preflight_computes_all_three_hashes():
    """evaluator, submit checker, data."""


@todo("D2")
def test_modified_evaluator_fails_preflight_closed():
    """A single changed byte in evaluate.py must abort the run, not warn."""


@todo("D2")
def test_metric_profile_is_read_from_the_shipped_evaluator():
    """Never from prose — the brief contradicts itself (Section 4.8)."""


# test_every_node_carries_the_manifest_hash lives in test_store.py — the
# stamping is the store's job (A2); D2 only supplies the hash.
