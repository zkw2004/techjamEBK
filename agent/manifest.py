"""MetricProfile, contract hashing, preflight.

Contract: AGENT_PLAN.md Section 8.3 (FROZEN). Owner: Workstream D. Task D2.

Preflight recomputes the evaluator, submit-checker and data hashes and FAILS
CLOSED on mismatch. Every node record carries the manifest hash, so a mid-run
evaluator change cannot silently invalidate earlier results.

The MetricProfile is populated FROM THE SHIPPED EVALUATOR, never from prose
(Section 4.8 — the brief contradicts itself on NDCG@10/Recall@50 vs
GAUC/nDCG@5; this is what makes the system profile-agnostic).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST_PATH = Path("logs/manifest.json")
EVALUATOR = Path("pipeline/evaluate.py")
SUBMIT_CHECKER = Path("pipeline/submit.py")

BASELINE_VALIDATION = 0.6016
BASELINE_SEED_STD = 0.0008
CONVERGENCE = {"epsilon": 0.002, "no_improvement_iterations": 3}

SUBMISSION = {
    "columns": ["row_id", "user_id", "video_id", "score"],
    "finite_scores_only": True,
    "preserve_repeated_pairs": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest() -> dict:
    """Written once at preflight, immutable for the run. See 8.3 for the shape."""
    raise NotImplementedError("D2")


def preflight() -> dict:
    """Recompute all three hashes; raise if they differ from logs/manifest.json."""
    raise NotImplementedError("D2")
