"""Immutable evaluation contract and per-run manifest preflight.

The starter-kit evaluator and submission checker are vendored unchanged in
``pipeline/``.  This module is the typed, project-owned boundary around them;
it deliberately fails closed if either file, the starter label definition, or
the KuaiRand archive differs from the contract recorded for a run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent import store

MANIFEST_PATH = Path("logs/manifest.json")
EVALUATOR = Path("pipeline/evaluate.py")
SUBMIT_CHECKER = Path("pipeline/submit.py")
STARTER_EVALUATOR = Path("kuairand-starter-kit/evaluate.py")
STARTER_SUBMIT_CHECKER = Path("kuairand-starter-kit/submit.py")
STARTER_DATA = Path("kuairand-starter-kit/data.py")
DATA_ARCHIVE = Path("data/KuaiRand-Pure.tar.gz")

BASELINE_VALIDATION = 0.6016
BASELINE_SEED_STD = 0.0008
CONVERGENCE = {"epsilon": 0.002, "no_improvement_iterations": 3}
SUBMISSION = {
    "columns": ["row_id", "user_id", "video_id", "score"],
    "finite_scores_only": True,
    "preserve_repeated_pairs": True,
}


class ManifestError(RuntimeError):
    """Raised when the immutable evaluation contract cannot be trusted."""


@dataclass(frozen=True)
class EvaluationResult:
    """Typed result from the otherwise untyped starter-kit evaluator."""

    gauc: float
    ndcg: float
    primary: float
    users: int
    rows: int


@dataclass(frozen=True)
class MetricProfile:
    """The exact metric and data contract used for a run."""

    source: str
    evaluator_sha256: str
    submit_checker_sha256: str
    data_sha256: str
    target_label: str
    group_key: str
    metrics: list[str]
    cutoffs: dict[str, int]
    aggregation: str
    zero_positive_rule: dict[str, str]
    baseline_validation: float
    baseline_seed_std: float
    convergence: dict[str, float | int]
    submission: dict[str, Any]


def sha256(path: Path | str) -> str:
    """Return the SHA-256 of a file, with a useful fail-closed error."""
    candidate = Path(path)
    if not candidate.is_file():
        raise ManifestError(f"required contract file is missing: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _starter_label() -> str:
    """Read the source-of-truth label assignment without importing starter code."""
    tree = ast.parse(STARTER_DATA.read_text(encoding="utf-8"), filename=str(STARTER_DATA))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "LABEL"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ManifestError(f"could not determine LABEL from {STARTER_DATA}")


def _evaluator_cutoff() -> int:
    """Read ``evaluate(..., k=...)`` directly from the immutable evaluator."""
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"), filename=str(EVALUATOR))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            if not node.args.defaults or not isinstance(node.args.defaults[-1], ast.Constant):
                break
            cutoff = node.args.defaults[-1].value
            if isinstance(cutoff, int):
                return cutoff
    raise ManifestError(f"could not determine evaluate() cutoff from {EVALUATOR}")


def verify_starter_kit() -> dict[str, str]:
    """Verify that the frozen pipeline copies are byte-identical to starter code."""
    hashes = {
        "evaluator_sha256": sha256(EVALUATOR),
        "submit_checker_sha256": sha256(SUBMIT_CHECKER),
    }
    if hashes["evaluator_sha256"] != sha256(STARTER_EVALUATOR):
        raise ManifestError("pipeline/evaluate.py differs from the immutable starter-kit evaluator")
    if hashes["submit_checker_sha256"] != sha256(STARTER_SUBMIT_CHECKER):
        raise ManifestError(
            "pipeline/submit.py differs from the immutable starter-kit submission checker"
        )
    return hashes


def evaluate_scores(
    user_ids: Sequence[object], labels: Sequence[int], scores: Sequence[float], *, k: int = 5
) -> EvaluationResult:
    """Evaluate scores with the frozen evaluator and return a typed result."""
    if not (len(user_ids) == len(labels) == len(scores)):
        raise ValueError("user_ids, labels, and scores must have equal lengths")

    # Import lazily so this wrapper remains the sole project-owned interface.
    from pipeline.evaluate import evaluate

    result = evaluate(user_ids, labels, scores, k=k)
    return EvaluationResult(
        gauc=float(result["GAUC"]),
        ndcg=float(result[f"nDCG@{k}"]),
        primary=float(result["primary"]),
        users=int(result["users"]),
        rows=int(result["rows"]),
    )


def build_manifest() -> dict[str, Any]:
    """Compute, but do not write, the immutable metric profile for this run."""
    hashes = verify_starter_kit()
    cutoff = _evaluator_cutoff()
    profile = MetricProfile(
        source="shipped_evaluate.py",
        evaluator_sha256=hashes["evaluator_sha256"],
        submit_checker_sha256=hashes["submit_checker_sha256"],
        data_sha256=sha256(DATA_ARCHIVE),
        target_label=_starter_label(),
        group_key="user_id",
        metrics=["GAUC", f"nDCG@{cutoff}"],
        cutoffs={"nDCG": cutoff},
        aggregation=f"mean(GAUC, nDCG@{cutoff})",
        zero_positive_rule={
            "GAUC": "excluded unless 0 < positives < impressions; weighted by positives",
            "nDCG": "0.0 and included in the mean",
        },
        baseline_validation=BASELINE_VALIDATION,
        baseline_seed_std=BASELINE_SEED_STD,
        convergence=CONVERGENCE,
        submission=SUBMISSION,
    )
    return {"metric_profile": asdict(profile)}


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create the manifest exactly once; never overwrite evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def manifest_sha256() -> str:
    """Return the identity propagated into every stored experiment node."""
    return sha256(MANIFEST_PATH)


def preflight() -> dict[str, Any]:
    """Establish or verify the run contract, then enable manifest propagation.

    An existing manifest must match freshly computed hashes exactly.  This
    avoids silently evaluating a run with a modified evaluator or data archive.
    """
    expected = build_manifest()
    _write_once(MANIFEST_PATH, expected)
    try:
        recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"unable to read run manifest: {MANIFEST_PATH}") from exc
    if recorded != expected:
        raise ManifestError(
            "run manifest does not match the current immutable evaluator/data contract"
        )
    store.set_manifest_provider(manifest_sha256)
    return recorded
