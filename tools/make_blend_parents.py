"""Build two accepted full-fidelity parents so C7 can be validated on demand.

C7 needs two accepted, full-fidelity, non-blend parent nodes. Waiting for the
agent loop to produce them blocks the whole blend path, but they can be built
directly: run two real experiments and record them.

**These nodes are validation fixtures, not experiment records.** They carry a
placeholder manifest hash and an `accepted` flag that no gate granted, so
they must never be written into `logs/nodes/` — that directory is the graded
run-log deliverable, and the trust boundary (Section 7.2) says the agent
cannot approve its own promotion. Everything here goes to a temporary store
that is discarded unless `--keep` names somewhere else.

Usage:
    python -m tools.make_blend_parents                 # temp store, prints results
    python -m tools.make_blend_parents --blend         # also run all four methods
    python -m tools.make_blend_parents --keep DIR      # persist to DIR (not logs/)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

BASE_FEATURES = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]

PARENTS = [
    ("n001", "FM", {"model": "fm", "features": BASE_FEATURES, "hparams": {}}),
    (
        "n002",
        "LGBM",
        {
            "model": "lgbm",
            "loss": "pointwise",
            "features": BASE_FEATURES,
            "hparams": {"num_boost_round": 150, "learning_rate": 0.1, "num_leaves": 63},
        },
    ),
]

FIXTURE_MANIFEST = "c7-validation-fixture"


def build(store_dir: Path, seed: int = 42, timeout_s: int = 3600) -> dict:
    from agent import store
    from agent.schema import Config
    from pipeline import train

    store.NODES_DIR = store_dir / "nodes"
    store.EVENT_LOG = store_dir / "run.jsonl"

    results = {}
    for node_id, tag, config in PARENTS:
        started = time.monotonic()
        result = train.run_experiment(config, fidelity="full", seed=seed, timeout_s=timeout_s)
        if result["status"] != "ok":
            raise SystemExit(
                f"{tag} failed: {result.get('error_class')} "
                f"{str(result.get('traceback'))[-400:]}"
            )
        results[node_id] = result
        print(
            f"{tag:5s} primary={result['primary']:.6f} "
            f"gauc={result['gauc']:.4f} ndcg={result['ndcg']:.4f} "
            f"{time.monotonic() - started:.0f}s -> {node_id}"
        )
        store.write(
            {
                "id": node_id,
                "parent": "n000",
                "family": "model",
                "hypothesis": f"{tag} full-fidelity parent, C7 validation fixture",
                "action_type": "config",
                "fidelity": "full",
                "status": "ok",
                "manifest_sha256": FIXTURE_MANIFEST,
                "accepted": True,
                "metrics": {
                    "gauc": result["gauc"],
                    "ndcg": result["ndcg"],
                    "primary": result["primary"],
                },
                "fold_primaries": result["fold_primaries"],
                "config": Config(**config).model_dump(),
            }
        )
    return results


def report_correlation(results: dict) -> float:
    from pipeline.models.blend import per_user_spearman

    first, second = (results[node_id] for node_id, _, _ in PARENTS)
    rho = per_user_spearman(first["val_scores"], second["val_scores"], first["val_user_ids"])
    print(
        f"\nper-user Spearman rho = {rho:.4f}"
        "   (>0.95 refuse, 0.7-0.9 sweet spot, <0.5 investigate)"
    )
    return rho


def run_blends(seed: int = 42, timeout_s: int = 3600) -> None:
    from pipeline import train

    print("\nblend methods, full tier:")
    for method in ("rank_avg", "logit_avg", "weighted_rank", "rrf"):
        result = train.run_experiment(
            {"model": "blend", "parents": ["n001", "n002"], "blend_method": method},
            fidelity="full",
            seed=seed,
            timeout_s=timeout_s,
        )
        if result["status"] != "ok":
            print(f"  {method:14s} ERROR {result.get('error_class')}")
            continue
        parents = [round(value, 6) for value in result.get("parent_primaries", [])]
        print(
            f"  {method:14s} primary={result['primary']:.6f} parents={parents} "
            f"accepted={result.get('blend_accepted')} gates={result.get('blend_gates')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", action="store_true", help="also run all four blend methods")
    parser.add_argument("--keep", type=Path, help="persist the store here instead of a temp dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    store_dir = args.keep or Path(tempfile.mkdtemp(prefix="c7-parents-"))
    if store_dir.resolve() == Path("logs").resolve():
        parser.error("refusing to write fixtures into logs/ — that is the graded ledger")
    store_dir.mkdir(parents=True, exist_ok=True)
    print(f"store: {store_dir}\n")

    results = build(store_dir, seed=args.seed)
    report_correlation(results)
    if args.blend:
        run_blends(seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
