"""run_experiment(): the single entry point the agent calls.

Contract: AGENT_PLAN.md Section 8.5 (FROZEN). Owner: Workstream C (Ethan). Task C1.

MUST NOT raise. MUST enforce timeout_s. MUST be deterministic given seed.
"""

from __future__ import annotations

FIDELITIES = ("smoke", "screen", "full", "confirm")

ERROR_CLASSES = ("syntax", "schema", "timeout", "oom", "transient", "leak_suspected")

LEAK_CANARY_PRIMARY = 0.75  # Section 3; realistic ceiling is nowhere near this


def run_experiment(
    config: dict,
    fidelity: str = "full",
    seed: int = 42,
    timeout_s: int = 1800,
) -> dict:
    """Run one experiment at the requested fidelity tier (Section 6.3).

      smoke   -> 1000-row fit, correctness checks only, no metrics
      screen  -> three internal folds, reduced budget
      full    -> full train prefix, all official validation rows
      confirm -> full, repeated across 5 seeds

    Success:
    {
      "status": "ok", "fidelity": str,
      "gauc": float, "ndcg": float, "primary": float,
      "fold_primaries": list[float],   # screen/full
      "segments": dict,                # full/confirm, see 6.6
      "val_scores": np.ndarray,        # aligned with val rows
      "val_user_ids": np.ndarray,
      "test_scores": np.ndarray,
      "seconds": float, "gpu_seconds": float, "peak_rss_mb": float,
    }

    Failure:
    {"status": "error", "stage": str, "error_class": str,
     "traceback": str, "seconds": float}

    MUST return status="error", error_class="leak_suspected" if primary > 0.75.
    """
    raise NotImplementedError("C1")
