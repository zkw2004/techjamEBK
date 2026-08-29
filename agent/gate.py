"""Bootstrap accept gate + segment report.

Contract: AGENT_PLAN.md Section 8.8 (FROZEN). Reference impl: Appendix A.3.
Owner: Workstream D (Pinxin). Tasks D3, D4.

Resample USERS, not rows. The metric is computed per user, so the user is the
unit of independence; resampling rows understates variance and lets noise
through (Section 5.3).
"""

from __future__ import annotations

import numpy as np

BASELINE_SEED_STD = 0.0008  # anything smaller than ~2x this is not an improvement
MIN_DELTA_FLOOR = 0.002


def accept(cand_scores, best_scores, user_ids, n_boot: int = 1000, seed: int = 0):
    """Bootstrap over USERS (not rows). Returns (accepted: bool, ci: tuple).

    accepted is True iff the 95% CI on (candidate - best) primary
    excludes zero on the low side.
    """
    raise NotImplementedError("D3")


def segments(scores, user_ids, meta) -> dict:
    """Primary metric by user-activity quartile, item-popularity
    quartile, and day within the evaluation window.
    """
    raise NotImplementedError("D4")
