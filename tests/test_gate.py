"""D3 acceptance: identical vectors rejected; a known +0.01 accepted; a
+0.0003 rejected; resampling is over users."""

from __future__ import annotations

from agent.gate import BASELINE_SEED_STD, MIN_DELTA_FLOOR
from tests.conftest import todo


def test_min_delta_floor_sits_above_the_noise_floor():
    """Trap 5: anything below ~2x seed std admits noise as improvement."""
    assert MIN_DELTA_FLOOR >= 2 * BASELINE_SEED_STD


@todo("D3")
def test_identical_score_vectors_are_rejected():
    """CI on a zero delta must straddle zero."""


@todo("D3")
def test_known_large_improvement_is_accepted():
    """A synthetic +0.01 delta must produce a CI strictly above zero."""


@todo("D3")
def test_sub_noise_improvement_is_rejected():
    """+0.0003 is below the seed std of 0.0008 — must not promote."""


@todo("D3")
def test_bootstrap_resamples_users_not_rows():
    """The metric is per-user, so the user is the unit of independence (5.3).
    Resampling rows understates variance."""


@todo("D4")
def test_segments_cover_activity_popularity_and_day():
    """Every full-fidelity result carries all three breakdowns (6.6)."""
