"""D3 acceptance: identical vectors rejected; a known improvement accepted; a
sub-noise improvement rejected; resampling is over users. D4 acceptance:
segments cover activity, popularity, and day."""

from __future__ import annotations

import numpy as np
import pytest

import agent.gate as gate
from agent.gate import BASELINE_SEED_STD, MIN_DELTA_FLOOR, accept, segments


def test_min_delta_floor_sits_above_the_noise_floor():
    """Trap 5: anything below ~2x seed std admits noise as improvement."""
    assert MIN_DELTA_FLOOR >= 2 * BASELINE_SEED_STD


def _ranking_fixture(n_users: int, improved_users: int):
    """Per user: 4 rows, one positive. `best` ranks the positive 2nd
    everywhere; `cand` ranks it 1st for the first `improved_users` users."""
    user_ids, labels, best, cand = [], [], [], []
    for user in range(n_users):
        user_ids += [user] * 4
        labels += [1, 0, 0, 0]
        best += [3.0, 4.0, 2.0, 1.0]  # positive is second-highest
        if user < improved_users:
            cand += [4.0, 3.0, 2.0, 1.0]  # positive promoted to the top
        else:
            cand += [3.0, 4.0, 2.0, 1.0]
    return (
        np.asarray(cand), np.asarray(best), np.asarray(user_ids),
        np.asarray(labels, dtype=float),
    )


def _patch_labels(monkeypatch, labels):
    monkeypatch.setattr(gate, "_validation_labels", lambda: np.asarray(labels, dtype=float))


def test_identical_score_vectors_are_rejected(monkeypatch):
    """CI on a zero delta must straddle zero."""
    cand, best, users, labels = _ranking_fixture(50, improved_users=0)
    _patch_labels(monkeypatch, labels)
    accepted, (low, high) = accept(best, best, users)
    assert not accepted
    assert low <= 0 <= high


def test_known_large_improvement_is_accepted(monkeypatch):
    """A consistent per-user improvement must produce a CI strictly above zero."""
    cand, best, users, labels = _ranking_fixture(200, improved_users=120)
    _patch_labels(monkeypatch, labels)
    accepted, (low, high) = accept(cand, best, users)
    assert accepted
    assert low > 0


def test_sub_noise_improvement_is_rejected(monkeypatch):
    """One improved user in 300 is inside the noise floor — must not promote."""
    cand, best, users, labels = _ranking_fixture(300, improved_users=1)
    _patch_labels(monkeypatch, labels)
    accepted, (low, _high) = accept(cand, best, users)
    assert not accepted
    assert low <= 0


def test_bootstrap_resamples_users_not_rows(monkeypatch):
    """The metric is per-user, so the user is the unit of independence (5.3).
    Permuting rows within users changes nothing; the CI depends only on the
    per-user structure the bootstrap resamples."""
    cand, best, users, labels = _ranking_fixture(120, improved_users=60)
    _patch_labels(monkeypatch, labels)
    _, ci_original = accept(cand, best, users, seed=3)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(users))
    _patch_labels(monkeypatch, labels[order])
    _, ci_permuted = accept(cand[order], best[order], users[order], seed=3)

    assert ci_original == pytest.approx(ci_permuted)


def test_accept_refuses_misaligned_vectors(monkeypatch):
    cand, best, users, labels = _ranking_fixture(10, improved_users=5)
    _patch_labels(monkeypatch, labels[:-4])
    with pytest.raises(ValueError, match="validation"):
        accept(cand, best, users)


def test_segments_cover_activity_popularity_and_day(monkeypatch):
    """Every full-fidelity result carries all three breakdowns (6.6)."""
    cand, _best, users, labels = _ranking_fixture(40, improved_users=40)
    meta = {
        "labels": labels,
        "activity_q": (users % 4) + 1,
        "pop_q": (users % 2) + 1,
        "day": (users % 3) + 1,
    }
    out = segments(cand, users, meta)
    for prefix, levels in (("activity_q", 4), ("pop_q", 2), ("day", 3)):
        found = [key for key in out if key.startswith(prefix)]
        assert len(found) == levels, (prefix, sorted(out))
    # every user's positive is ranked top -> perfect ranking in every segment
    assert all(value == pytest.approx(1.0) for value in out.values())
