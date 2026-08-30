"""C7 scale-safe blending, diagnostics, and fold-only weight fitting."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from pipeline.models.blend import Blend, blend_scores, per_user_spearman


def _matrix(users, first, second):
    return np.column_stack([users, first, second])


def test_rank_average_matches_hand_computed_per_user_ranks():
    users = np.array([1, 1, 1, 2, 2])
    first = np.array([0.1, 0.9, 0.4, 100.0, 10.0])
    second = np.array([2.0, 8.0, 5.0, -3.0, 7.0])

    actual = blend_scores([first, second], users, "rank_avg")

    np.testing.assert_allclose(actual, [0.0, 1.0, 0.5, 0.5, 0.5])


def test_per_user_spearman_matches_scipy_group_cross_check():
    users = np.repeat([10, 20], 4)
    first = np.array([0.1, 0.5, 0.2, 0.9, 4.0, 3.0, 2.0, 1.0])
    second = np.array([0.2, 0.1, 0.5, 0.8, 1.0, 3.0, 2.0, 4.0])
    expected = np.mean(
        [spearmanr(first[users == user], second[users == user]).statistic for user in [10, 20]]
    )

    assert per_user_spearman(first, second, users) == pytest.approx(expected)


def test_blend_refuses_near_identical_parents():
    users = np.repeat([1, 2], 4)
    first = np.arange(8, dtype=float)
    second = first * 10 + 3
    model = Blend(blend_method="rank_avg")

    with pytest.raises(ValueError, match="too similar"):
        model.fit(
            _matrix(users, first, second),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            None,
            None,
            groups=(users, None),
        )


def test_weighted_rank_selects_weight_using_only_supplied_internal_validation():
    users = np.repeat([1, 2], 4)
    labels = np.tile([0, 0, 1, 1], 2)
    good = np.tile([0.1, 0.2, 0.8, 0.9], 2)
    bad = np.tile([0.9, 0.8, 0.2, 0.1], 2)
    model = Blend(blend_method="weighted_rank")

    model.fit(
        _matrix(users, bad, good),
        1 - labels,
        _matrix(users, good, bad),
        labels,
        groups=(users, users),
    )

    assert model.weight == pytest.approx(0.9)
    assert model.weight_source == "internal_validation"


@pytest.mark.parametrize("method", ["rank_avg", "logit_avg", "weighted_rank", "rrf"])
def test_all_blend_methods_return_finite_continuous_scores(method):
    users = np.repeat([1, 2], 3)
    first = np.array([0.2, 0.7, 0.4, -3.0, 1.0, 8.0])
    second = np.array([8.0, 1.0, 4.0, 0.1, 0.9, 0.3])

    scores = blend_scores([first, second], users, method, weight=0.7)

    assert scores.shape == (6,)
    assert np.isfinite(scores).all()
    assert len(np.unique(scores)) > 2
