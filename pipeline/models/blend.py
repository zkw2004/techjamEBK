"""Blending accepted nodes. Task C7, spec in Section 6.9.

Four methods; rank_avg is the default. Trap 10: never majority-vote labels —
ties destroy NDCG. Blend weights are fitted on internal folds only (trap 6).
A blend is rejected unless it beats BOTH parents on folds and on the
official metric.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import register

METHODS = ("rank_avg", "logit_avg", "weighted_rank", "rrf")
RRF_K = 60.0
CORRELATION_REFUSAL = 0.95


def _validate_vectors(scores: list[np.ndarray], user_ids: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(values, dtype=float) for values in scores]
    users = np.asarray(user_ids)
    if users.ndim != 1:
        raise ValueError("user ids must be one-dimensional")
    if len(arrays) < 2:
        raise ValueError("a blend requires at least two parent score vectors")
    if any(values.ndim != 1 or len(values) != len(users) for values in arrays):
        raise ValueError("parent scores and user ids must be aligned one-dimensional arrays")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("parent scores must be finite")
    return arrays


def _user_groups(user_ids: np.ndarray):
    """One stable grouping sort, avoiding a full-array scan for every user."""
    if not len(user_ids):
        return
    order = np.argsort(user_ids, kind="stable")
    _, starts = np.unique(np.asarray(user_ids)[order], return_index=True)
    yield from np.split(order, starts[1:])


def _within_user_ranks(
    scores: np.ndarray,
    user_ids: np.ndarray,
    *,
    normalise: bool,
) -> np.ndarray:
    """Average ranks within each user; larger input always means larger rank."""
    scores = np.asarray(scores, dtype=float)
    users = np.asarray(user_ids)
    ranks = np.empty(len(scores), dtype=float)
    for positions in _user_groups(users):
        values = scores[positions]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            sorted_ranks[start:end] = (start + end - 1) / 2.0 + 1.0
            start = end
        group_ranks = np.empty(len(values), dtype=float)
        group_ranks[order] = sorted_ranks
        if normalise:
            group_ranks = (
                np.zeros_like(group_ranks)
                if len(values) == 1
                else (group_ranks - 1.0) / (len(values) - 1.0)
            )
        ranks[positions] = group_ranks
    return ranks


def _probability_scale(scores: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    if np.all((scores >= 0.0) & (scores <= 1.0)):
        return scores.copy()
    scaled = np.empty_like(scores, dtype=float)
    for positions in _user_groups(user_ids):
        values = scores[positions]
        low, high = float(np.min(values)), float(np.max(values))
        scaled[positions] = 0.5 if high == low else (values - low) / (high - low)
    return scaled


def blend_scores(
    parent_scores: list[np.ndarray],
    user_ids: np.ndarray,
    method: str = "rank_avg",
    *,
    weight: float = 0.5,
) -> np.ndarray:
    """Blend continuous parent scores without letting scale dominate rank."""
    if method not in METHODS:
        raise ValueError(f"unsupported blend method {method!r}; expected one of {METHODS}")
    users = np.asarray(user_ids)
    arrays = _validate_vectors(parent_scores, users)

    if method == "logit_avg":
        epsilon = np.finfo(float).eps
        probabilities = [
            np.clip(_probability_scale(values, users), epsilon, 1.0 - epsilon) for values in arrays
        ]
        mean_logit = np.mean([np.log(p / (1.0 - p)) for p in probabilities], axis=0)
        return 1.0 / (1.0 + np.exp(-mean_logit))

    ranks = [_within_user_ranks(values, users, normalise=True) for values in arrays]
    if method == "rank_avg":
        return np.mean(ranks, axis=0)
    if method == "weighted_rank":
        if len(ranks) != 2:
            raise ValueError("weighted_rank supports exactly two parents")
        if not 0.0 < weight < 1.0:
            raise ValueError("weighted_rank weight must be strictly between zero and one")
        return weight * ranks[0] + (1.0 - weight) * ranks[1]

    ordinal_ranks = [_within_user_ranks(values, users, normalise=False) for values in arrays]
    return np.sum(
        [1.0 / (RRF_K + counts_by_user(users) - ranks + 1.0) for ranks in ordinal_ranks],
        axis=0,
    )


def counts_by_user(user_ids: np.ndarray) -> np.ndarray:
    users = np.asarray(user_ids)
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    return counts[inverse].astype(float)


@register("blend")
class Blend:
    def __init__(
        self,
        parents: list[str] | None = None,
        blend_method: str = "rank_avg",
        seed: int = 42,
        **hparams,
    ):
        self.parents, self.blend_method = parents or [], blend_method
        self.weight = float(hparams.get("weight", 0.5))
        self.weight_source = "configured"
        self.parent_correlation: float | None = None

    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None:
        matrix = np.asarray(X_val if X_val is not None else X_train, dtype=float)
        labels = np.asarray(y_val if X_val is not None else y_train, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 3:
            raise ValueError("blend matrices must contain user_id then at least two parent scores")
        users = matrix[:, 0]
        parents = [matrix[:, index] for index in range(1, matrix.shape[1])]
        self.parent_correlation = per_user_spearman(parents[0], parents[1], users)
        if self.parent_correlation > CORRELATION_REFUSAL:
            raise ValueError(
                f"parent correlation {self.parent_correlation:.6f} is too similar to gain"
            )

        if self.blend_method == "weighted_rank":
            from pipeline.evaluate import evaluate

            candidates = np.arange(0.1, 1.0, 0.1)
            self.weight = max(
                candidates,
                key=lambda weight: (
                    evaluate(
                        users,
                        labels,
                        blend_scores(parents, users, "weighted_rank", weight=float(weight)),
                    )["primary"],
                    weight,
                ),
            )
            self.weight = float(self.weight)
            self.weight_source = "internal_validation"

    def predict(self, X) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 3:
            raise ValueError("blend matrices must contain user_id then at least two parent scores")
        return blend_scores(
            [matrix[:, index] for index in range(1, matrix.shape[1])],
            matrix[:, 0],
            self.blend_method,
            weight=self.weight,
        )


def per_user_spearman(a: np.ndarray, b: np.ndarray, user_ids: np.ndarray) -> float:
    """Mean within-user rank correlation between two score vectors.

    Reported alongside every blend: low correlation is the evidence that
    the parents' errors differ enough to be worth combining.
    """
    first, second = _validate_vectors([a, b], np.asarray(user_ids))
    users = np.asarray(user_ids)
    correlations = []
    first_ranked = _within_user_ranks(first, users, normalise=False)
    second_ranked = _within_user_ranks(second, users, normalise=False)
    for positions in _user_groups(users):
        if len(positions) < 2:
            continue
        first_ranks = first_ranked[positions]
        second_ranks = second_ranked[positions]
        if np.std(first_ranks) == 0.0 or np.std(second_ranks) == 0.0:
            continue
        correlations.append(float(np.corrcoef(first_ranks, second_ranks)[0, 1]))
    if not correlations:
        raise ValueError("Spearman correlation needs a non-constant multi-row user group")
    return float(np.mean(correlations))
