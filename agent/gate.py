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

_VALIDATION_LABEL_CACHE: np.ndarray | None = None


def _validation_labels() -> np.ndarray:
    """Labels for the official validation rows, in organiser row order.

    The frozen accept() signature carries scores and user ids only; per the
    Section 8.5 contract those are aligned with the official validation rows,
    so the matching labels are recoverable from the fixed split. Cached: the
    gate runs once per full evaluation, the loader reads 1.4M rows.
    """
    global _VALIDATION_LABEL_CACHE
    if _VALIDATION_LABEL_CACHE is None:
        from pipeline.data import LABEL, load

        _, validation, _ = load()
        _VALIDATION_LABEL_CACHE = validation[LABEL].to_numpy(dtype=np.float64)
    return _VALIDATION_LABEL_CACHE


def _per_user_stats(
    scores: np.ndarray, labels: np.ndarray, user_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-user (gauc, gauc_weight, ndcg) matching pipeline/evaluate.py.

    GAUC counts only users with 0 < positives < impressions, weighted by
    positive count. nDCG@5 uses gain 2^rel - 1 and counts zero-positive
    users as 0.0. Precomputing per user is what makes a 1000-resample
    bootstrap affordable.
    """
    from pipeline.evaluate import auc, ndcg_at_k

    row_order = np.argsort(user_ids, kind="stable")  # group rows without masking per user
    grouped_users = np.asarray(user_ids)[row_order]
    grouped_scores = scores[row_order]
    grouped_labels = labels[row_order]
    users, group_starts = np.unique(grouped_users, return_index=True)
    group_bounds = np.append(group_starts, len(grouped_users))

    gauc = np.zeros(len(users))
    weight = np.zeros(len(users))
    ndcg = np.zeros(len(users))
    for i in range(len(users)):
        begin, end = group_bounds[i], group_bounds[i + 1]
        user_scores = grouped_scores[begin:end]
        user_labels = grouped_labels[begin:end]
        order = np.argsort(-user_scores, kind="stable")
        sorted_labels = user_labels[order].tolist()
        positives = float(user_labels.sum())
        if 0 < positives < len(user_labels):
            gauc[i] = auc(sorted_labels, user_scores[order].tolist())
            weight[i] = positives
        ndcg[i] = ndcg_at_k(sorted_labels, 5)
    return users, gauc, weight, ndcg


def _primary_from_stats(
    gauc: np.ndarray, weight: np.ndarray, ndcg: np.ndarray, take: np.ndarray | None = None
) -> float:
    if take is not None:
        gauc, weight, ndcg = gauc[take], weight[take], ndcg[take]
    total_weight = weight.sum()
    gauc_value = float((gauc * weight).sum() / total_weight) if total_weight else 0.5
    ndcg_value = float(ndcg.mean()) if len(ndcg) else 0.0
    return (gauc_value + ndcg_value) / 2.0


def accept(cand_scores, best_scores, user_ids, n_boot: int = 1000, seed: int = 0):
    """Bootstrap over USERS (not rows). Returns (accepted: bool, ci: tuple).

    accepted is True iff the 95% CI on (candidate - best) primary
    excludes zero on the low side.
    """
    cand_scores = np.asarray(cand_scores, dtype=np.float64)
    best_scores = np.asarray(best_scores, dtype=np.float64)
    user_ids = np.asarray(user_ids)
    if not (len(cand_scores) == len(best_scores) == len(user_ids)):
        raise ValueError("candidate scores, best scores, and user ids must align")
    labels = _validation_labels()
    if len(labels) != len(user_ids):
        raise ValueError(
            f"scores cover {len(user_ids)} rows but the official validation window "
            f"has {len(labels)}; the gate only accepts full-validation score vectors"
        )

    _, cand_gauc, cand_weight, cand_ndcg = _per_user_stats(cand_scores, labels, user_ids)
    _, best_gauc, best_weight, best_ndcg = _per_user_stats(best_scores, labels, user_ids)

    rng = np.random.default_rng(seed)
    n_users = len(cand_gauc)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, n_users, n_users)
        deltas[i] = _primary_from_stats(
            cand_gauc, cand_weight, cand_ndcg, take
        ) - _primary_from_stats(best_gauc, best_weight, best_ndcg, take)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return bool(low > 0), (float(low), float(high))


def segments(scores, user_ids, meta) -> dict:
    """Primary metric by user-activity quartile, item-popularity
    quartile, and day within the evaluation window.

    `meta` maps a segment name to one per-row level array, e.g.
    {"activity_q": ..., "pop_q": ..., "day": ...}. The reserved key
    "labels" supplies per-row labels; when absent, the official
    validation labels are used (the frozen signature has no labels
    parameter, and full-fidelity segments are computed on that window).
    """
    scores = np.asarray(scores, dtype=np.float64)
    user_ids = np.asarray(user_ids)
    meta = dict(meta)
    labels = meta.pop("labels", None)
    labels = _validation_labels() if labels is None else np.asarray(labels, dtype=np.float64)
    if not (len(scores) == len(user_ids) == len(labels)):
        raise ValueError("scores, user ids, and labels must align")

    out: dict[str, float] = {}
    for name, keys in meta.items():
        keys = np.asarray(keys)
        if len(keys) != len(scores):
            raise ValueError(f"segment key array {name!r} must have one value per row")
        for level in np.unique(keys):
            mask = keys == level
            _, gauc, weight, ndcg = _per_user_stats(
                scores[mask], labels[mask], user_ids[mask]
            )
            out[f"{name}{level}"] = _primary_from_stats(gauc, weight, ndcg)
    return out
