"""C7 orchestration for fixed accepted parents, using C1/B's existing paths.

Score matrices for the Blend protocol are not feature matrices. This adapter
resolves node configs, fits weights on internal folds, and combines aligned
parent predictions without changing the frozen public runner/model contracts.
"""

from __future__ import annotations

import numpy as np

from pipeline.models.blend import CORRELATION_REFUSAL, blend_scores, per_user_spearman


def _parents(config: dict) -> list[dict]:
    from pipeline import train

    ids = config["parents"]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("C7 requires exactly two distinct accepted parent nodes")
    return [train._parent_config(node_id) for node_id in ids]


def _fold_evidence(config: dict, parents: list[dict], fidelity: str) -> dict:
    from pipeline import train

    folds = train._load_folds()
    if len(folds) != 3:
        raise ValueError("blending requires exactly three internal folds")
    fold_scores, users, parent_metrics, correlations = [], [], [], []
    for index, (training, validation) in enumerate(folds):
        scores = []
        for parent in parents:
            fitting_config = train._screen_config(parent) if fidelity == "screen" else parent
            (prediction,) = train._fit_and_predict(
                fitting_config, training, validation, [validation], parent["seed"] + index
            )
            scores.append(prediction)
        fold_users = train._column(validation, "user_id")
        fold_scores.append(scores)
        users.append(fold_users)
        parent_metrics.append([train._evaluate(validation, values)["primary"] for values in scores])
        correlations.append(per_user_spearman(scores[0], scores[1], fold_users))

    correlation = float(np.mean(correlations))
    if correlation > CORRELATION_REFUSAL:
        raise ValueError(f"parent correlation {correlation:.6f} is too similar to gain")

    def predictions(weight):
        return [
            blend_scores(scores, fold_users, config["blend_method"], weight=weight)
            for scores, fold_users in zip(fold_scores, users, strict=True)
        ]

    def metrics(predicted):
        return [
            train._evaluate(validation, scores)
            for (_, validation), scores in zip(folds, predicted, strict=True)
        ]

    weight = 0.5
    if config["blend_method"] == "weighted_rank":
        weight = max(
            (index / 10 for index in range(1, 10)),
            key=lambda candidate: (
                np.mean([metric["primary"] for metric in metrics(predictions(candidate))]),
                candidate,
            ),
        )
    predicted = predictions(weight)
    return {
        "metrics": metrics(predicted),
        "scores": np.concatenate(predicted),
        "users": np.concatenate(users),
        "parent_fold_primaries": np.asarray(parent_metrics).T.tolist(),
        "parent_correlation": correlation,
        "blend_weight": weight,
        "weight_source": "internal_folds" if config["blend_method"] == "weighted_rank" else "fixed",
        "blend_warning": "low correlation: investigate parent quality"
        if correlation < 0.5
        else None,
    }


def _full_parent_scores(parent: dict, validation, test=None) -> dict:
    from pipeline import train

    result = train._read_score_cache(
        parent, "full", parent["seed"], validation_frame=validation, test_frame=test
    )
    if result is None:
        result = train._run_full(parent, parent["seed"])
    if not np.array_equal(result["val_user_ids"], train._column(validation, "user_id")):
        raise ValueError("cached parent scores do not match official validation row order")
    return result


def _diagnostics(evidence: dict) -> dict:
    return {
        key: evidence[key]
        for key in (
            "parent_correlation",
            "parent_fold_primaries",
            "blend_weight",
            "weight_source",
            "blend_warning",
        )
    }


def _run_with_parents(config: dict, fidelity: str, seed: int, parents: list[dict]) -> dict:
    """Execute a two-parent blend; official labels are only used after fitting."""
    from agent import gate
    from pipeline import train

    if fidelity == "smoke":
        # Smoke never fits weights on the official validation sample.
        _, validation, test = train._load_data()
        validation, test = train._head(validation, 1_000), train._head(test, 1_000)
        results = [train._run_smoke(parent, parent["seed"]) for parent in parents]
        users = train._column(validation, "user_id")
        return train._success(
            "smoke",
            val_scores=blend_scores(
                [result["val_scores"] for result in results], users, config["blend_method"]
            ),
            val_user_ids=users,
            test_scores=blend_scores(
                [result["test_scores"] for result in results],
                train._column(test, "user_id"),
                config["blend_method"],
            ),
            blend_accepted=False,
        )

    evidence = _fold_evidence(config, parents, fidelity)
    fold_primaries = [metric["primary"] for metric in evidence["metrics"]]
    if fidelity == "screen":
        return train._success(
            "screen",
            **train._mean_metrics(evidence["metrics"]),
            fold_primaries=fold_primaries,
            val_scores=evidence["scores"],
            val_user_ids=evidence["users"],
            blend_accepted=False,
            **_diagnostics(evidence),
        )

    training, validation, test = train._load_data()
    results = [_full_parent_scores(parent, validation, test) for parent in parents]
    users = train._column(validation, "user_id")
    val_scores = blend_scores(
        [result["val_scores"] for result in results],
        users,
        config["blend_method"],
        weight=evidence["blend_weight"],
    )
    test_scores = blend_scores(
        [result["test_scores"] for result in results],
        train._column(test, "user_id"),
        config["blend_method"],
        weight=evidence["blend_weight"],
    )
    metrics = train._evaluate(validation, val_scores)
    if metrics["primary"] > train.LEAK_CANARY_PRIMARY:
        raise train.LeakSuspectedError(metrics["primary"])
    parent_primaries = [
        train._evaluate(validation, result["val_scores"])["primary"] for result in results
    ]
    min_delta = float(config["hparams"].get("min_blend_delta", gate.MIN_DELTA_FLOOR))
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_blend_delta must be a nonnegative finite margin")
    # Every temporal fold must clear both parents; one lucky slice is insufficient.
    fold_gate = bool(
        np.all(
            np.asarray(fold_primaries)
            > np.max(evidence["parent_fold_primaries"], axis=0) + min_delta
        )
    )
    official_gate = metrics["primary"] > max(parent_primaries) + min_delta
    accepted, ci = False, None
    if fold_gate and official_gate:
        best = int(np.argmax(parent_primaries))
        accepted, ci = gate.accept(val_scores, results[best]["val_scores"], users, seed=seed)
    return train._success(
        "full",
        **metrics,
        fold_primaries=fold_primaries,
        segments=train._segment_metrics(training, validation, val_scores),
        val_scores=val_scores,
        val_user_ids=users,
        test_scores=test_scores,
        blend_accepted=bool(accepted),
        blend_gate_ci=ci,
        blend_gates={
            "folds": fold_gate,
            "official": bool(official_gate),
            "bootstrap": bool(accepted),
        },
        parent_primaries=parent_primaries,
        **_diagnostics(evidence),
    )


def run_blend(config: dict, fidelity: str, seed: int) -> dict:
    """Run a fixed-node blend, or confirm the parent configurations over five seeds."""
    from pipeline import train

    parents = _parents(config)
    if fidelity != "confirm":
        return _run_with_parents(config, fidelity, seed, parents)

    runs = []
    for offset in range(5):
        train._seed_everything(seed + offset)
        seeded_parents = [{**parent, "seed": parent["seed"] + offset} for parent in parents]
        runs.append(_run_with_parents(config, "full", seed + offset, seeded_parents))
    segment_keys = sorted({key for run in runs for key in run["segments"]})
    return train._success(
        "confirm",
        **train._mean_metrics(runs),
        fold_primaries=np.mean([run["fold_primaries"] for run in runs], axis=0).tolist(),
        segments={
            key: float(np.mean([run["segments"][key] for run in runs])) for key in segment_keys
        },
        val_scores=np.mean([run["val_scores"] for run in runs], axis=0),
        val_user_ids=runs[0]["val_user_ids"],
        test_scores=np.mean([run["test_scores"] for run in runs], axis=0),
        parent_correlation=float(np.mean([run["parent_correlation"] for run in runs])),
        blend_accepted=all(run["blend_accepted"] for run in runs),
        blend_weights=[run["blend_weight"] for run in runs],
        weight_source="internal_folds" if config["blend_method"] == "weighted_rank" else "fixed",
    )
