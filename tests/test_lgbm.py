"""C4 LightGBM pointwise and LambdaRank behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.models.lgbm import LGBM

# Every test here drives lightgbm in-process. The conftest hook runs them
# in a forked child so the pytest process never co-loads torch and LightGBM
# (OMP Error #15 aborts the whole session).
pytestmark = pytest.mark.native_backend("lightgbm")


@pytest.mark.parametrize("budget, expected", [({"n_estimators": 7}, 7),
                                             ({"n_estimators": 7, "num_boost_round": 3}, 3)])
def test_estimator_alias_respects_explicit_final_refit_budget(budget, expected):
    X, y = _learnable_rows()
    model = LGBM(**budget, min_data_in_leaf=1)
    model.fit(X, y, None, None)
    assert model.best_epoch == expected


def test_lambdarank_direct_use_defaults_to_first_user_field():
    X, y = _learnable_rows()
    model = LGBM(loss="lambdarank", n_estimators=4, min_data_in_leaf=1)
    model.fit(X, y, None, None)
    assert model.predict(X[:2])[0] > model.predict(X[:2])[1]


@pytest.mark.parametrize("hparams, expected", [({"n_estimators": 7}, 7),
                                              ({"n_estimators": 7, "num_boost_round": 3}, 3),
                                              ({"n_estimators": 900}, 200)])
def test_screen_and_tuning_caps_preserve_requested_estimator_budget(hparams, expected):
    from agent.schema import Config
    from pipeline.train import _new_model, _screen_config

    config = Config(model="lgbm", hparams={**hparams, "min_data_in_leaf": 1}).model_dump()
    model = _new_model(_screen_config(config), seed=42)
    X, y = _learnable_rows()
    model.fit(X, y, None, None)
    assert model.best_epoch == expected
    assert config["hparams"] == {**hparams, "min_data_in_leaf": 1}


def _learnable_rows(repeats: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for index in range(repeats):
        rows.extend(
            [
                [f"u{index % 5}", "positive-item"],
                [f"u{index % 5}", "negative-item"],
            ]
        )
        labels.extend([1, 0])
    return np.asarray(rows, dtype=object), np.asarray(labels, dtype=float)


def test_pointwise_lightgbm_learns_positive_item_signal():
    X, y = _learnable_rows()
    model = LGBM(
        loss="pointwise",
        feature_names=["user_id", "video_id"],
        num_boost_round=30,
        early_stopping_rounds=5,
        min_data_in_leaf=1,
        num_leaves=7,
        seed=4,
    )

    model.fit(X, y, X, y, groups=(X[:, 0], X[:, 0]))
    scores = model.predict(
        np.asarray([["new-user", "positive-item"], ["new-user", "negative-item"]])
    )

    assert scores[0] > scores[1]


def test_pointwise_lightgbm_is_deterministic_and_handles_unseen_categories():
    X, y = _learnable_rows(10)
    target = np.asarray([["new-user", "new-item"]], dtype=object)
    models = [
        LGBM(
            feature_names=["user_id", "video_id"],
            num_boost_round=10,
            min_data_in_leaf=1,
            seed=8,
        )
        for _ in range(2)
    ]

    for model in models:
        model.fit(X, y, None, None, groups=(X[:, 0], None))

    first = models[0].predict(target)
    second = models[1].predict(target)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1,)
    assert np.isfinite(first).all()


def test_lambdarank_stably_groups_users_and_preserves_prediction_order(monkeypatch):
    import lightgbm as lgb

    captured = {}

    class Booster:
        best_iteration = 3

        def predict(self, matrix, num_iteration=None):
            return np.asarray(matrix)[:, 0]

    def fake_train(params, train_set, num_boost_round, valid_sets, callbacks):
        train_set.construct()
        valid_sets[0].construct()
        captured["objective"] = params["objective"]
        captured["train_group"] = train_set.get_group().tolist()
        captured["train_labels"] = train_set.get_label().tolist()
        captured["validation_group"] = valid_sets[0].get_group().tolist()
        return Booster()

    monkeypatch.setattr(lgb, "train", fake_train)
    X_train = np.asarray(
        [["u2", "v1"], ["u1", "v2"], ["u2", "v3"], ["u1", "v4"], ["u3", "v5"]],
        dtype=object,
    )
    y_train = np.asarray([0, 1, 1, 0, 0], dtype=float)
    X_val = np.asarray([["u2", "v1"], ["u1", "v2"], ["u2", "v3"]], dtype=object)
    y_val = np.asarray([0, 1, 1], dtype=float)
    model = LGBM(
        loss="lambdarank",
        feature_names=["user_id", "video_id"],
        num_boost_round=5,
        seed=2,
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        groups=(X_train[:, 0], X_val[:, 0]),
    )
    scores = model.predict(np.asarray([["u3", "v5"], ["u2", "v1"], ["u1", "v2"]]))

    assert captured == {
        "objective": "lambdarank",
        "train_group": [2, 2, 1],
        "train_labels": [1.0, 0.0, 0.0, 1.0, 0.0],
        "validation_group": [1, 2],
    }
    np.testing.assert_array_equal(scores, [2, 0, 1])


# --- C8: lambdarank group boundaries must equal user boundaries -------------

def test_group_boundary_assertion_accepts_correctly_grouped_users():
    LGBM._assert_group_boundaries(np.array(["u1", "u1", "u2", "u2", "u2"]), [2, 3])


@pytest.mark.parametrize(
    ("users", "counts", "message"),
    [
        (["u1", "u2", "u1", "u2"], [2, 2], "spans 2 users"),
        (["u1", "u1", "u2", "u2"], [4], "spans 2 users"),
        (["u1", "u2", "u1"], [1, 1, 1], "more than one lambdarank group"),
        (["u1", "u1", "u2"], [2, 2], "sum to 4 but there are 3 rows"),
        (["u1", "u1", "u2"], [2, 1, 0], "must all be positive"),
    ],
)
def test_group_boundary_assertion_fires_on_mis_grouped_input(users, counts, message):
    """LightGBM takes `group` positionally and validates nothing beyond the
    total, so every one of these mis-groupings would otherwise train silently:
    ranks optimised across user boundaries, with a quietly worse score as the
    only symptom."""
    with pytest.raises(ValueError, match=message):
        LGBM._assert_group_boundaries(np.array(users), counts)


def test_fit_actually_enforces_the_group_boundary_assertion(monkeypatch):
    """The assertion must guard the real fit() path, not sit unreferenced.

    AUDIT.md finding 1 was a regression test that passed with its own bug
    reinstated because it exercised a helper nothing called. This drives the
    production path with a deliberately corrupted grouping — `_group_order`
    returning a correct row order but counts that describe a different
    grouping — which is precisely the inconsistency the assertion exists to
    catch and which nothing else in fit() would notice.
    """
    X_train = np.asarray(
        [["u2", "v1"], ["u1", "v2"], ["u2", "v3"], ["u1", "v4"]], dtype=object
    )
    y_train = np.asarray([0, 1, 1, 0], dtype=float)

    # Bind the original before patching, or the replacement recurses into
    # itself rather than delegating.
    real_group_order = LGBM._group_order

    # Correct order (sorted by user), but counts claim four single-row groups.
    def corrupted_group_order(user_ids):
        order, _counts = real_group_order(user_ids)
        return order, [1, 1, 1, 1]

    monkeypatch.setattr(LGBM, "_group_order", staticmethod(corrupted_group_order))
    model = LGBM(loss="lambdarank", feature_names=["user_id", "video_id"], num_boost_round=3)

    with pytest.raises(ValueError, match="more than one lambdarank group"):
        model.fit(X_train, y_train, None, None, groups=(X_train[:, 0], None))


def test_the_real_group_order_satisfies_the_assertion_it_is_checked_against():
    """Guards against the assertion and the producer drifting apart: whatever
    `_group_order` returns must, by construction, pass the check `fit` applies
    to it."""
    users = np.array(["u3", "u1", "u2", "u1", "u3", "u3", "u2"])

    order, counts = LGBM._group_order(users)

    LGBM._assert_group_boundaries(users[order], counts)
