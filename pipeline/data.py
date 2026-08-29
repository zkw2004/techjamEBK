"""Fixed splits, expanding internal folds, date-order assertions,
negative-sampling strategies.

Contract: AGENT_PLAN.md Section 8.2 (FROZEN — do not rename or restructure).
Owner: Workstream B (Malvika). Tasks B1, B2, B7.

Trap 1 (Section 11) lives here: never shuffle, never k-fold, never re-split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame  # swap to polars if that is the Day-0 decision

# --- Official split boundaries (organiser-fixed, Section 4.3) ---------------
TRAIN_END = "2022-04-21"
VAL_START, VAL_END = "2022-04-22", "2022-04-28"
TEST_START = "2022-04-29"

# Expected row counts — B1 asserts these exactly.
N_TRAIN, N_VAL, N_TEST = 1_141_112, 124_909, 170_588

# The scored label, read from the shipped starter kit (data.py:5), NOT from
# the plan's prose — which says `click` and is wrong. See pipeline/features.py.
LABEL = "long_view"

# The five categorical fields the official FM baseline uses. The organisers
# measured that adding CWM's full 13 fields scores 0.5940 vs 0.5950 — inside
# noise, slightly worse. Start here; extra static fields are a known dead end.
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]

# Source log files, read in this order. Row order within a split is the
# original file order after date filtering — this is what row_id indexes.
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)

# `make data` extracts the organiser archive to this exact location.  Keep the
# raw dataset outside version control, but make the local loader agree with the
# documented layout rather than looking for CSVs directly under `data/`.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "KuaiRand-Pure" / "data"

# Randomised-exposure log: 1.18M rows dated 2022-04-22 to 2022-05-08, i.e.
# ENTIRELY after the training cutoff. Usable as an unbiased validation set
# only. Cannot be used for training-time debiasing (see B8 note in README).
RANDOM_LOG_FILE = "log_random_4_22_to_5_08_pure.csv"

# --- Internal folds (Section 6.2): expanding window, always train-earlier ---
INTERNAL_FOLDS = [
    ("2022-04-08", "2022-04-15", "2022-04-16", "2022-04-17"),
    ("2022-04-08", "2022-04-17", "2022-04-18", "2022-04-19"),
    ("2022-04-08", "2022-04-19", "2022-04-20", "2022-04-21"),
]  # (train_start, train_end, val_start, val_end)


def _assert_split_order(
    train: DataFrame,
    val: DataFrame,
    test: DataFrame,
) -> None:
    """Fail if the official splits are not strictly chronological."""
    assert train["date"].max() < val["date"].min() < test["date"].min()


def load() -> tuple[DataFrame, DataFrame, DataFrame]:
    """Returns (train, val, test) in the organisers' exact row order.

    MUST assert train.date.max() < val.date.min() < test.date.min().
    MUST NOT shuffle, resample, or re-split under any circumstances.
    Row order must match starter-kit data.load() so row_id aligns.
    """
    video_path = DATA_DIR / "video_features_basic_pure.csv"

    video_features = pd.read_csv(
        video_path,
        usecols=["video_id", "author_id", "tag"],
        dtype=str,
    )

    vid2author = dict(
        zip(video_features["video_id"], video_features["author_id"], strict=True)
    )
    vid2tag = dict(zip(video_features["video_id"], video_features["tag"], strict=True))

    frames = []

    for filename in LOG_FILES:
        df = pd.read_csv(DATA_DIR / filename)
        frames.append(df)

    rows = pd.concat(frames, ignore_index=True)

    video_ids = rows["video_id"].astype(str)
    rows["author_id"] = video_ids.map(vid2author).fillna("UNK")
    # Tags are static, pre-impression metadata used by B4's user-tag affinity.
    # Keep missing tags missing so the feature can apply its neutral fallback.
    rows["tag"] = video_ids.map(vid2tag)

    dates = pd.to_datetime(rows["date"].astype(str), format="%Y%m%d")

    train = rows.loc[dates <= TRAIN_END].copy()

    val = rows.loc[(dates >= VAL_START) & (dates <= VAL_END)].copy()

    test = rows.loc[dates >= TEST_START].copy()

    # B1 safety check: official splits must be strictly chronological.
    _assert_split_order(train, val, test)

    # B1 acceptance check: organiser-fixed row counts.
    assert len(train) == N_TRAIN, f"Expected {N_TRAIN} train rows, got {len(train)}"
    assert len(val) == N_VAL, f"Expected {N_VAL} val rows, got {len(val)}"
    assert len(test) == N_TEST, f"Expected {N_TEST} test rows, got {len(test)}"

    return train, val, test


def internal_folds() -> list[tuple[DataFrame, DataFrame]]:
    """Three expanding-window folds inside the training period.

    MUST assert fold_train.date.max() < fold_val.date.min() for each.
    Used for screening, early stopping, blend weights, and EB
    hyperparameters. NEVER touches the official validation window.
    """
    train, _, _ = load()
    dates = pd.to_datetime(train["date"].astype(str), format="%Y%m%d")
    official_val_start = pd.Timestamp(VAL_START)
    folds = []

    for train_start, train_end, val_start, val_end in INTERNAL_FOLDS:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        val_start_ts = pd.Timestamp(val_start)
        val_end_ts = pd.Timestamp(val_end)

        fold_train_mask = dates.between(train_start_ts, train_end_ts, inclusive="both")
        fold_val_mask = dates.between(val_start_ts, val_end_ts, inclusive="both")

        # Boolean selection preserves the organiser's original row order.
        fold_train = train.loc[fold_train_mask].copy()
        fold_val = train.loc[fold_val_mask].copy()
        fold_train_dates = dates.loc[fold_train_mask]
        fold_val_dates = dates.loc[fold_val_mask]

        assert not fold_train.empty, f"Empty internal training window: {train_start}–{train_end}"
        assert not fold_val.empty, f"Empty internal validation window: {val_start}–{val_end}"
        assert fold_train_dates.max() < fold_val_dates.min()
        assert fold_val_dates.max() < official_val_start

        folds.append((fold_train, fold_val))

    return folds


# --- Negative sampling (Section 6.5, B7) -------------------------------------
#
# "Which non-clicks to train against" is its own axis, independent of model
# class and loss framing (Section 6.5): Config.negative_sampling selects one
# of three strategies for which negative (LABEL == 0) rows join every
# positive row in a training frame. KuaiRand ships no explicit session id,
# so `tab` (the app tab/scenario, Section 6.8) is the closest real signal to
# a feed-context boundary and stands in for one here — documented, not
# invented: it is one of the five official FIELDS already in FM.fit()'s path.
NEGATIVE_SAMPLING_STRATEGIES = ("all", "in_session", "pop_weighted")

# BPR and similar pairwise setups commonly sample a handful of negatives per
# positive rather than one; 4 is a conventional starting point. Screen
# alternatives on the internal folds (6.2), never on the official validation
# window — same rule as alpha and half-life in B5.
DEFAULT_NEGATIVES_PER_POSITIVE = 4.0


def _positions(mask: pd.Series) -> np.ndarray:
    """0-based row positions where a boolean Series is True, in row order."""
    return np.flatnonzero(mask.to_numpy())


def _sample_in_session_negatives(
    train_df: DataFrame,
    positive_mask: pd.Series,
    negatives_per_positive: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Positions of negatives sharing a (user_id, date, tab) session with at
    least one of that user's positives, capped per user at
    `negatives_per_positive * that user's positive count`."""
    frame = pd.DataFrame(
        {
            "_position": np.arange(len(train_df)),
            "_user_id": train_df["user_id"].to_numpy(),
            "_session": list(
                zip(
                    train_df["user_id"].to_numpy(),
                    train_df["date"].to_numpy(),
                    train_df["tab"].to_numpy(),
                    strict=True,
                )
            ),
            "_is_positive": positive_mask.to_numpy(),
        }
    )
    positives = frame.loc[frame["_is_positive"]]
    positive_sessions_by_user = positives.groupby("_user_id")["_session"].apply(set)
    positive_counts_by_user = positives.groupby("_user_id").size()

    selected: list[int] = []
    negatives = frame.loc[~frame["_is_positive"]]
    for user_id, group in negatives.groupby("_user_id", sort=False):
        sessions = positive_sessions_by_user.get(user_id)
        if not sessions:
            continue
        eligible = group.loc[group["_session"].isin(sessions), "_position"].to_numpy()
        if len(eligible) == 0:
            continue
        budget = int(round(negatives_per_positive * positive_counts_by_user[user_id]))
        if budget <= 0:
            continue
        if len(eligible) > budget:
            eligible = rng.choice(eligible, size=budget, replace=False)
        selected.extend(eligible.tolist())
    return np.asarray(selected, dtype=np.int64)


def _sample_pop_weighted_negatives(
    train_df: DataFrame,
    positive_mask: pd.Series,
    negatives_per_positive: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Positions of negatives drawn without replacement from the whole
    frame, weighted by each row's video's training-set impression count.

    A popular-but-unclicked video is a harder, more informative negative
    than a uniformly drawn one (Section 6.5) — the inverse of down-weighting:
    popular items are *favoured* as negatives, not penalised.
    """
    negative_positions = _positions(~positive_mask)
    n_positive = int(positive_mask.sum())
    budget = min(int(round(negatives_per_positive * n_positive)), len(negative_positions))
    if budget <= 0:
        return np.asarray([], dtype=np.int64)

    video_ids = train_df["video_id"].to_numpy()
    video_counts = pd.Series(video_ids).value_counts()
    weights = pd.Series(video_ids[negative_positions]).map(video_counts).to_numpy(dtype=np.float64)
    weight_total = weights.sum()
    probabilities = weights / weight_total if weight_total > 0 else None

    chosen = rng.choice(negative_positions, size=budget, replace=False, p=probabilities)
    return np.sort(chosen)


def sample_negatives(
    train_df: DataFrame,
    strategy: str = "all",
    negatives_per_positive: float = DEFAULT_NEGATIVES_PER_POSITIVE,
    seed: int = 42,
) -> DataFrame:
    """Select which negative rows join every positive row in training.

    Every positive row (`LABEL == 1`) is always kept. Which negative rows
    (`LABEL == 0`) join them depends on `strategy` (Config.negative_sampling,
    Section 8.6):

    - `"all"` (default): every negative impression is kept, in original row
      order — plain pointwise training over the whole frame.
      `negatives_per_positive` is ignored.
    - `"in_session"`: a negative is eligible only if it shares a
      (user_id, date, tab) session with at least one of that same user's
      positives; up to `negatives_per_positive` are then sampled uniformly
      per user, without replacement.
    - `"pop_weighted"`: negatives are drawn without replacement from the
      whole frame, `negatives_per_positive` per positive overall, with
      inclusion probability proportional to each negative's video's
      training-set impression count.

    Sampling is seeded and therefore reproducible for a fixed `train_df`.
    Returns a new frame in ascending original-row-order (never mutates
    `train_df`, never reorders — matching B1's row-order contract). Raises
    `ValueError` for an unrecognised strategy, a non-positive
    `negatives_per_positive`, an empty `train_df`, or a `train_df` missing a
    column the chosen strategy requires.
    """
    if strategy not in NEGATIVE_SAMPLING_STRATEGIES:
        raise ValueError(
            f"unknown negative_sampling strategy {strategy!r}; "
            f"expected one of {NEGATIVE_SAMPLING_STRATEGIES}"
        )
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    if train_df.empty:
        raise ValueError("cannot sample negatives from an empty training frame")
    if LABEL not in train_df.columns:
        raise ValueError(f"train_df is missing the label column {LABEL!r}")

    if strategy == "all":
        return train_df.copy()

    required = {
        "in_session": ("user_id", "date", "tab"),
        "pop_weighted": ("video_id",),
    }[strategy]
    for column in required:
        if column not in train_df.columns:
            raise ValueError(f"{strategy!r} negative sampling requires a {column!r} column")

    positive_mask = train_df[LABEL].astype(bool)
    rng = np.random.default_rng(seed)
    if strategy == "in_session":
        negative_positions = _sample_in_session_negatives(
            train_df, positive_mask, negatives_per_positive, rng
        )
    else:
        negative_positions = _sample_pop_weighted_negatives(
            train_df, positive_mask, negatives_per_positive, rng
        )

    keep_positions = np.union1d(_positions(positive_mask), negative_positions)
    return train_df.iloc[keep_positions].copy()
