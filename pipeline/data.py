"""Fixed splits, expanding internal folds, date-order assertions.

Contract: AGENT_PLAN.md Section 8.2 (FROZEN — do not rename or restructure).
Owner: Workstream B (Malvika). Tasks B1, B2.

Trap 1 (Section 11) lives here: never shuffle, never k-fold, never re-split.
"""

from __future__ import annotations

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


def load() -> tuple[DataFrame, DataFrame, DataFrame]:
    """Returns (train, val, test) in the organisers' exact row order.

    MUST assert train.date.max() < val.date.min() < test.date.min().
    MUST NOT shuffle, resample, or re-split under any circumstances.
    Row order must match starter-kit data.load() so row_id aligns.
    """
    raise NotImplementedError("B1")


def internal_folds() -> list[tuple[DataFrame, DataFrame]]:
    """Three expanding-window folds inside the training period.

    MUST assert fold_train.date.max() < fold_val.date.min() for each.
    Used for screening, early stopping, blend weights, and EB
    hyperparameters. NEVER touches the official validation window.
    """
    raise NotImplementedError("B2")
