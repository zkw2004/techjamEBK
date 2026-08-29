"""Refit on train+val, 5 seeds, write and validate the submission. Task D8.

Validation ends 28 April; test starts 29 April. Refitting the chosen config on
train + validation is temporally legal and hands the model seven extra days of
data directly adjacent to the test window. Do not skip it.

Then: average 5 seeds, write `row_id,user_id,video_id,score`, run
`submit.py --check`, and verify row_id alignment. Join on row_id ONLY —
3.06% of test rows are duplicate (user_id, video_id) pairs (trap 4).
"""

from __future__ import annotations

N_SEEDS = 5
SUBMISSION_COLUMNS = ["row_id", "user_id", "video_id", "score"]


def main() -> None:
    raise NotImplementedError("D8")


if __name__ == "__main__":
    main()
