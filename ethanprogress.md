# Ethan Workstream Progress

Working status for Workstream C: experiment execution, models, losses, tuning,
and blending. Read `ethannotes.md` for the full technical brief and
`AGENT_PLAN.md` for frozen contracts.

## How to update this file

Update the date, task table, current blockers, and next action whenever a C task
changes state. Add one short entry to the update log after a meaningful code or
validation milestone. Record measured results and commands, not estimates.

## Current status

- Last updated: 2026-08-29
- Branch: `feat/C1-run-experiment`
- Active task: Prepare C3 FM baseline implementation
- Working tree: C1 and C2 code and tests are implemented but not committed

## Task tracker

| Task | Status | Evidence or next gate |
|---|---|---|
| C1 Experiment runner | Complete locally | Integrated with the pushed B1-B3 Pandas, label, fold, and feature-registry contracts. The merged repository test suite and Ruff pass. |
| C2 Random and Popularity | Complete locally | Random five-seed validation mean `0.4833884`; Popularity validation primary `0.5807219`. Both pass the reference tolerances and C1 full-tier execution. |
| C3 FM baseline | Ready | Reproduce validation primary near `0.6016` within one seed standard deviation. |
| C4 LightGBM | Blocked by C2 | Run pointwise and LambdaRank; verify per-user groups and internal-fold early stopping. |
| C5 DeepFM | Blocked by C2 | Complete CPU training within ten minutes with patience `1`. |
| C6 Optuna | Blocked by C2 | Run and resume a 20-trial TPE study with `MedianPruner` and SQLite. |
| C7 Blending | Blocked by C2 | Support four blend methods and require improvement over both parents. |

## C1 completed locally

- Added `smoke`, `screen`, `full`, and five-seed `confirm` tiers.
- Added subprocess isolation with bounded timeout termination.
- Added deterministic Python, NumPy, and Torch setup.
- Added structured syntax, schema, timeout, OOM, transient, and leakage errors.
- Added the `primary > 0.75` leakage canary, including per-seed confirmation checks.
- Added score alignment, finite-value, telemetry, and fold-count validation.
- Added safe raw-field restrictions and registered-feature leakage checks.
- Added C1 tests in `tests/test_models.py`, `tests/test_leakage.py`, and
  `tests/test_train.py`.

Verification on 2026-08-29:

```text
.venv/bin/pytest -q     exit 0; future-task acceptance tests remain skipped
.venv/bin/ruff check .  exit 0
git diff --check        exit 0
```

An independent code review found no Critical or Important issues after the fix
round.

## C2 completed locally

- Implemented deterministic seeded predictions for `RandomModel`.
- Implemented prior-20 empirical-Bayes item click rates for `PopularityModel`,
  with the global training click rate as the unseen-item fallback.
- Added synthetic unit tests, C1 smoke integration tests, and dataset-backed
  reference-gate tests.
- Linked the existing local KuaiRand-Pure checkout into the ignored project
  `data/` directory; no dataset files are tracked by Git.

Measured validation results on 2026-08-29:

```text
Random seeds 0-4 primary mean: 0.483388350987886
Popularity primary:            0.5807219293342971
Popularity GAUC:               0.6387257648986705
Popularity nDCG@5:             0.5227180937699238
C1 full tier, Random seed 0:   0.48265983817046704
C1 full tier, Popularity:      0.5807219293342971
```

## Current dependencies and limits

- B owns data loading, temporal folds, feature construction, and leakage checks.
  C1 must consume those interfaces without changing their frozen contracts.
- D4 owns populated segment metrics. C1 currently returns the required empty
  `segments` dictionary until D4 lands.
- The dataset remains local and uncommitted.
- C2's hard gate has passed; C3 may now begin. C4-C7 remain sequentially
  dependent on their preceding implementation and validation gates.

## Next action

Implement the C3 FM baseline through the C1 runner and reproduce validation
primary near `0.6016` before advancing to C4.

## Update log

- 2026-08-29: Implemented and verified the local C1 experiment runner. Began
  reconciliation with the pushed B workflow.
- 2026-08-29: Integrated C1 with B's Pandas frames, `LABEL`, internal folds,
  and B3 feature resolver. The merged suite and Ruff pass.
- 2026-08-29: Implemented C2 Random and Popularity models. Both dataset-backed
  reference gates and both C1 full-tier integration runs pass.
