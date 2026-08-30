# Ethan Workstream Progress

Working status for Workstream C: experiment execution, models, losses, tuning,
and blending. Read `ethannotes.md` for the full technical brief and
`AGENT_PLAN.md` for frozen contracts.

## How to update this file

Update the date, task table, current blockers, and next action whenever a C task
changes state. Add one short entry to the update log after a meaningful code or
validation milestone. Record measured results and commands; label runtime-saving
estimates explicitly rather than presenting them as measured speedups.

## Current status

- Last updated: 2026-08-30
- Branch: `codex/continue-c-workflow`
- Active task: OpenMP backend isolation fixed; cross-model integration suite added
- Implementation commits: `3fb2186` (C4-C7), `b081f1c` (latest-main integration),
  local only; not pushed or merged into local `main`.
- Integration base: `60cb273`, the latest fetched `main` when this branch began.
  Handoff commits `6c78c2d` and `a8fbc9a` were explicitly carried forward.
- Latest integration: fetched `origin/main` at `44c05e5` with a clean working
-  tree; merge was reconciled on this branch. Four overlapping files were
  reconciled, retaining both branches' model tests and C-workflow safeguards.
  No reset, stash, discard, or unrelated branch switch was used.

## Task tracker

| Task | Status | Evidence or next gate |
|---|---|---|
| C1 Experiment runner | Complete locally | Integrated with the pushed B1-B3 Pandas, label, fold, and feature-registry contracts. The merged repository test suite and Ruff pass. |
| C1b Self-audit battery | Complete locally | `pipeline/audit.py`: schema, determinism, label-shuffle canary, and evidence checks; `tools/leak_demo.py` exits 0 with safe=accepted, leaky=quarantined. 10 tests in `tests/test_audit.py`. |
| C2 Random and Popularity | Complete locally | Random five-seed validation mean `0.4833884`; Popularity validation primary `0.5807219`. Both pass the reference tolerances and C1 full-tier execution. |
| C3 FM baseline | Complete locally | Organiser-equivalent validation primary `0.6014695`, within the `0.6016 ± 0.0008` gate. Leakage-safe C1 full tier scores `0.6006120`. |
| C3b Evidence records | Complete locally | `pipeline/evidence.py`: full/confirm results carry config, fidelity, seed, fold and segment metrics, hypothesis, disproof condition, and a pass/fail decision; records are pure functions of (config, fidelity, seed). 7 tests. |
| C4b Generated features | Complete locally | `pipeline/codegen.py`: syntax → schema → leakage → smoke → screen → full gauntlet; safe `user_author_affinity` accepted end to end, leaky twin quarantined by the dynamic probe. 8 tests. |
| C4 LightGBM | Complete locally; runnable only after the OpenMP fix | Pointwise and LambdaRank both pass real internal-fold screens; group sorting, original prediction order, deterministic encoding, and fold-selected refit budgets are tested. Real full-tier `0.599210` pointwise / `0.597906` LambdaRank — both **below** the FM baseline, see the ladder note below. |
| C5 DeepFM | Complete locally | Deterministic CPU model uses patience `1` and the O(n) interaction identity; a one-epoch real screen completed in `5.61s`. |
| C6 Optuna | Implementation complete; performance gate unmet | Real 20-trial study finished, 8 pruned, 19.04% estimated time saved (below 30%). Actual owner-kill/resume and bounded native-failure containment tested. A-side dispatch pending. |
| C7 Blending | First pass complete; real lift pending | Four methods, all C1 tiers, parent/fold/official/bootstrap gates, cache provenance and C3b evidence integration tested. Requires two accepted full-config parents for real lift validation. |

## Critical fix: OpenMP backend isolation (2026-08-30)

**Every LightGBM experiment was failing on macOS, and the test suite could not
finish.** torch ships its own `libomp.dylib` while LightGBM links the system
one; a process that loads both aborts with `OMP: Error #15` in either import
order. `_seed_everything` imported torch unconditionally, so each C4 run
imported torch and then LightGBM and was killed by SIGABRT.

Why it went unnoticed:

- C1 saw only a closed pipe and classified the death `transient` — the one
  class A5 retries with backoff, so the agent would have spent its whole
  budget re-running an experiment that could never succeed.
- CI is `ubuntu-latest`, where one shared libgomp makes the conflict
  impossible. CI stayed green while every Mac aborted, including the machine
  the demo runs on.
- No test had ever run two model families in one session; each family's
  tests passed in isolation.

Fix:

- Models declare `native_backend` (`pipeline/models/__init__.py`);
  `lgbm` → `lightgbm`, `deepfm`/`deepfm_mtl` → `torch`, the numpy models
  declare nothing and can join any process.
- `_seed_everything(seed, config)` seeds torch only for runs that need it.
  Every production call site now passes its config — `_run_confirm`,
  `tune._fold_worker`, and `blending.run_blend` included, since a config-less
  call would silently leave DeepFM unseeded and break determinism rather than
  crash.
- `_assert_single_backend` refuses a config needing both runtimes (a blend of
  LightGBM and DeepFM parents) with a `BackendConflictError`, classified
  `schema` so A5 does not retry a permanently impossible run.
- `_child_death_report` names the killing signal and the likely cause instead
  of surfacing a bare `EOFError`.
- `tests/conftest.py` runs `native_backend`-marked tests in a forked child,
  and `tests/test_deepfm.py` imports torch lazily — a module-level import
  loaded torch into the pytest parent, which every later fork inherited.

Verification on 2026-08-30:

```text
.venv/bin/pytest        404 passed, exit 0   (was: hard SIGABRT crash mid-run)
.venv/bin/ruff check .  exit 0
```

## Real-data validation, 2026-08-30 (full tier, seed 42)

Reference gates re-confirmed after the fix, and C4 measured for the first time:

```text
C2 random              primary=0.484473  gauc=0.5015  ndcg=0.4675
C2 popularity          primary=0.580722  gauc=0.6387  ndcg=0.5227
C3 fm                  primary=0.601684  gauc=0.6673  ndcg=0.5361   42s
C4 lgbm pointwise      primary=0.599210  gauc=0.6639  ndcg=0.5346   99s
C4 lgbm lambdarank     primary=0.597906  gauc=0.6619  ndcg=0.5339   61s
```

FM at `0.601684` sits inside the `0.6016 ± 0.0008` gate. **LightGBM
underperforms FM on the five raw categorical fields** (−0.0025 pointwise),
which matches the plan's own warning in Section 6.7: trees cannot represent
the user×video interaction directly, so GBDT quality is almost entirely a
function of feature quality.

### C4b generated-feature lift benchmark (the pending innovation number)

`gen_user_author_affinity` passed the full gauntlet on real data — syntax,
schema, leakage (static scan + outcome-corruption probe), smoke, screen,
full. Measured lift, LightGBM pointwise, full tier:

```text
lgbm + raw ids                      primary=0.599210
lgbm + B4 aggregates                primary=0.599550   delta +0.000340
lgbm + aggregates + gen affinity    primary=0.599838   delta +0.000288
```

**Honest reading: neither delta clears the noise floor.** Seed std is
`0.0008` and `MIN_DELTA_FLOOR` is `0.002`, so both changes are inside noise
and the C3b decision records them as `fail` — the stated disproof condition
fires. The containment story is proven (a generated feature ran end to end
under audit, and its leaky twin is still quarantined); the *lift* story is
not. Do not present these numbers as an improvement.

Consequence for the ladder: on this dataset FM currently beats a lightly
tuned LightGBM, so effort is better spent on features and the agent loop than
on GBDT tuning. Section 6.4's probe ordering should be revisited with these
numbers rather than assumed.

## Live agent loop verified end to end, 2026-08-30

The A3 proposal path had only ever run against an injected fake client, so
its live behaviour was unverified. Running it for real surfaced two blockers
and then produced a complete closed loop.

Blockers found and fixed:

- `anthropic` was listed in `pyproject.toml` but **absent from the venv**, so
  `propose()` would have died with `ImportError` on its first real call.
- `_get_client()` built a bare `anthropic.Anthropic()`. Identity-linked API
  keys are rejected with a 400 — *"anthropic-workspace-id is required"* —
  unless the request names its workspace. It now forwards
  `ANTHROPIC_WORKSPACE_ID` as a header when set, and behaves exactly as
  before when unset. Two tests in `tests/test_propose.py` cover both paths;
  a mocked client could never have caught this.

Live proposal (claude-opus-5, 662 in / 1723 out / 4522 cache-write), given
the real n001-n004 history: it chose `objective` — the family with zero
coverage — and proposed LightGBM lambdarank as a clean single-variable test
against the pointwise parent, **stating its own disproof condition**: *"if it
lands within 0.002 of n004, the pointwise/ranking mismatch is not a real
source of headroom here and the loss-framing branch should be abandoned."*

Executing exactly that config on real data, full tier, seed 42:

```text
proposed lgbm lambdarank   primary=0.598165  gauc=0.6623  ndcg=0.5340
pointwise parent (n004)    primary=0.599210
delta                      -0.001045   (floor 0.002)
C3b decision               fail — not promoted, 15 segment metrics recorded
```

**The agent's own disproof condition fired.** The loss-framing branch is
falsified on this dataset: lambdarank does not beat pointwise here. That is a
real result to report, not a failure of the harness — propose → execute →
evidence → falsification ran unattended and refused to promote a change that
did not clear the noise floor.

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

## C3 completed locally

- Ported the organiser's NumPy factorization machine with five categorical
  fields, `k=16`, Adam updates, batch size `8192`, and patience `4`.
- Added train-fitted first-seen categorical vocabularies and reserved unknown
  slots for every field.
- Added train-only duration deciles for the official `dur_bucket` field.
- Added deterministic, unseen-category, learning-direction, runner-integration,
  and dataset-backed reference tests.
- Kept official validation labels out of production fitting. The final C1 refit
  uses the median best epoch selected by the three internal temporal folds.

Measured results on 2026-08-29:

```text
Organiser-equivalent validation primary: 0.6014695
C1 full-tier validation primary:         0.6006120484001147
C1 full-tier GAUC:                       0.6657969708252829
C1 full-tier nDCG@5:                     0.5354271259749465
Internal-fold best epochs:               [9, 6, 5]
Final leakage-safe refit epoch budget:   6
C1 full-tier wall time:                  41.02 seconds
```

## Innovation layer completed locally (C1b, C3b, C4b)

New modules (additive; no frozen contract was renamed or restructured):

- `pipeline/audit.py` — C1b self-audit battery: result-schema validity,
  determinism (two identical runs compared field by field), the label-shuffle
  canary (screen tier on permuted fold-training labels; the score must
  collapse to chance), and evaluation-evidence completeness. Every check
  returns `{check, passed, detail}` with an actionable message.
- `pipeline/evidence.py` — C3b ledger evidence: `run_with_evidence()` packages
  a full/confirm result with config, fidelity, seed, per-fold metrics,
  segment metrics, hypothesis, disproof condition, and a pass/fail decision
  citing its evidence. `canonical()` strips telemetry so records of the same
  (config, fidelity, seed) compare equal. `to_node()` shapes records for the
  append-only store; extra keys ride along without touching the 8.7 schema.
- `pipeline/codegen.py` — C4b generated-feature slice: `vet_generated_feature()`
  runs emitted source through syntax → schema → leakage → smoke → screen →
  full and returns accepted/rejected/quarantined with per-stage details.
  Ships `USER_AUTHOR_AFFINITY_SOURCE` (strict per-row date cutoff via
  `merge_asof(allow_exact_matches=False)`, EB smoothing alpha=20) and a
  deliberately leaky twin whose obfuscated target-label read evades a static
  grep and is caught by the dynamic outcome probe.
- `tools/leak_demo.py` — containment demo; synthetic by default, `--real` for
  the local extract; exit 0 iff safe=accepted and leaky=quarantined.

Stubs filled to their documented reference specs (signatures unchanged):

- `pipeline/features.py::leakage_check` (B6, Appendix A.2) — static source
  scan (fails closed on unauditable code; generated features attach
  `__leak_source__`), train-label-shuffle probe, and a target-outcome
  permutation probe, all on fixed-size samples.
- `agent/gate.py::accept` (D3, Appendix A.3) — bootstrap over users with
  per-user GAUC/nDCG precomputation; labels recovered from the fixed official
  split, since the frozen signature carries none.
- `agent/gate.py::segments` (D4) — primary by segment level; labels via the
  reserved `meta["labels"]` key or the official split.
- `pipeline/train.py` — full/confirm now populate the `segments` dict already
  required by the frozen 8.5 result contract (activity quartile, popularity
  quartile, day; quartiles fitted train-side); confirm averages them across
  seeds. Synthetic fixtures without meta columns still get `{}`.

Verification on 2026-08-29:

```text
.venv/bin/pytest         173 passed, 3 skipped (remaining task todos), exit 0
.venv/bin/ruff check .   exit 0
python -m tools.leak_demo  exit 0 — safe ACCEPTED, leaky QUARANTINED
python -m tools.leak_demo --real  exit 0 — same outcomes on the local
  KuaiRand-Pure extract; full-tier primary 0.4845 with the demo's random
  base model (at the Random reference, as expected — the feature's lift is
  measured when C4 LightGBM consumes it)
```

46 of the passing tests are new or were converted from `todo` skips:
`test_audit.py` (10), `test_codegen.py` (8), `test_evidence.py` (7),
`test_gate.py` (7), `test_leakage.py` (B6 todos + 2 hardening cases).
Numeric benchmark gates for the innovation layer on the real dataset (e.g.
whether `gen_user_author_affinity` lifts LightGBM) are pending C4; all
behaviour is verified on synthetic fixtures, and `tools/leak_demo.py --real`
exercises the gauntlet against the local extract.

## C4 completed locally

- Implemented deterministic LightGBM pointwise and LambdaRank objectives.
- Added train-fitted categorical vocabularies with unknown slots and native
  categorical feature indices; no one-hot encoding.
- Added stable per-user sorting and exact group counts for LambdaRank while
  preserving original target-row prediction order.
- Passed train and validation user IDs through the frozen model `groups` slot.
- Reused fold-selected `best_epoch` as the final `num_boost_round` budget.
- Installed the declared LightGBM dependency and its macOS `libomp` runtime
  locally; no environment artifact is tracked by Git.

Measured results on 2026-08-29:

```text
Baseline gates: Random, Popularity, FM       3 passed
C4 focused RED -> GREEN tests                6 passed
Pointwise screen primary                     0.5665747082411168
Pointwise screen fold primaries              [0.5893355741603625, 0.5622036788518762, 0.5481848717111115]
LambdaRank screen primary                    0.5709339734832006
LambdaRank screen fold primaries             [0.5925785156933101, 0.565158303147914, 0.5550651016083779]
Pointwise full primary                       0.5916666236670112
Pointwise full GAUC / nDCG@5                 0.6528876578120902 / 0.5304455895219322
Generated user-author affinity full primary  0.5916666236670112
Generated-feature delta                      0.0 (hypothesis disproved; do not promote)
Generated-feature containment                ACCEPTED through syntax/schema/leakage/smoke/screen/full
```

## C5 completed locally

- Implemented a deterministic PyTorch DeepFM with per-field embeddings,
  linear terms, MLP, and raw-logit output.
- Implemented the FM second-order term with the O(n) identity rather than a
  pairwise loop.
- Added train-fitted categorical unknown slots and quantile buckets for
  continuous generated features.
- Added seeded minibatch order, CPU thread control, validation-loss early
  stopping with patience exactly `1`, and fold-selected final epoch support.

Measured results on 2026-08-29:

```text
C5 focused RED -> GREEN tests                3 passed
One-epoch real screen primary                0.4893900293675011
One-epoch real screen GAUC / nDCG@5          0.5162375373837707 / 0.4625425213512315
One-epoch real screen fold primaries         [0.498886420332021, 0.4900394598809318, 0.4792442078895505]
One-epoch real screen wall time              5.607072499988135 seconds
```

This is a budget probe demonstrating the runnable CPU path, not a claim that
the untuned one-epoch DeepFM beats FM.

## C6 implemented and measured (2026-08-30)

- Added `pipeline/tune.py` and `tests/test_tune.py`: seeded single-worker TPE,
  MedianPruner reporting after each B temporal fold, persistent SQLite studies,
  total-budget resume, recorded errors and estimated pruning savings.
- Reuses C1 matrix construction, model fitting and official evaluator; no new
  loaders, splits or metrics. Official validation is not used by the objective.
- Trials execute in bounded children (`hparams.trial_timeout_s`, default 1800s).
  The owner receives each fold metric and decides whether to continue before
  the next fit starts. Native crashes, hangs and invalid scores become evidence.
- SQLite has a single-owner lock and config/search/seed identity guard. A killed
  owner's worker releases inherited descriptors and exits; interrupted attempts
  stay in the ledger but do not consume the resumed trial budget. Tests include
  killing an actual owner subprocess during a fit and resuming the same database.
- The real study's first attempt aborted because two OpenMP runtimes were loaded.
  It is retained as interrupted attempt 0, not hidden or counted as a valid trial.
  No unsafe duplicate-runtime override was used.
- A-side integration remains Kaiwen-owned: `agent.execute` does not yet route
  `type="tune"` to this harness; use `pipeline.tune.run_study` directly.

Real study: `logs/optuna/c6-real-20260830.db`, study `c6-real-20260830`, seed 42,
LightGBM pointwise with `FIELDS`, learning rate 0.05, min-data-in-leaf 50,
early-stopping rounds 5; searched leaves 15–63 and boosting rounds 10–30.

```text
Budget completed                    20 trials (12 completed, 8 pruned)
Ledger attempts                     21 (includes 1 interrupted attempt)
Best internal-fold primary          0.5654609124602713
Best parameters                     num_leaves=62, num_boost_round=20
Measured valid-attempt wall time    352.635093 seconds
Estimated avoided fold time         82.94865025009494 seconds
Estimated saving fraction           0.1904310055999246 (19.04%)
30% saving acceptance gate          NOT MET — pending, not a completion claim
```

The estimate excludes interruption downtime; no extra model evaluations were
run when reopening the study at its already-completed budget of 20.

## C7 first pass implemented (2026-08-30)

- Added four continuous per-user blend methods and Spearman diagnostics in
  `pipeline/models/blend.py`; near-identical parents are refused.
- Added `pipeline/blending.py` for node-based C1 integration. Only successful,
  accepted nodes qualify. Weight selection uses internal-fold labels only.
- Added atomic parent-score caching in `pipeline/train.py`, with cache artifacts
  ignored by Git. Full evaluation reports explicit fold/official/bootstrap gates
  and rejects a candidate that fails to beat both parents.
- Cache keys include implementation/source-file identity; ordered prediction-row
  fingerprints include only pre-outcome context. Stale, reordered, empty or
  corrupt cache artifacts trigger recomputation, without loading pickled arrays.
- Confirmation varies each parent configuration over five seeds. First-pass
  parents must be two distinct accepted successful `full` config nodes; confirmed
  seed-average parents, generated-code parents and nested blends fail explicitly
  rather than silently changing what the named parent represents.
- `pipeline/evidence.py` preserves blend diagnostics and refuses to promote a
  failed C7 gate even if the headline primary beats the stated baseline.
- C7 tests cover exact ranks, scipy-cross-checked Spearman, all four methods,
  similarity rejection, fold-only weight fitting, deterministic full execution,
  five-seed confirmation, each acceptance gate, and cache failure/replay behavior.
- Real blend lift remains pending: `logs/nodes/` has no accepted parent records.
  No synthetic node was presented as a real accepted experiment.

## Current dependencies and limits

- B owns data loading, temporal folds, feature construction, and leakage checks.
  C1 must consume those interfaces without changing their frozen contracts.
- D4 segment metrics are available and reused by full C1 runs.
- The dataset remains local and uncommitted.
- Multi-task DeepFM remains explicitly unimplemented/deprioritized; it cannot
  silently run the single-task model under a multi-task name.
- Smoke now passes no validation labels to model fitting; official-validation
  samples are prediction-only even in this correctness tier.
- The new `origin/main` includes A6/A7, B7/B8 and D-readiness, including overlapping
  LightGBM/DeepFM implementations. Reconcile deliberately before PR/merge; then
  rerun the expanded suite and ensure the A loop carries C7's gate evidence.

## Local macOS runtime

This machine's LightGBM resolves Homebrew `libomp`; PyTorch ships a second
`libomp`. Loading both caused native aborts (exit 134). Both libraries work when
they resolve the same bundled runtime. For this checkout/venv, prepend this
environment setting to tests and experiments:

```sh
DYLD_LIBRARY_PATH=/Users/quekee/Desktop/techjamEBK/.venv/lib/python3.14/site-packages/torch/lib .venv/bin/pytest -ra
```

Do not set `KMP_DUPLICATE_LIB_OK`. The environment setting is local to each
command; no shell profile or system library was changed. Linux does not require
this macOS-specific path.

## Final verification (2026-08-30)

Using the local runtime setting above:

```text
.venv/bin/pytest -ra                    286 passed in 67.33s; exit 0
.venv/bin/ruff check .                  All checks passed; exit 0
git diff --check                       exit 0
.venv/bin/python -m tools.leak_demo     safe ACCEPTED, leaky QUARANTINED; exit 0
.venv/bin/python -m tools.leak_demo --real
                                       safe ACCEPTED, leaky QUARANTINED; exit 0
```

The 286 tests include the real Random, Popularity and FM reference gates.
The real containment demo uses its random base model (full primary `0.4845`);
this is not a feature-lift claim. Frozen plan/evaluator/submit diffs are empty.
Independent review findings were reproduced with regression tests and fixed:
trial crash isolation, live owner-kill recovery, parent-fidelity identity and
corrupt-cache fallback. This verification covers this branch, not the newer
unmerged `origin/main` changes.

## Files changed in the C4–C7 continuation

- Models: `pipeline/models/lgbm.py`, `pipeline/models/deepfm.py`,
  `pipeline/models/fm.py`, `pipeline/models/blend.py`.
- Integration/evidence: `pipeline/train.py`, `pipeline/tune.py`,
  `pipeline/blending.py`, `pipeline/evidence.py`.
- Tests: `tests/test_lgbm.py`, `tests/test_deepfm.py`, `tests/test_models.py`,
  `tests/test_train.py`, `tests/test_tune.py`, `tests/test_blend.py`,
  `tests/test_blend_runner.py`, `tests/test_score_cache.py`, `tests/test_evidence.py`.
- Tracking: `.gitignore`, `ethanprogress.md`. Raw data, databases and score
  archives remain ignored. Frozen evaluator/submit files and `AGENT_PLAN.md`
  were not edited.

## Next action

Finish the running D5 real-data benchmark and record its results below.
Coordinate A-side tune dispatch and blend-gate evidence propagation through
the new A6 loop with Kaiwen (handoff section 11 reserves this hook for A).
Then register genuine accepted parents and measure C7 lift; continue C6
performance work toward the 30% saving gate without changing scores.

## Latest-main integration verification (2026-08-30)

Test runtime: Python 3.12.13 in `.venv-c-workflow`, isolated from the original
Python 3.14 environment. The latest main explicitly requires Python <3.13.
On this Mac, use the same single-OpenMP-runtime workaround with the new path:

```sh
DYLD_LIBRARY_PATH=/Users/quekee/Desktop/techjamEBK/.venv-c-workflow/lib/python3.12/site-packages/torch/lib .venv-c-workflow/bin/pytest -ra
```

- Full suite: **366 passed in 127.04s**, no skips; exit 0.
- `.venv-c-workflow/bin/ruff check .`: all checks passed; exit 0.
- `git diff --check`: exit 0; frozen plan/evaluator/submission files unchanged.
- `.venv-c-workflow/bin/python -m pip check`: no broken requirements.
- Synthetic and real `tools.leak_demo`: safe ACCEPTED, leaky QUARANTINED; exit 0.
- Refreshed official validation reference scores: Random five-seed mean
  `0.483388350987886`; Popularity `0.5807219293342971`; organiser-equivalent
  FM `0.6014687563529677` (within `0.6016 ± 0.0008`). The tiny FM difference
  from the earlier runtime does not change the gate. This reference FM test
  reproduces organiser stopping; production C1 still selects epochs on folds.
- Production C1 full FM (seed 0) also reproduced exactly:
  primary `0.6006120484001147`, folds `[0.6021106663245109,
  0.5709097464276729, 0.5599074916729097]`, `141.69s` during concurrent checks.
- Independent scoped review: two findings reproduced, fixed, retested, and
  approved on re-review. No remaining Critical/Important findings in the merge.
- D5 real screen benchmark with 30 FM tuning trials is running; scores pending.

Files in this integration, relative to the previous local commit:

- Reconciled/continued: `.gitignore`, `ethanprogress.md`,
  `pipeline/models/deepfm.py`, `pipeline/models/lgbm.py`, `pipeline/train.py`,
  `tools/probes.py`, `tests/test_deepfm.py`, `tests/test_lgbm.py`,
  `tests/test_models.py`, `tests/test_train.py`, `tests/test_probes.py`.
- Incoming main preserved: `agent/knowledge.md`, `agent/loop.py`,
  `pipeline/data.py`, `pipeline/features.py`, `pyproject.toml`, `uv.lock`,
  `tools/report.py`, `tests/test_knowledge.py`, `tests/test_leakage.py`,
  `tests/test_loop.py`, `tests/test_negative_sampling.py`,
  `tests/test_randomised_exposure.py`, `tests/test_report.py`,
  `tests/test_select_parent.py`.

## Update log

- 2026-08-30: Verified the live A3 path against the real Anthropic API for
  the first time. Installed the missing `anthropic` dependency and taught
  `_get_client()` to forward `ANTHROPIC_WORKSPACE_ID` for identity-linked
  keys. Ran a full closed loop on real data: the proposal's own disproof
  condition fired, falsifying the loss-framing branch. Suite 406 passed.
- 2026-08-30: Found and fixed the OpenMP backend conflict that made every
  LightGBM experiment fail on macOS and crashed the test suite mid-run.
  Added `tests/test_c_workflow_integration.py` (38 tests): every model family
  × every fidelity tier through the public runner, per-family determinism,
  both native families in one session, parent-process cleanliness, backend
  declarations, mixed-backend refusal, signal-level crash reporting, and an
  AST check that no `_seed_everything` call site omits its config. Suite
  404 passed, Ruff clean. Re-validated the reference gates on real data and
  measured C4 and the C4b lift benchmark for the first time.

- 2026-08-30: Reconciled incoming C4/C5 model interfaces and tests. LightGBM
  accepts direct categorical matrices, dict/tuple user groups and `n_estimators`
  while C1's fold-selected `num_boost_round` wins for final refits. DeepFM
  retains numeric bucketing, fixed-budget refits and explicit MTL deferral,
  and adopts main's lazy Torch loading. Regression run before lazy-load fix:
  40 passed, 1 expected failure; all three main LGBM failures were resolved.
  Added an ignored Python 3.12 `.venv-c-workflow` because main now requires
  Python <3.13; the old `.venv` is untouched. Final verification pending.
- 2026-08-30: Connected B7 sampling to C1 after building full historical
  training features. Validation/test rows and feature history remain unchanged.
  Tests first reproduced ignored `in_session`/`pop_weighted` settings (2 failed,
  1 passed). D5 P5 now uses the existing C6 tuner instead of its duplicate
  objective, preventing `--fidelity full` from tuning on official validation.
  Added SQLite resume and failed-study coverage. Targeted combined verification:
  33 passed in 19.16s. The integrated suite then passed 362 tests in 122.78s.
- 2026-08-30: Independent merge review exposed two interface defects, both
  reproduced before fixing: screen caps overrode a small `n_estimators` budget,
  and P5 study-lock errors prevented the results table. Normalize the budget
  alias before capping; retain P1-P4 and a visible P5 error on ordinary storage
  failures. Covering tests: 9 LightGBM tests passed in 0.70s, 4 probe tests
  passed in 1.15s. Ruff and diff checks passed. Real leakage demo again accepted
  the safe feature and quarantined its twin (not a numeric lift claim).
- 2026-08-30: Implemented C6 with subprocess-isolated fold pruning, real
  owner-kill/resume, identity-checked SQLite storage and measured 20-trial
  evidence. The 30% savings gate is explicitly unmet (19.04% estimated).
- 2026-08-30: Completed the C7 first pass and C3b gate-evidence bridge. Fixed
  smoke-tier early-stopping leakage, hardened score-cache provenance and
  corruption fallback, and preserved both newly advanced remote work and all
  local changes for deliberate reconciliation.

- 2026-08-29: Pulled and integrated the A and B workstreams. Merged
  origin/main (A4 executor, A5 recovery, B4 fixes, B5 smoothing, D1-D2
  manifest), origin/feat/B6-leakage-guard, and origin/feat/A3-propose into
  `feat/3`. Reconciled B6's `leakage_check` (their static-first bool
  contract and NaN-corruption probe kept as authority) with the C4b needs
  (`__leak_source__` recovery that fails closed, head-sampling). Wired
  `execute()` so `family="feature"` code actions register and use the
  generated feature (the exec'd namespace was previously discarded), and
  reclassified guard rejections as `leak_suspected` so A5 quarantines
  instead of repairing them. Suite 235 passed, Ruff clean, leak demo holds.

- 2026-08-29: Implemented the innovation layer: C1b self-audit battery,
  C3b hypothesis-ledger evidence, and the C4b generated-feature vertical
  slice, filling the B6/D3/D4 stubs to their Appendix A.2/A.3 reference
  specs. 173 tests and Ruff pass; `tools/leak_demo.py` (synthetic and
  `--real`) accepts the safe temporal feature and quarantines the leaky twin.
- 2026-08-29: Implemented and verified the local C1 experiment runner. Began
  reconciliation with the pushed B workflow.
- 2026-08-29: Integrated C1 with B's Pandas frames, `LABEL`, internal folds,
  and B3 feature resolver. The merged suite and Ruff pass.
- 2026-08-29: Implemented C2 Random and Popularity models. Both dataset-backed
  reference gates and both C1 full-tier integration runs pass.
- 2026-08-29: Implemented C3 FM, passed the organiser-equivalent `0.6016`
  reference gate, and verified leakage-safe C1 full-tier execution using an
  epoch budget selected only from internal temporal folds.
- 2026-08-29: Implemented C4 LightGBM pointwise and LambdaRank. Both objectives
  pass real internal-fold screens. The safe generated user-author affinity
  passed containment but produced zero full-tier lift, disproving its first
  benchmark hypothesis without promotion.
- 2026-08-29: Implemented C5 DeepFM with deterministic CPU training,
  patience `1`, and the O(n) FM identity. The first one-epoch real-data probe
  completed in `5.61s`; its low score is recorded without promotion.
