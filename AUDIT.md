# Workstream C audit

Adversarial review of the C changes merged in #25 and proposed in #26,
carried out on `audit/c-workflow-review`. The work being audited was written
in the same session, so the guiding question throughout was not "does it
pass?" but **"would this have caught the bug if the bug came back, and is the
number I reported the number reality would give?"**

Four findings. Three are in the *tests* rather than the production fixes —
the fixes were verified against real failures, while the tests guarding them
were not verified against anything. The fourth is a reported metric that does
not survive independent measurement, and it retracts a claim already merged
to `main`.

---

## Method

Three independent checks, each designed to fail if the work is decorative.

1. **Mutation.** Revert each production fix, run only the test that claims to
   guard it, and require that test to fail. A test that passes with the bug
   reinstated is not a regression test.
2. **Fault injection.** Inject the exact fault a test claims to detect (a
   descriptor leak, a memory leak, a zombie child) and require detection.
3. **Ground truth.** Re-derive a reported metric by independent measurement
   rather than trusting the instrument that produced it.

---

## Finding 1 — a regression test that could not regress

**Severity: medium. Status: fixed.**

`test_skipped_folds_are_priced_at_their_own_index_not_the_running_mean`
exercised `_skipped_fold_seconds` as a pure unit. Reverting the *call site* in
`_trial_fold_primaries` back to `mean(fold_seconds) * remaining` left it
green: the helper was tested, but nothing tested that anything used it.

Mutation result before the fix:

```
per-fold-index prune pricing    passed    *** MISSED — test is vacuous ***
```

Added `test_pruned_trials_record_per_index_pricing_not_the_running_mean`,
which drives a real prune through `_trial_fold_primaries` with fold costs
rising by index and asserts the recorded `seconds_saved`. All eight fixes now
have a test that genuinely catches their regression:

```
backend-scoped seeding                 CAUGHT
mixed-backend refusal                  CAUGHT
empty-eval-frame guard                 CAUGHT
non-finite train label guard           CAUGHT
missing blend parent classification    CAUGHT
per-fold-index prune pricing           CAUGHT
score cache eviction                   CAUGHT
native crash reporting                 CAUGHT
```

---

## Finding 2 — the soak test's memory check was decorative

**Severity: medium. Status: fixed.**

The leak assertion used `resource.getrusage().ru_maxrss`, a high-water mark
that never falls, against a flat 150MB bound over 48 iterations. A steady leak
and a single transient spike are indistinguishable under that measure, and
anything under ~3MB per iteration fits inside the bound.

Injecting a 2MB-per-run leak:

```
fd leak (1 pipe/run)       CAUGHT
memory leak (2MB/run)      *** MISSED — soak test is decorative ***
```

96MB of real leak passed. Replaced with current RSS sampled per iteration and
a linear fit, bounded at 0.5 MB/iteration — roughly 20x headroom over the
0.023 MB/iteration observed across 120 real runs, while still failing a
megabyte-scale per-run leak. Both injected faults are now caught.

---

## Finding 3 — the zombie check never ran at all

**Severity: high (as a test defect). Status: fixed.**

`_zombie_children` listed children with `ps -o pid= -P <pid>`. `-P` is an
illegal option on macOS and means "show processor" on Linux, so the helper
returned `[]` on both platforms and `assert not _zombie_children()` passed
unconditionally. It asserted nothing.

Demonstrated by creating a genuine zombie and asking the detector:

```
actual zombie pid : 28037
detected          : []          <- before
detected          : [28066]     <- after the fix
```

Rewritten to parse `ps -eo pid,ppid,stat` and filter on parent pid, which both
platforms understand. Verified against a real zombie, and verified to return
empty once reaped.

This is the finding that most deserved to be caught: the runner *does* reap
its workers correctly, so the broken check and a working one both showed
green. Nothing was wrong except the evidence.

---

## Finding 4 — the C6 gate claim was wrong

**Severity: high. Status: retracted, not yet re-fixed.**

C6 was reported as meeting its 30% pruning gate at 36.87%. That number came
from the estimator the same work had just modified, and was never checked
against an independent measurement. Running the identical 20-trial study
twice — `NopPruner` against `MedianPruner` — and comparing wall clock:

```
UNPRUNED  wall = 568.2s
PRUNED    wall = 433.7s
GROUND TRUTH wall-clock reduction = 23.67%
CLAIMED (estimator)               = 37.59%
discrepancy                       = +13.92 points
```

**The gate is not met.** Pruning saves real time — 23.7% of it — but not the
30% the criterion requires, and the estimator overstates by ~14 points.

Two supporting corrections:

- **The justification for the estimator change was empirically false.** It was
  argued that expanding windows make fold 3 substantially costlier than fold
  1, so pricing skipped folds at the running mean understates savings.
  Measured: `10.028 / 10.504 / 10.599s`, a 5.7% spread inside the ~22%
  run-to-run variation. The estimator is marginally more correct and
  practically inert; the improvement came from `n_startup_trials` 5 → 3.
- **Probable source of the overstatement.** Skipped folds are priced at the
  global mean fold cost. A trial is pruned for being bad, and bad configs here
  tend to be cheap (`max_epochs` is a searched parameter), so a cheap trial is
  credited with average-cost savings it never had.

Left open: the unpruned study reports `measured_trial_seconds` of 3103.5s
against 568.2s of wall clock (5.5x), while the pruned study is exact and a
controlled synthetic study accounts correctly at 0.93x. Unexplained.

Also noted: pruning changes what TPE samples next, so the two studies do not
run identical trials. The comparison is directionally sound, not controlled.

This is the finding that most justifies the audit. A metric was modified and
then reported as passing a gate on the strength of its own arithmetic, in a
pull request that has since been merged. Nothing caught it except measuring
the thing itself.

## Method note: an invalid measurement, and why

The first attempt at the ground-truth measurement above was run while the
full test suite, two mutation audits, and a duplicate copy of the audit script
itself were executing concurrently. Under that contention it reported a 1.69%
saving against a 27.47% ground truth, with per-trial durations 6x to 35x the
study's own wall clock — arithmetically impossible for sequential trials.

A controlled synthetic study on a quiet machine accounts correctly
(`sum(trial.duration)` 5.32s against 5.70s wall clock, ratio 0.93), so the
instrument was sound and the *measurement* was the defect. Repeated on an idle
machine, it produced the figures in Finding 4.

Recorded because it is the same error class the audit was looking for: a
number produced under conditions that invalidate it, reported as if it were
evidence.

---

## Reviewed and found sound

- **Score cache.** Atomic writes, per-frame fingerprints, finite-value checks,
  and a key that already includes a code fingerprint so an implementation
  change invalidates stale entries. Eviction is safe by construction: a miss
  costs a refit, never a wrong answer, and a test asserts the recomputed
  result matches what the cache would have served.
- **Runner input handling.** 17 malformed-input cases, none raising, all
  classified.
- **Process hygiene.** Across 120 real iterations: descriptors flat, workers
  reaped, zero errors.

---

## Known limitations, not fixed

Recorded rather than addressed, because each is a judgement call rather than a
defect.

- **`_evict_score_cache` orders by `st_atime`.** Filesystems mounted
  `noatime`/`relatime` do not update access time on read, degrading LRU toward
  insertion order. Still bounded, just less clever about what it drops.
- **`_config_backends` swallows parent-resolution errors** so backend
  detection can proceed; a malformed parent config yields an incomplete
  backend set and the conflict guard could miss. The error resurfaces
  immediately afterwards in `_parents`, so the window is narrow.
- **`_evaluate` now coerces labels with `np.asarray(..., dtype=float64)`,**
  so a non-numeric label column becomes a schema error where the starter-kit
  evaluator would previously have attempted it. Intentional, but a behaviour
  change.
- **The degenerate-data tie at exactly 0.750.** An all-positive or
  all-negative evaluation set scores exactly 0.75, and the canary fires on
  `> 0.75`, so it passes by an exact tie. Unreachable on real data and `>` is
  what the plan specifies, so the frozen semantics were left alone. Worth a
  team decision, not a unilateral change.
- **The soak test runs 48 iterations on small fixtures.** It detects leak
  *rates*, not absolute exhaustion, and does not cover DeepFM (too slow for
  the suite). A real overnight run remains the only full test.
