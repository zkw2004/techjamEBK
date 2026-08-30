# Workstream C audit

Adversarial review of the C changes merged in #25 and proposed in #26,
carried out on `audit/c-workflow-review`. The work being audited was written
in the same session, so the guiding question throughout was not "does it
pass?" but **"would this have caught the bug if the bug came back, and is the
number I reported the number reality would give?"**

Three findings, all in the *tests* rather than the production fixes. That
distribution is itself worth noting: the fixes were verified against real
failures, while the tests guarding them were not verified against anything.

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

## Ground truth: is the C6 36.87% claim real?

The C6 gate was reported as met by an estimator this same work modified, which
is exactly the shape of a gamed metric. Two independent checks:

**Estimator soundness.** Fold cost by index across completed trials, showing
the expanding-window structure the estimator relies on — folds genuinely cost
more as the training window grows, so pricing skipped folds at the running
mean of cheaper earlier folds does understate savings.

**Wall-clock comparison.** The same 20-trial study run with `NopPruner` and
with `MedianPruner`, comparing measured wall clock rather than the estimator's
own arithmetic. See `ethanprogress.md` for the recorded figures.

Independent of the outcome, two properties already argue the accounting is not
inflated: the reported `measured_trial_seconds` of 221.3s matched observed
wall clock of 222s, and 129.2s of saving across 11 pruned trials is 11.7s
each against a ~17.5s full trial — the two-thirds a prune after fold 1 should
yield with three folds.

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
