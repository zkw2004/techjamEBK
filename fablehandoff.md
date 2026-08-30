# fablehandoff — comprehensive continuation brief for the next agent

Written 2026-08-29, end of the session that delivered C3 + the innovation
layer and integrated the A/B workstreams. This document is self-contained:
it tells you where the repo stands, what was decided and why, exactly what
remains, and how to verify anything you touch.

Reading order after this file:

1. `AGENT_PLAN.md` — the specification. **Section 8 contracts are FROZEN**
   (function signatures, dict keys, file paths); Section 11 traps are
   mandatory; Section 9 has per-task acceptance criteria.
2. `ethannotes.md` — Workstream C brief + the innovation thesis and the
   C1b/C3b/C4b acceptance rows.
3. `ethanprogress.md` — measured results log (numbers + commands, never
   estimates). Keep updating it.

---

## 1. Project in one paragraph

TechJam 2026 Track 2. The deliverable is **an agent that builds recommender
systems**, not a recommender: a closed loop that proposes a hypothesis,
configures or writes the code, screens it cheaply on internal temporal
folds, evaluates survivors on the official validation window, statistically
gates promotion, records everything in an append-only node ledger, and
repeats. Dataset: KuaiRand-Pure (~1.4M impressions, 27K users, 7.6K items).
Scored label: **`long_view`** (the plan's prose says `click` — the prose is
wrong; the shipped starter kit wins). Metric: primary = mean(GAUC, nDCG@5),
computed **per user** then averaged. ~60% of the judging rubric is agent
behaviour (autonomy, recovery, insight, resource accounting), not model
quality — invest in the harness and the integrity story, cap model ambition.

Innovation thesis (argued and adopted this session):

> The agent invents temporal features as code, tests them under strict
> containment, records falsifiable hypotheses with evidence, and rejects
> results that appear to cheat — including its own.

Three pillars: **C4b** generated features (the one thing an LLM adds that
Optuna structurally cannot — expanding the search space), **C3b** the
hypothesis ledger (failed experiments are evidence, not noise), **C1b**
anti-self-deception (label-shuffle canary, leak quarantine — the headline
demo: "an agent that can't cheat itself").

---

## 2. Exact current state

### Git

- Branch **`feat/3`** pushed to `origin/feat/3`; working tree **clean**.
- **PR #19 → main is OPEN with green CI** (checks: `secrets` ✓ 25s,
  `tests` ✓ 3m0s). **Not merged** — it crosses ownership lines (see §5),
  so a human/teammate decision is pending. To land it: `gh pr merge 19`.
- `feat/3` is 10 commits ahead of `origin/main`. Highlights:
  - `6e58351` C3 FM + innovation layer (C1b, C3b, C4b) + B6/D3/D4 stub fills
  - `051d72c` merge origin/main (A4, A5, B4 fixes, B5, D1-D2)
  - `fc8a70a` merge feat/B6-leakage-guard (conflict resolved, see §5)
  - `a2a37a0` merge feat/A3-propose (clean)
  - `99cfb8f` execute()→codegen wiring + `leak_suspected` reclassification
- Remote branches already merged to main via PRs: A2, A4 (#11), A5 (#13),
  B1-B5 (#14-#16), D1-D2 (#17). Outstanding remotes now absorbed by #19:
  `feat/A3-propose`, `feat/B6-leakage-guard`. `fix/ci-data-tests` predates
  main and is stale.
- Git identity: eatenquek. Commit trailer required:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- Workflow rules: `main` only via PRs (CI runs on PRs), one branch per task
  id (`feat/C4-lightgbm`, …), small PRs, exclusive file ownership per
  workstream (§9 of the plan).

### Environment

- macOS, Python 3.14 venv at `.venv/` (`pytest`, `ruff==0.16.5`, pandas,
  numpy, lightgbm, torch, optuna, pydantic v2 installed).
- **Real dataset present** at `data/KuaiRand-Pure/data/` (gitignored):
  `log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`,
  `video_features_basic_pure.csv`, plus the randomised-exposure log
  (post-cutoff dates only — unusable for training-time debiasing).
- `multiprocessing` **fork** start method available and relied upon (child
  processes inherit monkeypatches and generated-feature registrations).
  A spawn-only platform breaks that assumption.

### Verification status at handoff

```text
.venv/bin/pytest                     235 passed, 0 skipped, exit 0
.venv/bin/ruff check .               exit 0
python -m tools.leak_demo            exit 0 (safe ACCEPTED, leaky QUARANTINED)
python -m tools.leak_demo --real     exit 0 (same outcomes on real data)
```

### Measured reference numbers (reproduced this session)

| Item | Value | Gate |
|---|---:|---|
| Random, 5-seed validation mean | 0.4833884 | ~0.4753 within noise ✓ |
| Popularity validation primary | 0.5807219 | ~0.5715 within noise ✓ |
| C3 FM organiser-equivalent validation primary | 0.6014695 | 0.6016 ± 0.0008 ✓ |
| C3 FM leakage-safe C1 full tier | 0.6006120 | (fold-selected epochs) |
| C1 full-tier FM wall time | ~41s | baseline ~40s ✓ |
| leak_demo --real full tier (random model + gen feature) | 0.4845 | ≈ Random reference, expected |

Other constants you will need: baseline seed std **0.0008**; promotion floor
**0.002** (`MIN_DELTA_FLOOR`); leak canary **primary > 0.75**; convergence
epsilon 0.002 / N=3; row counts 1,141,112 / 124,909 / 170,588
(train/val/test); submission header `row_id,user_id,video_id,score`, join on
`row_id` ONLY (3.06% of test rows are duplicate (user,video) pairs).

---

## 3. Task ledger — done vs remaining

### DONE (do not redo)

| Task | Where | Notes |
|---|---|---|
| A1 Config/Action schemas | `agent/schema.py` | pydantic, extra="forbid" |
| A2 node store | `agent/store.py` | append-only, atomic, `best_node()` = highest **accepted** primary |
| A3 propose() | `agent/propose.py` | merged via #19; two-tier routing |
| A4 execute() | `agent/execute.py` | + this session's feature-code routing (§5.3) |
| A5 recovery policy | `agent/recovery.py` | schema→1 repair; leak_suspected→quarantine, never retry |
| B1 load(), B2 internal_folds() | `pipeline/data.py` | date-cut, counts asserted |
| B3 registry, B4 features (+in-sample time_ms rework), B5 decay/EB | `pipeline/features.py` | |
| B6 leakage_check | `pipeline/features.py` | reconciled version, see §5.2 |
| C1 run_experiment, 4 tiers | `pipeline/train.py` | subprocess isolation, timeout, determinism, canary |
| C2 Random + Popularity | `pipeline/models/` | reference gates PASSED — downstream unblocked |
| C3 FM port | `pipeline/models/fm.py` | organiser-equivalent, epochs from internal folds |
| **C1b self-audit battery** | `pipeline/audit.py` | see §4.1 |
| **C3b evidence records** | `pipeline/evidence.py` | see §4.2 |
| **C4b generated-feature slice** | `pipeline/codegen.py`, `tools/leak_demo.py` | see §4.3 |
| D1-D2 evaluator copy + manifest/preflight | `pipeline/evaluate.py`, `agent/manifest.py` | evaluate.py/submit.py IMMUTABLE, hashed |
| D3 accept(), D4 segments() | `agent/gate.py` | I filled Pinxin's stubs to Appendix A.3 — flag to them (§5.4) |

Segment metrics (activity quartile / popularity quartile / day) now populate
in every full/confirm result via `pipeline/train.py::_segment_metrics`
(quartiles fitted train-side; returns `{}` when fixture frames lack meta
columns, so synthetic tests still pass).

### REMAINING, in recommended order

1. **Land PR #19** (user/teammate decision). Branch everything below from
   the updated main afterwards.
2. **C4 — LightGBM** (`feat/C4-lightgbm`; owner files `pipeline/models/lgbm.py`,
   `pipeline/train.py`). Acceptance (plan §9.3): both objectives run;
   lambdarank gets a correct per-user `group` array (impression counts per
   user, in row order — build from the training frame after matrix
   construction); early stopping on an internal fold, never official val.
   Practical notes: integer label-encode categoricals (no one-hot — trap 7);
   pass `categorical_feature` for the ID fields; screen-tier caps already
   clamp `num_boost_round`/`n_estimators` (see `SCREEN_BUDGET_CAPS`).
   **Then run the pending innovation benchmark**: LightGBM with vs without
   `gen_user_author_affinity` — this is the number the C4b story needs.
3. **C5 — DeepFM** (`feat/C5-deepfm`). PyTorch, CPU <10 min, patience 1,
   epochs 1-5, FM second-order via the O(n) identity — skeleton in plan
   Appendix A.4. `_seed_everything` already enables deterministic torch.
4. **C6 — Optuna** (`feat/C6-optuna`). TPE + `MedianPruner` + SQLite
   `RDBStorage`; 20 trials; resumable after kill; ≥30% runtime saved by
   pruning; objective evaluated on internal folds. Determinism is
   **per-experiment**, not per-study — run one worker when exact study
   reproduction matters (decision recorded in ethannotes).
5. **C7 — Blending** (`feat/C7-blending`). rank_avg default; per-user
   Spearman diagnostic (>0.95 skip, 0.7-0.9 sweet spot, <0.5 investigate);
   must beat BOTH parents on internal folds AND official metric; weights
   fitted on folds only; put a minimum-improvement margin in
   `Config.hparams` (do NOT change the frozen Config schema) and record it
   in node gates.
6. **A6 main loop + A7 parent selection** (Kaiwen's files — coordinate, do
   not implement unilaterally). Integration points already prepared:
   - `propose.md` prompt should REQUIRE a disproof condition and
     parent-evidence citation (the ethannotes checklist item still open);
     feed both into `pipeline/evidence.py::run_with_evidence`, write nodes
     via `evidence.to_node()` + `store.write()`.
   - Quarantine exclusion is already inherent: `best_node()` only considers
     accepted+ok nodes; quarantined features are deregistered.
   - `execute()` already handles all four action types incl. feature code.
7. **B7 negative sampling / B8 exposure debiasing** (Malvika's). B8 note:
   the randomised log is entirely post-cutoff → validation-side use only.
8. **D5 probes** (`tools/probes.py`) once C4/C5 exist — the five Day-1
   probes; their output reorders the model ladder (§6.7). Then **D6**
   knowledge.md, **D7** report generator, **D8** finalise/refit-on-train+val
   (do not skip the refit — 7 extra adjacent days of data), **D9** README.
9. **Demo assets** (§14): injected-failure demo is scripted deliberately
   (OOM via oversized batch), plus `tools/leak_demo.py` as the
   "can't cheat itself" 30-second segment.

Descope ladder if time runs short (cut top-first): B8, C7, A7, C5, B7, D4.
NEVER cut: B1, B2, B6, C2, D2, D3, D7, D8.

---

## 4. The innovation layer — API surface you'll build on

### 4.1 `pipeline/audit.py` (C1b)

- `audit_result_schema(result, fidelity) -> {check, passed, detail}` —
  validates against the 8.5 shape (wraps `train._result_schema_issue`).
- `audit_determinism(config, fidelity="smoke", seed, timeout_s)` — two runs,
  bit-identical metrics + score vectors required.
- `audit_label_shuffle(config, seed, threshold=0.55, timeout_s)` — patches
  `train._load_folds` to permute **fold-training** labels (fold-val labels
  intact for scoring), runs the **screen** tier (never smoke — smoke has no
  metrics; never official val), expects primary ≈ 0.475. A surviving high
  score means an input carries the label. In-process patch is inherited by
  the fork-started child.
- `audit_evidence(result)` — full/confirm must carry 3 finite fold
  primaries, non-empty segments, aligned non-empty val vectors.
- `self_audit(config, seed)` — runs schema → determinism → shuffle; returns
  `{"passed": bool, "checks": [...]}`.

### 4.2 `pipeline/evidence.py` (C3b)

- `run_with_evidence(hypothesis, disproof_condition, config, fidelity="full",
  seed, baseline_primary=0.6016, timeout_s)` → record with `config`,
  `fidelity`, `seed`, `metrics`, `fold_primaries`, `segments`, `decision`
  (`pass`/`fail`/`error`/`insufficient_evidence` + evidence string + delta
  vs baseline vs `MIN_DELTA_FLOOR`), `reproduce`, `telemetry`.
  Hypothesis and disproof condition are mandatory (raise if blank). Pilot
  fidelities (smoke/screen) are refused. Failed runs are recorded as
  evidence, not discarded.
- `canonical(record)` — record minus telemetry; equal across reruns of the
  same `(config, fidelity, seed)` (tested).
- `to_node(record, parent, family, ...)` — store-writable node; extra keys
  (`disproof_condition`, `decision`, `seed`) survive `store.normalise()`.
- Pass the parent node's primary as `baseline_primary` when branching.

### 4.3 `pipeline/codegen.py` (C4b)

- `vet_generated_feature(name, code, base_config=None, seed, log_events)` →
  report `{feature, registered_name, status, reason, stages, results}` with
  `status ∈ {accepted, rejected, quarantined}` through the gauntlet
  **syntax → schema → leakage → smoke → screen → full**.
  - `rejected` = broken code (repairable);
  - `quarantined` = leaking code (deregistered, logged to
    `store.append_event`, never "repaired" by loosening the guard);
  - `accepted` = stays registered as `gen_<fn name>`, usable in any config's
    `features` list.
- `load_feature(code, name=None)` — exec + frozen 8.4 signature check
  (exactly `(train_df, target_df)`), attaches `fn.__leak_source__ = code`.
- `register_generated` / `unregister_generated` — `gen_` namespace;
  generated code can never shadow a human-registered feature.
- `USER_AUTHOR_AFFINITY_SOURCE` — the deliverable feature: per-row strict
  temporal cutoff via `merge_asof(on=date, by=[user, author],
  allow_exact_matches=False)`, EB smoothing alpha=20 (B5 can fit alpha on
  folds later). `LEAKY_TWIN_SOURCE` — test fixture ONLY: assembles the label
  column name at runtime so a static grep misses it; the dynamic probe
  catches it. Never register it outside a vet run.
- `tools/leak_demo.py [--real]` — the containment demo; exit 0 iff
  safe=accepted AND leaky=quarantined. CI-friendly.

### 4.4 `agent/gate.py` (D3/D4, filled to Appendix A.3)

- `accept(cand_scores, best_scores, user_ids, n_boot=1000, seed=0)` →
  `(accepted, (lo, hi))`; bootstrap over USERS with per-user GAUC/nDCG
  precomputed once (sort-and-slice grouping, not per-user masks — this
  matters at 27K users). Frozen signature has no labels: they come from
  `_validation_labels()` (official split, cached) — tests monkeypatch
  `gate._validation_labels`. Scores must cover the full validation window.
- `segments(scores, user_ids, meta)` — primary per level per meta key;
  reserved `meta["labels"]` supplies labels (train.py passes it), else the
  official split's labels are used.

---

## 5. Integration decisions made this session (respect them)

### 5.1 Label + policy corrections

`LABEL = "long_view"` (starter kit wins over plan prose). `is_click` IS
forbidden as same-row input; `duration_ms` is NOT forbidden (it's a
pre-exposure video property and an official baseline field via
`dur_bucket`). `FORBIDDEN_SAME_ROW` and `EXCLUDED_SOURCES`
(`item_statistics_monthly`) live in `pipeline/features.py`.

### 5.2 B6 reconciliation (`leakage_check`)

Both sides had implemented it; **Malvika's B6 branch was kept as the
authority** because it fixes a real flaw in Appendix A.2's pseudocode (a
direct `target_df[LABEL]` read is invisible to the train-label-shuffle
probe; the static scan must run FIRST, unconditionally). Contract:

- returns **bool** (False on static hit or probe mismatch), raises only for
  empty frames and unauditable sources;
- static scan via `_reads_target_column` (both quote styles);
- dynamic probe: NaN-corrupt the label + every FORBIDDEN column on the
  target sample; output must not change;
- extensions folded in for C4b: source recovery prefers
  `fn.__leak_source__` (generated code has no file) and **fails closed**
  (ValueError) when no source is recoverable; probes run on head samples
  (2000 train / 1000 target rows) for cost.

My earlier permutation-probe implementation was discarded in its favour;
my hardening tests were re-added in B6's idiom (obfuscated runtime-assembled
column name → False; unauditable lambda → raises; `__leak_source__` audited).

### 5.3 Cross-ownership fixes (flag these in review)

- `agent/execute.py` (Kaiwen's): `_code_child` previously exec'd generated
  code into a **discarded namespace** — feature code could never affect a
  run. Now `family="feature"` routes through `codegen.load_feature` +
  `register_generated` (inside the worker child only; registration dies
  with the process) and appends `gen_<name>` to the config features. Other
  families keep the plain exec. Two integration tests added at the bottom
  of `tests/test_execute.py`.
- `pipeline/train.py` (mine): `FeatureLeakError(ValueError)` — leakage-guard
  rejections in `_matrix` classify as **`leak_suspected`, not `schema`**,
  because A5 grants schema errors a repair attempt and a leak must be
  quarantined, never regenerated until it evades the guard. Do not revert.
- `agent/gate.py` (Pinxin's): D3/D4 stubs filled (see §4.4).

### 5.4 Known open gaps

- **B4 aggregate features cannot build training matrices**:
  `_assert_historical_cutoff` rejects `_matrix(train, train, …)` because
  train's max date ≥ train's min date. B's newer in-sample `time_ms`
  features are the intended path for train-side history. Coordinate with
  Malvika before touching `pipeline/features.py`. C4b's affinity feature is
  unaffected (per-row cutoffs).
- **Innovation benchmark pending**: does `gen_user_author_affinity` lift a
  real model? Needs C4 LightGBM. The demo only proves containment.
- **Propose prompt** does not yet require a disproof condition (Kaiwen's
  `agent/prompts/propose.md`).
- **PR #19 unmerged** — everything below assumes it lands first.

---

## 6. Architecture crib (what calls what)

```text
agent/loop.py (A6, TODO)
  └─ propose() ──► Action{hypothesis, reasoning, type, family, parent, config|code}
  └─ execute(action, fidelity, timeout_s)          # agent/execute.py
        ├─ type=config/tune/blend ─► run_experiment(config, fidelity, seed, timeout_s)
        └─ type=code ─► outer subprocess:
              family=feature ─► codegen.load_feature + register_generated
              else           ─► plain exec
              then run_experiment(...)
  └─ gate.accept + gate.segments ─► promotion decision
  └─ evidence.to_node(...) ─► store.write(node)    # append-only ledger

run_experiment (pipeline/train.py, FROZEN 8.5)
  child process (fork) per call; timeout enforced by parent poll
  smoke: 1000-row fit, no metrics | screen: 3 internal folds, capped budget
  full: folds (epoch selection) + refit on full prefix, official val + test
  confirm: full × 5 seeds, averaged
  _matrix: FIELDS from columns; dur_bucket derived from train quantiles;
           registered features via features.get + leakage_check (B6) —
           failure ⇒ FeatureLeakError ⇒ error_class leak_suspected
  result: leak canary primary>0.75 ⇒ status=error, leak_suspected
  segments: _segment_metrics (activity_q/pop_q/day) on full/confirm
```

Fidelity/cost discipline: screen on internal folds; the official validation
window is a budgeted resource, spent only on survivors; never use it for
early stopping, tuning, blend weights, or EB fitting.

---

## 7. Test map (235 tests)

| File | Covers |
|---|---|
| test_data_split / test_folds | B1/B2 counts, chronology, fold windows |
| test_features / test_smoothing | B4 features, B5 decay+EB |
| test_leakage | B6 guard (bool contract) + hardening: obfuscated read, unauditable fail-closed, `__leak_source__` audit, >0.75 canary |
| test_train | C1 tiers via monkeypatched fixture backends (`_load_data`/`_load_folds`/`_get_model_class`); leak class now `leak_suspected` |
| test_models | C2/C3 unit + dataset-backed reference gates (skip if data absent) |
| test_audit | C1b battery: schema, determinism (urandom model fails), shuffle canary honest-vs-oracle, evidence |
| test_codegen | C4b gauntlet: syntax/schema rejection, safe accepted end-to-end, leaky quarantined, per-row cutoff math, vet determinism |
| test_evidence | C3b: required fields, pass/fail decisions, purity, error-as-evidence, store round-trip |
| test_gate | D3 bootstrap (zero delta rejected, clear win accepted, 1-in-300 rejected, row-permutation invariance), D4 three breakdowns |
| test_execute / test_recovery / test_propose / test_schema / test_store / test_manifest | A-side + D2 |

Fixture conventions: synthetic pandas frames with `date/user_id/video_id/
author_id/tab/duration_ms/long_view`; keep positive rates ~15% and scores
weakly correlated or the >0.75 canary trips on fixtures. Dict-frames are
used in some C1 tests — those monkeypatch `leakage_check` (the real guard is
pandas-only).

---

## 8. Verification ritual (all of it, every change)

```bash
.venv/bin/pytest                      # must be all-pass, no new skips
.venv/bin/ruff check .                # line length 100; E,F,I,UP,B
.venv/bin/python -m tools.leak_demo --real   # exit 0 = containment holds
```

Then update `ethanprogress.md` (measured numbers + exact commands) and tick
`ethannotes.md` checklist items. For dataset-backed claims, quote the actual
printed primary, never an estimate. Evidence before assertions, always.

## 9. Traps that actually bit (or nearly bit) this session

1. Never shuffle/k-fold; never touch `pipeline/evaluate.py` / `submit.py`
   (hashed; preflight fails closed on mismatch).
2. The >0.75 canary fires on ANY ok result — including your test fixtures.
3. Smoke returns `None` metrics — anything needing a score runs at screen+.
4. `_screen_config` requires positive numeric hparams for capped keys.
5. Per-user metric code must group by sort-and-slice, not per-user boolean
   masks (27K users × 125K rows will crawl).
6. `store.write()` refuses existing ids — the ledger is append-only;
   quarantine is a status + deregistration, never a deletion.
7. `merge_asof` needs globally sorted keys and consistent by-column dtypes
   (cast ids to str) — and `allow_exact_matches=False` is what makes the
   affinity feature leak-safe (same-day rows invisible).
8. Short fixture columns can survive a permutation draw unchanged — the B6
   probe NaN-corrupts instead, which sidesteps that class of flake.
9. `gh pr checks` + a bare `sleep` is blocked in this harness — poll with an
   `until` loop in a background Bash task.
10. A2's store and A5's recovery treat `manual_intervention` and error
    classes as graded output — log honestly, never hide a failure.

## 10. Immediate next action for you

If PR #19 is merged (check `gh pr view 19`): `git checkout main && git pull`,
branch `feat/C4-lightgbm`, implement C4 per §3 step 2, and run the
`gen_user_author_affinity` lift benchmark. If #19 is still open, ask the
user whether to merge it (`gh pr merge 19`) — do not push main directly, and
do not start C4 from a stale base.

---

## 11. C4→C7 implementation playbook

The per-task recipe to get from here to a complete Workstream C. Each task:
branch `feat/<ID>-…` from post-#19 main, work test-first against the §9.3
acceptance criteria, run the §8 ritual, update `ethanprogress.md` with
measured numbers, small PR. Models plug in via `pipeline/models/__init__.py`
`@register("<name>")` and must satisfy the frozen 8.9 protocol
`fit(X_train, y_train, X_val, y_val, groups=None)` / `predict(X) -> scores`.
`train.py::_new_model` passes `seed, loss, parents, blend_method,
negative_sampling` plus all hparams as constructor kwargs — accept and
ignore what you don't use (`**hparams`).

### C4 — LightGBM (`feat/C4-lightgbm`)

**Files:** `pipeline/models/lgbm.py`, plus a small `pipeline/train.py`
change (both Ethan-owned).

1. **Group plumbing (the one real design decision).** LambdaRank needs a
   per-user group array, and `_fit_and_predict` currently calls
   `model.fit(...)` without `groups`. Fix in `train.py`: pass
   `groups=(train_user_ids, val_user_ids_or_None)` — the frozen protocol
   already has the `groups=None` slot, and a tuple of id arrays keeps the
   signature unchanged. Models that don't rank ignore it.
2. **Inside the model:** rows are NOT contiguous by user (organiser row
   order). For lambdarank, stable-sort train rows by user id internally,
   build `group = counts per user in sorted order`, train on the sorted
   matrix; do the same for the eval set. `predict` must return scores in
   the ORIGINAL input row order (predict needs no groups).
3. **Objectives:** `binary` (pointwise) and `lambdarank`
   (`label_gain=[0,1]` for binary labels, `eval_at=[5]`). Integer
   label-encode the categorical FIELDS with train-fitted first-seen vocabs +
   UNK slot — copy the pattern from `fm.py` (trap 7: never one-hot). Pass
   `categorical_feature` for the ID columns.
4. **Early stopping:** `early_stopping_rounds` against the PROVIDED
   `X_val/y_val` — in screen/full flows that is an internal fold, never
   official validation (the runner guarantees this). Expose `best_epoch`
   (best_iteration) the way `fm.py` does so `_run_full`'s median-epoch
   refit keeps working; honour `num_boost_round` from hparams (screen caps
   clamp it via `SCREEN_BUDGET_CAPS`).
5. **Determinism:** `seed=seed`, `deterministic=True`,
   `force_row_wise=True`, `num_threads=1` (thread-order reductions break
   the C1 same-seed guarantee — verify with `audit_determinism`).
6. **Tests:** synthetic fixture (learnable pattern, ~15% positives);
   pointwise beats random on the fixture; lambdarank group array asserted
   correct via a spy; original-row-order prediction asserted after internal
   sorting; dataset-backed reference run recorded in ethanprogress.
7. **Then the innovation benchmark:** full-tier LightGBM with
   `features=[FIELDS..., "user_ctr", "video_ctr", ...]` vs the same +
   `"gen_user_author_affinity"` (vet it first via
   `codegen.vet_generated_feature`). Record both primaries and the delta in
   `ethanprogress.md`; judge with `gate.accept`, and write the pair as
   evidence records (`pipeline/evidence.py`) with the hypothesis "prior
   user-author exposure predicts long views". This number is the C4b story.

### C5 — DeepFM (`feat/C5-deepfm`)

**Files:** `pipeline/models/deepfm.py`.

1. **Architecture:** copy plan Appendix A.4 verbatim — per-field
   `nn.Embedding`, linear terms + bias, FM second-order via the O(n)
   identity `0.5 * ((Σv)² − Σv²)`, MLP `[256,128,64]` + dropout, raw logit
   out, `BCEWithLogitsLoss`, Adam.
2. **Encoding:** the same train-fitted per-field vocab + UNK pattern as
   `fm.py`/C4; `field_dims` from the vocab sizes; continuous feature
   columns (registered features like ctr rates) either bucketise to ids or
   concatenate to the MLP input — bucketising is simpler and consistent.
3. **Budget:** epochs 1–5, early-stop patience **1** on provided X_val
   (CTR models peak at 1–3 epochs, trap 11), batch 2048–4096, CPU
   (`torch.set_num_threads` modest); the plan requires <10 min CPU
   training. `_seed_everything` already forces deterministic torch — keep
   dataloader order fixed (no shuffling workers; shuffle via a seeded
   generator if at all).
4. Expose `best_epoch`; honour `max_epochs` hparam (screen cap = 3).
5. **Tests:** loss decreases on a learnable fixture; deterministic across
   two same-seed fits; unseen categories hit UNK; wall-time sanity marker
   for the real run recorded in ethanprogress (not a hard test).

### C6 — Optuna (`feat/C6-optuna`)

**Files:** new `pipeline/tune.py` (Ethan-owned; keep it out of frozen
modules). This is a HARNESS around run_experiment, not a model.

1. **API:** `run_study(base_config, search_space, budget=20, seed=42,
   storage="sqlite:///logs/optuna/<study_name>.db", study_name=...) ->
   {best_params, best_value, n_pruned, seconds_saved, trials}`.
   `search_space` uses the Action schema's shape:
   `{"lr": ["loguniform", 1e-4, 1e-2], "emb_dim": ["categorical", [8,16,32,64]],
   "num_leaves": ["int", 31, 255]}` — write one `_suggest(trial, name, spec)`
   translator.
2. **Objective:** merge suggested params into `base_config["hparams"]`, run
   `run_experiment(cfg, fidelity="screen", seed=seed)` (internal folds ONLY
   — official validation never enters tuning), return mean fold primary.
   Report per-fold values to the trial (`trial.report(value, step=fold)`)
   so `MedianPruner` can prune mid-trial; raise `optuna.TrialPruned` when
   `trial.should_prune()`. An error result → return a sentinel bad value
   (e.g. 0.0), never raise — a failed trial must not kill the study.
3. **Sampler/pruner:** `TPESampler(seed=seed)`, `MedianPruner
   (n_startup_trials=5)`. **One worker** (`n_jobs=1`) — study-level
   determinism was decided per-experiment; parallel trials reorder pruning.
4. **Resume:** `optuna.create_study(load_if_exists=True, storage=...)`;
   acceptance requires kill-and-resume to complete the remaining trials —
   test with a tiny study interrupted after N trials (drop the study object,
   recreate, continue).
5. **Accounting:** count pruned trials and estimate runtime saved (sum of
   pruned trials' completed-fold time vs full-trial mean) — ≥30% saving is
   an acceptance criterion; record in the returned dict and ethanprogress.
6. **A-side hook (coordinate, don't implement):** `execute()` currently
   ignores `action.search_space` for `type="tune"` — Kaiwen should route
   tune actions to `pipeline.tune.run_study` and put `best_params` into the
   node. Note it in the PR body.

### C7 — Blending (`feat/C7-blending`)

**Files:** `pipeline/models/blend.py`, small `train.py` support.

1. **Where parent scores come from (the design decision):** node records
   don't store score arrays. Use C1's determinism instead — a parent's
   `(config, fidelity, seed)` reproduces its exact `val_scores`/
   `test_scores`. Implement a score cache in `train.py` or `blend.py`:
   key `sha256(canonical_config_json + fidelity + seed)` →
   `logs/scores/<hash>.npz` (val_scores, val_user_ids, test_scores).
   `_run_full` writes it (cheap, ~2MB); the blend loads parents by
   recomputing the hash from the parent node's stored config, falling back
   to a fresh `run_experiment` if the file is missing. This keeps the
   frozen result contract untouched.
2. **Blend model (`model="blend"`):** constructor receives
   `parents=[node_ids]` and `blend_method`. It does not `fit` on matrices —
   its `fit` resolves parent score vectors (via store.read(node)["config"]
   → cache) and its `predict` combines them. All four methods:
   - `rank_avg` (DEFAULT, hardcoded): scipy-free per-user rank via
     argsort-of-argsort within user groups, averaged;
   - `logit_avg`: clip scores to (ε,1−ε) if probabilities, else min-max per
     user first; average log-odds;
   - `weighted_rank`: `w·rankA + (1−w)·rankB`, w ∈ {0.1..0.9} selected on
     INTERNAL FOLDS only;
   - `rrf`: Σ 1/(k + rank), k=60.
   Keep every output continuous — never threshold, never vote (trap 10).
3. **Diagnostic gate:** compute mean per-user Spearman between parents
   (rank-transform per user, Pearson on ranks). Report it in the result
   (extra key is fine) and in ethanprogress. >0.95 → refuse ("too similar
   to gain"); <0.5 → warn ("a parent may be broken").
4. **Acceptance rule:** the blend is accepted only if it beats BOTH parents
   on internal folds AND the official metric, by the margin set in
   `hparams["min_blend_delta"]` (default `MIN_DELTA_FLOOR`), then passes
   `gate.accept` against the best parent. Only accepted nodes may be
   parents (`store.best_node` semantics).
5. **Tests:** two synthetic parents with known score vectors → rank_avg
   reproduces a hand-computed blend exactly; per-user Spearman matches a
   scipy cross-check on a fixture; refusal at correlation >0.95; weighted
   blend's w chosen on folds (spy that official val labels are never read
   during weight fitting); determinism.

### Expected shape of results (so you can smell breakage)

FM baseline 0.6016. LightGBM pointwise with good temporal features should
land ~0.60–0.63; lambdarank similar ±; DeepFM near FM unless tuned (the
Dacrema/Rendle caution — don't chase it past its probe result); a good
blend adds ~0.002–0.01 over the best parent. Anything above **0.75 is a
leak until proven otherwise** — that's what the canary and the shuffle
audit are for. If a number looks too good, run
`audit_label_shuffle(config)` before believing it.

### Definition of done for Workstream C

All of C1–C7 + C1b/C3b/C4b merged; every model reachable through
`run_experiment` by name; the five D5 probes runnable; the lift benchmark
for `gen_user_author_affinity` recorded with an evidence record and a gate
decision; `ethanprogress.md` carries a measured number for every rung.
