# TechJam 2026 Track 2 — Requirements Specification

Every graded obligation, stated unambiguously with a stable ID.
Test procedures for each are in `03_TEST_SPEC.md`, keyed by the same IDs.

**Legend:** `R` = task requirement, `C` = constraint, `D` = deliverable, `K` = known constant.

---

## Part 1 — Constants (`K`)

These are fixed by the organisers. Hardcode them; do not recompute or guess.

**Provenance rule (added after the K10 incident below): no constant enters
this table without a `Source` naming a file, or "measured" naming the command
that produced it. A constant sourced only from the brief's prose is marked
`UNVERIFIED` and blocks the C2 gate until it is measured against the shipped
code.**

> **2026-08-31 correction.** K10 originally read `click (fixed)`, taken from
> the brief's prose with no citation. K12–K16 were taken from the same prose,
> uncited. The two are inconsistent: K12/K13's numbers are only reproducible
> using `long_view` as the label (verified below) — a genuine `click` scoring
> pass would produce different baseline numbers, and it doesn't. Whoever
> implements against this table next should not have to make that discovery
> the hard way; a downstream branch already lost time to it. Root cause and
> full chain: this session's transcript, 2026-08-31.
>
> While re-deriving K15/K16, a second issue surfaced and resolved cleanly:
> the brief's random/popularity figures (0.4753 / 0.5715) are the **hidden
> test** rungs, not validation. They are not wrong — they were just not
> labelled by split, and got miscited as validation numbers in that same
> session. Both splits are now given explicitly below.

| ID | Constant | Value | Source |
|---|---|---|---|
| K1 | Train window | 2022-04-08 to 2022-04-21 | `kuairand-starter-kit/data.py` `SPLITS['train']` |
| K2 | Train rows | 1,141,112 | measured, `pipeline.data.load()`, matches `data.py` |
| K3 | Validation window | 2022-04-22 to 2022-04-28 | `data.py` `SPLITS['valid']` |
| K4 | Validation rows | 124,909 | measured |
| K5 | Hidden test window | 2022-04-29 to 2022-05-08 | `data.py` `SPLITS['test']` |
| K6 | Hidden test rows | 170,588 | measured |
| K7 | Source file (train + validation) | `log_standard_4_08_to_4_21_pure.csv` | `data.py` |
| K8 | Source file (test) | `log_standard_4_22_to_5_08_pure.csv` | `data.py` |
| K9 | Users / items | 27K / 7.6K | measured on the standard logs |
| K10 | Scored label | **`long_view`** | `kuairand-starter-kit/data.py:5` (`LABEL = 'long_view'`), sha256 `1bf54f5f…d36e541`. `is_click` is a same-row post-exposure signal, forbidden as an input feature, not the label |
| K11 | Scored metrics | GAUC and nDCG@5, `primary = mean(GAUC, nDCG@5)` | `kuairand-starter-kit/evaluate.py`, sha256 `ecfde283…d195de` |
| K12 | Baseline validation | GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 | measured, `baseline.py --model fm`, `data.py` sha256 above, `baseline.py` sha256 `c8f7fc60…8fe18f8a` |
| K13 | Baseline hidden test | GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946 | same run as K12, test split |
| K14 | Baseline seed std | 0.0008 (over 5 seeds) | measured, FM, 5 seeds, valid split |
| K15 | Random reference | valid primary 0.4834 (std 0.0006) · **hidden test primary 0.4753** (std 0.0008) | measured, `label=long_view`, 5 seeds each split, 2026-08-31 |
| K16 | Popularity reference | valid primary 0.5808 · **hidden test primary 0.5715** | measured, `label=long_view`, 2026-08-31 |
| K17 | Attainable ceiling | GAUC 1.0000 / nDCG@5 0.7289 / primary 0.8645 | `baseline_scores.json`, `oracle_ceiling`, hidden test |
| K18 | Zero-positive users (hidden test) | 27.1% | `baseline_scores.json`, `test_set_composition` |
| K19 | All-positive users (hidden test) | 9.2% | `baseline_scores.json`, `test_set_composition` |
| K20 | Repeated `(user_id, video_id)` pairs in eval split | 3.06%, up to 12 repeats | `kuairand-starter-kit/README.md` |
| K21 | Convergence epsilon | 0.002 (≈ 2.5 × K14) | `AGENT_PLAN.md` §4.5, derived from K14 |
| K22 | Convergence N | 3 consecutive iterations | `AGENT_PLAN.md` §4.5 |
| K23 | Iteration cap | 50 per benchmark run (hard) | `UNVERIFIED` — not yet re-derived against the shipped kit; carried over from the brief |
| K24 | Wall-clock ceiling | 6 hours per run | `UNVERIFIED` — same as K23 |
| K25 | Baseline runtime | ~40s single CPU core; 100 iterations ≈ 28 min | measured, `baseline.py --model fm` |
| K26 | Submission header | `row_id,user_id,video_id,score` | `kuairand-starter-kit/submit.py` |
| K27 | Evaluator interface | `evaluate(user_ids, labels, scores)` — model-agnostic | `kuairand-starter-kit/evaluate.py` |

---

## Part 2 — Task requirements (`R`)

### R1 — Reproduce the official baseline

The agent must stand up a working end-to-end pipeline and confirm it reaches the official baseline's reported validation score.

**Precise meaning:**
- The reference is the **organiser-provided** FM (k=16, lr=0.001, 5 categorical fields), scoring K12 on validation.
- Any starter pipeline the team builds for itself is an internal step, **not** the reference. Improvements are measured against K13, never against a team-built baseline.
- The two harness self-check rungs (K15 random, K16 popularity) exist to verify the evaluation plumbing before any real experiment.

**Pass condition:** pipeline reproduces K12 within tolerance, and both reference rungs reproduce K15 and K16.

---

### R2 — Develop only on train + validation

The agent develops using only the training split and public validation feedback. It never has access to the hidden test set during development.

**Precise meaning:**
- The hidden test file (K8) must not be read at any point during the iterative loop.
- It is read exactly once, at final submission generation.
- Refitting the final chosen configuration on train + validation (K1 + K3) before predicting the test set is permitted and temporally legal, since validation ends before test begins.

**Pass condition:** the full agent loop completes with the test file physically absent from disk.

---

### R3 — Improve over the baseline, with a sustained trend

Through repeated iterations, drive the validation score above the official baseline. Improvement need not be strictly monotonic; the trajectory may fluctuate. The agent should show a clear, sustained ability to keep improving relative to the baseline.

**Precise meaning:**
- Fluctuation is expected and acceptable.
- A flat trajectory, or one with no net improvement, does not satisfy this.
- The best-so-far curve should be non-decreasing by construction if the accept gate is working.

**Pass condition:** net improvement exceeds the noise floor (K14), and the trajectory shows movement rather than a flat line.

---

### R4 — Run end-to-end and reach a converged result

The agent must run the full pipeline on KuaiRand-Pure and reach a converged result.

**Precise meaning:**
- **Converged** means any one of: (a) validation primary has not improved by more than K21 over the last K22 consecutive iterations, (b) the K23 iteration cap is reached, or (c) the K24 wall-clock ceiling is reached. Whichever comes first.
- The submission scored for ranking is the **validation-best checkpoint at that point** — not the final iteration's checkpoint, and not the peak if the peak was not accepted.
- Scoring uses the converged result, **not the peak and not the intermediate trajectory**.
- Falling short of the baseline is **not disqualifying**. The delta, positive or negative, feeds continuously into the Primary metric.

**Scoring formula:**
```
delta(m)      = score_agent(m) − score_baseline(m)     for m in {GAUC, nDCG@5}
score_dataset = mean over m of delta(m)
```
Both metrics are equal-weighted.

**Pass condition:** the loop stops for a legitimate documented reason, and the submitted checkpoint is provably the validation-best.

---

### R5 — Iterate autonomously across the full stack

The agent improves the solution on its own, driven by its own evaluation of results. Improvements may target any part of the algorithmic stack, not just model architecture. The goal is to minimise human intervention.

**Precise meaning:**
- Fully autonomous is ideal; a handful of interventions is acceptable and realistic.
- Measured by **number of manual interventions**. Fewer scores higher.
- "Full stack" means features, loss framing, model class, training strategy, evaluation loop, and ensembling are all legitimate targets. Model architecture alone is insufficient breadth.
- **The team must define "manual intervention" explicitly and publish that definition**, since the brief does not. Recommended definition:
  - **Counts:** restarting a crashed process, editing code mid-run, killing and adjusting a hung run, manually selecting a candidate.
  - **Does not count:** pre-run setup, seeding `knowledge.md` before the run starts, reading terminal output without acting.

**Pass condition:** a full run completes with the operator away from the keyboard, and the intervention count in the report matches the operator's own record.

---

### R6 — Robust operation

The pipeline runs reliably with minimal human intervention. Robustness is about how the agent handles difficulty, not how often it succeeds.

**Precise meaning:**
- **Not scored by failure count.** A capable agent may fail only on genuinely hard problems.
- What is scored: when a step fails (code error, timeout, unexpected input), the agent recovers, retries, or routes around it.
- Long iterative runs must neither **crash**, **stall**, nor **diverge**.

**Pass condition:** deliberately injected failures of each class are classified, recovered from, and logged, and the loop continues afterward.

---

## Part 3 — Constraints (`C`)

| ID | Constraint |
|---|---|
| C1 | **No external training data.** Only KuaiRand. No pretrained weights trained on these benchmarks' test labels. |
| C2 | Any open-source library, paper, public solution, or other pretrained weights are permitted. |
| C3 | Hidden test scored **once**, on the final submission. |
| C4 | KuaiRand-Pure is required and determines **100%** of the Primary metric. |
| C5 | KuaiRand-1k / 27k are bonus. Attempting them adds points; skipping them subtracts nothing. |
| C6 | Splits are fixed and date-based. Do not shuffle or re-split. |
| C7 | GPU-hours and LLM tokens are **reported, not capped**. They feed Feasibility (15%). |

---

## Part 4 — Evaluator conventions (pinned)

These are fixed in the shipped `evaluate.py` and must be matched exactly by any reimplementation.

| ID | Convention |
|---|---|
| E1 | Users with zero positives count as **nDCG = 0** and **are included** in the average. |
| E2 | GAUC counts **only** users with `0 < positives < impressions`, weighted by positive count. |
| E3 | nDCG gain = `2^rel − 1`. |
| E4 | Primary = mean of GAUC and nDCG@5. |
| E5 | Evaluator signature is `(user_ids, labels, scores)`. Model-agnostic. |

---

## Part 5 — Deliverables (`D`)

| ID | Deliverable | Content |
|---|---|---|
| D1 | Devpost writeup | How the solution addresses the problem; dev tools; APIs; libraries and frameworks; datasets and assets |
| D2 | Public GitHub repo | Well-structured commented code covering all components; README with overview, setup, reproduction steps, limitations reflection, per-member contributions |
| D3 | Run and iteration logs | Per iteration: **hypothesis**, **code diff**, **resulting metrics**, **error / recovery events**. Plus a summary of manual intervention count |
| D4 | Final submission | KuaiRand-Pure output in starter-kit schema (K26), validated by `submit.py --check` |
| D5 | Results summary | Validation-best component metrics and absolute delta over official baseline |
| D6 | Resource report | Total input + output tokens across all LLM calls; total GPU-hours to converged result |
| D7 | Secret hygiene | No secrets in source, git history, logs, traces, screenshots, or output |

**Note on D3:** this is not documentation. It is the mechanism by which **Autonomy** (under Impact & Relevance, 20%) and **Robustness** (under Technical Execution) are assessed. Without it those categories cannot be scored regardless of actual system quality.

**Note on demo video:** Tracks 3, 4 and 5 list a demo video as a deliverable. Track 2's list does not. Treat as optional pending confirmation.

---

## Part 6 — Judging weights

| Category | Weight | Driven by |
|---|---|---|
| Technical Execution | 35% | R4 delta on hidden test, plus R6 robustness |
| Innovation & Problem Insight | 20% | What the agent chose to try and why (visible in D3 hypotheses) |
| Impact & Relevance | 20% | R5 autonomy, measured by intervention count |
| Feasibility & Practicality | 15% | D6 tokens and GPU-hours |
| Presentation & Communication | 10% | Final event only |

**Implication:** roughly 60% of the score is agent *behaviour*, not model quality. The score delta is one ingredient inside the 35%, sharing that weight with robustness.

---

## Part 7 — Reading the numbers correctly

The metrics do not span [0, 1]. Because 27.1% of users have no positive label (K18) and 9.2% are all-positive (K19), an omniscient model reaches only primary **0.8645** (K17).

```
Usable range:            0.8645 − 0.4753 = 0.3892
Baseline position:      (0.5946 − 0.4753) / 0.3892 ≈ 31%
Remaining headroom:      0.8645 − 0.5946 = 0.2699
```

**Judge progress against 0.8645, never against 1.0.** A +0.02 delta is ~7.4% of the remaining attainable range, not "barely anything."
