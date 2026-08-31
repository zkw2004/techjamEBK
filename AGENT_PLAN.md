---
title: "Autonomous ML Research Agent for Recommender Systems"
subtitle: "Project Report and Implementation Plan | TechJam 2026, Track 2"
author: "Kaiwen · Malvika · Ethan · Pinxin"
date: "Version 2.0 — 28 August 2026"
---

| | |
|---|---|
| **Challenge** | TechJam 2026, Track 2 |
| **Benchmark** | KuaiRand-Pure (required). KuaiRand-1k / 27k (bonus, deprioritised) |
| **Team** | Kaiwen, Malvika, Ethan, Pinxin |
| **Duration** | 3 days |
| **Status** | Ready to implement pending webinar confirmation (Section 15) |

---

# PART I — ORIENTATION

## 1. How to use this document

### 1.1 For human readers

Read Part I and Part II in full. Read Part III for strategy. Read your own workstream in Section 9, and skim the others so you know what you depend on.

**Section 11 (known traps) is mandatory for everyone.** Several of those traps look like standard good ML practice and will silently invalidate the submission.

This document is self-contained. No prior recommender-systems knowledge is assumed: Section 5 explains the metrics, Section 6.5 explains the loss framings, and Appendix A gives reference implementations for the pieces most likely to be built subtly wrong.

### 1.2 For coding agents (Claude Code, Codex)

This document is your specification. Follow these rules:

1. **Contracts in Section 8 are frozen.** Every function signature, dict key, and file path in Section 8 is a hard interface that four people are building against in parallel. Do not rename, restructure, or "improve" them. If a contract seems wrong, stop and report it rather than changing it.
2. **Implement one task ID at a time.** Tasks are labelled `A1`, `B3`, etc. in Section 9. Each has explicit acceptance criteria. Do not start a task whose dependencies are unmet.
3. **Never modify `pipeline/evaluate.py` or `pipeline/submit.py`.** They are copied verbatim from the organiser starter kit and are the scoring ground truth. Their SHA-256 hashes are recorded in every run manifest.
4. **Never shuffle or re-split the data.** See Section 11, trap 1. Highest-severity failure mode in the project.
5. **Write tests alongside code.** Most acceptance criteria in Section 9 correspond to a pytest case.
6. When a task says "port from starter kit," read the actual starter kit file rather than reimplementing from memory.

Suggested prompt when starting work:

> Read AGENT_PLAN.md. Implement task \<ID\> only. Follow the contracts in Section 8 exactly. Write the pytest cases listed in the acceptance criteria. Do not touch files owned by other workstreams.

### 1.3 Git workflow

Four people, three days, one repo. Keep it simple:

- `main` stays green. Never push directly.
- One branch per task ID: `feat/A3-propose`, `feat/B4-leakage-guard`.
- Small PRs, merged as soon as tests pass. Do not batch a day's work into one PR.
- Workstream file ownership (Section 9) is exclusive. If you need a change in someone else's file, message them rather than editing it. This is what keeps merge conflicts near zero.
- The frozen contracts in Section 8 change only by agreement of all four.

---

## 2. Executive summary

### 2.1 What is being built

Not a recommender system. **An agent that builds recommender systems.**

A closed loop that autonomously reads the KuaiRand-Pure dataset, proposes an improvement hypothesis, writes or configures the code for it, screens it cheaply, evaluates survivors against the official metric, statistically tests whether the change was a real improvement, records the outcome, and repeats until convergence.

The recommender model is an artifact of the loop, not the deliverable.

### 2.2 Why the framing matters for scoring

| Judging category | Weight | What it actually measures |
|---|---|---|
| Technical Execution | 35% | Hidden-test delta over baseline, plus recovery from failures |
| Innovation & Problem Insight | 20% | What the agent chose to try and why |
| Impact & Relevance | 20% | Autonomy, measured by manual intervention count |
| Feasibility & Practicality | 15% | Total tokens and GPU-hours consumed |
| Presentation | 10% | Final event only |

Roughly 60% of the score is agent behaviour rather than model quality. A mediocre model driven by a genuinely autonomous, well-instrumented, error-resilient loop beats an excellent hand-tuned model.

**Strategic consequence:** invest disproportionately in the harness, the evaluation integrity layer, and the failure-recovery story. Cap model ambition deliberately.

### 2.3 The six design commitments

Everything in this plan follows from these:

1. **Contract first.** Read the metric, label, split, and schema out of the shipped code and hash it. Never from prose.
2. **Cheap screening, expensive confirmation.** Four fidelity tiers. Most candidates die in seconds.
3. **Protect the official validation set.** Screen on internal time folds; spend the official window only on survivors.
4. **Promote on evidence, not on score.** Bootstrap confidence intervals, segment checks, temporal consistency.
5. **Failures are graded output.** Classify, recover, and log them rather than hiding them.
6. **Account for every resource.** Tokens and compute logged from the first run.

### 2.4 Success criteria

- **Minimum viable:** loop runs 20+ unattended iterations, beats the official baseline on hidden test, produces complete run logs.
- **Target:** up to 50 iterations — the hard per-run cap (K23, 02_REQUIREMENTS.md) — clear score trajectory, at least one demonstrated failure-and-recovery, one novel methodological contribution. This document originally said "40+"; corrected 2026-08-31 to agree with the requirements doc rather than leave two authoritative-looking numbers in the repo.
- ~~**Stretch:** exposure-debiasing using KuaiRand's randomised-exposure slice.~~ **Closed 2026-08-31 (§9.2, B8):** `is_rand` is 0 across all 1,141,112 training rows — no unbiased slice exists in the training window. Replaced as the highest-upside item by B14/§9.2.

---

## 3. Quick reference card

Every constant you will need repeatedly, in one place.

| Item | Value |
|---|---|
| Train window / rows | 2022-04-08 to 2022-04-21 / 1,141,112 |
| Validation window / rows | 2022-04-22 to 2022-04-28 / 124,909 |
| Hidden test window / rows | 2022-04-29 to 2022-05-08 / 170,588 |
| Users / items | 27K / 7.6K |
| Feedback signals | 12 (only `click` is scored) |
| Baseline hidden test | GAUC 0.6610, nDCG@5 0.5282, **primary 0.5946** |
| Baseline validation | GAUC 0.6674, nDCG@5 0.5357, **primary 0.6016** |
| Baseline seed std | **0.0008** |
| Random reference | primary 0.4753 |
| Popularity reference | primary 0.5715 |
| Convergence | epsilon = 0.002, N = 3 consecutive iterations |
| Submission header | `row_id,user_id,video_id,score` |
| Leak canary threshold | validation primary > 0.75 |
| Baseline runtime | ~40s, 1 CPU core |

### 3.1 The five failure modes that will actually kill you

Full list in Section 11. These five are fatal rather than merely costly:

1. **Shuffled or k-fold splits.** Trains on the future. Validation looks great, hidden test collapses.
2. **Same-row post-click features.** Using `long_view` to predict `click` on the same row is reading the answer.
3. **The monthly aggregate statistics file.** May span the test period. Excluded by default.
4. **Joining on `(user_id, video_id)`.** Not unique. 3.06% of test rows are repeats. Join on `row_id` only.
5. **A promotion threshold below the noise floor.** Seed std is 0.0008. Anything smaller is not an improvement.

---

# PART II — THE PROBLEM

## 4. Problem context

### 4.1 The task

Each row of the dataset is one impression: on some date, user X was shown video Y, and clicked or did not click. You predict a score per row. Scoring sorts videos **within each user** and rewards putting actually-clicked videos near the top.

This is **supervised, tabular, binary-labelled, per-user ranking.**

### 4.2 Dataset scale

KuaiRand-Pure: ~1.4M interactions, **27K users, 7.6K items**. These two numbers size your embedding tables: 27,000 x 16 plus 7,600 x 16 is roughly 550K embedding parameters, and a DeepFM MLP adds maybe 100K more. That is a ~2MB model. It is small. Do not reach for anything larger.

The dataset carries 12 feedback signals per row (click, like, follow, comment, forward, long_view, play_time, and others) plus a randomised-exposure intervention slice that supports counterfactual evaluation. Only `click` is the scored label.

### 4.3 Data splits (fixed by organisers, date-based)

| Split | Dates | Rows | Use |
|---|---|---|---|
| Train | 2022-04-08 to 2022-04-21 | 1,141,112 | Fitting, and internal folds (6.2) |
| Validation | 2022-04-22 to 2022-04-28 | 124,909 | Official screening. **Budgeted resource** |
| Hidden test | 2022-04-29 to 2022-05-08 | 170,588 | Scored once |

### 4.4 Official baseline

Factorization Machine, k=16, lr=0.001, 5 categorical fields, numpy only, ~40s on one CPU core. Figures in Section 3.

Seed std across 5 seeds is **0.0008**. This single number drives the promotion gate (Section 6.6).

### 4.5 Convergence rule

epsilon = 0.002, N = 3. A run has converged when validation primary has not improved by more than 0.002 across 3 consecutive iterations. The scored submission is the validation-best checkpoint at that moment.

Note the interaction with search order: the run can be ended early by three consecutive low-value tweaks. Order the first full experiments by expected effect size (Section 6.4).

### 4.6 Submission format

CSV, header `row_id,user_id,video_id,score`. `row_id` is 0-based, strictly increasing, indexing into the split as produced by `data.load()`. `user_id` / `video_id` are redundant alignment assertions, **not a key**. `score` is any real number; only relative order matters. NaN and Inf rejected.

3.06% of test rows are duplicate `(user_id, video_id)` pairs, up to 12 repeats. Join and sort by `row_id` only. Validate with `python3 submit.py --check`.

### 4.7 Hard constraints

- **No external training data.** Only KuaiRand. No pretrained weights trained on these benchmarks' test labels. Everything else (any open-source library, any paper, any public solution, other pretrained weights) is permitted.
- **No hidden-test access during development.** Train plus validation only.
- Preserve temporal order. Every historical feature for a row must be computed strictly from earlier events.

### 4.8 Unresolved: the metric contradiction

The brief contains two incompatible contracts.

| Brief location | Contract stated |
|---|---|
| Task, constraints, benchmark table, judging | click; NDCG@10 and Recall@50 |
| Starter Kit resource section | GAUC and nDCG@5; primary is their mean; validation baseline 0.6016 |

These favour different objectives (top-K coverage vs within-user ordering) and use different K.

**Implementation response, not a guess:** build a versioned `MetricProfile` populated *from the shipped evaluator* (Section 8.3), store its SHA-256 in every run manifest, and confirm with the organisers at the webinar. The alternate prose profile may exist as a named diagnostic but must never mix into an official result. This makes the system profile-agnostic: if the answer changes, one config changes.

---

## 5. How the metrics work

Read this before writing any model code. The most common mistake for anyone coming from standard classification is assuming these metrics are global. **They are not. They are computed per user, then averaged.**

### 5.1 GAUC (grouped AUC)

Ordinary AUC is the probability that a randomly chosen positive is ranked above a randomly chosen negative, across the whole dataset. GAUC computes that *within each user's impression list*, then averages weighted by each user's positive count.

A model that only learns "popular videos get clicked" can score well on global AUC and badly on GAUC, because it never distinguishes items inside one person's feed.

Pinned convention: GAUC counts only users where `0 < positives < impressions`. Users with all-clicks or no-clicks are undefined and excluded.

### 5.2 nDCG@5

Quality of a ranked list with a position discount, so slot 1 is worth more than slot 5. Gain is `2^rel - 1`, discounted by log of position, normalised against the ideal ordering.

Pinned convention: users with zero positives count as nDCG = 0 and **are** included in the average.

### 5.3 Three consequences that shape everything downstream

1. **Only ordering matters.** Absolute values and calibration are irrelevant. Any monotonic transform of your output leaves the score unchanged.
2. **Ties are actively harmful.** Tied items get an arbitrary order. This is why ensembling by voting is disqualified (Section 6.9) and why you must never threshold.
3. **The user is the unit of independence.** This is why the bootstrap resamples users rather than rows (6.6), why internal folds must not split a user's history mid-stream, and why LightGBM needs a correct per-user `group` array for lambdarank.

### 5.4 The candidate lists are tiny — measured, not assumed

Measured on the official validation split (`RECSYS_COURSE_ANALYSIS.md` §2.1):

| Quantity | Value |
|---|---|
| Candidates per user, median | **4** (p90 = 12) |
| Users usable by GAUC (`0 < positives < impressions`) | 57.8% |
| Users excluded: all-negative | 30.3% |
| Users excluded: all-positive | 11.9% |

Read every result in this project through these numbers.

1. **A +0.002 delta is a large effect here, not a small one.** With a median of 4 items to order and 42% of users contributing nothing to GAUC, single-seed swings of ±0.001 are structurally expected. This is the frame for judging every experiment; it is also why `MIN_DELTA_FLOOR` at 0.002 is the right bar and not a timid one.
2. **The only thing that can move the score is within-user discrimination between ~4 items.** A feature that is constant across one user's candidates contributes exactly zero except through interactions — this is what B12 measures, and why B12 is a gate rather than a report.
3. **Report these numbers to judges.** Without them a +0.002 improvement looks trivial. With them it is legible.

---

# PART III — STRATEGY

## 6. Strategy and design decisions

### 6.1 Why this is not hyperparameter tuning

The tempting shortcut is to wrap `GridSearchCV` around a model, call it an agent, and go home. That scores near zero on 40% of the rubric.

| | Grid / random search | What is required here |
|---|---|---|
| Who defines the search space | A human wrote the grid | The agent proposes what is worth trying |
| What can change | Numeric values inside a fixed script | Features, loss, architecture, data handling, ensembling |
| Hypotheses | None. It is an exhaustive sweep | Required per iteration, and graded |
| Error handling | Never errors, so nothing to demonstrate | Must recover from code errors, timeouts, bad inputs |
| Autonomy score | Zero. The human is the loop | Measured by intervention count |

Innovation (20%) is judged on *what the agent identified as worth trying and why*. If you wrote the grid, the agent identified nothing.

**Correct framing:** hyperparameter tuning is one action *inside* the agent's action space, not the agent. The agent decides the current bottleneck is hyperparameters, and *then* calls Optuna as one tool among several. Next iteration it might decide the bottleneck is feature sparsity and go write cross features instead.

This is the AIDE formulation (Jiang et al. 2025, cited in the challenge brief): ML engineering as tree search over code, where each node is a working solution and the agent chooses whether to refine or branch. Say so in the README; Innovation is explicitly scored on drawing from published methods.

### 6.2 Evaluation protocol: rolling internal folds

**The problem this solves.** The official validation window (22 to 28 April) is your only clean signal, and the agent will evaluate against it up to 50 times. Repeated evaluation against one set is overfitting, regardless of how the set was constructed. The hidden test will not forgive it.

**The protocol.** Build three expanding-window folds *inside* the training period, always training earlier and validating later:

| Fold | Train | Validate |
|---|---|---|
| F1 | 04-08 to 04-15 | 04-16 to 04-17 |
| F2 | 04-08 to 04-17 | 04-18 to 04-19 |
| F3 | 04-08 to 04-19 | 04-20 to 04-21 |

Rules:

- **Never shuffle.** Folds are date-cut, expanding, never random.
- Screening happens on these folds. A candidate needs a **positive median delta across all three** to earn a full evaluation.
- The official 22 to 28 April window is a **budgeted resource**, spent only on survivors.
- Temporal consistency across folds is itself evidence. A candidate that wins on F3 alone is a fluke, not an improvement.
- Never use validation labels for feature fitting, early stopping, or blend-weight fitting. Use the internal folds for all three.

This is the constructive half of "never k-fold." Trap 1 tells you what not to do; this tells you what to do instead.

### 6.3 Multi-fidelity screening

Four tiers. Most candidates die in the first two, which is what keeps the Feasibility score healthy.

| Tier | Cost | What runs | Promote when |
|---|---|---|---|
| **Smoke** | Seconds | Imports, 1000-row fit, schema check, finite scores, leakage guard | All correctness checks pass |
| **Screen** | Minutes | Three rolling folds (6.2), reduced trees/epochs, cached features | Positive median delta, no red flags |
| **Full** | Budgeted | Full training prefix, all official validation rows | Accept gate passes (6.6) |
| **Confirm** | Selective | 5 seeds, only for candidates competing to be final | CI still excludes zero |

Honesty rule: **a pilot is still an attempt.** Log every smoke and screen run in the ledger. They are simply not full iterations for convergence purposes. Distinguish the three clearly rather than hiding cheap failures.

### 6.4 Calibrate the ladder before committing (Day 1)

Do not trust any assumed ordering of techniques. There is a substantial reproducibility literature (Dacrema et al. 2019; Rendle et al. 2020) showing well-tuned classical methods frequently beat neural recommenders. Assume nothing.

Run five manual probes before building the model ladder. Total compute under one hour.

| Probe | Config | Resolves |
|---|---|---|
| P1 | FM + 5 aggregate features | Does feature engineering alone beat baseline? |
| P2 | LightGBM pointwise, same features | Does GBDT beat FM? |
| P3 | LightGBM lambdarank, same features | Does a ranking loss actually help here? |
| P4 | DeepFM, light tuning | Is the neural branch worth pursuing? |
| P5 | FM + Optuna, 30 trials | Was the baseline simply undertuned? |

Reorder Section 6.7 from the results. If P5 lands near P2 to P4, pour effort into features and the agent rather than architectures.

Order the first *full* experiments by expected effect size, so the convergence rule (4.5) does not fire on three low-value tweaks.

### 6.5 Four independent action drawers

The dominant risk is not a slightly suboptimal technique. It is an action space so narrow the agent can only adjust numbers. Ensure all four exist:

1. **Features.** New derived signals. Likely the single highest-value axis at this data scale. See 6.8.

2. **Loss framing.** Three options, independent of model class:
   - *Pointwise:* each row as independent binary classification, BCE loss. What the FM baseline does.
   - *Pairwise (BPR):* sample one clicked and one non-clicked item **from the same user**, train so `score(clicked) > score(not clicked)`.
   - *Listwise (lambdarank):* optimise NDCG over a whole user's impression list. LightGBM implements this natively via `objective="lambdarank"` plus a per-user `group` array.

   GAUC is per-user while pointwise BCE optimises a global objective, so there is a plausible mismatch. Worth testing early, but **not** a guaranteed win: production CTR systems overwhelmingly use pointwise log-loss and it correlates well with GAUC in practice. Hypothesis, not fact.

3. **Model class.** The ladder in 6.7.

4. **Combination.** Blending accepted nodes. See 6.9.

Two under-explored ideas with high Innovation value, both belonging in `agent/knowledge.md`:

- **Negative sampling strategy.** Which non-clicks to train against (all impressions, in-session, popularity-weighted) materially affects recsys performance and few teams will touch it.
- ~~**Exposure debiasing via the randomised-exposure slice.**~~ **Closed 2026-08-31 (B8).** `is_rand` is 0 across all 1,141,112 training rows — the unbiased slice this idea depends on does not exist in the training window. Kept here only as a documented negative result; do not reattempt without new evidence the slice is present.

### 6.6 Promotion gates

A candidate that passes Full evaluation is not automatically promoted. It must clear every gate.

| Gate | Pass condition | On failure |
|---|---|---|
| **Contract** | Evaluator and data SHA-256 match the run manifest | Abort run; require an explicit new run version |
| **Leakage** | Every feature has a proven pre-row cutoff | Quarantine the candidate, record the lineage |
| **Execution** | Exit 0, no timeout, no OOM, artifacts hashed | Classify and apply bounded recovery |
| **Submission shape** | Correct schema, row count, ordering, finite scores | Reject before computing any metric |
| **Internal evidence** | Positive median delta across the three folds | Discard or revise the mechanism |
| **Statistical** | Bootstrap 95% CI on the delta excludes zero | Record the evidence, do not promote |
| **Segment** | No unacceptable collapse in any reported segment | Require a documented trade-off, or reject |
| **Resources** | Within per-experiment and per-run budget | Stop, shrink, or archive to the Pareto set |

**On the statistical gate.** Baseline seed noise is 0.0008 std and the convergence threshold is 0.002. Without a significance test the agent will spend iterations chasing +0.0004. A fixed minimum-delta threshold is acceptable only if it is at least 0.002; anything below one standard deviation of noise will admit noise as improvement. The bootstrap CI is preferred and is what this plan specifies. Resample **users**, not rows (5.3). Implementation in Appendix A.3.

**Both halves are required, not either.** A CI-only gate with no minimum-delta floor, or a minimum-delta floor applied only from the second full candidate on, both admit exactly the failure this paragraph warns about — see D12 and trap 5/13. The current implementation (`agent/loop.py::_accept_full`) applies both checks starting with the very first full-tier node measured against the official baseline, not only from the second candidate on.

**On the segment gate.** GAUC is weighted by positive count, so a model can lift the headline number while getting worse for most people. Report the primary metric broken down by: user activity quartile, item popularity quartile, and day within the validation window. A candidate that improves overall while collapsing the sparse-user segment is either rejected or promoted with an explicit logged trade-off. Cheap to compute, and it makes an excellent log entry.

### 6.7 Model ladder (provisional, reorder after probes)

| Rung | Model | Rationale |
|---|---|---|
| 0 | Random | Harness self-check, expect 0.4753 |
| 1 | Popularity | Harness self-check, expect 0.5715 |
| 2 | FM | Port of official baseline, 0.5946 hidden |
| 3 | LightGBM (+lambdarank) | Strong on tabular; depends heavily on feature quality |
| 4 | DeepFM | FM plus MLP; learns interactions trees cannot |
| 5 | Multi-task DeepFM | 12 sigmoid heads over feedback signals, read click head only |
| 6 | Sequence model (DIN/SASRec) | Highest ceiling, highest cost. **Skip unless ahead of schedule** |

Known GBDT weakness: trees split one feature at a time and cannot directly learn user-by-video interaction, the core recsys signal. GBDT performance is therefore almost entirely a function of feature quality. This dependency is itself a good thing for the agent to discover and log.

**Optional second GBDT:** CatBoost. Its ordered target statistics were designed for exactly the high-cardinality categorical problem here, and it often needs less manual feature engineering to be competitive. Slower to train. Worth having as an alternative branch rather than a replacement.

### 6.8 Feature policy

What may be used, and under what control. This table is the reference for the leakage guard (task B4).

| Block | Examples | Default | Control |
|---|---|---|---|
| **Identifiers** | user_id, video_id, author_id, music_id, tag | Use | Unknown/missing buckets; no target encoding fitted on evaluation rows |
| **Context** | date index, hour, weekday, tab/scenario, item age | Use | Only values knowable before the impression |
| **Static user** | activity level, creator flag, follow/fan buckets | Use | Record source file and missingness |
| **Static item** | type, upload date, duration, tags | Use | Prefer metadata known before exposure |
| **Temporal aggregates** | 1/3/7/14-day counts and smoothed rates by item, author, tag, user-cross | Use | Strictly expanding or cutoff-based, fitted on earlier rows only |
| **Sequence summaries** | last-N author/category affinity, time since prior event | Use after audit | Chronological construction, no look-ahead |
| **Same-row outcomes** | click, long_view, play_time, like, follow | **Never as input** | Legal as auxiliary training targets only |
| **Monthly aggregate stats file** | item behavioural statistics over one month | **Exclude by default** | Use only if the organisers confirm a pre-split cutoff |
| **Randomised-exposure log** | intervened records | Pre-cutoff rows only | Post-cutoff rows are diagnostic only |

**Time decay.** Prefer exponentially decayed aggregates over flat historical rates. Recent behaviour predicts near-future behaviour better, and the test window sits 8 to 17 days after the training cutoff.

**Empirical-Bayes smoothing.** Raw rates are unusable for sparse groups (an item with 2 impressions and 1 click has a 50% "rate"). Shrink toward the global rate:

```
r = (c + alpha * global_rate) / (n + alpha)
```

Fit `alpha` and the decay half-life on the internal folds (6.2), never on the official validation set.

**The governing rule:** if the time provenance of a feature cannot be proven, the feature is unavailable. A possibly-useful signal is not worth an invalid hidden-test result.

### 6.9 Blending

**Majority voting is categorically wrong here.** Voting emits a label. Labels produce ties. Ties destroy NDCG, which is position-sensitive (5.3). Keep every score continuous to the submission file. Never threshold.

Raw score averaging is also weak: lambdarank outputs unbounded reals, DeepFM outputs [0,1] probabilities. Averaging raw lets the larger-scaled model dominate.

| Method | Description | When it wins |
|---|---|---|
| **Rank average (default)** | Convert to ranks within each user, average ranks | Immune to scale and calibration differences |
| Logit average | Convert both to log-odds, average | Both genuinely calibrated |
| Weighted rank average | `w·A + (1-w)·B`, w tuned | One model clearly stronger |
| Reciprocal rank fusion | Sum of `1/(k+rank)` | Favours items either model ranks highly |
| Stacking | Learn combination weights | Most powerful, highest overfit risk |

**Rank averaging within user is the hardcoded default.**

**Fit blend weights on the internal folds, never on the official validation set.** That set already selects the best checkpoint; reusing it for weights inflates the estimate. Keep it to a single weight parameter.

**Diagnostic before blending at all:** compute mean per-user Spearman rank correlation between the two models.

- \> 0.95: near-identical, blending buys nothing, skip
- 0.7 to 0.9: sweet spot
- \< 0.5: one model likely broken, investigate first

Rules: a blend must beat **both** parents on the internal folds *and* the official metric, and only accepted nodes are eligible as parents. A blend that wins on one noisy slice is too fragile for a one-shot hidden test.

### 6.10 Hyperparameter search spaces

The agent proposes a *space*, never a value. Sensible starting bounds, ordered by expected impact.

**Neural (DeepFM):**

| Parameter | Range | Note |
|---|---|---|
| `emb_dim` | {8, 16, 32, 64} | Largest single lever |
| `l2` on embeddings | 1e-6 to 1e-3, log | Sparse ID embeddings overfit viciously |
| `lr` (Adam) | 1e-4 to 1e-2, log | |
| `epochs` | 1 to 5, early stop patience 1 | **CTR models at this scale usually peak after 1 to 3 epochs then degrade** |
| `batch_size` | {1024, 2048, 4096, 8192} | |
| `mlp_dims` | [256,128,64] and variants | |
| `dropout` | 0.0 to 0.5 | |

**LightGBM:**

| Parameter | Range |
|---|---|
| `num_leaves` | 31 to 255 |
| `min_data_in_leaf` | 20 to 500 |
| `learning_rate` | 0.01 to 0.2, log, paired with `num_boost_round` |
| `feature_fraction` | 0.6 to 1.0 |
| `bagging_fraction` | 0.6 to 1.0 |
| `lambda_l2` | 1e-3 to 10, log |

Use Optuna's TPE sampler with `MedianPruner`. TPE is the Bayesian-optimisation approach: it models which regions produced good scores and samples from the promising ones, converging far faster than random search and vastly faster than grid. `MedianPruner` kills underperforming trials partway through, typically halving compute on a 20-trial study.

Early stopping always uses an internal fold, never the official validation set.

### 6.11 Compute posture

**Assume CPU.** LightGBM and a 16-dim DeepFM both run fine on CPU at 1.1M rows. Cheap iterations determine how many the agent completes, and GPU-hours are directly scored at 15%. Use GPU only for DeepFM epochs if available.

Cache immutable feature blocks by `(data_cutoff, feature_spec_hash)`. Model-only experiments must not recompute features.

---

## 7. System architecture

### 7.1 Control flow

```
                    ┌─────────────────────────────┐
                    │      agent/loop.py          │
                    │  propose → build → smoke →  │
                    │  screen → full → gate →     │
                    │  record → reflect           │
                    └──────────┬──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐   ┌────────────────────┐   ┌────────────────┐
│ agent/        │   │ pipeline/train.py  │   │ agent/gate.py  │
│ propose.py    │   │  run_experiment()  │   │ bootstrap CI   │
│  (Claude)     │   │  fidelity tiers    │   │ + segments     │
└───────────────┘   └─────────┬──────────┘   └────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌──────────────┐ ┌────────────┐ ┌──────────────┐
      │ pipeline/    │ │ pipeline/  │ │ pipeline/    │
      │ data.py      │ │ features.py│ │ models/      │
      │ fixed split  │ │ registry + │ │ fm, lgbm,    │
      │ + folds      │ │ EB/decay   │ │ deepfm, blend│
      └──────────────┘ └────────────┘ └──────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ pipeline/          │
                    │ evaluate.py        │
                    │ STARTER KIT,       │
                    │ HASHED, IMMUTABLE  │
                    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ logs/nodes/*.json  │
                    │ = run-log          │
                    │   deliverable      │
                    └────────────────────┘
```

### 7.2 Trust boundary

The LLM has broad reasoning freedom inside a narrow typed contract. It **cannot**: modify the evaluator, read hidden-test data, approve its own promotion, or delete a failed experiment from the ledger.

Draw this boundary explicitly in the architecture diagram for the pitch. "The LLM proposes; locked services own data access, execution, scoring, and promotion" is a one-sentence answer to the obvious judge question about whether the agent is gaming its own metric.

### 7.3 Two-tier action space

The brief says the agent must write the code for each stage. Free-form generation is expressive but fragile, and robustness is graded. The resolution:

- **Tier 1, typed configuration.** Model class, loss framing, feature set, hyperparameter search spaces. The agent emits validated JSON. Failures are schema errors, not tracebacks. Expect ~80% of iterations here.
- **Tier 2, code generation.** New feature transforms and model classes, written against a frozen interface, executed in an isolated subprocess. Where novel work and visible recovery happen.

**Document this trade-off in the README and pitch.** A judge seeing only Tier 1 calls it a config sweeper. A judge seeing only Tier 2 watches it crash live.

---

# PART IV — BUILD SPECIFICATION

## 8. Frozen contracts

Everything here is a hard interface. Do not change without team agreement.

### 8.1 Repository layout

```
pyproject.toml
.env.example                  # ANTHROPIC_API_KEY=
.gitignore                    # .env, data/ excluded
README.md
AGENT_PLAN.md                 # this file

pipeline/
  data.py                     # fixed splits, folds, date-order assertion
  features.py                 # registry, leakage guard, decay/EB helpers
  train.py                    # run_experiment() with fidelity tiers
  evaluate.py                 # STARTER KIT, IMMUTABLE, HASHED
  submit.py                   # STARTER KIT, IMMUTABLE, HASHED
  models/
    __init__.py               # MODEL_REGISTRY
    popularity.py  fm.py  lgbm.py  deepfm.py  blend.py

agent/
  loop.py                     # driver
  schema.py                   # Pydantic Config + Action
  propose.py                  # LLM call
  execute.py                  # Action -> node
  store.py                    # node tree persistence
  gate.py                     # bootstrap CI + segment checks
  manifest.py                 # MetricProfile, hashing, preflight
  knowledge.md                # recsys priors for the LLM
  prompts/  propose.md  repair.md

tools/
  probes.py                   # the five Day-1 probes
  report.py                   # deliverable tables
  finalise.py                 # refit + 5-seed + submission

tests/
  test_data_split.py  test_folds.py  test_leakage.py
  test_gate.py  test_schema.py  test_models.py  test_manifest.py

logs/
  nodes/                      # one JSON per node
  run.jsonl                   # append-only event stream
  manifest.json               # run-level contract fingerprint
```

### 8.2 Data loader and folds

```python
# pipeline/data.py

TRAIN_END = "2022-04-21"
VAL_START, VAL_END = "2022-04-22", "2022-04-28"
TEST_START = "2022-04-29"

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

def internal_folds() -> list[tuple[DataFrame, DataFrame]]:
    """Three expanding-window folds inside the training period.

    MUST assert fold_train.date.max() < fold_val.date.min() for each.
    Used for screening, early stopping, blend weights, and EB
    hyperparameters. NEVER touches the official validation window.
    """
```

### 8.3 MetricProfile and run manifest

```yaml
# logs/manifest.json, written once at preflight, immutable for the run
metric_profile:
  source: shipped_evaluate.py
  evaluator_sha256: <computed at preflight>
  submit_checker_sha256: <computed at preflight>
  data_sha256: <computed at preflight>
  target_label: <read from starter kit>
  group_key: user_id
  metrics: <read from starter kit>
  cutoffs: <read from starter kit>
  aggregation: <read from starter kit>
  zero_positive_rule: <read from starter kit>
  baseline_validation: 0.6016
  baseline_seed_std: 0.0008
  convergence: {epsilon: 0.002, no_improvement_iterations: 3}
  submission:
    columns: [row_id, user_id, video_id, score]
    finite_scores_only: true
    preserve_repeated_pairs: true
```

Preflight recomputes all three hashes and **fails closed** on mismatch. Every node record carries the manifest hash, so a mid-run evaluator change cannot silently invalidate earlier results.

### 8.4 Feature registry

```python
# pipeline/features.py

FEATURES: dict[str, Callable] = {}

def feature(name: str):
    """Decorator registering a feature builder."""

# Every feature has this exact signature:
def my_feature(train_df, target_df) -> np.ndarray:
    """Fit statistics on train_df ONLY. Apply to target_df.
    Return a 1-D array of len(target_df).
    """

FORBIDDEN_SAME_ROW = [
    "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "long_view", "is_profile_enter", "is_click_pause",
    "play_time_ms", "duration_ms", "profile_stay_time",
]
# Legal as auxiliary TRAINING TARGETS and as HISTORICAL AGGREGATES
# over a user's past rows. Never as same-row input features.

EXCLUDED_SOURCES = ["item_statistics_monthly"]  # see 6.8
```

### 8.5 Experiment runner

```python
# pipeline/train.py

def run_experiment(config: dict, fidelity: str = "full",
                   seed: int = 42, timeout_s: int = 1800) -> dict:
    """The single entry point the agent calls.

    fidelity in {"smoke", "screen", "full", "confirm"} per 6.3.
      smoke   -> 1000-row fit, correctness checks only, no metrics
      screen  -> three internal folds, reduced budget
      full    -> full train prefix, all official validation rows
      confirm -> full, repeated across 5 seeds

    Success:
    {
      "status": "ok", "fidelity": str,
      "gauc": float, "ndcg": float, "primary": float,
      "fold_primaries": list[float],   # screen/full
      "segments": dict,                # full/confirm, see 6.6
      "val_scores": np.ndarray,        # aligned with val rows
      "val_user_ids": np.ndarray,
      "test_scores": np.ndarray,
      "seconds": float, "gpu_seconds": float, "peak_rss_mb": float,
    }

    Failure:
    {"status": "error", "stage": str, "error_class": str,
     "traceback": str, "seconds": float}

    error_class in {"syntax", "schema", "timeout", "oom",
                    "transient", "leak_suspected"}

    MUST NOT raise. MUST enforce timeout_s. MUST be deterministic given seed.
    MUST return status="error", error_class="leak_suspected" if primary > 0.75.
    """
```

### 8.6 Config and Action schema

```python
# agent/schema.py
from pydantic import BaseModel
from typing import Literal

class Config(BaseModel):
    model: Literal["random", "popularity", "fm", "lgbm", "deepfm",
                   "deepfm_mtl", "blend"]
    loss: Literal["pointwise", "pairwise", "lambdarank"] = "pointwise"
    features: list[str] = ["user_id", "video_id"]
    negative_sampling: Literal["all", "in_session", "pop_weighted"] = "all"
    hparams: dict = {}
    parents: list[str] = []                 # blend only
    blend_method: Literal["rank_avg", "logit_avg",
                          "weighted_rank", "rrf"] = "rank_avg"
    seed: int = 42

class Action(BaseModel):
    hypothesis: str                          # required, graded
    reasoning: str
    type: Literal["config", "tune", "code", "blend"]
    family: Literal["feature", "model", "objective", "training", "ensemble"]
    parent: str                              # node id to branch from
    config: Config | None = None
    search_space: dict | None = None         # tune only
    budget: int = 20                         # tune only, trial count
    code: str | None = None                  # code only, Python source
```

`family` exists so the search controller can enforce diversity: cover all five families before repeatedly refining one.

### 8.7 Node record

Written to `logs/nodes/<id>.json` on completion. **This file is the run-log deliverable.**

```json
{
  "id": "n017",
  "parent": "n009",
  "family": "objective",
  "hypothesis": "lambdarank should beat pointwise BCE since GAUC is per-user",
  "reasoning": "...",
  "action_type": "config",
  "fidelity": "full",
  "config": {},
  "diff": "loss: pointwise -> lambdarank",
  "manifest_sha256": "...",
  "metrics": {"gauc": 0.681, "ndcg": 0.547, "primary": 0.614},
  "fold_primaries": [0.609, 0.612, 0.617],
  "segments": {"activity_q1": 0.58, "activity_q4": 0.64,
               "pop_q1": 0.60, "pop_q4": 0.63},
  "delta_vs_best": 0.0124,
  "ci_95": [0.008, 0.017],
  "gates": {"contract": true, "leakage": true, "execution": true,
            "submission": true, "internal": true, "statistical": true,
            "segment": true, "resources": true},
  "accepted": true,
  "status": "ok",
  "errors": [],
  "repair_attempted": false,
  "tokens": {"in": 4210, "out": 890, "model": "claude-opus-4-5"},
  "seconds": 94.2, "gpu_seconds": 0.0,
  "manual_intervention": false,
  "timestamp": "2026-08-29T14:03:11Z"
}
```

### 8.8 Accept gate

```python
# agent/gate.py

def accept(cand_scores, best_scores, user_ids, n_boot=1000, seed=0):
    """Bootstrap over USERS (not rows). Returns (accepted: bool, ci: tuple).
    accepted is True iff the 95% CI on (candidate - best) primary
    excludes zero on the low side.
    """

def segments(scores, user_ids, meta) -> dict:
    """Primary metric by user-activity quartile, item-popularity
    quartile, and day within the evaluation window.
    """
```

### 8.9 Model registry

```python
# pipeline/models/__init__.py

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}

class BaseModel(Protocol):
    def fit(self, X_train, y_train, X_val, y_val, groups=None) -> None: ...
    def predict(self, X) -> np.ndarray: ...   # raw scores, higher = more likely
```

---

## 9. Workstreams and task assignment

**Assignment rationale.** Kaiwen owns the agent loop because it is backend orchestration (subprocess management, retries, state, API integration) and is on the critical path for 40% of the score. The other three are a proposal, swap them on Day 0 if they do not match actual strengths. Each workstream owns its files exclusively to minimise merge conflicts.

**Critical path:** B1 → C1 → C2 → (everything else). C2 is a hard gate: if random and popularity do not reproduce 0.4753 and 0.5715, nothing downstream means anything.

### 9.1 Workstream A: Agent loop and orchestration
**Owner: Kaiwen** · Files: `agent/loop.py`, `propose.py`, `execute.py`, `store.py`, `schema.py`, `prompts/*`

| ID | Task | Depends on | Acceptance criteria |
|---|---|---|---|
| A1 | Pydantic `Config` and `Action` schemas per 8.6 | none | `test_schema.py`: valid JSON parses; unknown model name rejected readably; missing `hypothesis` or `family` rejected |
| A2 | Node store: write/read/list per 8.7 | A1 | Write-read round-trips identically; `best_node()` returns highest accepted primary; tree survives process restart |
| A3 | `propose()` LLM call with structured output | A1, A2 | Given a fake node history, returns a valid `Action`; token usage captured; two-tier routing (Opus propose, Haiku repair) |
| A4 | `execute()`: Action to node, subprocess isolation for `type="code"` | A1, C1 | Code that raises returns `status="error"` with traceback rather than crashing the parent; code that hangs is killed at timeout |
| A5 | Recovery policy per error class | A4 | `syntax` gets one repair; `oom` retries at reduced batch size; `timeout` reduces fidelity; `transient` backs off. All attempts logged. 3 consecutive dead nodes force a branch |
| A6 | Main loop with fidelity escalation (smoke→screen→full) | A3, A4, A5, D2 | Runs 10 iterations unattended, zero human input; `manual_intervention` false on every node; cheap tiers filter the majority |
| A7 | Family-diverse + epsilon-greedy parent selection | A2, A6 | All five families covered before any is refined twice; over 40 iterations, ≥15% of nodes branch from a non-best parent |
| A10 | Ablation-driven drawer targeting. `ablate.py` takes the current best node and generates cheap variants: each registered feature group disabled in turn; loss swapped to pointwise; each ensemble member dropped. Runs them via `execute()` at screening fidelity, ranks component sensitivity (abs validation-primary delta), stores the sensitivity table on the node, and injects it into the `propose()` prompt. | A2, A4, B2, C1 | `test_ablate.py`: on a synthetic pipeline where one feature group carries all the signal, that group ranks first by sensitivity. Sensitivity table present in node JSON. `propose()` prompt string contains the rendered table. Total ablation wall-clock for one round ≤ 5 min at screening fidelity. |
| A11 | Convergence-aware scheduler. `schedule.py` maintains (a) a strike counter: consecutive iterations with validation-primary improvement ≤ 0.002, reset on any improvement > 0.002; (b) a hedge queue of high-probability actions (Optuna refinement of the current best config, adding the top unused feature from the last ablation table). When strike counter = 2, the next action is forced from the hedge queue instead of free exploration. Counter and forced-hedge events are logged per iteration. **Motivated by a real incident:** a run converged after only 4 full evaluations — the earliest point the rule can fire — see trap 13 in Section 11. | A6, A10, C4 | `test_schedule.py`: in a simulated run where free exploration fails 6 times in a row, the run never reaches 3 strikes because hedges fire at strike 2. Strike counter value appears in every iteration's log line. Hedge-forced iterations are marked `scheduler_forced: true` in the node. |
| A12 | Citations in hypotheses. Add optional `references: list[str]` to the `Action` schema. `propose()` prompt instructs the model to name the published method behind each hypothesis (e.g. "D2Q duration grouping, Zhan et al. KDD 2022") drawing from the idea bank (Appendix D). Run-log renderer prints the citation with the hypothesis. | A1, A3 | Schema round-trips with and without `references`. A propose call against the idea-bank prompt returns at least one populated citation in 3 of 5 sampled generations. Rendered run log shows `[ref: ...]` next to the hypothesis. `agent/prompts/idea_bank.md` not yet created — content is in Appendix D, pending wiring into `propose()`. |

**Kaiwen also owns** `.env` handling and `pre-commit` with `detect-secrets` from the first commit. Secrets in git history are an unrecoverable scored failure.

**Open question carried from A10–A12 planning:** does refitting the validation-best config on train+validation before final submission comply with "the scored submission is the validation-best checkpoint," or must the submitted checkpoint be literally the one selected on validation? Unlike an earlier draft of this addition, **do not freeze D8** over this — `tools/finalise.py`'s refit is already built and shipped; see the open question in Section 15 instead of blocking already-working code on an unconfirmed reading of the brief.

### 9.2 Workstream B: Data, features, and leakage defence
**Owner: Malvika** · Files: `pipeline/data.py`, `pipeline/features.py`

| ID | Task | Depends on | Acceptance criteria |
|---|---|---|---|
| B1 | `load()` per 8.2 with date-order assertion | none | `test_data_split.py`: row counts exactly 1,141,112 / 124,909 / 170,588; assertion fires if splits swapped; row order matches starter kit |
| B2 | `internal_folds()` per 8.2 | B1 | `test_folds.py`: three folds; each asserts train.max < val.min; no fold touches 22 April or later; expanding window verified |
| B3 | Feature registry per 8.4 | B1 | Register-then-call returns correct-length array; unknown feature name raises before any training starts |
| B4 | Seven baseline aggregate features | B3 | `user_ctr`, `video_ctr`, `video_impressions`, `user_activity`, `user_tag_affinity`, `hour_of_day`, `day_of_week`. Each fits on train only, each unit-tested on a 100-row fixture |
| B5 | Time-decay + empirical-Bayes smoothing helpers | B4, B2 | Exponential decay with configurable half-life; `r = (c + α·g)/(n + α)`; α and half-life fitted on internal folds only; sparse-group test (n=2) returns near-global rate |
| B6 | Leakage guard | B3 | `test_leakage.py`: a feature reading `target_df["click"]` rejected; a `FORBIDDEN_SAME_ROW` column of `target_df` rejected; `EXCLUDED_SOURCES` rejected; legitimate historical aggregates pass |
| B7 | Negative sampling strategies | B1 | Three strategies (all, in_session, pop_weighted) selectable via config |
| B8 | ~~**Stretch:** randomised-exposure slice loader + IPS reweighting~~ **WONTFIX — impossible on this data.** | B1, B6 | **Closed by measurement, not by descoping.** `is_rand` is **0 for all 1,141,112 training rows** — the randomised-exposure slice KuaiRand documents is absent from the Pure variant's training window. There is no unbiased slice from which to estimate an exposure propensity, so IPS reweighting has no estimable weights. Independently confirmed twice: once by this workstream's own investigation, once by the data verification in `RECSYS_COURSE_ANALYSIS.md` §2.5. **This is a deliverable, not a gap** — a documented negative result with the row count that produced it. Cite it in the writeup; do not reopen. |
| B10 | Duration-bias feature pack: `video_duration`, `duration_bucket` (train-quantile groups, default 10, D2Q-style), `pcr_hist` (historical play_time/duration ratio aggregates per user and per video, train-only, with the existing time-decay + EB smoothing), `long_view_rate_by_duration_group` (target encoding of `long_view` within duration bucket, train-only, EB-smoothed). | B2, B3 | Each feature fits on train only and passes the existing leakage tests. Quantile bucket edges computed on train are reused unchanged for validation/test (assert no re-fitting). All four registered and callable by name. |
| B11 | Auxiliary-signal historical rates: per-user and per-video past rates for `click`, `like`, `follow`, `long_view`, time-decayed and EB-smoothed, where the value for a row on date *t* uses only rows with date < *t*. | B2, B3 | `test_aux_rates.py` on synthetic data: a feature value never changes when future-dated rows are edited; same-day rows are excluded. Registered in the feature registry. |
| B12 | Within-user variance screen. `screen.py` computes, for every registered feature, its mean within-user variance on the validation split. Features below threshold (default 1e-9) are flagged `metric_inert: true` — constant within user, hence invisible to GAUC/nDCG@5 except through interactions (5.3, trap re: pure user-side terms). The report feeds A10's prompt injection. | B2, B3, B10, B11 | `test_screen.py`: a pure user-level feature (e.g. `user_activity`) is flagged inert; a video-varying feature is not. Report file schema documented in Section 8 style. |
| B13 | Data-usage checker. `data_usage.py` runs pre-flight: lists every data file shipped in the starter kit / KuaiRand download (user features, video features, both log files) and asserts each is either joined into the pipeline or explicitly listed in an `EXCLUDED = {...: reason}` map. | B1 | Running with a shipped file neither joined nor excluded exits non-zero and names the file. The exclusion map with reasons is printed into the run log (judge-visible). |
| B14 | **New — cheap probe, gates C10.** `sim_to_history`: item–item co-occurrence similarity (cosine on co-view/co-like counts; optionally Swing) between a candidate video and the user's train-only interaction history, aggregated (mean of top-k, or decayed mean) into a single registered feature. This is the organisers' #2-ranked headroom direction (user-history × candidate interaction) at feature cost instead of model cost — see `RECSYS_COURSE_ANALYSIS.md` §3.1. | B2, B3, B11 | Fits on train-only co-occurrence counts, passes leakage tests. `test_sim_to_history.py`: a candidate video that co-occurs heavily with a user's history scores higher than one that never does, on a synthetic fixture. Registered and callable by name. **Decision gate:** FM + 5 official fields + `sim_to_history`, confirm-tier (5 seeds) vs. baseline. Clears `MIN_DELTA_FLOOR` with a CI excluding zero → funds C10. Lands inside noise → C10 is not built; go straight to B10's re-spec and C11/C12. |
| B10-fix | **New — correction to B10, not new work. Upgraded from "likely mis-specified" to confirmed dead by `tools/screen.py` (run 2026-08-31): `pcr_hist` is `metric_inert` with mean within-user variance exactly `0.000e+00` — it contributes literally nothing to ranking, not just less than hoped.** Root cause is structural, not the duration confound alone: `pcr_hist` (`pipeline/features.py::pcr_hist`) is a per-**user** aggregate (`groupby("user_id")`), so every candidate video for a given user receives the identical value — it cannot discriminate within a user's 4-candidate list by construction, independent of the duration-confound issue described in `RECSYS_COURSE_ANALYSIS.md` §2.4/§3.4 (raw completion ratio duration-confounded 5.2× vs. the label's 1.4×). Re-normalising the formula without changing the grain would stay inert. | B10 | Add a per-**video** completion-ratio history (varies by candidate, not just by user) — the "natural follow-up... not attempted" already flagged in the `pcr_hist` docstring — duration-normalised per §3.4 (compare against the bucket's own median completion, not raw play_time/duration). Confirm via `tools/screen.py` that the new feature is not `metric_inert`. Ablate old vs new via A10 at screen tier; report the delta. A regression is an acceptable, reportable outcome. The other 7 features `screen.py` flagged inert (`user_ctr`, `user_activity`, `user_ctr_decayed`, `user_{click,like,follow,long_view}_rate_decayed`) are pure per-user aggregates with the same structural cause — expected, not bugs; leave as-is, they may still contribute through blending/interactions per §5.4. |

**Note on B8:** closed — see row above. **Highest current upside item for Innovation scoring is now B14.**

### 9.3 Workstream C: Models, losses, and tuning
**Owner: Ethan** · Files: `pipeline/models/*`, `pipeline/train.py`

| ID | Task | Depends on | Acceptance criteria |
|---|---|---|---|
| C1 | `run_experiment()` per 8.5 with all four fidelity tiers | B1, B2, B3 | Never raises; enforces timeout; identical output across two same-seed runs; leak canary fires above 0.75; smoke tier completes in under 10s |
| C2 | Random + popularity models | C1 | Reproduce 0.4753 and 0.5715 within noise. **Gate: nothing else proceeds until this passes** |
| C3 | FM ported from starter kit | C1 | Reproduces validation primary 0.6016 within one seed-std |
| C4 | LightGBM, pointwise and lambdarank | C1, B4 | Both objectives run; lambdarank gets a correct per-user `group` array; early stopping on an internal fold, never on official validation |
| C5 | DeepFM in PyTorch, pointwise | C1, B4 | Trains on CPU in under 10 min; early stopping patience 1; FM second-order term uses the O(n) identity |
| C6 | Optuna with `MedianPruner` and SQLite storage | C1, B2 | A 20-trial study completes; pruning cuts total time by 30%+; study resumable after kill; objective evaluated on internal folds |
| C7 | Blending per 6.9, four methods, `rank_avg` default | C1, D2 | Blend of two fixed nodes reproduces exactly; per-user Spearman reported; blend rejected unless it beats both parents on folds and official metric; weights fitted on internal folds only |
| C8 | Within-user pairwise loss as a loss-drawer entry: BPR-style logistic pairwise loss with pairs drawn only within the same user, for the FM stack. Also, LightGBM LambdaRank `group` boundaries must equal user boundaries — the structural match between loss and the per-user metrics. | C1, C3 | **Mostly implemented.** `test_pairwise.py`: no training pair spans two users; BPR variant trains end-to-end through the standard `(user_ids, labels, scores)` interface. — `pipeline/models/fm.py` has `loss="pairwise"` (BPR sampled within-user via `_build_pair_index`/`_sample_pairs`), covered by `tests/test_models.py::test_fm_pairwise_*`: determinism, a "not a silent no-op" regression test, and the within-user-pairs guard. Confirm-tier check on real data: delta over baseline +0.0008, ~1σ of seed noise — a real result (not yet a proven win), see trap 13. **Now complete:** `lgbm.py::_assert_group_boundaries` checks each lambdarank group is exactly one user's contiguous rows and is called from `fit()` on both the train and validation groupings; `tests/test_lgbm.py` covers five mis-grouping modes plus a guard test that drives the production path with a corrupted `_group_order`, verified to fail when the assertion call is removed. LightGBM takes `group` positionally and validates nothing beyond the total, so a mismatched grouping would otherwise train silently with ranks optimised across user boundaries. |
| C9 | *(Stretch — only after C1–C8 green.)* Multi-task DeepFM: shared embedding table, auxiliary sigmoid heads for `click` and `like` with Optuna-tunable loss weights. Inference emits only the `long_view` head score. | C6 | **Implemented and merged.** `DeepFMMultiTask` (`pipeline/models/deepfm.py`) has `AUX_TARGETS = ("is_click", "is_like")`, Optuna-tunable `aux_click_weight`/`aux_like_weight`, primary-head-only early stopping. RNG-quarantined aux-head construction (saving/restoring `torch.get_rng_state()` around `nn.Linear` init) so aux weights = 0 is bit-exact to the single-task model. `deepfm_mtl` removed from `UNIMPLEMENTED_MODELS` in `agent/schema.py`. **Not yet used:** this is the single-task-baseline half of Phase 4 (§3.5) of `RECSYS_COURSE_ANALYSIS.md` — fusing the aux head *outputs* into the ranking score at inference, rather than only using them as training regularisers. That fusion is C13 below. |
| C10 | **New — gated on B14.** DIN: attention over the user's LastN history (n ≈ 10–30, median 31 rows/user measured) with a time-since-interaction embedding, replacing plain mean-pooling for the history representation used alongside `sim_to_history`. See §3.2. **Do not build if B14 lands inside noise** — SIM-style truncation is explicitly out (sequences are too short to need it; see §2.3). | B14, C5 | Only funded if B14 clears its confirm-tier gate. Attention form only, never plain average-pooling (that variant is not a test of the hypothesis). Trains on CPU in reasonable time given n≤31. Confirm-tier (5 seeds) before any claim of improvement. |
| C11 | **New — independent of C10, cheap.** LHUC / PPNet-style user-conditioned modulation on DeepFM: a small per-user gating vector rescales hidden-layer activations. Cheapest structural change aimed at within-user discrimination. See §3.3. | C5 | **Implemented.** `hparams: {"lhuc": true}` on `deepfm`/`deepfm_mtl` (not a new model name — `Config.model` is a frozen Literal, §8.6). `LHUCGate` in `pipeline/models/deepfm.py::_build_network` projects the row's *detached* `user_id` embedding (PPNet stop-gradient) to one `2*sigmoid` gate per MLP hidden unit; parameterised on the embedding rather than a per-user table, so it costs `emb_dim x sum(mlp_dims)` and generalises to held-out users. Zero-initialised, so it is the exact identity before training and `on` vs `off` measures the mechanism, not a different random start. Ablatable via A10 as `block:lhuc`. Raises rather than silently disabling when `user_id` is absent. **Not yet claimed as a win — needs confirm tier (5 seeds).** |
| C12 | **New — independent of C10/C11, cheap.** SENet field-weighting: a squeeze-excitation block over the field embeddings before the FM/DeepFM interaction layer, learning per-field importance weights per example. See §3.6. | C5 | **Implemented.** `hparams: {"senet": true}`, composable with `lhuc`. Reweights the embeddings feeding both the second-order term and the MLP; the first-order table is untouched, as in FiBiNET. Zero-initialised to the exact identity, same rationale as C11. Ablatable via A10 as `block:senet`. **Two deliberate deviations from the paper, forced by having only five fields:** a floor of two bottleneck units and a `2*sigmoid` output instead of ReLU — at `n_fields=5` the paper's `n_fields/r` bottleneck is a *single* ReLU that, sitting negative, emits zero for every row and collapses the block to one constant weight per field, i.e. a silent no-op that still reports itself active (caught by `test_c12_senet_reweights_fields_per_row`, which asserts the weights vary *across rows*). **Not yet claimed as a win — needs confirm tier.** |

**C11/C12 first measurement (screen tier, seed 0, capped budget — direction only, not a result).** Absolute values are far below 0.6016 because screen tier trains a capped model on internal folds, so read only the deltas:

| variant | screen primary | delta |
|---|---|---|
| `deepfm` plain | 0.530672 | — |
| `+ lhuc` (C11) | 0.531053 | +0.000381 |
| `+ senet` (C12) | 0.531052 | +0.000380 |
| `+ both` | 0.531613 | +0.000942 |

All three sit **inside the noise floor** — seed std is 0.0008 and `MIN_DELTA_FLOOR` is 0.002, so the best of them is roughly 1.2σ and a fifth of the promotion bar. Both blocks are confirmed live (they move the score, and the ablation/mutation tests prove they are wired to something), and they compose roughly additively, which is what independent mechanisms should do. But **nothing here is a win**, and on this evidence neither should be promoted. The open question is whether the deltas grow at full budget — screen tier caps epochs and MLP width, and a gating mechanism has the least to work with when the network it modulates is smallest, so this is exactly the case A10's table warns can look inert at screen and matter at full. Next step is a full-tier run, then confirm tier only if full clears the floor.
| C13 | **New — Phase 4, after C9's single-task baseline is trustworthy.** Multi-task score fusion: combine the `long_view` head with the `is_click`/`is_like` aux head outputs into the final ranking score via rank-based fusion (reuse C7's blending machinery), weights fit on internal folds only. See §3.5. | C9, C7 | Fusion weights fit on internal folds, never official validation. Rejected unless it beats the `long_view`-only head on folds and on the official metric — same acceptance discipline as C7. |

**Deprioritised:** sequence models beyond DIN (SASRec-style, rung 6) not attempted — median 31 rows/user does not justify them. CatBoost optional. Session-position proxy (§3.7) unscheduled, attempt only if time allows.

### 9.4 Workstream D: Integrity, statistics, reporting, demo
**Owner: Pinxin** · Files: `pipeline/evaluate.py` (copy only), `agent/gate.py`, `agent/manifest.py`, `tools/*`, `README.md`, `agent/knowledge.md`

| ID | Task | Depends on | Acceptance criteria |
|---|---|---|---|
| D1 | Copy `evaluate.py` / `submit.py` verbatim; typed wrapper | B1 | Wrapper reproduces starter-kit numbers exactly on a fixture; originals unmodified |
| D2 | `MetricProfile` + run manifest + preflight hashing per 8.3 | D1 | `test_manifest.py`: hashes computed at preflight; a modified evaluator fails preflight closed; every node carries the manifest hash |
| D3 | Bootstrap accept gate per 8.8 | D1 | `test_gate.py`: identical vectors rejected; a known +0.01 accepted; a +0.0003 rejected; resampling is over users |
| D4 | Segment metrics per 6.6 | D3 | Primary by activity quartile, popularity quartile, and day; returned in every full-fidelity result |
| D5 | The five Day-1 probes in `tools/probes.py` | C2, C3, C4, C5 | Single command runs all five, prints a comparison table. **Output determines Section 6.7 ordering** |
| D6 | `agent/knowledge.md`: recsys priors | none | Covers FM, DeepFM, LightGBM lambdarank, multi-task heads, BPR, negative sampling, time decay, EB smoothing, exposure debiasing, blending. One line of rationale each. Leakage rules and the feature policy table in bold |
| D7 | `tools/report.py`: all deliverable tables from `logs/nodes/*.json` | A2 | Emits results table, absolute delta vs baseline, total tokens, total GPU-hours, manual intervention count, pilot-vs-full breakdown. Zero hand-assembly |
| D8 | `tools/finalise.py`: refit on train+val, 5 seeds, submission | C1, D1 | Refits the chosen config on the full permitted training period (train + validation) before predicting test; averages 5 seeds; `submit.py --check` passes; `row_id` alignment verified |
| D9 | README, trajectory plot, Devpost writeup | D7, D8 | One-command startup verified from a clean clone; README covers overview, setup, reproduction, limitations, contributions, and the unresolved metric discrepancy |
| D10 | *(Optional)* walkthrough video with injected-failure demo | D9 | See Section 14.2 |
| D12 | Promotion-threshold correction (**amends 6.6's statistical gate — closes the noise-floor issue named in trap 5**). Accept requires BOTH: user-level bootstrap 95% CI of the primary-delta excludes 0, AND point delta ≥ 0.002 (= ε, ≈ 2.5σ of the baseline's 5-seed std 0.0008). | D2 | **Implemented, at the call site rather than in `gate.accept()` itself.** `test_gate.py`: synthetic +0.001 delta with tight CI is rejected (threshold), +0.004 with CI spanning 0 is rejected (noise), +0.004 with CI > 0 is accepted. — The frozen 8.8 contract for `gate.accept()` is untouched (it still returns the bootstrap-CI verdict alone); the combined rule lives in `agent/loop.py::_accept_full`, which requires both the CI check and `primary - max(baseline, incumbent) >= MIN_DELTA_FLOOR` before accepting a full-tier node, and records `gates`/`ci_95`/`delta_vs_best` on the node so the decision is auditable. `tests/test_loop.py` covers all three cases from this row's acceptance criteria by name. This closes a real incident: a run once accepted its first full node (primary 0.5837, *below* the shipped 0.6016 baseline) unconditionally and used it as the reference for the rest of the run. |
| D13 | Leakage canary probes, pre-flight: (a) permute `long_view` labels within user on train, retrain the cheapest model — validation GAUC must land in 0.5 ± 0.02; (b) a deliberately leaky fixture feature (built from future `long_view`) must trip the alarm. Run once before iteration 1 and record the result in the run log. | B2, C2 | Both canaries implemented as pytest + a `--preflight` CLI mode. Injected leaky feature raises; clean pipeline passes. Preflight result appears in the run log header. |
| D14 | Judge-visible reporting of the new machinery: run-log renderer shows, per iteration, the strike counter (A11), any `scheduler_forced` flag, the latest ablation sensitivity table (A10), citations (A12), and the metric-inert feature list (B12). **New:** also surface the candidate-count diagnostics from §5.4 (median 4 candidates/user, 57.8% GAUC-usable) once, in the report header — without this, a +0.002 delta reads as trivial to a judge; with it, it's legible as a large effect on this task. Add the ablation table + solution tree to the demo script. | A10, A11, A12, B12, D7 | Rendered log for a 5-iteration mock run contains all five elements plus the header diagnostics. Demo script section drafted with one screenshot placeholder per element. |

**Pinxin also owns** the live Rich terminal view of the node tree. Roughly 40 lines, and it materially improves the demo over scrolling logs.

**On D8, the refit:** validation ends 28 April, test starts 29 April. Refitting the final configuration on train + validation is temporally legal and hands the model seven extra days of data directly adjacent to the test window. On a dataset with this much drift that is likely worth real points. Do not skip it.

---

## 10. Timeline, descope, and risk

### 10.1 Day 0 (today, ~3 hours) — blocking

Nothing else starts until this is done.

- [ ] All four: run `python3 baseline.py --model fm`, confirm validation primary 0.6016
- [ ] Confirm random 0.4753 and popularity 0.5715
- [ ] Record SHA-256 of the archive, `evaluate.py`, `baseline.py`, and `submit.py`
- [ ] Read `evaluate.py`, `submit.py`, and `data.load()` properly
- [ ] Run `submit.py --make` then `--check`, confirm the round trip
- [ ] **Attend the 14:00 Track 2 webinar.** Questions in Section 15
- [ ] Repo scaffolded, `.gitignore` and `pre-commit` in place, workstreams confirmed or swapped

### 10.2 Day 1

- **Morning:** B1, B2, B3, A1, D1, D2, C1 (contracts layer). **C2 is the gate for everything downstream.**
- **Afternoon:** B4, C3, C4, D3, A2, A3
- **Evening:** **D5, run the five probes.** Reorder Section 6.7 from the results.

### 10.3 Day 2

- **Morning:** A4, A5, A6. First unattended 10-iteration run.
- **Afternoon:** C5, C6, B5, B6, B7, D4, D6
- **Evening:** A7, C7. First unattended 40-iteration run overnight.

### 10.4 Day 3

- **Morning:** B14 (sim_to_history + confirm-tier gate) if time, D7, D8, final run
- **Afternoon:** D9, optional D10
- **Buffer:** 3 hours minimum before deadline

### 10.5 Descope ladder

Cut in this order. Each cut costs less than the one below it.

1. **B14** `sim_to_history` + DIN (C10). High upside, gated probe by design — cut the DIN half first, B14 itself is cheap.
2. **C7** blending. Nice demo moment, modest score impact.
3. **A7** family-diverse selection. Fall back to always extending the best node.
4. **C5** DeepFM. Ship LightGBM plus FM only. Costs headroom, keeps the agent story intact.
5. **B7** negative sampling axis.
6. **D4** segment metrics. Report overall only.

**Never cut:** B1 (date assertion), B2 (internal folds), B6 (leakage guard), C2 (reference rungs), D2 (manifest hashing), D3 (accept gate), D7 (report generation), D8 (validated submission). These are correctness-critical or graded deliverables.

**If one person's workstream is at risk**, reassign toward D. A working loop with two models and complete logs scores far better than four models with a half-built loop.

### 10.6 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Metric ambiguity unresolved | Medium | High | Ask at webinar. `MetricProfile` makes the system profile-agnostic. Note the discrepancy in the README |
| Temporal leakage undetected | Low, if B1/B2/B6 done | Fatal | Assertions, internal folds, leakage guard, 0.75 canary |
| Official validation overfit | Medium | High | Internal folds absorb screening; full evaluations budgeted |
| Agent stalls on repeated errors | Medium | Medium | A5 per-class recovery, dead-node cap, forced branch |
| Compute budget announced and exceeded | Medium | High | CPU-first, fidelity tiers, per-node resource logging from day 1 |
| Overnight run crashes silently | Medium | Medium | Optuna SQLite storage, nodes written on completion, loop resumable from store |
| Single lucky validation gain | Medium | High | Three folds, bootstrap CI, 5-seed confirm tier |
| Submission misalignment | Medium | Fatal | `row_id`-only join, `submit.py --check` in CI |
| Merge conflicts across four people | Medium | Low | Exclusive file ownership, frozen contracts |
| Bonus dataset distraction | Medium | Medium | Hard gate: nothing on 1k/27k until the primary package is complete |

---

# PART V — DELIVERY

## 11. Known traps

Ranked by severity. **Mandatory reading for all four.**

1. **Temporal leakage via k-fold.** Standard practice says cross-validate. Here it is fatal: shuffling puts late-April rows in train and mid-April rows in validation, training on the future. Never shuffle. Use the given date splits and the expanding internal folds in 6.2. Enforced by B1 and B2.

2. **Same-row post-click features.** The 12 feedback signals occur at or after the click on that row. Using `long_view` to predict `click` on the same row is reading the answer. Legal as auxiliary training targets and as historical aggregates over *past* rows. Enforced by B6 plus the 0.75 canary in C1.

3. **The monthly aggregate statistics file.** KuaiRand ships item behavioural statistics aggregated over a month. If that window spans the test period, joining it is leakage, and it looks completely innocuous. Excluded by default via `EXCLUDED_SOURCES`. Use only on confirmed organiser guidance about its cutoff.

4. **Joining on `(user_id, video_id)`.** Not unique. 3.06% of test rows are duplicate pairs, up to 12 repeats. A join silently collapses or duplicates rows. Sort and align on `row_id` only; treat the ID columns as assertions.

5. **A promotion threshold below the noise floor.** Seed std is 0.0008. Any fixed `min_delta` below ~0.002 admits noise as improvement, repeatedly. Use the bootstrap CI (D3), or a threshold of at least 0.002.

6. **Reusing the official validation set.** It selects the best checkpoint. Also using it for early stopping, blend weights, or EB hyperparameters reuses it and inflates the estimate. Use internal folds for all fitting.

7. **One-hot encoding IDs.** 27K users plus 7.6K videos would produce 35K near-empty columns. Use embeddings for neural models, integer label encoding plus native categorical handling for LightGBM.

8. **Oversampling for class imbalance.** Correct for accuracy-based classification, wrong here. Ranking metrics care only about within-user ordering. Oversampling distorts scores and inflates training time.

9. **Threshold-based metrics.** Accuracy, precision, recall, F1 are global and threshold-based. A model predicting "no click" everywhere scores high accuracy and zero on what matters. Use `evaluate.py` only.

10. **Majority voting for ensembling.** Produces labels, labels produce ties, ties destroy NDCG. Rank-average continuous scores instead.

11. **Overtraining.** CTR models at this scale typically peak after 1 to 3 epochs then degrade. Early stopping with patience 1.

12. **Retrofitting logs.** The run-log is a graded deliverable requiring per-iteration hypothesis, diff, metrics, and error events. Build the node schema on Day 1 or the data will not exist on Day 3.

13. **Premature convergence on the first few tries.** The convergence rule (4.5: ε=0.002, N=3) checks only `full`/`confirm` nodes and can fire as early as the 4th one — right after the 3rd, if none improved on the 1st by more than ε. Observed for real: a run converged after exactly 4 full evaluations (the earliest mathematically possible point), having tried only one idea per family. This is not evidence the search space is dry — the plan already names the fix in 4.5 ("order the first full experiments by expected effect size... so the convergence rule does not fire on three low-value tweaks"), and A11's convergence-aware scheduler (a strike counter plus a forced hedge at strike 2) is the proposed follow-up, not yet built. Until then: don't read an early "converged" as a negative result on the whole space — read the individual full-tier deltas and their CIs (D12), and if a result looks ambiguous rather than clearly negative, re-run it at `confirm` tier (5 seeds) before concluding anything. A candidate with a positive point-delta but a CI straddling zero, averaged over 5 seeds, may resolve either way — that's a data point, not a guess.

---

## 12. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Better tracebacks matter when feeding errors to an LLM |
| Dependencies | `uv` | Fast; `uv run` gives one-command startup. `pip` + `venv` acceptable |
| Data | `polars` (or `pandas`) | Polars is 5-10x faster on groupby and has no silent index alignment. Use pandas if the learning cost is not worth it |
| Arrays | `numpy` | Score vectors, bootstrap |
| GBDT | `lightgbm` | `objective="lambdarank"` with per-user `group` |
| Neural | `torch` | CPU wheel sufficient. Not Keras: recsys references are all PyTorch |
| Tuning | `optuna` | TPE sampler, `MedianPruner`, SQLite `RDBStorage` for crash resilience |
| LLM | `anthropic` | Opus 4.5 for propose/reflect, Haiku 4.5 for repair. Log `usage.input_tokens` / `output_tokens` on every call |
| Validation | `pydantic>=2` | Highest-leverage reliability choice in the stack |
| Terminal UI | `rich` | Live node tree for the demo |
| Reporting | `jinja2`, `matplotlib` | Templated tables, trajectory plot |
| Dev | `pytest`, `ruff`, `pre-commit`, `detect-secrets` | |

**Deliberately excluded:** LangChain or any agent framework (the loop is ~250 lines; frameworks cost debuggability and obscure token accounting), Weights & Biases (node JSONs already hold everything, and a reviewer cannot see your project), Docker (subprocess isolation suffices for buggy-not-malicious code), any web framework or database server.

**Optional:** `catboost` as a second GBDT. `recbole` for config-driven access to many architectures at the cost of a one-time data conversion.

```toml
[project]
requires-python = ">=3.11"
dependencies = ["numpy", "polars", "lightgbm", "torch", "optuna",
                "anthropic", "pydantic>=2", "rich", "jinja2", "matplotlib"]

[dependency-groups]
dev = ["pytest", "ruff", "pre-commit", "detect-secrets"]
```

---

## 13. Deliverables

Mapped to the challenge brief, Section 2.5. **Note: unlike Tracks 3, 4 and 5, Track 2 does not list a demo video.** Verify at the webinar, but as written the brief specifies only four deliverables for this track. A walkthrough is still worth making for the finals pitch; treat it as optional and schedule it last.

- [ ] **Devpost writeup:** how the solution addresses the problem, dev tools, APIs, libraries and frameworks, datasets and assets
- [ ] **Public GitHub repo:** well-structured commented code covering all components; README with overview, setup and installation, steps to reproduce, reflection on limitations and future improvements, per-member contributions
- [ ] **Run and iteration logs:** per iteration, the hypothesis, the code diff, resulting metrics, and any error or recovery events with how they were handled. Plus a summary of manual interventions, with an explicit definition of what counts as one
- [ ] **Final submission:** KuaiRand-Pure output in the starter-kit schema, validated with `submit.py --check`
- [ ] **Results summary:** validation-best component metrics and the absolute delta over the official baseline
- [ ] **Resource report:** total input + output tokens across all LLM calls, and total GPU-hours to the converged result
- [ ] **No secrets** in source, git history, logs, traces, screenshots, or output
- [ ] *(Optional)* walkthrough video

---

## 14. Demo and pitch

Only scored at the final event (10%), but the assets double as README material.

### 14.1 Narrative

One causal story, not a tour of screens: **baseline → hypothesis → failure → recovery → improvement → converged result.**

| Time | Show | Say |
|---|---|---|
| 0:00-0:20 | Problem and the metric contradiction | Optimising the wrong evaluator is worse than no automation |
| 0:20-0:45 | Architecture with the trust boundary | The LLM proposes; locked services own data, execution, scoring, promotion |
| 0:45-1:15 | Run manifest and first hypothesis | The evaluator, split, and baseline are fingerprinted before the agent starts |
| 1:15-1:50 | Node tree and score trajectory | It explored feature, model, objective, and ensemble families, not a blind sweep |
| 1:50-2:20 | Injected failure and recovery | Failure is classified, budget reduced, run resumes with no human repair |
| 2:20-2:45 | Validation-best checkpoint, submission check, resource card | The converged audited result, with exact tokens and interventions |
| 2:45-3:00 | Limitations | No hidden labels, no unproven aggregates, no bonus distraction |

### 14.2 The injected-failure demo

Do not wait for an accidental failure on camera. Engineer a deterministic one:

Set the first neural candidate's batch size deliberately above the available memory limit. Show OOM classification, automatic batch-size reduction, checkpoint-safe retry, successful evaluation, and the two linked events in the node tree.

Faster and far more convincing than hoping something breaks. Keep a recorded replay of a completed run as a network-failure fallback.

### 14.3 Assets

- Architecture diagram with the LLM outside the evaluator and data trust boundary
- Node tree view with hypotheses, parents, deltas, failures, recoveries
- Results table separating internal folds, official validation, organiser baseline, hidden-test status
- Resource card: tokens, GPU-hours, wall time, retries, manual interventions
- One reproduce command and the checked submission artifact

---

## 15. Open questions

| Question | Owner | Resolve by |
|---|---|---|
| Which metric does `evaluate.py` actually score, given the NDCG@10/Recall@50 vs GAUC/nDCG@5 discrepancy? | Webinar attendee | Day 0, 14:45 |
| What is the compute budget (listed as TBD)? | Same | Day 0, 14:45 |
| Does seeding the agent with a static knowledge file count as a manual intervention? | Same | Day 0, 14:45 |
| Is a demo video required for Track 2? | Same | Day 0, 14:45 |
| What is the cutoff of the monthly item statistics file? | Same | Day 0, 14:45 |
| Polars or pandas? | Malvika | Day 0 |
| Is a GPU available to anyone on the team? | All | Day 0 |
| Hand-rolled PyTorch or RecBole? | Ethan | After probes, Day 1 evening |
| Attempt bonus benchmarks (1k / 27k)? | All | Day 2 evening, default no |
| Does refitting the validation-best config on train+validation before final submission comply with "the scored submission is the validation-best checkpoint," or must the submitted checkpoint be literally the one selected on validation? **Note:** the refit (D8) is already built and shipped in `tools/finalise.py` and used in the reproduction steps in `README.md`/`SETUP.md` — resolve this by confirming existing behaviour, not by freezing it unbuilt. | Pinxin | Before the final submission run |

---

# PART VI — APPENDICES

## Appendix A: Reference implementations

Provided because these pieces are the ones most likely to be built subtly wrong. Treat as the intended shape, adapting names only where Section 8 contracts require.

### A.1 Feature registry with time decay and EB smoothing (B3, B4, B5)

```python
# pipeline/features.py
FEATURES = {}

def feature(name):
    def wrap(fn):
        FEATURES[name] = fn
        return fn
    return wrap

def eb_smooth(clicks, impressions, global_rate, alpha=20.0):
    """Empirical-Bayes shrinkage toward the global rate.
    Sparse groups (small n) collapse to global_rate; dense groups
    keep their own rate. Fit alpha on internal folds only."""
    return (clicks + alpha * global_rate) / (impressions + alpha)

def decay_weights(dates, cutoff, half_life_days=7.0):
    """Exponential recency weights. Recent behaviour predicts the
    near future better, and the test window sits 8-17 days out."""
    age = (cutoff - dates).dt.total_seconds() / 86400.0
    return 0.5 ** (age / half_life_days)

@feature("user_ctr_decayed")
def user_ctr_decayed(train_df, target_df):
    """Time-decayed, EB-smoothed click rate per user.
    FIT ON train_df ONLY."""
    cutoff = train_df["date"].max()
    w = decay_weights(train_df["date"], cutoff)
    g = train_df.assign(w=w, wc=w * train_df["click"]).groupby("user_id")
    clicks, impr = g["wc"].sum(), g["w"].sum()
    global_rate = train_df["click"].mean()
    rates = eb_smooth(clicks, impr, global_rate)
    return target_df["user_id"].map(rates).fillna(global_rate).values
```

Every feature takes `(train_df, target_df)`, fits on train, applies to target. This shape makes leakage the exception rather than the default.

### A.2 Leakage guard (B6)

Two static checks plus a runtime canary.

```python
def leakage_check(fn, train_df, target_df):
    # 1. Label-independence probe: shuffling train labels must change
    #    the output of any label-dependent feature. If it does not, the
    #    feature is either label-free (safe) or reading labels off
    #    target_df (unsafe) — the static check below disambiguates.
    shuffled = train_df.assign(click=train_df["click"].sample(frac=1).values)
    if np.allclose(fn(train_df, target_df), fn(shuffled, target_df)):
        return True                                  # label-independent
    src = inspect.getsource(fn)
    # 2. Static check: must not read labels or forbidden columns off target_df
    assert 'target_df["click"]' not in src
    for col in FORBIDDEN_SAME_ROW:
        assert f'target_df["{col}"]' not in src
    for src_name in EXCLUDED_SOURCES:
        assert src_name not in src
    return True

# 3. Runtime canary, inside run_experiment. Realistic ceiling is
#    nowhere near 0.75.
if primary > 0.75:
    return {"status": "error", "error_class": "leak_suspected",
            "primary": primary}
```

### A.3 Bootstrap accept gate and segment report (D3, D4)

```python
# agent/gate.py
def accept(cand_scores, best_scores, user_ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    users = np.unique(user_ids)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(users, len(users), replace=True)
        deltas[i] = primary(cand_scores, sample) - primary(best_scores, sample)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo > 0, (lo, hi)

def segments(scores, user_ids, meta):
    """Primary broken down by user-activity quartile, item-popularity
    quartile, and day. A candidate that lifts the overall number while
    collapsing a segment is either rejected or promoted with a logged
    trade-off (6.6)."""
    out = {}
    for name, keys in meta.items():          # e.g. "activity_q": array
        for level in np.unique(keys):
            mask = keys == level
            out[f"{name}{level}"] = primary(scores[mask], user_ids[mask])
    return out
```

**Resample users, not rows.** The metric is computed per user, so the user is the unit of independence. Resampling rows understates variance and lets noise through.

### A.4 DeepFM skeleton (C5)

```python
import torch, torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, field_dims, emb_dim=16, mlp=[256,128,64], dropout=0.2):
        super().__init__()
        # one embedding table per categorical field (user_id, video_id, tag, ...)
        self.emb  = nn.ModuleList([nn.Embedding(d, emb_dim) for d in field_dims])
        self.lin  = nn.ModuleList([nn.Embedding(d, 1) for d in field_dims])
        self.bias = nn.Parameter(torch.zeros(1))

        layers, inp = [], len(field_dims) * emb_dim
        for h in mlp:
            layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
            inp = h
        layers += [nn.Linear(inp, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):                    # x: (batch, num_fields) int ids
        v = torch.stack([e(x[:, i]) for i, e in enumerate(self.emb)], dim=1)

        order1 = sum(l(x[:, i]) for i, l in enumerate(self.lin)) + self.bias
        # FM pairwise term via the O(n) identity, not O(n^2)
        order2 = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1, keepdim=True)
        deep   = self.mlp(v.flatten(1))

        return (order1 + order2 + deep).squeeze(1)      # raw logit
```

Train with `BCEWithLogitsLoss` and Adam, early stopping on an internal fold with patience 1.

The `order2` line is the factorization-machine trick: the sum of all pairwise dot products equals half of (sum-squared minus sum-of-squares), turning a quadratic computation into a linear one. This is why FMs are fast and it must not be reimplemented naively.

**Multi-task variant (rung 5):** keep everything up to `v.flatten(1)`, then replace the single head with 12 heads, one per feedback signal. Loss becomes a weighted sum of 12 BCE terms. At inference read only the click head. The other 11 exist purely to force the shared embeddings to learn richer representations.

---

## Appendix B: Action examples

What the agent should emit. Use for prompt design (A3) and `agent/prompts/propose.md`.

**Config change:**
```json
{"hypothesis": "GAUC is per-user, so a listwise objective may align better than pointwise BCE",
 "reasoning": "lambdarank optimises NDCG within each group directly...",
 "type": "config", "family": "objective", "parent": "n009",
 "config": {"model": "lgbm", "loss": "lambdarank",
            "features": ["user_id","video_id","user_ctr_decayed",
                         "video_ctr_decayed","hour_of_day"]}}
```

**Tune:**
```json
{"hypothesis": "embeddings are overfitting; the current L2 range is too narrow",
 "reasoning": "fold GAUC rose on train while falling on F3 for n014...",
 "type": "tune", "family": "training", "parent": "n014", "budget": 20,
 "search_space": {"lr": ["loguniform", 1e-4, 1e-2],
                  "emb_dim": ["categorical", [8,16,32,64]],
                  "l2": ["loguniform", 1e-6, 1e-3]}}
```

**Blend:**
```json
{"hypothesis": "GBDT and DeepFM rank-correlate at 0.81 per user, low enough that their errors differ",
 "reasoning": "...",
 "type": "blend", "family": "ensemble", "parent": "n022",
 "config": {"model": "blend", "parents": ["n014","n022"],
            "blend_method": "rank_avg"}}
```

The `hypothesis` field is not decoration. It is copied verbatim into the run-log deliverable and is what Innovation is scored on.

---

## Appendix C: References

Cited in the challenge brief:

1. Chan et al., *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering*, OpenAI 2024. arXiv:2410.07095
2. Jiang et al., *AIDE: AI-Driven Exploration in the Space of Code*, 2025. arXiv:2502.13138. **Primary architectural reference for the tree-search loop.**
3. Yamada et al., *The AI Scientist-v2*, 2025. arXiv:2504.08066

Supporting the caution in Section 6.4:

4. Dacrema, Cremonesi & Jannach, *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*, RecSys 2019. Most of 18 audited neural recommenders were beaten by well-tuned simple baselines.
5. Rendle et al., *Neural Collaborative Filtering vs. Matrix Factorization Revisited*, RecSys 2020. A properly tuned dot product beat the neural alternative.
6. Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022.

Model and method references worth citing if the corresponding branch is built:

7. Guo et al., *DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction*, 2017. arXiv:1703.04247
8. Wang et al., *DCN V2: Improved Deep & Cross Network*, 2020. arXiv:2008.13535
9. Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features*, 2017. arXiv:1706.09516
10. Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*, 2016. arXiv:1603.06560
11. Gao et al., *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*, 2022. arXiv:2208.08696

Cite 2, 4 and 5 in the README at minimum. Innovation is explicitly scored on "originality in drawing on published methods," and showing that you knew about the neural-recommender reproducibility problem and designed the probes to test for it is exactly the judgement being looked for.

---

## Appendix D: Idea bank

Feeds A12's propose prompt. Once A12 is built, move this table verbatim into `agent/prompts/idea_bank.md` and wire it into `propose()`'s system prompt (not done yet — this is source content for that task, not a description of current behaviour).

| Idea | Drawer | Reference to cite | One-line why |
|---|---|---|---|
| Duration-quantile grouping / duration-normalized engagement features (B10) | features | D2Q — Zhan et al., KDD 2022 | Duration confounds watch-time labels; `long_view` is duration-derived |
| Historical play-completion-ratio aggregates (B10) | features | PCR baseline family; CWM — Zhao et al., KDD 2024 (Appendix C ref. 4's neighbour in the same literature) | Completion behaviour predicts `long_view` beyond raw counts |
| Auxiliary-signal historical rates (B11) | features | ESMM-style multi-feedback usage | 11 unscored signals carry information about the scored one |
| Within-user pairwise / listwise objectives grouped by user (C8) | loss | BPR (Rendle 2009); LambdaRank | Structurally matched to per-user GAUC/nDCG@5. **C8 is mostly implemented** — see 9.3 |
| Multi-task shared-embedding heads (C9) | model | ESMM / shared-bottom MTL | Densifies sparse `long_view` signal via shared representations. **C9 is not implemented** — see 9.3 |
| Ablation-guided component targeting (A10) | (meta) | MLE-STAR — Yoon & Nam, NeurIPS 2025 | Evidence-driven choice of what to improve next |
| Tree search over solutions, branch from global best | (meta) | AIDE — Jiang et al., 2025 (Appendix C ref. 2) | Already the A2/A7 design; cite it in the writeup |

**Screened out (log these as considered-and-rejected — cheap Innovation points):** full CWM censored regression (`torch==1.6.0` dependency, heavy for the delta); MCTS à la ML-Master (overkill at 50 iterations); Recall@50 optimisation (≈0.999 for all models per the organisers — see 4.8 on the Recall@50/NDCG@10 prose being stale against the shipped `evaluate.py`).

---

*Version 2.1, 31 August 2026. Sections 9.1/9.2/9.3/9.4, 6.6, 11 (trap 13), 15, and Appendix D merged from `AGENT_PLAN_v2.1_ADDENDUM.md` — additive per that document's own rule; no Section 8 contract was changed. Update Section 6.7 ordering after the Day-1 probes and this document stays the single source of truth.*
