# Research analysis: Shusen Wang's industrial recsys course → our KuaiRand-Pure ranker

Source: `RecommenderSystem/` (王树森《工业界的推荐系统》) — 3 lecture notes + 45 slide
decks, ~950 pages, read in full. This maps that material onto our task, separates what
transfers from what does not, and proposes a sequenced plan.

**Read the "Scope" and "Five measurements" sections before the recommendations.** Most of
the course does not apply to us, and the reason it doesn't is also the reason our recent
experiments keep landing inside the noise floor.

---

## 1. Scope: what this course is actually about

The course describes a complete **online** industrial pipeline:

```
召回 (retrieval, ~1e8→1e3) → 粗排 (pre-rank, 1e3→1e2) → 精排 (rank) → 重排 (re-rank)
```

optimised for **DAU / retention / publishing**, validated by **online A/B tests**, with
separate machinery for **item cold-start** and **creator-side incentives**.

Our task is one box in that diagram: **精排 (ranking) only**, offline, on a candidate set
that is fixed by the log, scored once by a frozen offline metric. We generate no
candidates, serve no traffic, run no A/B tests, and have no author-side metrics.

So roughly **60% of the course is out of scope by construction** — not "lower priority",
genuinely inapplicable. Section 4 lists those explicitly, because knowing what *not* to
try is worth as much here as knowing what to try: our binding constraint is iterations,
not ideas.

The parts that do transfer are concentrated in **Section 3 (排序)**, **Section 4 (交叉结构)**,
**Section 5 (用户行为序列建模)**, and **Section 8.3 (排序涨指标)**.

---

## 2. Five measurements that should drive every decision

I measured these against our actual data before writing any recommendation. Several
contradict assumptions in the brief, in the course, and in our own plan.

### 2.1 The eval task is far smaller than it looks — median **4** candidates per user

| | valid split |
|---|---|
| rows | 124,909 |
| users | 22,377 |
| **median candidates per user** | **4** (p90 = 12) |
| users usable by GAUC (`0 < pos < n`) | **57.8%** |
| all-negative users (nDCG ≡ 0) | 30.3% |
| all-positive users (nDCG ≡ 1) | 11.9% |

This is the most important fact in this document. We are re-ordering a **median of four
items per user**, and only 57.8% of users contribute to GAUC at all. Per-user GAUC on a
4-item list takes very few discrete values, so aggregate deltas are dominated by which
users happen to *flip* — which is exactly why our measured effects keep landing at
~1σ (BPR: +0.0008 over 5 seeds, against a 0.0008 seed std).

Consequences:
- The realistic ceiling is 0.8484 on valid, not 1.0 — already known, but with 4 candidates
  the *reachable* band above 0.6016 is narrower still.
- **Anything that reorders by pushing items apart globally cannot help.** Only signals
  that discriminate *between a specific user's ~4 candidates* can move the metric.
- This is the formal statement of the organisers' "pure user-side first-order terms
  contribute exactly zero": with 4 candidates from one user, any user-constant term is a
  constant added to all 4 scores and cancels in the ordering.

**Every recommendation in §3 is filtered by this test: does it vary within a user's
candidate list?**

### 2.2 Cold-start is a non-issue for us

| | valid rows |
|---|---|
| video unseen in train | **0.01%** |
| user unseen in train | **1.6%** |

The entire cold-start section (Notes 07 + 6 decks — default embedding, category/cluster
recall, look-alike, 保量/差异化保量 traffic control) addresses a problem we do not have.
Skip it.

### 2.3 User history is plentiful; sequences are short — DIN yes, SIM no

| | train split |
|---|---|
| rows / users / videos | 1,141,112 / 26,210 / 7,538 |
| **median rows per user** | **31** (p90 97, max 809) |
| median rows per video | 44 |
| median train-history behind a valid row | **51** |
| valid rows whose user has ≥10 history items | **91.4%** |

The organisers' brief claims "每用户在 train 里有上百到上千条交互" (hundreds to thousands).
**That is false for KuaiRand-Pure** — the median is 31. It is presumably true of
KuaiRand-1K/27K.

This matters for choosing between the course's two sequence models:
- **DIN** (n ≈ tens, attention over LastN) — feasible; 91.4% of eval rows have ≥10 history.
- **SIM** (n ≈ thousands, search-then-attend) — **not applicable**. Its entire purpose is
  cutting a 1e6-length sequence down to k. We have 31. Building SIM here would be
  machinery with nothing to do.

### 2.4 `long_view` is a completion label that is *already* duration-normalised

| long_view | median play/duration ratio |
|---|---|
| 0 | 0.031 |
| 1 | **0.977** |

So `long_view` ≈ "watched essentially the whole video". But the duration confound behaves
very differently for the label than for the raw ratio:

| duration decile | 0 (7.9s) | 3 (39.7s) | 6 (104s) | 9 (287s) | spread |
|---|---|---|---|---|---|
| `long_view` rate | 0.281 | 0.367 | 0.376 | 0.318 | **1.4×**, non-monotone |
| completion ratio | 0.729 | 0.418 | 0.285 | 0.140 | **5.2×**, monotone ↓ |

The label is only mildly duration-dependent (1.4×, inverted-U) because KuaiRand defines
`long_view` with duration-dependent thresholds. The **raw completion ratio is 5.2×
duration-confounded**.

Two consequences, one of them a warning about existing code:
- The course's D2Q-style correction (`Rank_04`: divide predicted completion by
  `f(duration)`) has **less headroom than the course implies**, because our label already
  absorbs most of that correction.
- **Our `pcr_hist` feature (B10) is probably mis-specified.** It aggregates a raw
  completion ratio carrying a 5.2× duration signal, to predict a target carrying only a
  1.4× one — a systematic mismatch. See §3.4.

### 2.5 `is_rand` is all-zero in training — IPS debiasing is dead, confirmed

`is_rand.value_counts()` over the 1,141,112 training rows returns `{0: 1141112}`. There is
**no randomly-exposed data in the training window**. This independently confirms the B8
conclusion by a second route (previously argued from the separate `log_random` file's date
range). Exposure debiasing / IPS has no substrate here. Close it.

Class balance, incidentally, is fine: `long_view` rate = 0.337. The course's negative
down-sampling + calibration material (`Rank_01`) addresses an imbalance we don't have.

---

## 3. What transfers, ranked

Ordered by (expected value × probability of working) ÷ cost. Every item passes the §2.1
test: it varies *within* a user's candidate list.

### 3.1 Item–item co-occurrence similarity to the user's history — **do this first**

*Course source: `02_Retrieval_01` (ItemCF), `02_Retrieval_02` (Swing).*

The course teaches ItemCF/Swing as **retrieval channels**, which we don't have. But the
object they build — an item→item similarity index — is directly usable as a **ranking
feature**:

```
sim_to_history(user, candidate) = max  (or top-k mean) over the user's training-history
                                  items h of  sim(candidate, h)
```

with `sim(i₁,i₂) = |W₁∩W₂| / √(|W₁|·|W₂|)` (ItemCF), or Swing's variant that down-weights
co-occurrences from users who overlap heavily (`1/(α + overlap(u₁,u₂))`) to suppress
small-clique artefacts.

Why this is first:
- **It is DIN's signal without DIN's cost.** DIN's attention weight is exactly
  "similarity between the candidate and each history item"; this computes that
  non-parametrically, as a feature, with no model change.
- Varies strongly within a user's 4 candidates — passes §2.1.
- Feasible: 7,538 items → a 7.5K×7.5K similarity matrix is trivial; 91.4% of eval rows
  have ≥10 history items.
- Leakage-safe under our existing discipline: fit co-occurrence on train only, and for
  in-sample rows use the strictly-prior pattern the B11 helpers already implement.
- Cost: one feature builder. No new model, no new training path.

**It also functions as a cheap probe.** If `sim_to_history` moves nothing, the entire
"user-history × candidate" family is unlikely to pay off, and we should not spend the much
larger DIN budget. This is exactly the ablation-driven targeting logic of our own A10,
applied to the plan rather than to a config.

### 3.2 DIN — attention over LastN

*Course source: `05_LastN_01/02`, `08_Improvement_03`.*

The organisers rank user-history sequences **#2 of 5** headroom directions and nothing is
implemented. The course's recipe:

- LastN item IDs → embedding → **weighted** average, weights = softmax similarity between
  the candidate item's embedding and each history item's embedding → user feature.
- Course's own summary: *长序列优于短序列，注意力机制优于简单平均，使用时间信息有提升*.

The critical detail for us, which the course states plainly (`05_LastN_02`, p18):
> 简单平均只需要用到 LastN，属于用户自身的特征 … 注意力机制需要用到 LastN + 候选物品

A **plain LastN average is user-constant → provably contributes zero** under §2.1. Only
the attention form is eligible. If we build this, build the attention; do not ship the
average as a stepping stone.

Add the course's time-embedding refinement (`05_LastN_03`): concatenate a bucketed
"time since that interaction" embedding to each history item vector. Cheap, and the course
reports it helps.

Cost: high — a causal per-row history builder plus an attention layer in DeepFM. Gate it
on §3.1 showing signal.

### 3.3 LHUC / PPNet — user-conditioned modulation

*Course source: `04_Cross_03`; endorsed again in `08_Improvement_03` as base-model
improvement #2.*

```
user features → FC → sigmoid × 2 → ⊙ (Hadamard) with the item-side hidden layer
```

A **multiplicative user×item interaction**: the user vector rescales each dimension of the
item representation, so two candidates with different item features get different
treatment for the same user. Passes §2.1 by construction.

Kuaishou runs this in production ranking (PPNet). It is ~15 lines inside our existing
DeepFM, needs no new data path, and is the single cheapest structural change that is
*targeted* at the within-user problem. Low risk: with the gate saturated it degenerates
toward the current model.

### 3.4 Fix `pcr_hist`'s duration confound (correction to existing work)

*Course source: `03_Rank_04` (完播率 must be divided by `f(video length)` before use).*

Per §2.4, `pcr_hist` aggregates a signal that is 5.2× duration-confounded to predict a
target that is 1.4× duration-confounded. The course is explicit that raw predicted
completion **cannot** enter the score directly and must be normalised by a
duration-conditional baseline:

> 不能直接把预估的完播率用到融分公式 … `p_finish = 预估完播率 / f(视频长度)`

Proposed fix: normalise each row's completion ratio by its duration-bucket mean before
aggregating, i.e. aggregate `pcr / mean_pcr(duration_bucket)` rather than `pcr`.

This is cheap, it is a *correction* rather than new surface area, and the current version
may be actively injecting noise — which would also partly explain why the B10 pack has not
helped. Worth measuring both ways.

### 3.5 Use the multi-task heads in the score, not only as regularisers

*Course source: `03_Rank_01` (multi-objective), `03_Rank_03` (预估分数融合).*

Our new `deepfm_mtl` trains `is_click`/`is_like` heads but reads only `long_view` at
inference — the heads exist purely to shape the shared embeddings. The industrial pattern
in the course is to **fuse** the predictions:

```
p_longview · (1 + w₁·p_click) · (1 + w₂·p_like) …          (multiplicative form)
Σᵢ wᵢ / (rankᵢ^αᵢ + βᵢ)                                     (rank-based form)
```

The rank-based form is notable for us: it is scale-free, which matters because our metric
is rank-only. Fusion weights must be fitted on **internal folds**, never on the official
validation window.

Temper expectations with the course's own verdict (`08_Improvement_03`): *MMoE、PLE 等结构
可能有效，但往往无效*. Treat the fusion as the experiment; treat MMoE as unlikely.

### 3.6 SENet field-weighting

*Course source: `04_Cross_04` (SENet / FiBiNET).*

SENet learns a per-**field** weight (AvgPool → FC+ReLU → FC+Sigmoid → row-wise multiply).
Our own A10 ablation of the baseline already showed field importances differ by ~30×
(`tab` −0.0125 vs `video_id` −0.0004 at screen budget) — SENet learns that reweighting
automatically instead of us hand-tuning the feature list. Cheap; moderate expected value.
The bilinear-cross half of FiBiNET is more expensive and can be deferred.

### 3.7 Session-position proxy from `time_ms` (speculative)

*Course source: `08_Improvement_03` (position bias — "可能有效，也可能无效").*

KuaiRand-Pure has no explicit position column, but `time_ms` orders a user's impressions
within a day, giving a derivable within-session rank. Position varies within a user's
candidate list, so it is eligible. Cheap to build; the course itself is agnostic about
whether it pays. Lowest priority of the seven.

---

## 4. Verified dead ends — do not spend iterations here

Each of these is either measured against our data or ruled out by the metric's definition.

| Course material | Why it cannot help us |
|---|---|
| **Section 6 — 多样性 (MMR, DPP, MGS, rules)** | **Actively harmful.** MMR/DPP deliberately demote high-scoring items to increase variety; our metric rewards putting positives at the top. With a median of 4 candidates there is nothing to diversify. |
| **Section 7 — 物品冷启动** (default embedding, cluster/look-alike recall, 保量, 差异化保量) | §2.2: 0.01% unseen items, 1.6% unseen users. No author-side or publishing metrics exist in this task. |
| **Section 2 — 召回** as retrieval (two-tower, ANN, Deep Retrieval, exposure filtering) | We generate no candidates; the candidate set is fixed by the log. *(The item–item similarity object is still reusable as a feature — §3.1.)* |
| **`Rank_01` — 预估值校准** | GAUC and nDCG@5 are invariant under any monotone transform of the scores. Calibration is monotone, so its effect on our metric is **provably exactly zero**. |
| **Negative down-sampling for class balance** | `long_view` rate is 0.337 — already balanced. |
| **`05_LastN_03` — SIM** | Designed for n in the thousands; our median is 31 (§2.3). |
| **IPS / exposure debiasing** | §2.5: `is_rand` is all-zero across all 1.14M training rows. |
| **`Basics_03`, 在线学习, 老汤模型** (A/B layering, holdout, 反转实验, online learning) | Offline, single-shot, no serving. *(Conceptual note: the course's holdout principle — a bucket that receives no experiments — is what our seeded baseline anchor already implements.)* |

---

## 5. Plan

Sequenced so that each cheap step informs whether the expensive step is worth funding.
Every step obeys the existing discipline: fit on train / internal folds only, never on the
official validation window; promote only on `MIN_DELTA_FLOOR` (0.002) **and** a bootstrap
CI excluding zero; confirm-tier (5 seeds) before believing any single-seed result.

### Phase 0 — re-baseline the reading of results (no code)

Record §2.1 in `AGENT_PLAN.md` as a metric-shape note. With a median of 4 candidates and
57.8% GAUC-usable users, a +0.002 delta is a *large* effect here, not a small one, and
single-seed swings of ±0.001 are structurally expected. This is the frame for judging
everything below.

### Phase 1 — the cheap probe (§3.1)

Build `sim_to_history` (ItemCF cosine; optionally a Swing variant) as a registered feature.
Evaluate FM + 5 official fields + `sim_to_history` at full tier, then **confirm tier**.

- **If it clears +0.002 with a CI excluding zero** → the user-history × candidate family is
  live. Fund Phase 2.
- **If it lands inside noise** → do *not* build DIN. Skip to Phase 3.

Either outcome is a reportable result: it is a direct test of the organisers' #2 ranked
headroom direction, at a fraction of DIN's cost.

### Phase 2 — DIN, gated on Phase 1 (§3.2)

Attention over LastN (n ≈ 10–30) with the time-since-interaction embedding. Attention
form only — never ship the plain average (§3.2). Confirm tier before any claim.

### Phase 3 — cheap structural wins, in parallel (§3.3, §3.4, §3.6)

Independent of Phases 1–2, and independent of each other:
1. **LHUC** on DeepFM (§3.3) — cheapest structural change aimed at the within-user problem.
2. **`pcr_hist` duration normalisation** (§3.4) — measure old vs new; the current version may
   be hurting.
3. **SENet** field weighting (§3.6).

### Phase 4 — multi-task fusion (§3.5)

Only after `deepfm_mtl` has a trustworthy single-task baseline to be compared against.
Fuse `is_click`/`is_like` head outputs into the ranking score with rank-based fusion;
fit weights on internal folds.

### Not scheduled

§3.7 (position proxy) if time allows. Everything in §4: never.

---

## 6. Honest summary

The course is a high-quality description of a problem substantially larger than ours. Its
main value to this project is **three specific mechanisms** (item–item similarity as a
ranking feature, DIN attention, LHUC modulation) that all attack the one thing our metric
actually measures — discriminating between a single user's handful of candidates — plus a
**correction to `pcr_hist`** and a clear, defensible **list of things not to try**.

The measurements in §2 are the more durable contribution. Two of them contradict stated
assumptions we have been working under (the brief's "hundreds to thousands of interactions
per user"; the implicit assumption that raw completion ratio is a clean signal), and one
of them — median 4 candidates per user — reframes how every result in this project should
be read.

Nothing here is a guaranteed win. Our own confirm-tier result for BPR (+0.0008 over 5
seeds, ≈1σ) is a fair prior for how these land: the honest expectation is that most of
these also fail, and that the value lies in failing them *cheaply and in a defensible
order*.
