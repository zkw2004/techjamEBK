# Recsys priors for the proposing LLM

Owner: Workstream D (Pinxin). Task D6. One line of rationale per entry.
This file is read into the propose prompt; it is static, seeded knowledge,
not something the agent writes to.

> **Open question (Section 15):** does seeding the agent with this file count
> as a manual intervention? Confirm at the webinar.

## Leakage rules — non-negotiable

**Never shuffle or k-fold.** Splits are date-based. Shuffling trains on the future.

**Never use a same-row post-exposure signal as an input feature.** The feedback
signals occur at or after the outcome on that row.

The scored label is **`long_view`**, read from the shipped starter kit
(`data.py:5`, `baseline_scores.json`). AGENT_PLAN.md's prose says `click`; it is
wrong, and this table is inverted relative to it.

| Column | As input feature (same row) | As training target | As historical aggregate over past rows |
|---|---|---|---|
| `long_view` | **the label** | — | yes |
| `is_click` | **never** | yes (multi-task) | yes |
| `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate` | **never** | yes (multi-task) | yes |
| `is_profile_enter`, `is_click_pause` | **never** | yes | yes |
| `play_time_ms`, `profile_stay_time` | **never** | yes | yes |
| `duration_ms` | **yes** — video property, known before exposure; the official baseline uses it as `dur_bucket` | — | — |
| `item_statistics_monthly` | **never** | never | never (window may span test) |

**Never fit on the official validation window** — not features, not early
stopping, not blend weights, not EB hyperparameters. Use the internal folds.

## Model priors

- **FM** — Use low-dimensional embeddings to model sparse user-item and
  context interactions cheaply; it is the official baseline and the first
  trustworthy parent for feature experiments.
- **DeepFM** — Add an MLP over the same embeddings when higher-order feature
  interactions appear useful, but keep the model small and verify the gain
  against a well-tuned FM.
- **LightGBM lambdarank** — Use per-user groups so the objective optimises
  within-user ranking; without the correct group array it does not match GAUC
  or nDCG and the result is invalid.
- **Multi-task heads** — Predict legal auxiliary feedback signals only as
  training targets to enrich shared embeddings, while using only the
  `long_view` head at inference.
- **BPR / pairwise** — Compare a positive impression with sampled negatives
  from the same user's eligible history when a ranking-aligned objective may
  outperform pointwise BCE.
- **Negative sampling** — Treat which non-clicked impressions are retained as
  an experiment axis: all negatives are faithful, while in-session and
  popularity-weighted negatives can focus learning on harder comparisons.
- **Time decay** — Weight recent historical events more heavily because user
  interests and item popularity drift, fitting the half-life on internal
  temporal folds only.
- **Empirical-Bayes smoothing** — Shrink sparse group rates toward the global
  rate with `(clicks + alpha * global_rate) / (impressions + alpha)`, fitting
  `alpha` on internal folds only.
- **Exposure debiasing (randomised slice)** — Use the randomised-exposure log
  for unbiased diagnostics or IPS only when its propensity and temporal
  assumptions are satisfied; post-training-cutoff rows never enter fitting.
- **Blending** — Rank-average continuous predictions from accepted,
  sufficiently diverse parents; skip near-identical models and reject a blend
  unless it beats both parents on internal folds and official validation.

## Cautions from the literature

Well-tuned classical methods frequently beat neural recommenders
(Dacrema et al. 2019; Rendle et al. 2020). Assume no ordering; the Day-1
probes decide it.
