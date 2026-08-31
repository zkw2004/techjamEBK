# Recsys priors for the proposing LLM

Owner: Workstream D (Pinxin). Task D6. One line of rationale per entry.
This file is read into the propose prompt; it is static, seeded knowledge,
not something the agent writes to.

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
  trustworthy parent for feature experiments. The baseline is FM over the
  **five** official fields (`user_id`, `video_id`, `author_id`, `tab`,
  `dur_bucket`) — `Config.features` defaults to only two, which is a much
  weaker model and not a valid comparison against 0.6016.
- **DeepFM** — Add an MLP over the same embeddings when higher-order feature
  interactions appear useful, but keep the model small and verify the gain
  against a well-tuned FM.
- **LightGBM lambdarank** — Use per-user groups so the objective optimises
  within-user ranking; without the correct group array it does not match GAUC
  or nDCG and the result is invalid.
- **Multi-task heads** — Predict legal auxiliary feedback signals as extra
  training targets so the shared embeddings have to explain more than one
  behaviour. Implemented as `deepfm_mtl` with `is_click`/`is_like` heads and
  tunable `aux_click_weight`/`aux_like_weight`; only the `long_view` head is
  scored. Weights of 0 reproduce plain `deepfm` exactly, so the weights are the
  experiment.
- **LHUC / PPNet gating** — Set `hparams: {"lhuc": true}` on `deepfm` or
  `deepfm_mtl` to scale every MLP hidden unit by a gate learned from the row's
  own user embedding. Worth trying because the metric only ever compares items
  *within* one user, so a mechanism that lets the same feature count differently
  for different users is aimed at the right target. Requires `user_id` in
  `features`.
- **SENet field weighting** — Set `hparams: {"senet": true}` to learn per-row,
  per-field importance weights over the embeddings before the interaction term.
  Same motivation as LHUC and composable with it; both default off and are
  exactly the identity at initialisation, so turning one on is a clean A/B.
- **BPR / pairwise** — Compare a positive impression with sampled negatives
  from the same user's eligible history when a ranking-aligned objective may
  outperform pointwise BCE. Implemented for `fm` only (`loss="pairwise"`);
  LightGBM offers `lambdarank` instead, and DeepFM is pointwise-only.
- **Negative sampling** — Treat which non-clicked impressions are retained as
  an experiment axis: all negatives are faithful, while in-session and
  popularity-weighted negatives can focus learning on harder comparisons.
- **Time decay** — Weight recent historical events more heavily because user
  interests and item popularity drift, fitting the half-life on internal
  temporal folds only.
- **Empirical-Bayes smoothing** — Shrink sparse group rates toward the global
  rate with `(clicks + alpha * global_rate) / (impressions + alpha)`, fitting
  `alpha` on internal folds only.
- ~~**Exposure debiasing (randomised slice)**~~ — **Impossible on this data.**
  `is_rand` is 0 for all 1,141,112 training rows, so there is no unbiased slice
  and no estimable propensity. Do not propose IPS or exposure reweighting.
- **Blending** — Combine continuous predictions from **exactly two** distinct
  parents, each an accepted, successful, full-tier node; reject a blend unless
  it beats both parents. Two measured points on *which* blend is worth trying:
  parents must genuinely disagree, and the method matters. FM × DeepFM+SENet
  failed (per-user Spearman **+0.7975** — DeepFM contains FM over the same
  fields, so "different family" was a label, not a mechanism). FM × LightGBM,
  a factorisation against trees, correlated less (**+0.7637**) and worked:
  `logit_avg` gave **+0.001965 over FM with a 95% CI excluding zero**, while
  the default `rank_avg` found nothing on the same pair. **Report the parent
  correlation before blending, and do not assume `rank_avg` is the right
  method** — it was the only one of three that missed the real effect here.

## Measured results specific to this project — do not re-derive these

Each cost real compute; repeating one wastes an iteration.

- **The completed single-feature sweep is a boundary, not a verdict that
  static or engineered features are a dead end.** All 20 features registered
  at the time were tested one at a time on top of `FIELDS` with FM at full
  tier (`scripts/feature_sweep.py`). Every one landed inside noise except
  `hour_of_day` (+0.001686) and `day_of_week` (+0.001635), whose CIs exclude
  zero but sit under the 0.002 floor; together they give +0.001644 and are
  redundant. Do not repeat those exact FM experiments, but do consider new
  **candidate-varying** features, interactions, and combinations. In
  particular, B10's later `video_completion_ratio_hist` is per-video rather
  than a user scalar and was not part of that sweep; it needs its own screen
  and ablation evidence before the agent assigns it a prior.
- **B14's specific `sim_to_history` implementation carries no usable signal**
  at confirm tier (delta −0.000561, CI [−0.001456, +0.001082]). This triggers
  the declared C10 go/no-go: do not build DIN on top of that representation.
  It does **not** prove every possible user-history encoding or every
  candidate-conditioned feature is useless; reopen the direction only when a
  materially different cheap representation first clears the 0.002 gate.
- **The per-user aggregates are `metric_inert`.** `user_ctr`, `user_activity`,
  `pcr_hist` and the `user_*_rate_decayed` family are `groupby("user_id")`
  values, identical across every candidate a user sees, so they cannot move a
  within-user metric except through a cross. `tools/screen.py` measures this;
  the sweep confirms it end to end.
- **Nothing has ever been tuned.** There are zero `tune` nodes in the ledger
  and no Optuna study on disk, for any model. Every result in this project is
  on hand-picked defaults. `type="tune"` is wired and works — proposing a tuning
  action is among the few genuinely unexplored moves available.

## Cautions from the literature

Well-tuned classical methods frequently beat neural recommenders
(Dacrema et al. 2019; Rendle et al. 2020). Assume no ordering; the Day-1
probes decide it.
