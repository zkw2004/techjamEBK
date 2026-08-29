# Recsys priors for the proposing LLM

Owner: Workstream D (Pinxin). Task D6. One line of rationale per entry.
This file is read into the propose prompt; it is static, seeded knowledge,
not something the agent writes to.

> **Open question (Section 15):** does seeding the agent with this file count
> as a manual intervention? Confirm at the webinar.

## Leakage rules — non-negotiable

**Never shuffle or k-fold.** Splits are date-based. Shuffling trains on the future.

**Never use a same-row post-click signal as an input feature.** The 12 feedback
signals occur at or after the click on that row.

| Column | As input feature (same row) | As training target | As historical aggregate over past rows |
|---|---|---|---|
| `click` | **the label** | yes | yes |
| `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate` | **never** | yes (multi-task) | yes |
| `long_view`, `is_profile_enter`, `is_click_pause` | **never** | yes | yes |
| `play_time_ms`, `duration_ms`, `profile_stay_time` | **never** | yes | yes |
| `item_statistics_monthly` | **never** | never | never (window may span test) |

**Never fit on the official validation window** — not features, not early
stopping, not blend weights, not EB hyperparameters. Use the internal folds.

## Model priors

- **FM** — TODO (D6)
- **DeepFM** — TODO
- **LightGBM lambdarank** — TODO
- **Multi-task heads** — TODO
- **BPR / pairwise** — TODO
- **Negative sampling** — TODO
- **Time decay** — TODO
- **Empirical-Bayes smoothing** — TODO
- **Exposure debiasing (randomised slice)** — TODO
- **Blending** — TODO

## Cautions from the literature

Well-tuned classical methods frequently beat neural recommenders
(Dacrema et al. 2019; Rendle et al. 2020). Assume no ordering; the Day-1
probes decide it.
