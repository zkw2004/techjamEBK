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
