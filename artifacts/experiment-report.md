# Experiment report

## Evaluation context

- Median candidates per user: **4**
- GAUC-usable users: **57.8%**
- Interpretation: A +0.002 primary delta is material when users rank a median of four candidates and only 57.8% contribute to GAUC.

## Results

| Node | Fidelity | Status | Accepted | GAUC | nDCG | Primary | Δ baseline |
|---|---|---|---:|---:|---:|---:|---:|
| n001 | smoke | ok | no | — | — | — | — |
| n002 | screen | ok | no | 0.652110 | 0.497915 | 0.575012 | -0.026588 |
| n003 | full | ok | yes | 0.667311 | 0.536057 | 0.601684 | 0.000084 |
| n004 | full | ok | no | 0.662896 | 0.534317 | 0.598607 | -0.002993 |
| n005 | smoke | ok | no | — | — | — | — |
| n006 | smoke | ok | no | — | — | — | — |
| n007 | screen | error | no | — | — | — | — |
| n008 | screen | error | no | — | — | — | — |
| n009 | smoke | ok | no | — | — | — | — |
| n010 | screen | ok | no | 0.627956 | 0.490008 | 0.558982 | -0.042618 |
| n011 | smoke | ok | no | — | — | — | — |
| n012 | screen | ok | no | 0.650454 | 0.497008 | 0.573731 | -0.027869 |
| n013 | smoke | ok | no | — | — | — | — |
| n014 | screen | ok | no | 0.652942 | 0.498044 | 0.575493 | -0.026107 |
| n015 | screen | ok | no | 0.651868 | 0.497160 | 0.574514 | -0.027086 |
| n016 | smoke | ok | no | — | — | — | — |
| n017 | screen | ok | no | 0.650968 | 0.497098 | 0.574033 | -0.027567 |
| n018 | smoke | ok | no | — | — | — | — |
| n019 | screen | ok | no | 0.653765 | 0.497922 | 0.575843 | -0.025757 |
| n020 | full | ok | no | 0.666200 | 0.535074 | 0.600637 | -0.000963 |
| n021 | full | ok | no | 0.669110 | 0.536810 | 0.602960 | 0.001360 |

## Run totals

- Nodes: 21
- Tokens: 40642 (32796 in, 7846 out)
- GPU-hours: 0.000000
- Agent wall-clock: 00:08:26 (506 seconds)
- Iterations used: 12 / 50
- Manual interventions: 0
- Pilot iterations: 17
- Full/confirm iterations: 4
- Other iterations: 0

Manual intervention means: A completed node whose manual_intervention field is true; automated proposals, bounded retries, and recovery actions are not interventions.

## Search controller

| Iteration | Strikes | Scheduler forced | Hedges fired | Node |
|---:|---:|---:|---:|---|
| 1 | 1 | no | 0 | — |
| 2 | 1 | no | 0 | — |
| 3 | 2 | no | 0 | — |
| 4 | 2 | no | 0 | — |
| 5 | 1 | yes | 1 | — |
| 6 | 2 | no | 1 | — |
| 7 | 1 | yes | 2 | — |
| 8 | 2 | no | 2 | — |
| 9 | 1 | yes | 3 | — |
| 10 | 1 | yes | 1 | — |
| 11 | 2 | no | 3 | — |
| 12 | 2 | no | 1 | — |

## Latest ablation sensitivity

Latest table: node `n003`; base primary 0.574109.

| Component | Ablated primary | Delta | Sensitivity |
|---|---:|---:|---:|
| feature:tab | 0.561579 | -0.012531 | 0.012531 |
| feature:user_id | 0.568532 | -0.005577 | 0.005577 |
| feature:dur_bucket | 0.571213 | -0.002896 | 0.002896 |
| loss:pairwise | 0.575843 | 0.001734 | 0.001734 |
| feature:author_id | 0.573589 | -0.000521 | 0.000521 |
| feature:video_id | 0.573706 | -0.000403 | 0.000403 |

## Research citations

- `n001` — BPR \u2014 Rendle 2009; LambdaRank
- `n002` — BPR \u2014 Rendle 2009; LambdaRank
- `n004` — BPR \u2014 Rendle 2009; LambdaRank
- `n009` — BPR — Rendle 2009; LambdaRank
- `n010` — BPR — Rendle 2009; LambdaRank
- `n013` — D2Q \u2014 Zhan et al., KDD 2022
- `n015` — D2Q \u2014 Zhan et al., KDD 2022
- `n018` — BPR — Rendle 2009; LambdaRank
- `n019` — BPR — Rendle 2009; LambdaRank
- `n021` — BPR — Rendle 2009; LambdaRank

## Metric-inert features

No metric-inert feature report was recorded.
