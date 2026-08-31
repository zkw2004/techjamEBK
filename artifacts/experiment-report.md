# Experiment report

## Evaluation context

- Median candidates per user: **4**
- GAUC-usable users: **57.8%**
- Interpretation: A +0.002 primary delta is material when users rank a median of four candidates and only 57.8% contribute to GAUC.

## Results

| Node | Fidelity | Status | Accepted | GAUC | nDCG | Primary | Δ baseline |
|---|---|---|---:|---:|---:|---:|---:|
| n001 | smoke | ok | no | — | — | — | — |
| n002 | screen | ok | no | 0.605364 | 0.484544 | 0.544954 | -0.056646 |
| n003 | full | ok | no | 0.637701 | 0.521686 | 0.579694 | -0.021906 |
| n004 | smoke | error | no | — | — | — | — |
| n005 | smoke | error | no | — | — | — | — |

## Run totals

- Nodes: 5
- Tokens: 17147 (7456 in, 9691 out)
- GPU-hours: 0.000000
- Manual interventions: 0
- Pilot iterations: 4
- Full/confirm iterations: 1
- Other iterations: 0

Manual intervention means: A completed node whose manual_intervention field is true; automated proposals, bounded retries, and recovery actions are not interventions.

## Search controller

No iteration-controller events were recorded.

## Latest ablation sensitivity

No ablation sensitivity table was recorded.

## Research citations

No structured `[ref: ...]` citations were recorded.

## Metric-inert features

No metric-inert feature report was recorded.
