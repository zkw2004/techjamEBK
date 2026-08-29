# Day-0 verification record

Section 10.1 of [AGENT_PLAN.md](AGENT_PLAN.md). Run 2026-08-29 on macOS,
Python 3.11.2, numpy only. All checks reproduce.

## Contract hashes

Recomputed at preflight by `agent/manifest.py` (D2), which **fails closed** on
mismatch. A changed evaluator must invalidate the run, not silently alter it.

| Artifact | SHA-256 |
|---|---|
| `KuaiRand-Pure.tar.gz` | `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0` |
| `evaluate.py` | `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de` |
| `submit.py` | `ab01bb2b970ae2a9f2ead299f5240b71ff4126c2d9bb0e0c4de6c7e245dc148c` |
| `baseline.py` | `c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a` |
| `data.py` | `1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541` |

## Split sizes — match the plan exactly

`{'train': 1141112, 'valid': 124909, 'test': 170588}`,
fields `['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']`

## Baselines reproduced (seed 0)

| Model | valid primary | reference | test primary | reference |
|---|---|---|---|---|
| random | 0.4827 | 0.4834 | 0.4757 | 0.4753 |
| item popularity | 0.5807 | 0.5807 | 0.5715 | 0.5715 |
| **FM (to beat)** | **0.6015** | 0.6016 | **0.5953** | 0.5946 |

All inside the 0.0008 seed std. The harness is sound — the starter kit's own
self-check is that random must land at primary ≈ 0.475 ± 0.001.

FM early-stopped at epoch 11, peaking around epoch 7 (0.6015) and decaying
after. Consistent with trap 11: CTR models peak after 1–3 effective epochs
then degrade. Whole run: 22s on CPU.

## Submission round trip

`--make` → `--check` passes on both splits: 170,588 rows (test), 124,909
(valid). `--score --split valid` returns GAUC 0.6671 | nDCG@5 0.5358 |
primary 0.6015, agreeing with `baseline.py`.

## Metric profile, read from `evaluate.py`

- `primary = mean(GAUC, nDCG@5)`, k=5
- label **`long_view`** — not `click`; see the corrections in the README
- nDCG gain `2^rel - 1`, identity under binary labels
- zero-positive users: nDCG 0.0, counted in the mean
- GAUC counts only users with `0 < positives < impressions`, weighted by
  positive count

## Ceiling

Oracle primary is **0.8645** on test (nDCG@5 caps at 0.7289 because 27.1% of
test users are all-negative), not 1.0. FM has already taken 30.7% of the
available range. Measure progress against 0.8645, not 1.0.

## Still open (Section 15, for the webinar)

Compute budget; whether seeding `agent/knowledge.md` counts as a manual
intervention; whether a demo video is required for Track 2; the cutoff of
`video_features_statistic_pure.csv` (excluded by default).
