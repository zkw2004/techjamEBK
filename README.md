# Autonomous ML Research Agent for Recommender Systems

TechJam 2026, Track 2 · Kaiwen · Malvika · Ethan · Pinxin

**Not a recommender system. An agent that builds recommender systems.**

A closed loop that reads KuaiRand-Pure, proposes an improvement hypothesis,
writes or configures the code for it, screens it cheaply, evaluates survivors
against the official metric, statistically tests whether the change was real,
records the outcome, and repeats until convergence. The recommender model is
an artifact of the loop, not the deliverable.

Full specification: [AGENT_PLAN.md](AGENT_PLAN.md). It is the single source of
truth — the contracts in Section 8 are frozen.

## Status

Scaffolded; Day-0 verification complete and reproducing — see [DAY0.md](DAY0.md).
Next on the critical path: **B1** ([pipeline/data.py](pipeline/data.py)), then
C1, then the C2 gate.

### Corrections to AGENT_PLAN.md, read out of the shipped code

The plan's Section 8 contracts are frozen, so these are reported rather than
silently applied — but the scaffold already reflects them, because code built
against the plan's prose would be wrong.

1. **The label is `long_view`, not `click`** (`data.py:5`, `baseline_scores.json`).
   The plan lists `long_view` as forbidden and `click` as the label; it is
   inverted. `is_click` is now the forbidden same-row signal.
2. **`duration_ms` is not leakage.** It is a video property known before
   exposure and is one of the five official baseline fields (`dur_bucket`).
3. **The C2 gate should read `valid`** (random 0.4834, popularity 0.5807), not
   the plan's 0.4753 / 0.5715, which are test figures.
4. **Two of the plan's axes are measured dead ends.** The organisers report
   that adding CWM's 13 feature fields scores 0.5940 vs 0.5950, and that
   embedding capacity k=8/16/32 is flat. Pure user-side first-order terms
   contribute *exactly* zero, since ranking is within-user — they can only act
   through crosses with item-side terms.
5. **B8 cannot debias training.** `log_random_4_22_to_5_08_pure.csv` is
   entirely after the 21 April training cutoff, so under the plan's own rule
   it is an unbiased validation set only.

The organisers' own ranked headroom is loss function → user history sequences
→ multi-task → watch-time censored regression → model class. The plan's ladder
is close to the reverse of this; Section 6.7 should be reordered before Day 1
rather than after.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: uv sync
pre-commit install
cp .env.example .env             # add ANTHROPIC_API_KEY
```

Then install the two immutable starter-kit files and the data — see
[pipeline/README_STARTER_KIT.md](pipeline/README_STARTER_KIT.md). Neither the
data nor `.env` is ever committed.

```bash
pytest                           # green; unbuilt tasks show as skips
ruff check .
```

CI runs both on every PR to `main`, plus `pre-commit run --all-files` as a
secrets gate. Neither job needs the dataset — every test runs against fixtures
and tmp dirs, so CI stays in seconds.

## Layout

| Path | Contains | Owner |
|---|---|---|
| [pipeline/](pipeline/) | Locked services: splits, features, training, scoring | B / C |
| [pipeline/evaluate.py](pipeline/) · `submit.py` | **Starter kit, immutable, hashed** | D |
| [agent/](agent/) | The loop: propose, execute, gate, manifest, store | A / D |
| [tools/](tools/) | Probes, report generation, finalisation | D |
| [logs/nodes/](logs/nodes/) | One JSON per node — **the run-log deliverable** | — |

## Trust boundary

The LLM has broad reasoning freedom inside a narrow typed contract. It cannot
modify the evaluator, read hidden-test data, approve its own promotion, or
delete a failed experiment from the ledger.

## Unresolved: the metric contradiction

The brief specifies NDCG@10 and Recall@50 in its task and judging sections,
while the Starter Kit section specifies GAUC and nDCG@5 with their mean as the
primary metric. These favour different objectives and use different K.

Rather than guess, the `MetricProfile` is populated from the shipped
`evaluate.py` and its SHA-256 is recorded in every run manifest. If the answer
changes, one config changes. See Section 4.8.

**Read out of the shipped `evaluate.py`** (`sha256 ecfde283…d195de`), which is
the contract we build against:

- `primary = (GAUC + nDCG@5) / 2`, `k=5`
- nDCG gain `2^rel - 1`, equivalent to identity under binary labels
- zero-positive users: nDCG recorded as 0.0 and counted in the mean
- GAUC counts only users with `0 < positives < impressions`, weighted by
  positive count

Still worth confirming at the webinar that the prose (NDCG@10 / Recall@50) is
stale rather than a second scored profile.

## Reproduction, limitations, contributions

TODO — task D9.

## References

Dacrema et al., *Are We Really Making Much Progress?*, RecSys 2019 ·
Rendle et al., *Neural Collaborative Filtering vs. Matrix Factorization
Revisited*, RecSys 2020 · Jiang et al., *AIDE: AI-Driven Exploration in the
Space of Code*, 2025 (arXiv:2502.13138).
