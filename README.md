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

The complete research loop is implemented: immutable data/scoring contracts,
leakage-checked features, four model families, internal-fold screening,
bootstrap promotion, append-only experiment records, and final submission
generation. Day-0 contract checks and baseline reproduction are recorded in
[DAY0.md](DAY0.md).

The active deliverable is an evidence-backed run, not a claim that every model
is universally best. Run the probes, inspect the ledger, then finalise the
accepted validation-best configuration.

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
make setup                       # deps + pre-commit hooks
make data                        # fetch + checksum KuaiRand-Pure (45MB)
cp .env.example .env             # add ANTHROPIC_API_KEY
make check                       # what CI runs
make preflight                   # independent leakage canaries
```

`make data` is idempotent, so re-running it is a no-op once the dataset is
there. It verifies the archive against the SHA-256 in [DAY0.md](DAY0.md) and
fails loudly on a mismatch — every recorded baseline assumes that archive.

**The dataset is deliberately not committed.** It is 194MB extracted with an
83MB largest file, and a blob that size is permanent in git history: removing
it later needs a rewrite that breaks everyone's clone. It is one public file
from a stable Zenodo DOI, so fetching beats vendoring. `.env` is never
committed either.

`pipeline/evaluate.py` and `submit.py` are vendored from the starter kit and
must stay byte-identical — see
[pipeline/README_STARTER_KIT.md](pipeline/README_STARTER_KIT.md).

```bash
pytest                           # test suite
ruff check .
```

CI runs both on every PR to `main`, plus `pre-commit run --all-files` as a
secrets gate. The loader also has full-archive integration checks, but they
skip when the git-ignored dataset is absent; its split logic is exercised
against temporary CSV fixtures in every CI run. This keeps CI in seconds while
`make data && pytest` validates the organiser-fixed row counts locally.

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

## Reproduction

From a clean clone, install the development dependencies and organiser data:

```bash
make setup
make data
make check
```

Run the five model-direction probes on internal temporal folds (the default
does not spend the official validation window):

```bash
python -m tools.probes --fidelity screen --fm-trials 30
```

The unattended agent uses the Anthropic key in your ignored `.env` and writes
an append-only record for every proposal, pilot, failure, and recovery.

`cli.py` is the supported entry point: it loads `.env`, archives any prior
ledger under `--fresh`, and parses these flags. **`python -m agent.loop` also
starts a run, but its `main()` takes Python keyword arguments and never reads
`argv`** — so `python -m agent.loop --max-iterations 10` silently runs the
default 50 instead of failing. Use `cli.py`.

```bash
python3 cli.py run --fresh --max-iterations 50 --max-hours 6
python -m tools.report \
  --markdown-output artifacts/experiment-report.md \
  --json-output artifacts/experiment-report.json \
  --trajectory artifacts/trajectory.png
```

The report and trajectory are generated only from the append-only node ledger;
failed or metric-less attempts remain in the ledger and are never interpolated
into the chart. A submission-ready project narrative is maintained in
[DEVPOST.md](DEVPOST.md).
The repository's generated snapshot is indexed in
[DELIVERABLES.md](DELIVERABLES.md); regenerate it after any ledger change.

During an unattended run, open a second terminal for the live experiment tree:

```bash
python -m tools.live --watch
```

After the statistical gate accepts a full-fidelity winner, create the final
test submission. This refits the chosen configuration on the permitted
`train + validation` period, averages exactly five seeds, preserves the
loader's test order, and calls the untouched organiser checker:

```bash
python -m tools.finalise --node n017 --output submission.csv
```

Replace `n017` with the accepted full/confirm node ID. The output has exactly
`row_id,user_id,video_id,score`; `row_id` is the positional index from
`pipeline.data.load()`, never a join on user/video pairs.

For the optional finals demo, [WALKTHROUGH.md](WALKTHROUGH.md) provides the
three-minute shot list and a safe deterministic OOM-recovery replay:

```bash
python -m tools.oom_demo
```

## Limitations

- The organiser brief conflicts with its shipped evaluator. This project uses
  the hashed evaluator contract: GAUC and nDCG@5, not prose-only alternatives.
- The randomised-exposure log starts after the training cutoff, so it is used
  only as a diagnostic/validation slice—not for training-time debiasing.
- `video_features_statistic_pure.csv` remains excluded because its aggregation
  cutoff is not proven to be before the test window.
- Validation is a budgeted selection resource. Internal expanding folds, not
  the official validation labels, tune feature parameters and blend weights.
- Results are data- and compute-budget-specific; model probes decide the
  search order rather than assuming neural models outperform classical ones.

## Contributions

| Member | Contribution |
|---|---|
| Kaiwen | Agent schemas, node ledger, structured proposal/repair calls, isolated execution, recovery, fidelity loop, and search control. |
| Malvika | Fixed temporal data splits, internal folds, feature registry, historical aggregates, smoothing, leakage defence, negative sampling, and exposure diagnostics. |
| Ethan | Experiment runner, reference baselines, FM/LightGBM/DeepFM models, tuning, score caching, and blending. |
| Pinxin | Evaluation integrity, metric manifest, promotion/segment gates, probes, reporting, finalisation workflow, README, and research priors. |

## References

Dacrema et al., *Are We Really Making Much Progress?*, RecSys 2019 ·
Rendle et al., *Neural Collaborative Filtering vs. Matrix Factorization
Revisited*, RecSys 2020 · Jiang et al., *AIDE: AI-Driven Exploration in the
Space of Code*, 2025 (arXiv:2502.13138).
