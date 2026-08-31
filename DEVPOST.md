# Devpost submission draft

## Submission links

- Public repository: `[PASTE PUBLIC GITHUB URL]`
- Optional recommended demo: `[PASTE VERIFIED YOUTUBE URL OR REMOVE THIS LINE]`

Do not publish the Devpost entry until the repository placeholder is replaced
and opens in a signed-out browser. Track 2 does not require a video; include
the recommended approximately three-minute demo only if it improves clarity.

## Inspiration

Recommendation benchmarks often reward one final model while hiding the
engineering process that produced it. TechJam Track 2 asks for the opposite:
an autonomous agent whose hypotheses, failures, resource use, recovery, and
promotion decisions are inspectable. We built a research loop that treats the
recommender as an artifact of evidence-driven search rather than a manually
selected endpoint.

## What it does

The agent reads the organiser-fixed KuaiRand-Pure temporal splits, proposes a
falsifiable model or feature hypothesis, executes it in an isolated worker,
and escalates promising candidates from smoke checks to three internal
expanding-window folds and then full validation. It promotes a candidate only
when the contract, leakage, execution, submission, internal-evidence,
bootstrap, segment, and resource gates pass. Every attempt—including errors
and repairs—is written to an append-only node ledger.

The finalisation workflow selects an accepted full-fidelity node, refits its
configuration on the temporally legal train-plus-validation period, averages
five distinct seeds, writes scores in untouched test-row order, and delegates
the final check to the organiser's immutable submission script.

## How we built it

- Python 3.11/3.12 with Pydantic contracts for agent actions and configs.
- NumPy and pandas for aligned score vectors, temporal features, and
  user-level bootstrap statistics.
- Factorization Machines, LightGBM pointwise/LambdaRank, and CPU DeepFM as the
  bounded model ladder.
- Optuna TPE with median pruning and SQLite storage for resumable tuning.
- Anthropic structured generation for hypothesis proposal and bounded repair.
- Rich for the live node-tree view; Matplotlib and Jinja-friendly Markdown for
  generated reporting.
- Pytest, Ruff, pre-commit, and detect-secrets in GitHub Actions.

The only dataset is KuaiRand-Pure. The randomised-exposure slice is kept as a
diagnostic because it begins after the training cutoff; it is never smuggled
into training. The monthly item-statistics file remains excluded because its
temporal provenance cannot be proved safe.

## Evaluation integrity

We copy the organiser's `evaluate.py` and `submit.py` byte-for-byte and hash
them at preflight together with the canonical dataset archive. A typed
`MetricProfile` is read from shipped code—not prose—and every experiment node
carries the manifest hash. This matters because the written brief mentions
NDCG@10 and Recall@50, while the shipped evaluator computes GAUC and nDCG@5
and averages them. We fail closed on a contract change instead of silently
mixing results.

Statistical promotion bootstraps users rather than rows because the metric's
unit of independence is the user. Full results also report user-activity
quartiles, item-popularity quartiles, and validation day so a headline lift
cannot conceal a collapsed segment.

## Challenges

The hardest failures looked reasonable: random cross-validation leaked future
events, same-row engagement columns exposed post-outcome information, and
repeated `(user_id, video_id)` pairs made that pair unsafe as a submission
key. We enforce expanding date folds, historical cutoffs, and positional
`row_id` alignment. Native ML backends also conflicted on Intel macOS, so the
lockfile selects compatible Torch/NumPy versions and the runner prevents mixed
native backends in one worker.

## Accomplishments

- Immutable, hash-stamped metric and submission contracts.
- Cheap-to-expensive fidelity escalation with classified, bounded recovery.
- User-bootstrap promotion and three-dimensional segment reporting.
- Resumable model probes and tuning without spending official validation on
  hyperparameter fitting.
- Zero-hand-assembly result, resource, intervention, and trajectory reports.
- Five-seed, train-plus-validation finalisation with organiser validation.

## Results (converged run, `artifacts/experiment-report.{md,json}`)

- **21 nodes**, stop reason **`converged`** (three consecutive full-tier
  evaluations failed to clear the accept gate by more than 0.002 — see
  "What we'd do with more runway" below).
- **1 accepted node**: the seeded FM baseline anchor, validation primary
  **0.601684** (matches the organiser-published 0.6016 within one seed-std).
  No candidate the loop proposed cleared the promotion gate, so the anchor
  remains the validation-best checkpoint and is what `cli.py finalize`
  refit for the submission.
- **Best measured (not promoted)**: an FM run with BPR pairwise loss instead
  of pointwise BCE, delta **+0.000816** over the anchor at 5-seed confirm
  tier, 95% bootstrap CI **[+0.000193, +0.003474]** — a real, reproducible
  effect, excluding zero, that nonetheless falls under the project's
  `MIN_DELTA_FLOOR = 0.002` (≈2.5× the measured 0.0008 seed-noise floor).
  We report it as a verified non-win rather than round it up.
- **8m26s total agent wall-clock**, **12 of 50 iterations**, **0 GPU-hours**
  (CPU-only models throughout), and **40,642 LLM tokens** (32,796 in /
  7,846 out) across the whole run.
- **0 manual interventions** — the operator definition (restarting a
  crashed process, editing code mid-run, killing a hung run, hand-picking a
  candidate) never fired; one screen-tier node hit a schema error and was
  auto-repaired by the recovery policy, and the loop continued unattended.
- Citations attached to 10 of the 21 hypotheses (BPR — Rendle 2009;
  LambdaRank; D2Q — Zhan et al., KDD 2022), each rendered inline in the
  hypothesis text and visible in the run log.
- The incumbent's own ablation flagged `tab` as the most sensitive feature
  (removing it cost 0.0125 primary), ahead of `user_id` (0.0056) and
  `dur_bucket`; 8 registered features (`pcr_hist`, `user_activity`, and the
  `user_*_rate_decayed` family) are structurally `metric_inert` — constant
  within a user's own candidate list, and therefore invisible to a
  within-user metric regardless of model.
- Read the delta in context: validation candidates are a median of **4 per
  user**, and only **57.8%** of users contribute to GAUC at all, so a
  +0.002 delta is a large effect on this task, not a small one.

## What we'd do with more runway

The loop's convergence rule fired at the earliest point it mathematically
could — 3 consecutive full-tier misses right after the seeded anchor, with
only 4 full-tier evaluations total. That is a known, named failure mode
(not a search-space verdict): the fix already scoped is widening the
no-improvement window so a few unlucky early proposals cannot end the run
before the scheduler's hedge queue gets enough tries. A same-day
hand-driven search (documented in the run log's companion notes) covered
~40 additional configurations under the identical statistical bar and found
the same shape of result — several real, reproducible, sub-floor effects,
and no single change that cleared the promotion gate — which is why we
report this as a genuine ceiling on this feature set rather than an
under-searched run.

## What we learned

Autonomy is not the absence of constraints. It comes from giving the agent a
wide hypothesis space inside narrow, testable interfaces. Typed configs handle
routine iterations reliably; isolated code generation remains available for
novel features, where failures become logged evidence rather than hidden
manual debugging.

## What's next

The unattended search has run to convergence, the report and trajectory are
generated from the ledger, and the validation-best accepted node has been
refit and submitted (`submission.csv`, validated against the organiser's own
`submit.py --check`). Remaining work is widening the convergence window
described above and re-running with the larger iteration/hour budget still
available, to confirm the current result against a longer search rather than
one that stopped at the earliest legal point. Organizer confirmation of the
prose-versus-evaluator metric discrepancy remains the only external contract
question.
