# Devpost submission draft

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

Quantitative claims should be copied from `python -m tools.report` only after
the converged run is complete; this draft intentionally does not fabricate a
best score.

## What we learned

Autonomy is not the absence of constraints. It comes from giving the agent a
wide hypothesis space inside narrow, testable interfaces. Typed configs handle
routine iterations reliably; isolated code generation remains available for
novel features, where failures become logged evidence rather than hidden
manual debugging.

## What's next

Run the complete probe suite to reorder the provisional model ladder, execute
the unattended search to convergence, generate the final report and
trajectory from the ledger, and use the validation-best accepted node for the
five-seed submission. Organizer confirmation of the prose-versus-evaluator
metric discrepancy remains the only external contract question.
