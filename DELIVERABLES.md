# Deliverables

This index separates generated evidence from presentation material. The report
files and plot are snapshots of the append-only ledger at the commit that
contains them; rerun the command below after adding experiment nodes.

## Evidence

- [Experiment report](artifacts/experiment-report.md) — human-readable node,
  metric, resource, and iteration totals.
- [Machine-readable report](artifacts/experiment-report.json) — the same data
  for downstream checks.
- [Validation trajectory](artifacts/trajectory.png) — measured primary scores
  only; missing and failed runs are not interpolated.
- [Metric manifest](logs/manifest.json) — hashes and the frozen evaluator
  profile used by recorded experiments.
- [Node ledger](logs/nodes/) — append-only experiment records.

Regenerate all three report artifacts from the ledger:

```bash
python -m tools.report \
  --markdown-output artifacts/experiment-report.md \
  --json-output artifacts/experiment-report.json \
  --trajectory artifacts/trajectory.png
```

## Submission and presentation

- [Devpost narrative](DEVPOST.md)
- [Three-minute YouTube walkthrough and recovery demo](WALKTHROUGH.md) —
  recording and public upload still required; do not mark complete until the
  signed-out link works and the final runtime is at most 3:00.
- [Final submission](submission.csv) — generated with `cli.py finalize` (`make
  finalize`) only after the ledger contained an accepted, successful `full`
  node. This safeguard prevents an unpromoted pilot or fabricated winner from
  reaching the hidden test set.

## Results summary (D5)

Converged run: 21 nodes, stop reason `converged`. Full figures and the run's
best measured-but-unpromoted candidate are in
[DEVPOST.md](DEVPOST.md#results-converged-run-artifactsexperiment-reportmdjson);
the numbers below are the two the submission is scored on.

| | Validation primary | GAUC | nDCG@5 | Δ vs. published baseline (0.6016) |
|---|---|---|---|---|
| Accepted node (n003, submitted) | 0.601684 | 0.667311 | 0.536057 | +0.000084 (within one seed-std) |

`submission.csv` is the five-seed, train+validation refit of this node,
validated against the organiser's unmodified `submit.py --check`
(170,588 rows, `split=test`).

No candidate the loop proposed cleared the project's own promotion gate
(point delta ≥ 0.002 **and** a bootstrap 95% CI excluding zero), so the
seeded baseline anchor remains the validation-best accepted checkpoint and
is what was finalised. See DEVPOST.md for the closest candidate that did
not clear the gate, and why the run converged after only 21 nodes.
