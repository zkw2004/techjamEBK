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
- [Three-minute walkthrough and recovery demo](WALKTHROUGH.md)
- Final submission: generated with `python -m tools.finalise` only after the
  ledger contains an accepted, successful `full` or `confirm` node. This
  safeguard prevents an unpromoted pilot or fabricated winner from reaching
  the hidden test set.

At this snapshot the ledger has no accepted full/confirm node, so a valid final
CSV cannot yet be generated. Run the autonomous full-fidelity workflow to earn
an acceptance, then invoke finalisation with that node ID and retain the
organiser check output alongside the submission.
