# Submission checklist

TechJam 2026, Track 2. Verified against `02_REQUIREMENTS.md` Part 5
(Deliverables) as of the converged run (21 nodes, `logs/nodes/n001`–`n021`).
Re-run the commands under "Verify" yourself before submitting — this file is
a snapshot, not a live check.

## The 7 required deliverables

- [x] **D1 — Devpost writeup** → [`DEVPOST.md`](DEVPOST.md)
      Real numbers as of the converged run: 21 nodes, accepted primary
      0.601684, best non-promoted candidate +0.000816 (CI excludes zero,
      under the 0.002 floor), 40,642 tokens, 0 GPU-hours, 0 manual
      interventions.

- [x] **D2 — Public GitHub repo** → this repo
      README has `## Contributions` and `## Limitations` sections.
      **Action needed:** confirm the repo is set to Public — GitHub
      Settings → General → Danger Zone. Not verifiable from the CLI.

- [x] **D3 — Run and iteration logs** → [`logs/nodes/`](logs/nodes/) +
      [`logs/run.jsonl`](logs/run.jsonl)
      This is the graded artifact for Autonomy (20%) and Robustness
      (part of Technical Execution, 35%) — not documentation. 21 node
      files, one per attempt, each carrying hypothesis / diff / metrics /
      error-recovery events.

- [x] **D4 — Final submission** → [`submission.csv`](submission.csv)
      Five-seed train+validation refit of the accepted node. Passed the
      organiser's own unmodified checker: 170,588 rows, `split=test`.

- [x] **D5 — Results summary** → [`DELIVERABLES.md`](DELIVERABLES.md#results-summary-d5)
      Validation-best component metrics + absolute delta vs. published
      baseline, in a table.

- [x] **D6 — Resource report** → [`artifacts/experiment-report.md`](artifacts/experiment-report.md)
      Total tokens (in/out) and GPU-hours across every LLM call and
      training run.

- [x] **D7 — Secret hygiene** → verified below
      `.env` never committed; not present anywhere in git history.

## Verify each one yourself

```bash
# D2 — confirm README sections exist
grep -n "^## Contributions\|^## Limitations" README.md

# D3 — count nodes, confirm the run log exists
ls logs/nodes/*.json | wc -l
tail -5 logs/run.jsonl

# D4 — re-check the submission against the organiser's own script
python3 kuairand-starter-kit/submit.py submission.csv --check \
  --data_dir data/KuaiRand-Pure/data --split test

# D6 — see the resource totals directly
grep -A5 "^## Run totals" artifacts/experiment-report.md

# D7 — confirm .env was never tracked
git log --all --diff-filter=A --name-only | grep -x '\.env' || echo "clean"
git ls-files | grep -x '\.env' || echo "not tracked"
```

## Before you hand over the repo URL

- [ ] `git push origin main` — confirm `git status -sb` shows no
      `[ahead N]`
- [ ] Repo visibility set to Public (D2, not machine-checkable)
- [ ] Optional: `WALKTHROUGH.md` demo video — **not required for Track 2**
      per `02_REQUIREMENTS.md`'s own note ("Tracks 3, 4 and 5 list a demo
      video as a deliverable. Track 2's list does not.")

## What "final" means for this run

No candidate the loop proposed cleared the project's own promotion gate
(point delta ≥ 0.002 **and** a bootstrap 95% CI excluding zero), so the
seeded FM baseline anchor is the validation-best accepted checkpoint, and
it is what `submission.csv` contains. This is a legitimate, documented R4
outcome per the requirements doc: *"Falling short of the baseline is not
disqualifying. The delta, positive or negative, feeds continuously into
the Primary metric."* The best candidate the run found (BPR pairwise loss,
+0.000816, CI excluding zero) is reported in DEVPOST.md as a verified
non-win rather than rounded up.
