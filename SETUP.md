# Setup Guide

How to get this repo running end to end: environment, data, verification, and
the actual agent run. This is an operational how-to; see
[README.md](README.md) for what the project is and
[AGENT_PLAN.md](AGENT_PLAN.md) for the frozen specification.

## 1. Prerequisites

- Python 3.11 or 3.12 (`pyproject.toml` pins `>=3.11,<3.13`)
- An Anthropic API key

## 2. Clone and create a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -e ".[dev]"
pre-commit install
```

Or equivalently:

```bash
make setup
```

## 4. Fetch the dataset

```bash
make data
```

This downloads and checksums KuaiRand-Pure (~45MB archive) into `data/`,
which is git-ignored. The command is idempotent — safe to re-run, it no-ops
once the archive is present and verified. It fails loudly if the checksum
does not match the one recorded in [DAY0.md](DAY0.md).

## 5. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

`ANTHROPIC_WORKSPACE_ID` is only required for identity-linked keys (personal
Console login keys rather than keys issued directly to a workspace) — leave
it blank otherwise. `.env` is git-ignored; never commit it.

## 6. Verify the install

```bash
make check          # ruff + full test suite — what CI runs
```

Then sanity-check the evaluation harness itself against the two reference
rungs, and reproduce the official baseline:

```bash
python3 cli.py selfcheck    # random ≈0.4834, popularity ≈0.5807 on validation
python3 cli.py baseline     # FM ≈0.6016 ± noise — the number the agent must beat
```

If `selfcheck` fails, the evaluation plumbing is broken and nothing
downstream is trustworthy — fix this before running the agent.

## 7. Run the agent

```bash
python3 cli.py run --max-iterations 50 --max-hours 6
```

`cli.py` is the maintained entry point: it loads `.env`, archives any prior
run under `logs/archive/<timestamp>/`, and exposes the real CLI flags.
(`python -m agent.loop` also executes but silently ignores CLI flags — use
`cli.py`.)

Useful flags:

| Flag | Effect |
|---|---|
| `--max-iterations N` | Hard iteration cap (default 50, per K23) |
| `--max-hours H` | Wall-clock ceiling (default 6, per K24) |
| `--fresh` | Archive any existing run before starting (default) |
| `--resume` | Continue an existing run instead of archiving it |
| `--fake-provider` | Use a deterministic stub proposer, no API calls — for a dry run only, not for a scored submission |

The run requires a real `ANTHROPIC_API_KEY` unless `--fake-provider` is set.

## 8. Watching progress while it runs

`cli.py run` prints nothing to stdout while it works — it can look hung for
minutes even when it isn't. Don't just stare at a blank terminal: everything
the loop does is written to `logs/` as it happens, so watch that instead, in
a second terminal.

**Live event stream** (promotions, recoveries, dead-ends — appended as they
happen):

```bash
tail -f logs/run.jsonl
```

**Node count so far** (one JSON file per completed node, `n000`, `n001`, ...):

```bash
watch -n 5 'ls logs/nodes | wc -l'
```

**Latest node's status and score at a glance:**

```bash
python3 - <<'PY'
import json, pathlib
nodes = sorted(pathlib.Path("logs/nodes").glob("n*.json"))
if not nodes:
    print("no nodes written yet")
else:
    n = json.loads(nodes[-1].read_text())
    print(f"{n['id']}  fidelity={n['fidelity']}  status={n['status']}  "
          f"accepted={n.get('accepted')}  primary={n.get('metrics', {}).get('primary')}")
PY
```

**Best-so-far and full trajectory**, refreshed periodically (cheap to
re-run; it only reads `logs/nodes/`):

```bash
watch -n 30 python3 cli.py report
```

If you see nothing appear in `logs/run.jsonl` or `logs/nodes/` within the
first minute or two, the run has likely stalled on the first `propose()`
call — check `.env` for a valid `ANTHROPIC_API_KEY` (see Troubleshooting
below) rather than waiting it out.

## 9. After the run

Generate the report tables:

```bash
python3 cli.py report            # markdown
python3 cli.py report --json     # JSON
```

Produce the final submission from the accepted node (replace `n017` with the
actual accepted full/confirm node ID from your run):

```bash
python3 cli.py finalize --node n017 --output submission.csv
```

Validate the submission against the organiser's own checker:

```bash
python3 kuairand-starter-kit/submit.py --check --split test submission.csv
```

## 10. Other useful commands

```bash
python3 cli.py probes --fidelity screen --fm-trials 30   # the 5 Day-1 model-direction probes
pytest -q                                                  # test suite directly
ruff check .                                                # lint directly
```

## Troubleshooting

- **`selfcheck` or `baseline` fails tolerance:** re-run `make data` to
  confirm the archive checksum, then re-read [DAY0.md](DAY0.md) for the
  exact reference numbers this repo expects.
- **macOS + LightGBM + PyTorch in the same process aborts (`OMP: Error #15`):**
  both libraries ship their own OpenMP runtime. See `ethanprogress.md` for
  the backend-isolation fix already applied in `pipeline/models/__init__.py`.
- **`ImportError: anthropic`:** confirm `pip install -e ".[dev]"` completed
  inside the active virtualenv, not a different Python.
- **A run does nothing and returns almost instantly:** usually a missing or
  invalid `ANTHROPIC_API_KEY` — `cli.py` loads `.env` automatically, so
  check the file exists and the key is valid.
