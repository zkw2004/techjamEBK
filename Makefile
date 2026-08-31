.PHONY: setup data test lint check selfcheck preflight baseline probes run report finalize help

# Every target below shells out to `cli.py`, which is the supported entry
# point: it loads `.env`, archives a prior ledger under `--fresh`, and actually
# parses its flags. `python -m agent.loop` also starts a run, but its `main()`
# takes Python keyword arguments and never reads `argv`, so flags passed to it
# are silently ignored -- do not wrap it here.

help:  ## List available targets
	@grep -hE '^[a-z][a-zA-Z_-]*:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install deps and hooks
	pip install -e ".[dev]"
	pre-commit install

data:  ## Fetch and verify KuaiRand-Pure (git-ignored, ~45MB download)
	bash scripts/get_data.sh

test:
	pytest -q

lint:
	ruff check .

check: lint test  ## What CI runs

selfcheck:  ## Preflight: contracts, hashes, and data integrity
	python3 cli.py selfcheck

preflight:  ## Run D13 label-permutation and injected-leak canaries
	python3 cli.py --preflight

baseline:  ## Reproduce the official FM baseline (validation primary ~0.6016)
	python3 cli.py baseline

probes:  ## Five model-direction probes on internal folds, never on validation
	python3 cli.py probes --fidelity screen --fm-trials 30

# ITERATIONS/HOURS are overridable: `make run ITERATIONS=25 HOURS=2`. Compute is
# only ~9 minutes across a 50-iteration run; the wall clock is dominated by
# propose() latency, so HOURS bounds the API spend more than the training.
ITERATIONS ?= 50
HOURS ?= 6

run:  ## Unattended agent run (override: make run ITERATIONS=25 HOURS=2)
	python3 cli.py run --fresh --max-iterations $(ITERATIONS) --max-hours $(HOURS)

report:  ## Regenerate report + trajectory from the append-only ledger
	python -m tools.report \
		--markdown-output artifacts/experiment-report.md \
		--json-output artifacts/experiment-report.json \
		--trajectory artifacts/trajectory.png

# Defaults to the best accepted node when NODE is unset. Fails closed if the
# ledger holds no accepted full/confirm node (tools/finalise.py::FinaliseError),
# which is the intended behaviour -- there is nothing valid to submit.
NODE ?=
finalize:  ## Refit the accepted winner and write submission.csv
	python3 cli.py finalize $(if $(NODE),--node $(NODE),) --output submission.csv
	python3 kuairand-starter-kit/submit.py --check --split test submission.csv
