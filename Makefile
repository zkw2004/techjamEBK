.PHONY: setup data test lint check

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
