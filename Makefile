.PHONY: install lint format typecheck test run-local clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest tests -v

run-local:
	credittarget run-all --config configs/local.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info artifacts
