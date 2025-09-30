.PHONY: setup fmt lint test test-fast test-all run-cli run-api predict generate benchmark monitor stress-test rapid-test

setup:
	uv sync --group dev --group test
	uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
	uv run pre-commit install

fmt:
	ruff format .

lint:
	ruff check .

test:
	uv run pytest tests/unit -v

test-fast:
	uv run pytest tests/unit -v -m "not slow"

test-all:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/unit -v --cov=ruvonvllm --cov-report=term-missing

run-cli:
	uv run python cli.py

run-api:
	uv run python cli.py serve

predict:
	uv run python cli.py predict

generate:
	uv run python cli.py generate

sample:
	uv run python cli.py sample

compare:
	uv run python cli.py compare

benchmark:
	uv run python cli.py benchmark

monitor:
	uv run python cli.py monitor

stress-test:
	uv run python cli.py stress-test

rapid-test:
	uv run python cli.py rapid-test
