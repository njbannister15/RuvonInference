.PHONY: setup fmt lint test run-cli run-api day1

setup:
	uv sync
	uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
	uv run pre-commit install

fmt:
	ruff format .

lint:
	ruff check .

run-cli:
	uv run python cli.py

day1:
	uv run python cli.py day1
