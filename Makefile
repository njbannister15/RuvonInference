.PHONY: setup fmt lint test run-cli run-api

setup:
	uv sync
	uv run pre-commit install

fmt:
	ruff format .

lint:
	ruff check .
