.PHONY: setup fmt lint test test-fast test-all run-cli run-api predict generate benchmark monitor stress-test rapid-test test-api test-deployment

setup:
	uv sync --group dev --group test
	uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
	uv run pre-commit install

fmt:
	ruff format .

lint:
	ruff check .

test:
	uv run pytest -v

test-cov:
	uv run pytest -v --cov=ruvoninference --cov-report=term-missing

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

# API Testing with Newman (Postman CLI)
test-api: check-newman
	@echo "🧪 Testing API at http://127.0.0.1:8000"
	@mkdir -p test-reports
	newman run postman_collection.json \
		--env-var "base_url=http://127.0.0.1:8000" \
		--timeout 60000 \
		--reporters cli,html,json \
		--reporter-html-export test-reports/api-test-report.html \
		--reporter-json-export test-reports/api-test-results.json \
		--color on
	@echo "📊 Detailed report: test-reports/api-test-report.html"

# Test deployed API (pass URL as BASE_URL env var)
test-deployment: check-newman
	@if [ -z "$(BASE_URL)" ]; then \
		echo "❌ Please provide BASE_URL: make test-deployment BASE_URL=https://your-api.com"; \
		exit 1; \
	fi
	@echo "🧪 Testing deployed API at: $(BASE_URL)"
	@mkdir -p test-reports
	newman run postman_collection.json \
		--env-var "base_url=$(BASE_URL)" \
		--timeout 60000 \
		--reporters cli,html,json \
		--reporter-html-export test-reports/deployment-test-report.html \
		--reporter-json-export test-reports/deployment-test-results.json \
		--color on
	@echo "📊 Detailed report: test-reports/deployment-test-report.html"

# Quick health check only
test-health: check-newman
	@echo "🏥 Running health check tests..."
	newman run postman_collection.json \
		--folder "Health & Info Tests" \
		--env-var "base_url=$(or $(BASE_URL),http://127.0.0.1:8000)" \
		--timeout 30000 \
		--reporters cli \
		--color on

# Check if Newman is installed
check-newman:
	@which newman > /dev/null || (echo "❌ Newman not found. Run: npm install -g newman" && exit 1)
