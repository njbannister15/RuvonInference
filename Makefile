.PHONY: setup fmt lint test test-fast test-all run-cli run-api predict generate benchmark monitor stress-test rapid-test test-api test-deployment docker-build-cpu docker-build-gpu docker-run-cpu docker-run-gpu create-ecr delete-ecr docker-push-ecr

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


# Build Docker image for CPU deployment
docker-build-cpu:
	@echo "🐳 Building CPU Docker image..."
	docker build \
		--build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" \
		--tag ruvoninference-cpu:latest \
		--file Dockerfile \
		.
	@echo "✅ CPU Docker image built: ruvoninference-cpu:latest"

# Build Docker image for GPU deployment
docker-build-gpu:
	@echo "🐳 Building GPU Docker image..."
	docker build \
		--build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" \
		--tag ruvoninference-gpu:latest \
		--file Dockerfile \
		.
	@echo "✅ GPU Docker image built: ruvoninference-gpu:latest"

# Run CPU container locally
docker-run-cpu: docker-build-cpu
	@echo "🚀 Running CPU container on port 8000..."
	docker run -p 8000:8000 \
		-e DEVICE=cpu \
		-e MODEL_NAME=gpt2 \
		--name ruvoninference-cpu-test \
		--rm \
		ruvoninference-cpu:latest

# Run GPU container locally (requires nvidia-docker)
docker-run-gpu: docker-build-gpu
	@echo "🚀 Running GPU container on port 8000..."
	docker run -p 8000:8000 \
		-e DEVICE=cuda \
		-e MODEL_NAME=gpt2 \
		--gpus all \
		--name ruvoninference-gpu-test \
		--rm \
		ruvoninference-gpu:latest

# Create ECR repositories (idempotent)
create-ecr:
	@if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "❌ Please set AWS_ACCOUNT_ID and AWS_REGION environment variables"; \
		echo "Example: make create-ecr AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1"; \
		exit 1; \
	fi
	@echo "🏗️  Creating ECR repositories..."
	@for repo in ruvoninference-cpu ruvoninference-gpu; do \
		echo "Creating repository: $$repo"; \
		aws ecr create-repository \
			--repository-name $$repo \
			--region $(AWS_REGION) \
			--image-scanning-configuration scanOnPush=true 2>/dev/null \
		|| echo "Repository $$repo already exists (skipping)"; \
		echo "Setting lifecycle policy for $$repo"; \
		aws ecr put-lifecycle-policy \
			--repository-name $$repo \
			--region $(AWS_REGION) \
			--lifecycle-policy-text '{"rules":[{"rulePriority":1,"selection":{"tagStatus":"untagged","countType":"sinceImagePushed","countUnit":"days","countNumber":1},"action":{"type":"expire"}},{"rulePriority":2,"selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' \
			> /dev/null || echo "Failed to set lifecycle policy for $$repo"; \
	done
	@echo "✅ ECR repositories ready"

# Delete ECR repositories (cleanup)
delete-ecr:
	@if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "❌ Please set AWS_ACCOUNT_ID and AWS_REGION environment variables"; \
		exit 1; \
	fi
	@echo "🗑️  Deleting ECR repositories..."
	@for repo in ruvoninference-cpu ruvoninference-gpu; do \
		echo "Deleting repository: $$repo"; \
		aws ecr delete-repository \
			--repository-name $$repo \
			--region $(AWS_REGION) \
			--force 2>/dev/null \
		|| echo "Repository $$repo doesn't exist (skipping)"; \
	done
	@echo "✅ ECR repositories deleted"

# Push images to ECR (requires AWS_ACCOUNT_ID and AWS_REGION env vars)
docker-push-ecr: docker-build-cpu docker-build-gpu create-ecr
	@if [ -z "$(AWS_ACCOUNT_ID)" ] || [ -z "$(AWS_REGION)" ]; then \
		echo "❌ Please set AWS_ACCOUNT_ID and AWS_REGION environment variables"; \
		echo "Example: make docker-push-ecr AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1"; \
		exit 1; \
	fi
	@echo "🚀 Pushing images to ECR..."
	@ECR_REGISTRY=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com; \
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $$ECR_REGISTRY; \
	docker tag ruvoninference-cpu:latest $$ECR_REGISTRY/ruvoninference-cpu:latest; \
	docker tag ruvoninference-gpu:latest $$ECR_REGISTRY/ruvoninference-gpu:latest; \
	docker push $$ECR_REGISTRY/ruvoninference-cpu:latest; \
	docker push $$ECR_REGISTRY/ruvoninference-gpu:latest; \
	echo "✅ Images pushed to ECR: $$ECR_REGISTRY"
