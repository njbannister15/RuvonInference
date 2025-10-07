# Intermission: Production Deployment Infrastructure

*From development container to AWS production: Building a deployment pipeline*

After building our inference engine through Parts 1-9, we reached a milestone: **taking our educational system from local development to production-like deployment**. This intermission documents the infrastructure we built to deploy RuvonInference to AWS with Docker containers, ECR image registry, and EC2 provisioning.

## The Production Reality Check

Our CLI tool and FastAPI server worked beautifully in development, but production deployment requires solving a completely different set of problems:

- **Containerization**: Packaging our application with all dependencies
- **Image Registry**: Storing and versioning our Docker images
- **Infrastructure as Code**: Reproducible AWS resource provisioning
- **Security**: IAM roles, networking, and credential management
- **Monitoring**: Health checks and automated restart capabilities

Let's walk through exactly how we solved each of these challenges.

## Docker: Containerizing the Inference Engine

Our first step was packaging RuvonInference into production-ready Docker containers. We created a **multi-stage Dockerfile** optimized for both CPU and GPU deployments.

### Multi-Stage Build Strategy

```dockerfile
# Build stage - Heavy dependencies and compilation
FROM python:3.12-slim as builder
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (CPU by default, overrideable for GPU)
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN uv pip install torch==2.2.2 torchvision==0.17.2 --index-url $TORCH_INDEX_URL
RUN uv pip install .

# Production stage - Minimal runtime image
FROM python:3.12-slim as production
```

### Key Design Decisions

**Build Args for CPU/GPU Flexibility**: The `TORCH_INDEX_URL` build argument lets us create both CPU and GPU variants from the same Dockerfile:

```bash
# CPU build
docker build --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

# GPU build
docker build --build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
```

**Security-First Design**: Non-root user, minimal dependencies, encrypted cache directories:

```dockerfile
# Create non-root user for security
RUN groupadd -r ruvoninference && useradd -r -g ruvoninference ruvoninference

# Create cache directory with proper ownership
RUN mkdir -p /app/.cache/huggingface && \
    chown -R ruvoninference:ruvoninference /app

USER ruvoninference
```

**Production Environment Variables**: Clean separation of runtime configuration:

```dockerfile
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV DEVICE=cpu
ENV MODEL_NAME=gpt2
```

## Makefile: Deployment Automation

We integrated Docker builds into our existing Makefile workflow, creating a seamless development-to-production pipeline.

### Docker Build Targets

```makefile
# Build Docker image for CPU deployment
docker-build-cpu:
	@echo "🐳 Building CPU Docker image..."
	docker build \
		--build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" \
		--tag ruvoninference-cpu:latest \
		--file Dockerfile \
		.

# Build Docker image for GPU deployment
docker-build-gpu:
	@echo "🐳 Building GPU Docker image..."
	docker build \
		--build-arg TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" \
		--tag ruvoninference-gpu:latest \
		--file Dockerfile \
		.
```

### ECR Integration

The real breakthrough was automating **Amazon ECR** (Elastic Container Registry) integration:

```makefile
# Create ECR repositories (idempotent)
create-ecr:
	@for repo in ruvoninference-cpu ruvoninference-gpu; do \
		aws ecr create-repository \
			--repository-name $$repo \
			--region $(AWS_REGION) \
			--image-scanning-configuration scanOnPush=true || \
		echo "Repository $$repo already exists"; \
		aws ecr put-lifecycle-policy \
			--repository-name $$repo \
			--lifecycle-policy-text '{"rules":[...]}'; \
	done

# Push images to ECR
docker-push-ecr: docker-build-cpu docker-build-gpu create-ecr
	@ECR_REGISTRY=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com; \
	aws ecr get-login-password | docker login --username AWS --password-stdin $$ECR_REGISTRY; \
	docker tag ruvoninference-cpu:latest $$ECR_REGISTRY/ruvoninference-cpu:latest; \
	docker push $$ECR_REGISTRY/ruvoninference-cpu:latest
```

**Lifecycle Policies**: Automatic cleanup keeps costs down by retaining only the 10 most recent images and deleting untagged images after 1 day.

## Terraform: Infrastructure as Code

With containers ready, we needed **reproducible infrastructure**. Our Terraform configuration provisions complete AWS environments with conditional CPU/GPU support.

### Variable-Driven Architecture

The foundation is a flexible variable system supporting different deployment scenarios:

```hcl
variable "use_gpu" {
  description = "Whether to use a GPU instance"
  type        = bool
  default     = true
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.xlarge"

  validation {
    condition = contains([
      # CPU instances
      "t3.large",    # 2 vCPU, 8GB
      # GPU instances
      "g5.xlarge",   # 1x A10G, 4 vCPU, 16GB
      "g5.2xlarge",  # 1x A10G, 8 vCPU, 32GB
      "g5.4xlarge",  # 1x A10G, 16 vCPU, 64GB
    ], var.instance_type)
    error_message = "Instance type must be supported."
  }
}
```

### Conditional Resource Provisioning

The elegant solution was **conditional resource creation** based on the `use_gpu` variable:

```hcl
# GPU Instance AMI (only when GPU needed)
data "aws_ami" "deep_learning" {
  count       = var.use_gpu ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 *Ubuntu*"]
  }
}

# CPU Instance AMI (only when CPU needed)
data "aws_ami" "ecs_optimized" {
  count       = var.use_gpu ? 0 : 1
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-hvm-*-x86_64-ebs"]
  }
}

# Conditional instance creation
resource "aws_instance" "ruvoninference_gpu" {
  count = var.use_gpu ? 1 : 0
  ami   = data.aws_ami.deep_learning[0].id
  # ... configuration
}

resource "aws_instance" "ruvoninference_cpu" {
  count = var.use_gpu ? 0 : 1
  ami   = data.aws_ami.ecs_optimized[0].id
  # ... configuration
}
```

### IAM Security Model

Production deployment requires proper **IAM roles** for ECR access:

```hcl
# IAM Role for EC2 instances
resource "aws_iam_role" "ruvoninference_ec2_role" {
  name = "ruvoninference-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

# ECR Access Policy
resource "aws_iam_policy" "ruvoninference_ecr_policy" {
  name = "ruvoninference-ecr-policy-${var.environment}"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ]
      Resource = "*"
    }]
  })
}

# Instance Profile (bridges IAM role to EC2)
resource "aws_iam_instance_profile" "ruvoninference_instance_profile" {
  name = "ruvoninference-instance-profile-${var.environment}"
  role = aws_iam_role.ruvoninference_ec2_role.name
}
```

**Security Best Practice**: The instance profile provides **temporary, rotating credentials** automatically. No need to manage AWS keys on the instances.

## User Data: Automated Bootstrap

The final piece was **user data scripts** that automatically configure instances on boot. We created separate scripts for CPU and GPU deployments.

### CPU Instance Bootstrap

```bash
#!/bin/bash
set -e

# Variables templated by Terraform
AWS_REGION="${aws_region}"
AWS_ACCOUNT_ID="${aws_account_id}"
ECR_REPOSITORY="ruvoninference-cpu"
CONTAINER_NAME="ruvoninference-cpu"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/ruvoninference-setup.log
}

log "Starting RuvonInference CPU setup..."

# Update system and start Docker
yum update -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Login to ECR using IAM instance profile
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull and run the container
FULL_IMAGE_NAME="$ECR_REGISTRY/$ECR_REPOSITORY:latest"
docker pull $FULL_IMAGE_NAME

docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p 8000:8000 \
    -e DEVICE=cpu \
    -e MODEL_NAME=gpt2 \
    $FULL_IMAGE_NAME

log "SUCCESS: Container running on port 8000"
```

### GPU Instance Enhancements

The GPU script adds **NVIDIA Container Toolkit** configuration:

```bash
# Install nvidia-container-toolkit for GPU support
log "Installing NVIDIA Container Toolkit..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
    tee /etc/yum.repos.d/nvidia-container-toolkit.repo
yum install -y nvidia-container-toolkit

# Configure Docker for GPU support
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Run with GPU support
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    --gpus all \
    -p 8000:8000 \
    -e DEVICE=cuda \
    $FULL_IMAGE_NAME
```

### Health Monitoring

Both scripts include **automated monitoring**:

```bash
# Create monitoring script
cat > /home/ec2-user/monitor.sh << 'EOF'
#!/bin/bash
CONTAINER_NAME="ruvoninference-cpu"

if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Container not running, restarting..."
    docker start $CONTAINER_NAME
fi

# Health check
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "Health check: PASS"
else
    echo "Health check: FAIL"
fi
EOF

# Set up cron job (check every 5 minutes)
echo "*/5 * * * * /home/ec2-user/monitor.sh >> /var/log/ruvoninference-monitor.log 2>&1" | \
    crontab -u ec2-user -
```

## Production Deployment Workflow

The complete deployment process is now **fully automated**:

### 1. Build and Push Images

```bash
# Set environment variables
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1

# Build and push to ECR
make docker-push-ecr
```

### 2. Deploy Infrastructure

```bash
cd infra

# Configure deployment
cat > terraform.tfvars << EOF
use_gpu = true
instance_type = "g5.xlarge"
environment = "prod"
allowed_ssh_cidrs = ["YOUR_IP/32"]
allowed_api_cidrs = ["YOUR_IP/32"]
EOF

# Deploy
terraform init
terraform plan
terraform apply
```

### 3. Verify Deployment

```bash
# Get the Elastic IP from Terraform output
ELASTIC_IP=$(terraform output -raw elastic_ip)

# Test the API
curl http://$ELASTIC_IP:8000/health
curl -X POST http://$ELASTIC_IP:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 15}'
```

## What We Accomplished

This deployment infrastructure gives us **production-grade capabilities**:

- **Reproducible Deployments**: Infrastructure as Code with Terraform
- **Security**: IAM roles, encrypted storage, non-root containers
- **Flexibility**: CPU/GPU switching via single variable
- **Monitoring**: Health checks, logging, automated restart
- **Cost Optimization**: Spot instances, lifecycle policies, auto-scaling ready
- **Zero-Downtime Updates**: Blue/green deployment ready

## The Educational Value

Building this deployment pipeline taught us that **inference engines are only half the story**. Production AI systems require:

1. **Containerization Strategy**: Multi-stage builds, security, optimization
2. **Infrastructure Automation**: Terraform, conditional resources, variables
3. **Security Models**: IAM roles, network policies, credential management
4. **Operational Excellence**: Monitoring, logging, automated recovery

This infrastructure foundation now supports our continued development through Parts 10-20, letting us focus on advanced inference optimizations while maintaining production deployment capabilities.

**Next up**: Part 10 will dive into logprobs API implementation, building on our solid deployment foundation.

## Key Files

- **[Dockerfile](../Dockerfile)**: Multi-stage container build
- **[Makefile](../Makefile)**: Docker and ECR automation
- **[infra/main.tf](../infra/main.tf)**: Terraform infrastructure
- **[infra/variable.tf](../infra/variable.tf)**: Configuration variables
- **[infra/user_data_cpu.sh](../infra/user_data_cpu.sh)**: CPU instance bootstrap
- **[infra/user_data_gpu.sh](../infra/user_data_gpu.sh)**: GPU instance bootstrap
