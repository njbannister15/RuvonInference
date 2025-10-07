#!/bin/bash

# User data script for CPU EC2 instances running RuvonInference.
#
# This script:
# 1. Updates the system and installs dependencies
# 2. Configures Docker and ECR authentication
# 3. Pulls the CPU Docker image from ECR
# 4. Runs the container with proper configuration
# 5. Sets up health monitoring and auto-restart

set -e

# Variables (will be templated by Terraform)
AWS_REGION="${aws_region}"
AWS_ACCOUNT_ID="${aws_account_id}"
ECR_REPOSITORY="ruvoninference-cpu"
CONTAINER_NAME="ruvoninference-cpu"
IMAGE_TAG="latest"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/ruvoninference-setup.log
}

log "Starting RuvonInference CPU setup..."

# Update system packages
log "Updating system packages..."
yum update -y

# Install additional packages
log "Installing additional packages..."
yum install -y htop zip unzip jq

# Start Docker service (should already be installed on ECS-optimized AMI)
log "Starting Docker service..."
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install AWS CLI v2 if not present
if ! command -v aws &> /dev/null; then
    log "Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# Configure AWS region
log "Configuring AWS region..."
aws configure set region $AWS_REGION

# Login to ECR
log "Logging into ECR..."
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull the latest image
log "Pulling Docker image from ECR..."
FULL_IMAGE_NAME="$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
docker pull $FULL_IMAGE_NAME

# Stop and remove existing container if it exists
log "Stopping existing container if running..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run the container
log "Starting RuvonInference CPU container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p 8000:8000 \
    -e DEVICE=cpu \
    -e MODEL_NAME=gpt2 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    $FULL_IMAGE_NAME

# Wait for container to be healthy
log "Waiting for container to be healthy..."
sleep 30

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    log "SUCCESS: RuvonInference CPU container is running successfully!"

    # Test the health endpoint
    if curl -f http://localhost:8000/health &>/dev/null; then
        log "SUCCESS: Health check passed - API is responding"
    else
        log "WARNING: Health check failed - API may still be starting up"
    fi
else
    log "ERROR: Container failed to start"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Create a simple monitoring script
log "Creating monitoring script..."
cat > /home/ec2-user/monitor.sh << 'EOF'
#!/bin/bash
# Simple monitoring script for RuvonInference container

CONTAINER_NAME="ruvoninference-cpu"

if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME is not running, attempting to restart..."
    docker start $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME is running normally"
fi

# Check health endpoint
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "Health check: PASS"
else
    echo "Health check: FAIL"
fi
EOF

chmod +x /home/ec2-user/monitor.sh
chown ec2-user:ec2-user /home/ec2-user/monitor.sh

# Set up crontab for monitoring (check every 5 minutes)
log "Setting up monitoring cron job..."
echo "*/5 * * * * /home/ec2-user/monitor.sh >> /var/log/ruvoninference-monitor.log 2>&1" | crontab -u ec2-user -

log "SUCCESS: RuvonInference CPU setup completed successfully!"
log "Container: $CONTAINER_NAME"
log "Image: $FULL_IMAGE_NAME"
log "API endpoint: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
log "Health check: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/health"
