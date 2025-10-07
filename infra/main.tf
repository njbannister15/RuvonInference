terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
provider "aws" {
  region = var.aws_region
}


data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# IAM Role for EC2 instances to access ECR
resource "aws_iam_role" "ruvoninference_ec2_role" {
  name = "ruvoninference-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "ruvoninference-ec2-role-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
  }
}

# IAM Policy for ECR access
resource "aws_iam_policy" "ruvoninference_ecr_policy" {
  name        = "ruvoninference-ecr-policy-${var.environment}"
  description = "Policy for RuvonInference EC2 instances to access ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "ruvoninference-ecr-policy-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
  }
}

# Attach ECR policy to the role
resource "aws_iam_role_policy_attachment" "ruvoninference_ecr_policy_attachment" {
  role       = aws_iam_role.ruvoninference_ec2_role.name
  policy_arn = aws_iam_policy.ruvoninference_ecr_policy.arn
}

# Attach AWS managed policy for EC2 basic operations
resource "aws_iam_role_policy_attachment" "ruvoninference_ec2_policy_attachment" {
  role       = aws_iam_role.ruvoninference_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess"
}

# Instance profile for the IAM role
resource "aws_iam_instance_profile" "ruvoninference_instance_profile" {
  name = "ruvoninference-instance-profile-${var.environment}"
  role = aws_iam_role.ruvoninference_ec2_role.name

  tags = {
    Name        = "ruvoninference-instance-profile-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
  }
}

# Security Group
resource "aws_security_group" "ruvoninference" {
  name_prefix = "ruvoninference-${var.environment}-"
  description = "Security group for ruvoninference instance"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
    description = "SSH access"
  }

  ingress {
    from_port   = 8000
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = var.allowed_api_cidrs
    description = "ruvoninference API ports"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name        = "ruvoninference-${var.environment}"
    Environment = var.environment
    Project     = "RuvonCode"
  }
}


# Key Pair
resource "aws_key_pair" "ruvoninference" {
  count      = var.create_key_pair ? 1 : 0
  key_name   = "ruvoninference-${var.environment}"
  public_key = var.public_key
}

# GPU Instance - Deep Learning AMI with GPU support
data "aws_ami" "deep_learning" {
  count       = var.use_gpu ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 *Ubuntu*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# GPU Instance (conditional)
resource "aws_instance" "ruvoninference_gpu" {
  count                  = var.use_gpu ? 1 : 0
  ami                    = data.aws_ami.deep_learning[0].id
  instance_type          = var.instance_type
  key_name              = var.create_key_pair ? aws_key_pair.ruvoninference[0].key_name : var.existing_key_name
  vpc_security_group_ids = [aws_security_group.ruvoninference.id]
  availability_zone      = data.aws_availability_zones.available.names[0]
  iam_instance_profile   = aws_iam_instance_profile.ruvoninference_instance_profile.name

  ebs_optimized = true

  # Spot config - enables spot pricing (conditional)
  dynamic "instance_market_options" {
    for_each = var.use_spot_instance ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    iops                  = var.root_iops
    throughput            = var.root_throughput
    delete_on_termination = true
    encrypted             = true
  }

  user_data = templatefile("${path.module}/user_data_gpu.sh", {
    aws_region     = var.aws_region
    aws_account_id = data.aws_caller_identity.current.account_id
  })

  tags = {
    Name        = "ruvoninference-gpu-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
    Purpose     = "ruvoninference GPU Testing"
  }

  lifecycle {
    create_before_destroy = true
  }
  user_data_replace_on_change = true
}

# CPU Instance - ECS optimized AMI with Docker pre-installed
data "aws_ami" "ecs_optimized" {
  count       = var.use_gpu ? 0 : 1
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-hvm-*-x86_64-ebs"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# CPU Instance (conditional)
resource "aws_instance" "ruvoninference_cpu" {
  count                  = var.use_gpu ? 0 : 1
  ami                    = data.aws_ami.ecs_optimized[0].id
  instance_type          = var.instance_type
  key_name              = var.create_key_pair ? aws_key_pair.ruvoninference[0].key_name : var.existing_key_name
  vpc_security_group_ids = [aws_security_group.ruvoninference.id]
  availability_zone      = data.aws_availability_zones.available.names[0]
  iam_instance_profile   = aws_iam_instance_profile.ruvoninference_instance_profile.name

  ebs_optimized = true

  # Spot config - enables spot pricing (conditional)
  dynamic "instance_market_options" {
    for_each = var.use_spot_instance ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    iops                  = var.root_iops
    throughput            = var.root_throughput
    delete_on_termination = true
    encrypted             = true
  }

  user_data = templatefile("${path.module}/user_data_cpu.sh", {
    aws_region     = var.aws_region
    aws_account_id = data.aws_caller_identity.current.account_id
  })

  tags = {
    Name        = "ruvoninference-cpu-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
    Purpose     = "ruvoninference CPU Testing"
  }

  lifecycle {
    create_before_destroy = true
  }
  user_data_replace_on_change = true
}




# Elastic IP (optional)
resource "aws_eip" "ruvoninference" {
  count    = var.create_elastic_ip ? 1 : 0
  instance = var.use_gpu ? aws_instance.ruvoninference_gpu[0].id : aws_instance.ruvoninference_cpu[0].id
  domain   = "vpc"

  tags = {
    Name        = "ruvoninference-${var.environment}"
    Environment = var.environment
    Project     = "RuvonInference"
  }
}
