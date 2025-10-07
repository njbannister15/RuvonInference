variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.xlarge"

  validation {
    condition = contains([
      # CPU instances
      "t3.large",    # 2 vCPU, 8GB - medium CPU
      # GPU instances
      "g5.xlarge",   # 1x A10G, 4 vCPU, 16GB
      "g5.2xlarge",  # 1x A10G, 8 vCPU, 32GB
      "g5.4xlarge",  # 1x A10G, 16 vCPU, 64GB
    ], var.instance_type)
    error_message = "Instance type must be a supported CPU or GPU instance."
  }
}


# Large models + Docker layers + HF cache are multi-GB. 200 GB gp3 avoids
# “disk full” during first model pull and future upgrades.
variable "root_volume_size" {
  type = number
  default = 200
  }

# root_iops (e.g., 3000) / root_throughput (e.g., 250)
# Higher baseline IO improves first-time model downloads/unpacks and container pulls,
# and reduces cold-start time after scaling or replacement.
variable "root_iops"       {
  type = number
  default = 3000
}
variable "root_throughput" {
  type = number
  default = 250
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # SECURITY: Change this to your IP in terraform.tfvars!
}

variable "allowed_api_cidrs" {
  description = "CIDR blocks allowed for API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # SECURITY: Change this to your IP in terraform.tfvars!
}

variable "create_key_pair" {
  description = "Whether to create a new key pair"
  type        = bool
  default     = false
}

variable "public_key" {
  description = "Public key content (required if create_key_pair is true)"
  type        = string
  default     = ""
}

variable "existing_key_name" {
  description = "Name of existing key pair (used if create_key_pair is false)"
  type        = string
  default     = "RuvonInference"
}

variable "create_elastic_ip" {
  description = "Whether to create and associate an Elastic IP"
  type        = bool
  default     = true
}

variable "use_spot_instance" {
  description = "Whether to use spot instance pricing"
  type        = bool
  default     = true
}

variable "use_gpu" {
  description = "Whether to use a GPU instance"
  type        = bool
  default     = true
}
