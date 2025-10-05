terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }

  # Optional: Configure remote state
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "lambda/ruvoninference/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RuvonInference"
      Environment = var.environment
      Deployment  = "Lambda"
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "function_name" {
  description = "Lambda function name"
  type        = string
  default     = "ruvoninference-demo"
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 900  # 15 minutes - max for Lambda
}

variable "lambda_memory" {
  description = "Lambda memory in MB"
  type        = number
  default     = 3008  # Near maximum for better CPU performance
}

variable "deployment_package" {
  description = "Path to deployment package"
  type        = string
  default     = "./deployment-package.zip"
}

# IAM role for Lambda execution
resource "aws_iam_role" "lambda_role" {
  name = "${var.function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Basic execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda function
resource "aws_lambda_function" "ruvoninference" {
  filename         = var.deployment_package
  function_name    = var.function_name
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.12"
  timeout         = var.lambda_timeout
  memory_size     = var.lambda_memory

  # Important: Use source_code_hash for deployments
  source_code_hash = filebase64sha256(var.deployment_package)

  environment {
    variables = {
      MODEL_NAME = "gpt2"
      LOG_LEVEL  = "INFO"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic,
  ]
}

# API Gateway for HTTP access
resource "aws_api_gateway_rest_api" "ruvoninference_api" {
  name        = "${var.function_name}-api"
  description = "RuvonInference Lambda API"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# API Gateway resource for completions
resource "aws_api_gateway_resource" "completions" {
  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id
  parent_id   = aws_api_gateway_rest_api.ruvoninference_api.root_resource_id
  path_part   = "completions"
}

# POST method for completions
resource "aws_api_gateway_method" "completions_post" {
  rest_api_id   = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id   = aws_api_gateway_resource.completions.id
  http_method   = "POST"
  authorization = "NONE"
}

# OPTIONS method for CORS
resource "aws_api_gateway_method" "completions_options" {
  rest_api_id   = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id   = aws_api_gateway_resource.completions.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

# Lambda integration for POST
resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id = aws_api_gateway_resource.completions.id
  http_method = aws_api_gateway_method.completions_post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.ruvoninference.invoke_arn
}

# CORS integration for OPTIONS
resource "aws_api_gateway_integration" "cors_integration" {
  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id = aws_api_gateway_resource.completions.id
  http_method = aws_api_gateway_method.completions_options.http_method

  type = "MOCK"
  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# CORS response for OPTIONS
resource "aws_api_gateway_method_response" "cors_response" {
  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id = aws_api_gateway_resource.completions.id
  http_method = aws_api_gateway_method.completions_options.http_method
  status_code = "200"

  response_headers = {
    "Access-Control-Allow-Headers" = true
    "Access-Control-Allow-Methods" = true
    "Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "cors_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id
  resource_id = aws_api_gateway_resource.completions.id
  http_method = aws_api_gateway_method.completions_options.http_method
  status_code = aws_api_gateway_method_response.cors_response.status_code

  response_headers = {
    "Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT'"
    "Access-Control-Allow-Origin"  = "'*'"
  }
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ruvoninference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.ruvoninference_api.execution_arn}/*/*"
}

# API Gateway deployment
resource "aws_api_gateway_deployment" "ruvoninference_deployment" {
  depends_on = [
    aws_api_gateway_integration.lambda_integration,
    aws_api_gateway_integration.cors_integration,
  ]

  rest_api_id = aws_api_gateway_rest_api.ruvoninference_api.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.completions.id,
      aws_api_gateway_method.completions_post.id,
      aws_api_gateway_method.completions_options.id,
      aws_api_gateway_integration.lambda_integration.id,
      aws_api_gateway_integration.cors_integration.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

# API Gateway stage
resource "aws_api_gateway_stage" "ruvoninference_stage" {
  deployment_id = aws_api_gateway_deployment.ruvoninference_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.ruvoninference_api.id
  stage_name    = var.environment

  xray_tracing_enabled = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      responseTime   = "$context.responseTime"
      error          = "$context.error.message"
    })
  }
}

# CloudWatch log group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${var.function_name}"
  retention_in_days = 7
}

# CloudWatch log group for Lambda
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.function_name}"
  retention_in_days = 7
}

# Outputs
output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.ruvoninference.arn
}

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.ruvoninference.function_name
}

output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = "https://${aws_api_gateway_rest_api.ruvoninference_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}"
}

output "api_endpoint" {
  description = "Complete API endpoint for completions"
  value       = "https://${aws_api_gateway_rest_api.ruvoninference_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}/completions"
}

output "test_command" {
  description = "Example curl command to test the API"
  value       = "curl -X POST https://${aws_api_gateway_rest_api.ruvoninference_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Once upon a time\", \"max_tokens\": 10}'"
}
