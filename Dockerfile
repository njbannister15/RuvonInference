# Multi-stage Dockerfile for RuvonInference deployment to AWS
# Optimized for both CPU and GPU instances

# Build stage
FROM python:3.12-slim as builder

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU by default (can be overridden for GPU)
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN uv pip install torch==2.2.2 torchvision==0.17.2 --index-url $TORCH_INDEX_URL

# Install project dependencies
RUN uv pip install .

# Production stage
FROM python:3.12-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ruvoninference && useradd -r -g ruvoninference ruvoninference

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY ruvoninference/ ./ruvoninference/
COPY commands/ ./commands/
COPY cli.py ./

# Create necessary directories including model cache
RUN mkdir -p logs test-reports /app/.cache/huggingface && \
    chown -R ruvoninference:ruvoninference /app

# Switch to non-root user
USER ruvoninference

# Environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace cache configuration
ENV HF_HOME=/app/.cache/huggingface

# Default to CPU mode, can be overridden
ENV DEVICE=cpu
ENV MODEL_NAME=gpt2
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose the port
EXPOSE 8000

# Default entrypoint - run the FastAPI server
CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
