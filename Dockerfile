# syntax=docker/dockerfile:1.4

# ============================================================================
# Stage 1: Builder - Install dependencies in virtual environment
# ============================================================================
FROM python:3.14-slim AS builder

# Install build dependencies if needed for compiled packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install production dependencies only
COPY requirements-prod.txt /tmp/requirements-prod.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-prod.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.14-slim AS runtime

# Build arguments for metadata
ARG BUILD_DATE
ARG VERSION=dev
ARG VCS_REF

# OCI labels for image metadata
LABEL org.opencontainers.image.title="HuggingFace Model Cache Loader" \
      org.opencontainers.image.description="Kubernetes init container for caching HF models to S3-compatible storage" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/yourusername/hf-model-downloader" \
      org.opencontainers.image.licenses="MIT"

# Security: Install security updates and ca-certificates only
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Environment variables for Python optimization and hf_transfer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Create non-root user with specific UID/GID for security
RUN groupadd -g 1000 modelcache && \
    useradd -m -u 1000 -g 1000 -s /bin/bash modelcache

# Set working directory
WORKDIR /app

# Copy application with proper ownership
COPY --chown=modelcache:modelcache cache_model.py .

# Create dedicated temp directory with proper permissions
RUN mkdir -p /home/modelcache/tmp && \
    chown -R modelcache:modelcache /home/modelcache/tmp && \
    chmod 700 /home/modelcache/tmp

# Set environment variable for default download directory
ENV DOWNLOAD_DIR=/home/modelcache/tmp

# Security: Run as non-root user
USER modelcache

# Health check for debugging
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run the script as entrypoint
ENTRYPOINT ["python", "cache_model.py"]
