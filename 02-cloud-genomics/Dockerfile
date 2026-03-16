# ===========================================================================
# CloudGenomics - Multi-stage production Dockerfile
# Security-hardened: non-root user, read-only filesystem, minimal attack surface
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir ".[dev]" && \
    uv pip install --no-cache-dir gunicorn

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Security: metadata labels
LABEL maintainer="CloudGenomics Team" \
      description="Genomic Variant Classification Service" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/cloudgenomics/cloud-genomics"

# Install only runtime dependencies (libpq for psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 tini && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y --auto-remove

# Security: create non-root user and group
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY --chown=appuser:appgroup src/ /app/src/

# Create required directories with proper ownership
RUN mkdir -p /app/tmp /app/models /app/logs && \
    chown -R appuser:appgroup /app

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # DataDog APM configuration
    DD_SERVICE="cloud-genomics" \
    DD_ENV="production" \
    DD_VERSION="1.0.0" \
    DD_TRACE_ENABLED="true" \
    DD_LOGS_INJECTION="true" \
    # Application configuration
    APP_PORT=8000 \
    APP_WORKERS=4 \
    LOG_LEVEL=info \
    # Temp directory for read-only filesystem
    TMPDIR="/app/tmp"

WORKDIR /app

# Security: switch to non-root user
USER appuser

# Security: drop capabilities (done at container runtime, documented here)
# Run with: --cap-drop=ALL --read-only --tmpfs /app/tmp:rw,noexec,nosuid,size=100m

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:${APP_PORT}/health'); assert r.status_code == 200"

# Expose port
EXPOSE ${APP_PORT}

# Use tini as init process (handles signals properly for containers)
ENTRYPOINT ["tini", "--"]

# Run with gunicorn + uvicorn workers
CMD ["sh", "-c", "gunicorn cloud_genomics.api.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${APP_WORKERS} \
    --bind 0.0.0.0:${APP_PORT} \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL}"]
