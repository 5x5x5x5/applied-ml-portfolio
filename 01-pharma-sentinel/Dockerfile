# =============================================================================
# PharmaSentinel - Multi-stage Docker Build
# =============================================================================
# Stage 1: Build dependencies with uv
# Stage 2: Slim runtime image with non-root user
# =============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency specification first (layer caching)
COPY pyproject.toml ./

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# Copy source code
COPY src/ ./src/

# Install the package itself
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps .

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: Don't run as root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

# Install runtime system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up environment
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Application configuration defaults
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=2 \
    APP_LOG_LEVEL=INFO \
    DD_SERVICE=pharma-sentinel \
    DD_ENV=production \
    DD_LOGS_INJECTION=true

WORKDIR /app

# Copy application code from builder
COPY --from=builder /build/src ./src

# Create directories for model artifacts and temp data
RUN mkdir -p /app/models /app/tmp && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check - verifies the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init process for proper signal handling
ENTRYPOINT ["tini", "--"]

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", \
     "pharma_sentinel.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log", \
     "--log-level", "info"]
