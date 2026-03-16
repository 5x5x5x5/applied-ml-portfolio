# PharmAssistAI - Multi-stage Docker build
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir -r pyproject.toml

# Stage 2: Runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Create directories for persistent data
RUN mkdir -p /app/data/chromadb /app/.cache/embeddings

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN groupadd -r pharmassist && \
    useradd -r -g pharmassist -d /app -s /sbin/nologin pharmassist && \
    chown -R pharmassist:pharmassist /app
USER pharmassist

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "pharm_assist.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
