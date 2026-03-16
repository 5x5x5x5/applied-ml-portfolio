# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency specification
COPY pyproject.toml .

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml

# Copy application source
COPY src/ src/

# Install the application
RUN uv pip install --python /opt/venv/bin/python --no-deps .

# Pre-compile Python bytecode for faster startup
RUN /opt/venv/bin/python -m compileall -q /opt/venv/lib/ src/ 2>/dev/null || true

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r rxpredict && useradd -r -g rxpredict -d /app -s /sbin/nologin rxpredict

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ src/
COPY frontend/ frontend/

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    ENV=production

# Expose port
EXPOSE 8000

# Switch to non-root user
USER rxpredict

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with optimized uvicorn settings
CMD ["python", "-m", "uvicorn", \
     "rx_predict.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log", \
     "--timeout-keep-alive", "30"]
