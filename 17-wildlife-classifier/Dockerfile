# WildEye - Wildlife Species Classifier
# Multi-stage build for optimized production image.
#
# Stage 1: Build dependencies with uv
# Stage 2: Production image with FastAPI + model

# ---------------------------------------------------------------------------
# Stage 1: Dependency installation
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency spec first for layer caching.
COPY pyproject.toml ./

# Install production dependencies into a virtual environment.
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache .

# Copy application source.
COPY src/ ./src/
COPY app.py ./

# ---------------------------------------------------------------------------
# Stage 2: Production runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Security: run as non-root user.
RUN groupadd --gid 1000 wildeye && \
    useradd --uid 1000 --gid wildeye --shell /bin/bash --create-home wildeye

# Install minimal runtime dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/app.py ./

# Create directory for model artifacts.
RUN mkdir -p /app/models && chown -R wildeye:wildeye /app

# Activate virtual environment via PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    WILDEYE_MODEL_PATH="/app/models/wildeye.onnx" \
    WILDEYE_ENV="production"

USER wildeye

# Health check for container orchestration.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FastAPI on port 8000, Streamlit on port 8501.
EXPOSE 8000 8501

# Default: run FastAPI API server.
# Override with: docker run wildeye streamlit run app.py
CMD ["uvicorn", "wild_eye.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
