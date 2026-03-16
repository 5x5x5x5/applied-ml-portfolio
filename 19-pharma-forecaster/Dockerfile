FROM python:3.11-slim AS base

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by statsmodels and prophet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency specification first for layer caching
COPY pyproject.toml ./

# Install dependencies with uv
RUN uv pip install --system --no-cache -e ".[dev]"

# --- Production stage ---
FROM base AS production

COPY . .

# Install the package
RUN uv pip install --system --no-cache -e .

# Create non-root user
RUN groupadd -r pharma && useradd -r -g pharma -d /app pharma
RUN chown -R pharma:pharma /app
USER pharma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "pharma_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# --- Test stage ---
FROM base AS test

COPY . .
RUN uv pip install --system --no-cache -e ".[dev]"

CMD ["pytest", "tests/", "-v", "--tb=short"]
