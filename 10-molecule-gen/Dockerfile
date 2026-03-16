FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv pip install --system --no-cache .

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints /app/results

EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "molecule_gen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
