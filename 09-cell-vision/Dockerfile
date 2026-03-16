# CellVision - Microscopy Cell Type Classifier
# Multi-stage build: install dependencies, then copy app code

FROM python:3.11-slim AS base

# System dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependency installation stage ----

FROM base AS deps

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency files first (cache layer)
COPY pyproject.toml ./

# Install dependencies (CPU-only PyTorch to keep image smaller)
RUN uv pip install --system --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache .

# ---- Application stage ----

FROM base AS app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY app.py ./
COPY pyproject.toml ./

# Create directory for model weights
RUN mkdir -p /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default: run the FastAPI server
CMD ["uvicorn", "cell_vision.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# To run Streamlit instead:
# docker run -p 8501:8501 cellvision streamlit run app.py --server.port 8501 --server.address 0.0.0.0
