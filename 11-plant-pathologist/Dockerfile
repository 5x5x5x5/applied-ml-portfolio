FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project definition and install dependencies
COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir . 2>/dev/null || \
    uv pip install --system --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache-dir \
    fastapi uvicorn[standard] pillow numpy pydantic streamlit python-multipart

# Copy application code
COPY src/ src/
COPY app.py .
COPY frontend/ frontend/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Default: run FastAPI server
CMD ["uvicorn", "plant_pathologist.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
