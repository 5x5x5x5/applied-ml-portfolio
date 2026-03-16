FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata first for layer caching
COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir .

# Copy application source
COPY src/ ./src/
COPY frontend/ ./frontend/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/agents')" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "pharma_agents.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
