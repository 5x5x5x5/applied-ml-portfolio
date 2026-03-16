FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY frontend/ frontend/

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "nutri_optimize.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
