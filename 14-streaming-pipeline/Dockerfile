FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["stream-rx-api"]
