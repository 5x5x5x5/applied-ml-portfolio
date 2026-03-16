# BiomarkerDash - Real-time Biomarker Monitoring Dashboard

A full-stack clinical monitoring application that streams patient biomarker data in real-time, detects anomalies using ML models, and provides an interactive dashboard for healthcare professionals.

## Architecture

```
                    +------------------+
                    |   Frontend       |
                    |  (HTML/CSS/JS)   |
                    +--------+---------+
                             |
                     WebSocket / REST
                             |
                    +--------+---------+
                    |   FastAPI App    |
                    |  - REST API      |
                    |  - WebSocket     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------+--+  +-------+---+  +-------+---+
     | Anomaly   |  | Trend     |  | Alert     |
     | Detector  |  | Analyzer  |  | Engine    |
     | (sklearn) |  | (numpy)   |  | (rules+ML)|
     +-----------+  +-----------+  +-----------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------+---------+
                    | Event Processor  |
                    | (async pipeline) |
                    +--------+---------+
                             |
                    +--------+---------+
                    |     Redis        |
                    | (time-series)    |
                    +------------------+
```

## Features

- **Real-time streaming** via WebSocket with automatic reconnection
- **ML anomaly detection**: Isolation Forest (multivariate) + Z-score (univariate)
- **Patient-specific baselines** that adapt over time
- **Contextual detection**: age/sex-adjusted normal ranges
- **Trend analysis**: time-series decomposition, rate of change, 24h predictions
- **Clinical alerts**: rule-based + ML-based with deduplication and escalation
- **Interactive dashboard**: Canvas-based charts, dark mode, responsive layout
- **Backpressure handling**: bounded queue with graceful degradation
- **CI/CD pipeline**: lint, test, build, deploy stages

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Redis (or Docker)

### Local Development

```bash
# Install dependencies
uv sync --dev

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run the application
PYTHONPATH=src uv run uvicorn biomarker_dash.api.main:app --reload --port 8000

# Open dashboard
open http://localhost:8000
```

### Generate Sample Data

```bash
# Save to file
uv run python scripts/generate_sample_data.py --output data/sample.json

# Send to running API
uv run python scripts/generate_sample_data.py --send --hours 24
```

### Run Tests

```bash
uv run pytest tests/ -v --cov=biomarker_dash
```

### Lint & Type Check

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/biomarker_dash/
```

### Docker

```bash
# Build and run all services
docker compose up --build

# Access dashboard at http://localhost (via Nginx)
# API directly at http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/biomarkers` | Submit biomarker reading |
| GET | `/api/patients/{id}/history` | Historical data |
| GET | `/api/patients/{id}/anomalies` | Detected anomalies |
| GET | `/api/alerts` | Active clinical alerts |
| POST | `/api/alerts/{id}/acknowledge` | Acknowledge alert |
| POST | `/api/alerts/{id}/resolve` | Resolve alert |
| GET | `/api/patients` | List patients |
| POST | `/api/patients` | Register patient |
| GET | `/api/metrics` | Processing metrics |
| WS | `/ws/biomarkers/{patient_id}` | Real-time stream |

## ML Models

### Anomaly Detection

1. **Range-based**: Population and sex-adjusted normal ranges with severity scoring based on deviation magnitude.

2. **Z-score**: Patient-specific baseline learning. After collecting sufficient samples (10+), detects deviations from the patient's own baseline.

3. **Isolation Forest**: Multivariate anomaly detection across all biomarkers. Trains on 30+ data points and considers correlations between biomarkers.

### Trend Analysis

- Linear regression-based trend detection with R-squared significance testing
- Time-series decomposition (trend + seasonal + residual)
- 24-hour forward prediction with normal-range exit warnings
- Volatility tracking for instability detection

## Project Structure

```
06-biomarker-dash/
  src/biomarker_dash/
    api/main.py              # FastAPI application
    streaming/event_processor.py  # Real-time pipeline
    models/anomaly_detector.py    # ML anomaly detection
    models/trend_analyzer.py      # Trend analysis
    data/biomarker_store.py       # Redis storage layer
    alerts/alert_engine.py        # Clinical alert system
    schemas.py                    # Pydantic models
  frontend/
    index.html               # Dashboard HTML
    styles.css                # Clinical UI theme
    dashboard.js              # Real-time chart & WebSocket
  tests/                     # pytest test suite
  scripts/                   # Data generation utilities
  .github/workflows/         # CI/CD pipeline
```
