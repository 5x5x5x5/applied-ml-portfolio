# ModelLab - A/B Testing Platform for ML Models

Production-grade experimentation platform for running A/B tests on machine learning models. Supports frequentist and Bayesian statistical analysis, consistent-hashing traffic routing, model registry with champion/challenger management, and comprehensive experiment monitoring.

## Architecture

```
src/model_lab/
  experiments/
    experiment_manager.py   # Experiment lifecycle, state machine, mutual exclusion
    traffic_router.py       # Consistent hashing, sticky sessions, overrides
  analysis/
    statistical_engine.py   # Frequentist + Bayesian tests, sequential testing
    metrics_calculator.py   # Conversion, latency, revenue, CUPED
  models/
    model_registry.py       # Champion/challenger, rollback, SageMaker
  monitoring/
    experiment_monitor.py   # SRM detection, guardrails, early stopping
  api/
    main.py                 # FastAPI REST endpoints
frontend/                   # Dashboard with Chart.js visualizations
tests/                      # pytest test suite
```

## Features

- **Experiment Lifecycle**: Full state machine (draft -> running -> analyzing -> completed) with mutual exclusion groups and gradual rollout (1% -> 5% -> 25% -> 100%)
- **Traffic Routing**: Consistent hashing for stable user assignment, sticky sessions, admin overrides, feature flag integration, multi-armed bandit
- **Statistical Analysis**: Two-sample t-test, chi-squared, z-test for proportions, Beta-Binomial Bayesian, Normal-Normal Bayesian, sequential testing with O'Brien-Fleming/Pocock spending functions
- **Multiple Comparison Correction**: Bonferroni and Benjamini-Hochberg FDR
- **Power Analysis**: Sample size calculation for proportions and continuous metrics
- **Metrics**: Conversion rate, latency quantiles (p50/p95/p99), model quality (accuracy/precision/recall/F1), revenue impact estimation, CUPED variance reduction
- **Monitoring**: Sample Ratio Mismatch detection, data quality checks, guardrail metrics, early stopping
- **Model Registry**: Version management, champion/challenger, automated promotion, rollback, SageMaker endpoint integration

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync --all-extras

# Run the API server
uv run uvicorn model_lab.api.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Docker

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:8000/frontend/index.html`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/experiments` | Create a new experiment |
| POST | `/experiments/{id}/start` | Start an experiment |
| POST | `/experiments/{id}/pause` | Pause a running experiment |
| POST | `/experiments/{id}/stop` | Stop and move to analysis |
| POST | `/experiments/{id}/complete` | Mark as completed |
| GET | `/experiments` | List all experiments |
| GET | `/experiments/{id}` | Get experiment details |
| GET | `/experiments/{id}/results` | Get statistical analysis |
| GET | `/experiments/{id}/health` | Run health checks |
| POST | `/route` | Route a request to a variant |
| POST | `/events` | Log experiment events |
| GET | `/power-analysis` | Calculate required sample size |
| GET | `/health` | Application health check |

## Example: Create and Run an Experiment

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Create experiment
exp = client.post("/experiments", json={
    "name": "New Recommendation Model v2",
    "hypothesis": "v2 improves click-through rate by 5%",
    "variants": [
        {"name": "Control", "traffic_percentage": 50, "is_control": True, "model_version_id": "rec-v1"},
        {"name": "Treatment", "traffic_percentage": 50, "is_control": False, "model_version_id": "rec-v2"},
    ],
    "success_metrics": [
        {"name": "ctr", "metric_type": "conversion", "minimum_detectable_effect": 0.02}
    ],
    "traffic_allocation": "random",
}).json()

# Start it
client.post(f"/experiments/{exp['id']}/start")

# Route users
decision = client.post("/route", json={
    "user_id": "user-12345",
    "experiment_id": exp["id"],
}).json()

# Log events
client.post("/events", json={
    "experiment_id": exp["id"],
    "variant_id": decision["variant_id"],
    "user_id": "user-12345",
    "event_type": "conversion",
    "value": 1.0,
})

# Check results
results = client.get(f"/experiments/{exp['id']}/results").json()
```

## Technology Stack

- **Backend**: FastAPI, Pydantic, SQLAlchemy, structlog
- **Statistics**: SciPy, NumPy, Pandas
- **Infrastructure**: PostgreSQL, Redis, Docker
- **ML Integration**: boto3 (SageMaker)
- **Frontend**: HTML/CSS/JS, Chart.js
- **CI/CD**: GitHub Actions, AWS ECR/ECS
