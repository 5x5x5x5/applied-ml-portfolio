## PharmaForecast
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy. Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn pharma_forecast.api.main:app --reload`

## Key Architecture
- Ensemble forecaster combining ARIMA, Prophet, and ML models in `src/pharma_forecast/models/`
- Pharma use cases (demand, adverse events) in `use_cases/`
- Airflow DAG in `dags/`, CloudFormation in `infrastructure/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
