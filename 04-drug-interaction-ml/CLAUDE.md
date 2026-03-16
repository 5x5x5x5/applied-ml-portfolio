## DrugInteractionML
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`

## Key Architecture
- XGBoost interaction predictor in `src/drug_interaction/models/`, MLflow registry
- Feature engineering from Snowflake + molecular descriptors in `features/`
- AWS deployment: SageMaker + Step Functions in `deployment/`, Airflow DAGs in `dags/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
