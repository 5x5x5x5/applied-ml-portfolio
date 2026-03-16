## FeatureForge
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict, pydantic plugin). Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`

## Key Architecture
- Feature store with Snowflake backend in `src/feature_forge/feature_store/`
- Drift detection (feature + model) in `drift/`, feature extractors in `extractors/`
- Airflow DAGs in `dags/` for pipeline and drift monitoring orchestration

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
