## ModelLab
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn model_lab.api.main:app --reload`

## Key Architecture
- Bayesian statistical engine + metrics calculator in `src/model_lab/analysis/`
- Experiment management + traffic routing in `experiments/`
- FastAPI service in `api/`, model registry in `models/`, frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
