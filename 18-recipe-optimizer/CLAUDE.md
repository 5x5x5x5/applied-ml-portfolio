## NutriOptimize
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn nutri_optimize.api.main:app --reload`

## Key Architecture
- SciPy-based recipe optimizer in `src/nutri_optimize/optimizer/`
- Nutrition database + taste model in `knowledge/`, ML predictor in `models/`
- FastAPI service in `api/`, frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
