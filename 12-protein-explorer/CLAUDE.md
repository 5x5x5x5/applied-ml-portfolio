## ProteinExplorer
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn protein_explorer.api.main:app --reload`

## Key Architecture
- BioPython sequence analysis + alignment in `src/protein_explorer/analysis/`
- Structure prediction module in `analysis/structure_predictor.py`
- FastAPI service in `api/`, interactive frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
