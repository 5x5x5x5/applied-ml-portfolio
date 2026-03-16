## PlantPathologist
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run API: `uv run uvicorn plant_pathologist.api.main:app --reload`
- Run Streamlit app: `uv run streamlit run app.py`

## Key Architecture
- EfficientNet disease classifier in `src/plant_pathologist/models/`
- Leaf image preprocessing in `preprocessing/`, disease knowledge base in `knowledge/`
- FastAPI endpoint in `api/`, Streamlit mobile-friendly frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
