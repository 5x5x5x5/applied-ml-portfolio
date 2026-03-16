## CellVision
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy. Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`
- Run API: `uv run uvicorn cell_vision.api.main:app --reload`
- Run Streamlit app: `uv run streamlit run app.py`

## Key Architecture
- PyTorch cell classifiers (CellNet CNN + ResNet transfer) in `src/cell_vision/models/`
- GradCAM visualization in `visualization/`, image preprocessing in `preprocessing/`
- FastAPI inference endpoint in `api/`, Streamlit demo app at project root

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
