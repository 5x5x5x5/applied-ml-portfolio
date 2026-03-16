## BiomarkerDash
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn biomarker_dash.api.main:app --reload`

## Key Architecture
- WebSocket real-time dashboard via FastAPI in `src/biomarker_dash/api/`
- ML anomaly detection + trend analysis in `models/`, alert engine in `alerts/`
- Redis-backed biomarker store in `data/`, frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
