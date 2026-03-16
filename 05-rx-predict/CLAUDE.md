## RxPredict
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn rx_predict.api.main:app --reload`
- Run benchmarks: `uv run python benchmarks/latency_benchmark.py`

## Key Architecture
- Sub-100ms FastAPI prediction service in `src/rx_predict/api/`
- Redis caching layer in `cache/`, Prometheus metrics in `monitoring/`
- Performance benchmarks in `benchmarks/`, frontend dashboard in `frontend/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
