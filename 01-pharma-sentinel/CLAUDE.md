## PharmaSentinel
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn pharma_sentinel.api.main:app --reload`

## Key Architecture
- FastAPI service in `src/pharma_sentinel/api/`, NLP classifier in `models/`
- Pipeline layer (`pipeline/`) handles FDA FAERS data ingestion and daily processing
- CloudFormation infra in `infrastructure/`, Datadog monitoring in `monitoring/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
