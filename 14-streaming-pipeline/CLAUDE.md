## StreamRx
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn stream_rx.api.main:app --reload`

## Key Architecture
- Kafka producers in `src/stream_rx/producers/`, Faust stream consumers in `consumers/`
- Kinesis adapter in `kinesis/` for AWS integration, Redis-backed storage
- CloudFormation infra in `infrastructure/`, CLI entrypoints in pyproject.toml scripts

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
