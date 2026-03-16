## PharmaFlow
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`

## Key Architecture
- Informatica PowerCenter-style ETL framework in `src/pharma_flow/framework/`
- Concrete pipelines (adverse event, clinical trial, drug master) in `pipelines/`
- PL/SQL in `sql/`, Control-M XML job definitions in `controlm/`, scripts in `scripts/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
