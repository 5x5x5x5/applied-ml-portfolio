## RegRecord
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn reg_record.api.main:app --reload`

## Key Architecture
- FastAPI compliance service in `src/reg_record/api/`, SQLAlchemy models in `models/`
- Pseudo record + submission + compliance services in `services/`
- PL/SQL triggers/packages in `sql/`, Control-M jobs in `controlm/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
