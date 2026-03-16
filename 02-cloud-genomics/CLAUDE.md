## CloudGenomics
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn cloud_genomics.api.main:app --reload`

## Key Architecture
- FastAPI service in `src/cloud_genomics/api/`, variant classifier in `models/`
- Pipeline layer handles VCF processing and Step Functions orchestration
- HIPAA-compliant: encryption module in `security/`, CloudFormation infra

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
