## PharmaDataVault
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`

## Key Architecture
- Data Vault 2.0 ETL framework in `src/pharma_vault/etl/`, quality checks in `quality/`
- PL/SQL: DDL in `sql/ddl/`, stored procedures in `sql/etl/`, queries in `sql/queries/`
- Control-M job definitions in `controlm/`, UNIX shell scripts in `scripts/unix/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
