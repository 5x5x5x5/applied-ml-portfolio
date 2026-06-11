## Clinical Trial EDA
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`
- Run notebooks: `uv run jupyter lab notebooks/`

## Key Architecture
- Four-notebook EDA narrative in `notebooks/` (data generation → EDA → statistical testing → biomarker discovery)
- Reusable analysis logic in `src/clinical_eda/` (data_generator, stats, visualization) so notebooks stay thin and testable
- Synthetic Phase III trial data: RX-7281 (anti-inflammatory) vs placebo, seeded for reproducibility

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
