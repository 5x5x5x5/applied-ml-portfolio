## WildEye
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn wild_eye.api.main:app --reload`

## Key Architecture
- MobileNetV3 species classifier + ONNX export in `src/wild_eye/models/`
- Camera trap preprocessing + EXIF extraction in `preprocessing/`, S3 pipeline in `pipeline/`
- Biodiversity analytics in `analytics/`, CloudFormation infra in `infrastructure/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
