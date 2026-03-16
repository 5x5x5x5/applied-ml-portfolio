## MoleculeGen
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy. Formatting: ruff format
- Tests: pytest in tests/ directory
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn molecule_gen.api.main:app --reload`

## Key Architecture
- Variational Autoencoder (VAE) for molecules in `src/molecule_gen/models/mol_vae.py`
- SMILES processing + molecular descriptors (RDKit) in `chemistry/`
- Molecule generation pipeline in `generation/`, FastAPI serving in `api/`

## Conventions
- Type hints on all public functions
- Use logging module, never print()
- Default branch: main
