# MoleculeGen - AI-Powered Drug Molecule Generator

A deep generative model for novel drug-like molecule design using Variational Autoencoders (VAE). Encodes SMILES molecular representations into a continuous latent space, enabling targeted generation, molecular optimization, and property-conditioned sampling for pharmaceutical lead discovery.

## Architecture

**Molecular VAE** (SMILES-based sequence-to-sequence):
- **Encoder**: Bidirectional GRU maps tokenized SMILES to latent distribution parameters (mu, logvar)
- **Decoder**: Autoregressive GRU reconstructs SMILES from latent vectors with optional property conditioning
- **Training**: ELBO loss with beta-VAE KL annealing (linear and cyclical schedules) to prevent posterior collapse
- **Latent space**: Supports interpolation (molecular morphing), random sampling, and property-targeted search

**Property Prediction** (multi-task feedforward):
- Shared trunk with property-specific heads for LogP, MW, QED, and SA prediction
- Heteroscedastic uncertainty weighting for automatic multi-task loss balancing

## Key Features

- **Random generation**: Sample z ~ N(0, I) and decode to novel drug-like molecules
- **Targeted generation**: Rejection sampling with property scoring toward desired profiles
- **Molecular optimization**: Iterative local search in latent space around a seed molecule (lead optimization)
- **Diversity filtering**: Tanimoto distance thresholds ensure structural diversity in generated sets
- **Drug-likeness filtering**: Lipinski Ro5, Veber rules, Ghose filter, lead-likeness, PAINS alerts
- **Molecular descriptors**: Full physicochemical profiling (MW, LogP, TPSA, HBD/HBA, QED, SA)
- **REST API**: FastAPI service for generation, optimization, property prediction, and SVG visualization

## Project Structure

```
10-molecule-gen/
├── src/molecule_gen/
│   ├── models/
│   │   ├── mol_vae.py              # VAE encoder/decoder/loss
│   │   └── property_predictor.py   # Multi-task property prediction
│   ├── chemistry/
│   │   ├── smiles_processor.py     # Tokenization, vocabulary, fingerprints
│   │   └── molecular_descriptors.py # Descriptors, drug-likeness, PAINS
│   ├── generation/
│   │   └── generator.py            # Generation pipeline with filtering
│   └── api/
│       └── main.py                 # FastAPI REST endpoints
├── scripts/
│   ├── train_vae.py                # Training with checkpointing
│   └── evaluate_generation.py      # Validity/uniqueness/novelty/diversity
├── tests/
│   ├── conftest.py                 # Shared fixtures and drug SMILES
│   ├── test_vae.py                 # VAE architecture tests
│   ├── test_chemistry.py           # SMILES/descriptor tests
│   └── test_generation.py          # Generation pipeline tests
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

## Usage

### Training

```bash
uv run python scripts/train_vae.py \
    --data data/molecules.csv \
    --epochs 100 \
    --batch-size 256 \
    --latent-dim 128 \
    --kl-annealing-steps 5000 \
    --checkpoint-dir checkpoints/
```

### Evaluation

```bash
uv run python scripts/evaluate_generation.py \
    --checkpoint checkpoints/mol_vae_best.pt \
    --training-data data/molecules.csv \
    --num-samples 10000 \
    --output results/evaluation.json
```

### API Server

```bash
uv run uvicorn molecule_gen.api.main:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Generate N novel molecules with property constraints |
| POST | `/optimize` | Optimize an existing molecule via latent space search |
| POST | `/predict-properties` | Full physicochemical profiling of a SMILES |
| GET | `/molecule/{smiles}/visualize` | 2D structure rendering as SVG |
| GET | `/health` | Health check |

### Example API Calls

```bash
# Generate 10 drug-like molecules
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_molecules": 10, "temperature": 1.0, "logp_max": 5.0, "mw_max": 500}'

# Predict properties of aspirin
curl -X POST http://localhost:8000/predict-properties \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'

# Optimize ibuprofen
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "num_candidates": 20}'
```

### Docker

```bash
docker build -t molecule-gen .
docker run -p 8000:8000 molecule-gen
```

## Testing

```bash
uv run pytest
uv run pytest --cov=molecule_gen
```

## Drug-Likeness Filters

| Filter | Criteria | Reference |
|--------|----------|-----------|
| Lipinski Ro5 | MW<=500, LogP<=5, HBD<=5, HBA<=10 | Lipinski et al., 1997 |
| Veber | TPSA<=140, RotBonds<=10 | Veber et al., 2002 |
| Ghose | 160<=MW<=480, -0.4<=LogP<=5.6 | Ghose et al., 1999 |
| Lead-likeness | MW<=350, LogP<=3.5, RotBonds<=7 | Teague et al., 1999 |
| PAINS | Substructure alert screen | Baell & Holloway, 2010 |

## Evaluation Metrics

Following the MOSES benchmark (Polykovskiy et al., 2020):
- **Validity**: Fraction of generated SMILES that parse to valid molecules
- **Uniqueness**: Fraction of valid molecules that are structurally distinct
- **Novelty**: Fraction of unique molecules absent from the training set
- **Internal Diversity**: Average pairwise Tanimoto distance (Morgan FP)

## Technologies

- **PyTorch**: VAE model architecture and training
- **RDKit**: Cheminformatics (SMILES parsing, descriptors, fingerprints, rendering)
- **FastAPI**: REST API for model serving
- **scikit-learn**: Auxiliary ML utilities
- **pandas/numpy**: Data processing
