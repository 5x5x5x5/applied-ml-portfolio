# ProteinExplorer

Interactive Protein Structure Analysis Tool combining bioinformatics, machine learning, and full-stack development for protein sequence analysis and structure prediction visualization.

## Features

### Sequence Analysis
- **Amino acid composition** with percentage breakdown
- **Molecular weight** calculation (average and monoisotopic masses)
- **Isoelectric point (pI)** estimation via Henderson-Hasselbalch bisection
- **Hydrophobicity profile** using the Kyte-Doolittle scale
- **Secondary structure prediction** via the Chou-Fasman method
- **Signal peptide detection** using Von Heijne's method
- **Disulfide bond prediction** based on cysteine pairing heuristics
- **GRAVY score**, aromaticity, instability index, net charge at pH 7

### Structure Feature Prediction
- **Contact map prediction** using statistical potentials
- **Solvent accessibility** prediction (buried/exposed/intermediate)
- **Intrinsic disorder** prediction (IUPred-like approach)
- **Domain boundary** detection via multi-signal analysis

### Pairwise Alignment
- **Needleman-Wunsch** global alignment with affine gap penalties (Gotoh modification)
- **BLOSUM62** scoring matrix (full 20x20 from Henikoff & Henikoff 1992)
- **Gap penalty optimization** via grid search
- Color-coded alignment viewer with identity/similarity statistics

### Interactive Dashboard
- Sequence input with FASTA parsing and validation
- SVG-based hydrophobicity plot with hydrophobic/hydrophilic shading
- Amino acid composition pie chart with hover effects
- Secondary structure visualization with helix/sheet/coil bars
- Color-coded residue display (hydrophobic, polar, charged, aromatic, special)
- Pairwise alignment viewer with match highlighting

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn protein_explorer.api.main:app --reload

# Open http://localhost:8000 in your browser
```

## Docker

```bash
docker build -t protein-explorer .
docker run -p 8000:8000 protein-explorer
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Full sequence analysis |
| POST | `/align` | Pairwise Needleman-Wunsch alignment |
| POST | `/predict-structure` | Structure feature prediction |
| GET | `/protein/{uniprot_id}` | Fetch protein by UniProt ID (mock data) |
| GET | `/health` | Health check |

### Example: Analyze a Sequence

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"}'
```

## Testing

```bash
uv run pytest tests/ -v
```

## Project Structure

```
12-protein-explorer/
├── pyproject.toml
├── Dockerfile
├── src/protein_explorer/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── sequence_analyzer.py   # Sequence properties and predictions
│   │   ├── structure_predictor.py # Structure feature prediction
│   │   └── alignment.py          # Needleman-Wunsch alignment
│   └── api/
│       └── main.py               # FastAPI application
├── frontend/
│   ├── index.html                # Dashboard UI
│   ├── styles.css                # Scientific dark theme
│   └── protein.js                # Interactive charts and API client
└── tests/
    ├── conftest.py               # Shared fixtures (real protein sequences)
    ├── test_sequence_analyzer.py # Sequence analysis tests
    └── test_alignment.py         # Alignment algorithm tests
```

## Biochemical References

- **BLOSUM62**: Henikoff & Henikoff (1992) PNAS 89:10915-10919
- **Kyte-Doolittle**: Kyte & Doolittle (1982) J Mol Biol 157:105-132
- **Chou-Fasman**: Chou & Fasman (1978) Adv Enzymol 47:45-148
- **Needleman-Wunsch**: Needleman & Wunsch (1970) J Mol Biol 48:443-453
- **Affine gaps (Gotoh)**: Gotoh (1982) J Mol Biol 162:705-708
- **pKa values**: Lehninger Principles of Biochemistry
- **Max ASA**: Tien et al. (2013) PLOS ONE 8:e80635
- **Instability index**: Guruprasad et al. (1990) Protein Eng 4:155-161

## License

MIT
