# Clinical Trial EDA & Biomarker Discovery

Notebook-driven exploratory data analysis of a synthetic Phase III trial for a novel anti-inflammatory compound (RX-7281) vs placebo. Demonstrates the biostatistics workflow behind trial readouts: baseline balance checks, hypothesis testing, subgroup analysis, and ML-based biomarker discovery.

## Notebooks

| Notebook | Contents | Status |
|----------|----------|--------|
| `01_data_generation.ipynb` | Generate the synthetic trial dataset (demographics, biomarkers, outcomes, adverse events) | Complete |
| `02_exploratory_analysis.ipynb` | Demographics, biomarker distributions, correlations, missing data patterns | Complete |
| `03_statistical_testing.ipynb` | Chi-squared, Mann-Whitney U, response rate comparison, baseline balance table | In progress |
| `04_biomarker_discovery.ipynb` | Predictive biomarker identification with scikit-learn + SHAP, survival curves with lifelines | In progress |

The statistical and biomarker logic these notebooks exercise already lives in `src/clinical_eda/stats.py` and is covered by tests; notebooks 03–04 wire it into the narrative.

## Supporting Package

Reusable logic lives in `src/clinical_eda/` so notebooks stay readable and the logic stays testable:

- `data_generator.py` - synthetic Phase III dataset (default 1,200 patients, seeded for reproducibility) with inflammatory biomarker panel: CRP, IL-6, TNF-alpha, ESR
- `stats.py` - chi-squared test, Mann-Whitney U test, response rate comparison with confidence intervals, subgroup analysis, baseline balance table
- `visualization.py` - demographics grid, biomarker distributions, correlation heatmap, response rates, missing data map, subgroup forest plot

## Setup

```bash
uv sync --extra dev
uv run jupyter lab notebooks/
```

## Testing

```bash
uv run pytest tests/
```

## Key Dependencies

- pandas + numpy + scipy (analysis and statistics)
- scikit-learn + shap (biomarker discovery and interpretability)
- lifelines (survival analysis)
- matplotlib + seaborn (visualization)
- jupyter (notebooks)
