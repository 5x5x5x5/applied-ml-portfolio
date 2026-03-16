# DrugInteractionML

Drug-Drug Interaction Prediction Pipeline that predicts potential adverse interactions between drug pairs using molecular features and patient data.

## Architecture

```
Snowflake (Features) ─┐
                       ├──► Airflow DAG ──► Step Functions ──► SageMaker
Drug SMILES (RDKit) ──┘       │                                   │
                               │                                   ▼
                          MLflow Registry              Model Monitor + CloudWatch
                               │                                   │
                               ▼                                   ▼
                          Model Comparison              Drift Detection + Alerting
```

### Components

- **Feature Extraction**: Molecular descriptors (RDKit) and patient-level features (Snowflake)
- **Model**: XGBoost classifier predicting interaction severity and type
- **Training Pipeline**: Airflow DAG orchestrating weekly retraining
- **Deployment**: AWS Step Functions state machine with SageMaker endpoints, A/B testing, and auto-scaling
- **Monitoring**: Drift detection (PSI, KS test), performance tracking, SNS alerting, and automatic retraining triggers

## Project Structure

```
04-drug-interaction-ml/
├── pyproject.toml
├── src/drug_interaction/
│   ├── features/
│   │   ├── molecular_features.py    # RDKit molecular descriptors + Morgan fingerprints
│   │   └── snowflake_features.py    # Patient features via Snowflake SQL
│   ├── models/
│   │   ├── interaction_predictor.py # XGBoost model + SHAP explanations
│   │   └── model_registry.py       # MLflow model versioning + promotion
│   ├── deployment/
│   │   ├── step_functions.py        # AWS Step Functions state machine
│   │   └── sagemaker_deploy.py      # SageMaker endpoints + A/B testing
│   └── monitoring/
│       ├── drift_detector.py        # PSI, KS test, label drift
│       └── alerting.py              # SNS, CloudWatch, retraining triggers
├── dags/
│   ├── training_pipeline_dag.py     # Weekly training Airflow DAG
│   └── monitoring_dag.py            # Daily monitoring Airflow DAG
├── step_functions/
│   └── training_workflow.json       # Step Functions state machine definition
├── sql/
│   ├── feature_extraction.sql       # Snowflake feature extraction queries
│   └── drift_analysis.sql           # Snowflake drift analysis queries
└── tests/
    ├── conftest.py
    ├── test_features.py
    ├── test_model.py
    └── test_deployment.py
```

## Setup

```bash
cd 04-drug-interaction-ml
uv sync
uv sync --extra dev   # for development dependencies
```

## Running Tests

```bash
uv run pytest
```

## Configuration

Set the following environment variables or Airflow Variables:

| Variable | Description |
|----------|-------------|
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier |
| `SNOWFLAKE_USER` | Snowflake username |
| `SNOWFLAKE_PASSWORD` | Snowflake password |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL |
| `AWS_DEFAULT_REGION` | AWS region for SageMaker/Step Functions |

## Pipeline Schedules

- **Training Pipeline**: Weekly (Sunday 02:00 UTC) via `drug_interaction_training_pipeline` DAG
- **Monitoring Pipeline**: Daily (06:00 UTC) via `drug_interaction_monitoring` DAG

## Drift Detection

The system monitors three types of drift:

1. **Feature Drift**: Population Stability Index (PSI) and Kolmogorov-Smirnov test on each feature
2. **Prediction Drift**: Distribution shift in model output probabilities
3. **Label Drift**: Chi-squared test on ground-truth label distributions

When drift exceeds configured thresholds, the system:
- Publishes CloudWatch custom metrics
- Sends SNS notifications
- Optionally triggers automatic retraining via Step Functions
