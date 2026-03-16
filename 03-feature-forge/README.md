# FeatureForge - ML Feature Store and Drift Detection System

An end-to-end ML feature engineering platform built on **Snowflake**, orchestrated with **Apache Airflow**, and integrated with **AWS SageMaker** for model monitoring and automated retraining.

## Architecture

```
                        +-------------------+
                        |   Airflow DAGs    |
                        | (Orchestration)   |
                        +--------+----------+
                                 |
              +------------------+------------------+
              |                  |                   |
    +---------v-------+  +------v--------+  +-------v--------+
    |   Structured    |  | Semi-Struct.  |  |    Drift       |
    |   Extractor     |  | Extractor     |  |    Detection   |
    +--------+--------+  +------+--------+  +-------+--------+
              \                  |                   /
               +--------+-------+-------+---------+
                        |               |
               +--------v------+  +-----v---------+
               |   Feature     |  |   SageMaker   |
               |   Store       |  |   Monitor     |
               | (Snowflake)   |  |   (AWS)       |
               +---------------+  +---------------+
```

## Components

### Feature Extractors
- **StructuredFeatureExtractor** - Extracts features from relational Snowflake tables (patient demographics, lab results, prescriptions) using parameterized SQL with window functions and multi-table joins
- **SemiStructuredFeatureExtractor** - Extracts features from JSON/XML data in Snowflake VARIANT columns using LATERAL FLATTEN and PARSE_JSON

### Feature Store
- **FeatureRegistry** - Feature catalog with metadata (name, type, source, freshness SLA, owner), versioning, and lineage tracking stored in Snowflake
- **FeatureServingLayer** - Point-in-time correct feature retrieval (prevents data leakage), batch and online serving modes, TTL-based caching

### Drift Detection
- **FeatureDriftDetector** - Statistical drift detection using PSI, KS test, and chi-squared test with severity classification
- **ModelDriftDetector** - Monitors prediction distribution shifts, accuracy degradation, concept drift, and data drift
- **SageMakerModelMonitor** - AWS SageMaker Model Monitor integration for monitoring schedules, baseline constraints, alerting, and retraining triggers

### Airflow DAGs
- **feature_forge_pipeline** - Daily: extract features, validate quality, register in store, detect drift, alert, retrain
- **drift_monitoring_hourly** - Hourly: feature drift, model drift, SageMaker monitor checks

## Project Structure

```
03-feature-forge/
  pyproject.toml
  src/feature_forge/
    __init__.py
    extractors/
      structured_extractor.py
      semi_structured_extractor.py
    feature_store/
      registry.py
      serving.py
    drift/
      feature_drift_detector.py
      model_drift_detector.py
      sagemaker_monitor.py
  dags/
    feature_pipeline_dag.py
    drift_monitoring_dag.py
  sql/
    feature_queries/
      patient_features.sql
      lab_features.sql
      semi_structured_features.sql
    schema/
      feature_store_tables.sql
  tests/
    conftest.py
    test_extractors.py
    test_drift.py
    test_feature_store.py
```

## Setup

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/
```

## Configuration

### Snowflake
Set the following Airflow Variables (or environment variables for local development):
- `snowflake_account`, `snowflake_user`, `snowflake_password`
- `snowflake_warehouse`, `snowflake_database`, `snowflake_schema`

### AWS SageMaker
- `sagemaker_endpoint` - SageMaker endpoint name
- `sagemaker_pipeline` - Retraining pipeline name
- `sagemaker_role_arn` - IAM role ARN
- `s3_monitoring_path` - S3 path for monitoring outputs
- `sns_drift_topic` - SNS topic ARN for drift alerts

### Database Setup
Run `sql/schema/feature_store_tables.sql` against your Snowflake instance to create all required tables, views, and permissions.

## Healthcare Domain Context

This project is designed for a healthcare ML use case (e.g., hospital readmission prediction). The feature extractors handle:
- Patient demographics and chronic conditions
- Lab results with rolling statistics and trend computation
- Prescription history with adherence metrics
- Clinical notes (JSON) with diagnosis and procedure extraction
- Social determinants of health from medical records

All features are designed with clinical relevance in mind and extracted using point-in-time correct joins to prevent data leakage in training datasets.
