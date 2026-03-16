# PharmaSentinel - Drug Adverse Event Detection Pipeline

An end-to-end AI/ML pipeline for processing FDA FAERS (FDA Adverse Event Reporting System) data using NLP to classify and detect adverse drug events by severity.

## Architecture Overview

```
                     +------------------+
                     |   S3 (Input)     |
                     | FAERS CSV/XML    |
                     +--------+---------+
                              |
                     +--------v---------+
                     |  Lambda Trigger   |
                     | (Daily Processor) |
                     +--------+---------+
                              |
                  +-----------v-----------+
                  |   NLP Pipeline        |
                  | - Tokenization        |
                  | - Drug Name Extraction|
                  | - Event Classification|
                  +-----------+-----------+
                              |
              +---------------+---------------+
              |                               |
     +--------v---------+           +--------v---------+
     |  S3 (Output)     |           | SQS (Critical    |
     | Classified Events|           |  Notifications)  |
     +------------------+           +------------------+
              |
     +--------v---------+
     | FastAPI Service   |
     | (ECS Fargate)     |
     | - /predict        |
     | - /batch-predict  |
     | - /health         |
     | - /metrics        |
     +--------+---------+
              |
     +--------v---------+
     | DataDog          |
     | Monitoring       |
     +------------------+
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Model | scikit-learn (TF-IDF + LogisticRegression) | Classify adverse events by severity |
| API | FastAPI on ECS Fargate | REST endpoints for real-time classification |
| Pipeline | AWS Lambda + S3 | Daily batch processing of FAERS data |
| Infrastructure | CloudFormation | VPC, ECS, ALB, S3, SQS, Lambda, KMS |
| Monitoring | DataDog + CloudWatch | APM, custom metrics, dashboards, alerts |
| CI/CD | GitHub Actions | Lint, test, build, deploy |

## Severity Levels

The classifier categorizes adverse events into four severity levels:

- **Mild** - Minor symptoms (headache, nausea, dizziness) not requiring medical intervention
- **Moderate** - Significant symptoms (vomiting, rash, fever) requiring treatment
- **Severe** - Serious events (hospitalization, seizures, organ damage) requiring intensive care
- **Critical** - Life-threatening or fatal events (death, cardiac arrest, anaphylaxis)

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker and Docker Compose
- AWS CLI (for deployment)

### Local Development

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/pharma_sentinel/

# Start local development environment
docker compose up -d

# Run the API locally (without Docker)
uv run uvicorn pharma_sentinel.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient experienced severe seizure and was hospitalized after taking the medication.",
    "drug_name": "DrugX"
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "reports": [
      {"text": "Mild headache and nausea reported by patient."},
      {"text": "Fatal cardiac arrest after drug administration."}
    ]
  }'

# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

## Deployment

### AWS Infrastructure

Deploy the full stack using CloudFormation:

```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name pharma-sentinel-production \
  --parameter-overrides \
    Environment=production \
    ContainerImage=<ECR_IMAGE_URI> \
    CertificateArn=<ACM_CERT_ARN> \
    DataDogApiKeyArn=<DD_SECRET_ARN> \
  --capabilities CAPABILITY_NAMED_IAM
```

### Infrastructure includes:

- **VPC** with public/private subnets across 2 AZs, NAT gateway, flow logs
- **ECS Fargate** with auto-scaling (CPU and request-based)
- **ALB** with HTTPS (TLS 1.3), HTTP-to-HTTPS redirect
- **S3 buckets** (input, output, models) with KMS encryption, versioning, lifecycle policies
- **SQS queues** with dead letter queues for critical event notifications
- **Lambda** for daily FAERS data processing (S3 trigger + scheduled)
- **CloudWatch alarms** for CPU, memory, 5xx errors, DLQ depth
- **KMS** key with automatic rotation
- **IAM roles** with least-privilege policies
- **Secrets Manager** for API keys and credentials

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automates:

1. **Lint** - ruff, ruff format, mypy
2. **Test** - pytest with coverage
3. **Build** - Multi-stage Docker build, push to ECR
4. **Deploy** - CloudFormation stack update, ECS service stability check, health verification

## Monitoring

DataDog integration provides:

- **APM Tracing** - Distributed traces across API requests and AWS service calls
- **Custom Metrics** - Prediction latency, confidence scores, severity distribution, pipeline throughput
- **Dashboard** - Pre-built dashboard definition in `infrastructure/datadog-dashboard.json`
- **Alerts** - Error rate thresholds, latency SLOs, pipeline failure detection

## Project Structure

```
01-pharma-sentinel/
├── pyproject.toml                    # Project config (uv, dependencies, tools)
├── Dockerfile                        # Multi-stage Docker build
├── docker-compose.yml                # Local dev environment
├── src/pharma_sentinel/
│   ├── __init__.py
│   ├── config.py                     # Pydantic Settings configuration
│   ├── models/
│   │   └── adverse_event_classifier.py  # ML classifier (TF-IDF + LogReg)
│   ├── pipeline/
│   │   ├── data_ingestion.py         # S3-based FAERS data ingestion
│   │   └── daily_processor.py        # Lambda handler for daily processing
│   ├── api/
│   │   └── main.py                   # FastAPI application
│   └── monitoring/
│       └── datadog_config.py         # DataDog metrics & APM config
├── infrastructure/
│   ├── cloudformation.yaml           # AWS infrastructure (VPC, ECS, ALB, S3, SQS, Lambda)
│   ├── datadog-dashboard.json        # DataDog dashboard definition
│   └── localstack-init.sh            # LocalStack initialization script
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── test_classifier.py            # ML model tests
│   ├── test_api.py                   # API endpoint tests
│   └── test_pipeline.py              # Data pipeline tests
└── .github/workflows/
    └── ci-cd.yml                     # GitHub Actions CI/CD pipeline
```

## Security

- All S3 buckets use KMS encryption with automatic key rotation
- HTTPS-only ALB with TLS 1.3 policy
- ECS tasks run in private subnets with no public IP
- IAM roles follow least-privilege principle
- No hardcoded secrets; all sensitive values from Secrets Manager or environment
- VPC Flow Logs enabled for network auditing
- Input validation on all API endpoints (Pydantic)
- PHI pattern detection and removal in text preprocessing

## License

MIT
