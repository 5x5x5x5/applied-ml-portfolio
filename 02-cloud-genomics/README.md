# CloudGenomics - Genomic Variant Classification Service

ML-powered classification of genetic variants (SNPs, indels) into ACMG/AMP categories (**benign**, **likely benign**, **VUS**, **likely pathogenic**, **pathogenic**) deployed on AWS with HIPAA-compliant security and DataDog monitoring.

## Architecture Overview

```
                         +------------------+
                         |   API Gateway    |
                         |   + WAF v2       |
                         +--------+---------+
                                  |
                         +--------v---------+
                         |  Internal ALB    |
                         |  (Multi-AZ)      |
                         +--------+---------+
                                  |
                   +--------------+--------------+
                   |                             |
            +------v------+              +-------v-----+
            | ECS Fargate |              | ECS Fargate |
            | (AZ-1)      |              | (AZ-2)      |
            +------+------+              +-------+-----+
                   |                             |
         +---------+---------+-----------+-------+
         |                   |                   |
   +-----v-----+    +-------v-------+    +------v------+
   | RDS PG    |    | S3 (VCF +     |    | KMS CMK     |
   | Multi-AZ  |    | Results)      |    | Encryption  |
   +-----------+    +-------+-------+    +-------------+
                            |
                    +-------v-------+
                    | Step Functions|
                    | Pipeline      |
                    +---------------+
```

**Key design decisions:**
- **Private subnets only** -- no public internet access; all AWS service access via VPC endpoints
- **Envelope encryption** -- KMS-managed keys with AES-256-GCM for genomic data at rest
- **Field-level encryption** -- PHI fields encrypted individually within classification records
- **HIPAA audit trail** -- every encrypt/decrypt/access operation is logged

## Components

| Component | Purpose |
|-----------|---------|
| `src/cloud_genomics/models/variant_classifier.py` | Random Forest classifier with feature engineering, training, prediction, and explanation |
| `src/cloud_genomics/pipeline/vcf_processor.py` | VCF parser, quality filter, variant annotator, feature extractor |
| `src/cloud_genomics/pipeline/step_functions.py` | AWS Step Functions state machine (Python-generated ASL/JSON) |
| `src/cloud_genomics/api/main.py` | FastAPI REST API with rate limiting and DataDog tracing |
| `src/cloud_genomics/security/encryption.py` | KMS envelope encryption, field-level encryption, audit logging |
| `src/cloud_genomics/monitoring/metrics.py` | DataDog custom metrics and dashboard definition |
| `infrastructure/cloudformation.yaml` | Full HA AWS infrastructure (VPC, ECS, RDS, S3, WAF, Step Functions) |
| `Dockerfile` | Multi-stage, security-hardened container (non-root, read-only FS) |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- AWS CLI (for deployment)

### Local Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Start local stack (app + PostgreSQL + LocalStack)
docker compose up -d

# API is available at http://localhost:8000
# Health check: http://localhost:8000/health
```

### API Usage

**Classify a single variant:**

```bash
curl -X POST http://localhost:8000/classify-variant \
  -H "Content-Type: application/json" \
  -d '{
    "chrom": "chr17",
    "pos": 7675088,
    "ref": "G",
    "alt": "A",
    "sift_score": 0.01,
    "polyphen2_score": 0.99,
    "cadd_phred": 35.0,
    "revel_score": 0.9,
    "gnomad_af": 0.0,
    "consequence": "missense"
  }'
```

**Upload a VCF file:**

```bash
curl -X POST http://localhost:8000/upload-vcf \
  -F "file=@sample.vcf"
```

**Retrieve a classified variant:**

```bash
curl http://localhost:8000/variant/{variant_id}
```

### AWS Deployment

```bash
# Deploy infrastructure
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name cloudgenomics-production \
  --parameter-overrides \
    Environment=production \
    ECRRepositoryUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudgenomics \
    DBMasterPassword=<secure-password> \
    AlertEmailAddress=ops@example.com \
  --capabilities CAPABILITY_NAMED_IAM
```

## ML Model

The variant classifier uses a **Random Forest** ensemble with 500 trees trained on 25 engineered features spanning:

- **Conservation**: PhyloP, PhastCons, GERP++ scores
- **Population frequency**: gnomAD global and population-specific allele frequencies
- **In-silico predictions**: SIFT, PolyPhen-2, CADD, REVEL, MutationTaster
- **Protein structure**: domain membership, active site proximity, Pfam domains
- **Variant characteristics**: consequence type, amino acid change scores, splice impact

Feature engineering includes log-transformed allele frequencies, composite pathogenicity scores, conservation-pathogenicity interaction terms, and domain impact scores. Probability calibration via isotonic regression ensures well-calibrated confidence estimates.

Each prediction includes a human-readable **clinical explanation** citing relevant ACMG/AMP evidence criteria (PVS1, PM1, PM2, PP3, BS1, BA1, BP4).

## Security

- **Encryption at rest**: S3 SSE-KMS, RDS storage encryption, ECS tmpfs
- **Encryption in transit**: TLS everywhere, VPC endpoints for AWS services
- **Field-level encryption**: PHI fields (patient ID, MRN, DOB) encrypted individually
- **KMS CMK**: Customer-managed key with automatic annual rotation
- **IAM**: Least-privilege roles for every service component
- **WAF**: SQL injection, known bad inputs, and IP rate limiting rules
- **Network**: Private subnets only, no internet egress, no public IPs
- **Container hardening**: Non-root user, read-only filesystem, dropped capabilities
- **Audit logging**: Every data access event logged with principal, resource, classification

## Monitoring

DataDog integration provides:

- Classification latency (p50/p95/p99)
- Variant class distribution over time
- Model confidence histograms
- VCF processing throughput (variants/second)
- API request latency by endpoint
- Error rates by type
- Infrastructure resource utilization (CPU, memory)

## Testing

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/cloud_genomics --cov-report=term-missing

# Specific test module
uv run pytest tests/test_variant_classifier.py -v

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## License

MIT
