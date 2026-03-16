# StreamRx - Real-time Pharmaceutical Event Streaming Pipeline

A production-grade streaming architecture for real-time processing of pharmaceutical events including prescription fills, adverse event reports, and pharmacovigilance safety signal detection. Demonstrates big data patterns with both Apache Kafka and AWS Kinesis.

## Architecture

```
Pharmacy POS ──► Kafka (rx.prescriptions) ──► Prescription Processor ──► Alerts
                                                   │
                                                   ├──► Drug Interaction Check
                                                   ├──► Windowed Aggregation
                                                   └──► Anomaly Detection

FAERS Reports ──► Kafka (rx.adverse_events) ──► Signal Detector ──► Safety Signals
                                                    │
                                                    ├──► PRR / ROR Computation
                                                    ├──► Sliding Window Analysis
                                                    └──► Dead Letter Queue

                     ┌──► Kinesis Data Streams ──► Enhanced Fan-Out Consumers
All Events ──────────┤
                     └──► Kinesis Firehose ──► S3 Data Lake (Parquet)

FastAPI Dashboard ──► /metrics/throughput
                  ──► /metrics/lag
                  ──► /alerts/active
                  ──► /ws/events (WebSocket)
```

## Components

| Component | Description |
|---|---|
| `producers/prescription_producer.py` | Kafka producer simulating pharmacy POS prescription events |
| `producers/adverse_event_producer.py` | Kafka producer for FAERS-like adverse event submissions |
| `consumers/prescription_processor.py` | Faust stream processor with drug interaction checking and anomaly detection |
| `consumers/signal_detector.py` | Real-time pharmacovigilance signal detection (PRR, ROR, chi-square) |
| `kinesis/kinesis_adapter.py` | AWS Kinesis Data Streams, Firehose, Enhanced Fan-Out, DynamoDB checkpoints |
| `storage/s3_sink.py` | S3 Parquet writer with partitioning, micro-batch buffering, and compaction |
| `monitoring/stream_metrics.py` | Consumer lag, throughput, latency, error rate tracking + DataDog integration |
| `api/main.py` | FastAPI dashboard with REST metrics endpoints and WebSocket live stream |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- uv (Python package manager)

### Local Development

```bash
# Start infrastructure
docker compose up -d zookeeper kafka redis kafka-init

# Install dependencies
uv sync --dev

# Run the API dashboard
uv run stream-rx-api

# Run producers (in separate terminals)
uv run stream-rx-prescriptions --eps 100
uv run stream-rx-adverse --eps 10

# Run load test
uv run python scripts/generate_load.py --rx-eps 500 --ae-eps 50 --duration 60
```

### Full Stack with Docker

```bash
docker compose up --build
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/metrics/throughput` | GET | Current processing rates |
| `/metrics/lag` | GET | Consumer lag per partition |
| `/metrics/latency` | GET | Processing latency percentiles |
| `/metrics/errors` | GET | Error rate and breakdown |
| `/metrics/summary` | GET | Comprehensive metrics snapshot |
| `/alerts/active` | GET | Active safety signals and interaction alerts |
| `/alerts/{id}/acknowledge` | POST | Acknowledge an alert |
| `/alerts/dlq` | GET | Dead letter queue entries |
| `/ws/events` | WebSocket | Live event stream |

## AWS Deployment

The `infrastructure/cloudformation.yaml` template provisions:

- **MSK** (Managed Streaming for Apache Kafka) with 3 brokers
- **Kinesis Data Streams** for prescriptions and adverse events
- **Kinesis Data Firehose** for S3 delivery with GZIP compression
- **S3 Data Lake** with lifecycle policies (Standard -> IA -> Glacier)
- **DynamoDB** tables for checkpoints and signal state
- **ECS Fargate** tasks for producers, consumers, and API
- **CloudWatch Alarms** for consumer lag, error rate, and safety signals
- **VPC** with public/private subnets, NAT gateway, and security groups

```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name streamrx-production \
  --parameter-overrides EnvironmentName=production \
  --capabilities CAPABILITY_IAM
```

## Signal Detection

The safety signal detector computes disproportionality metrics in sliding windows:

- **PRR (Proportional Reporting Ratio)**: Compares reporting rate for a drug-reaction pair vs all other drugs
- **ROR (Reporting Odds Ratio)**: Odds ratio from a 2x2 contingency table
- **Chi-square**: Yates-corrected test for statistical significance

A signal is triggered when PRR >= 2.0 (or ROR >= 2.0) AND case count >= 3 AND chi-square >= 3.84.

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `AWS_REGION` | `us-east-1` | AWS region |
| `REDIS_HOST` | `localhost` | Redis host |
| `PRR_THRESHOLD` | `2.0` | PRR signal threshold |
| `ROR_THRESHOLD` | `2.0` | ROR signal threshold |
| `MIN_CASE_COUNT` | `3` | Minimum cases for signal |
| `LAG_CRITICAL_THRESHOLD` | `100000` | Consumer lag critical alert |
| `S3_BUFFER_SIZE_MB` | `128` | S3 write buffer size |
| `LOG_LEVEL` | `INFO` | Logging level |
