# WildEye - Wildlife Species Classifier

A conservation biology toolkit that combines deep learning with ecological analytics to identify wildlife species from camera trap imagery and compute biodiversity metrics for ecosystem monitoring.

## Overview

WildEye processes camera trap images through a MobileNetV3-based multi-label classifier to identify 22+ North American wildlife species, then feeds detection data into ecological analytics pipelines for biodiversity assessment. Designed for deployment on AWS with Docker, it supports both cloud-scale processing and edge inference via ONNX export.

### Supported Species

| Mammals | Birds |
|---------|-------|
| White-tailed Deer (*Odocoileus virginianus*) | Wild Turkey (*Meleagris gallopavo*) |
| Mule Deer (*Odocoileus hemionus*) | Bald Eagle (*Haliaeetus leucocephalus*) |
| Elk (*Cervus canadensis*) | Great Horned Owl (*Bubo virginianus*) |
| Moose (*Alces alces*) | |
| American Black Bear (*Ursus americanus*) | |
| Grizzly Bear (*Ursus arctos horribilis*) | |
| Gray Wolf (*Canis lupus*) | |
| Coyote (*Canis latrans*) | |
| Red Fox (*Vulpes vulpes*) | |
| Bobcat (*Lynx rufus*) | |
| Mountain Lion (*Puma concolor*) | |
| Raccoon (*Procyon lotor*) | |
| Striped Skunk (*Mephitis mephitis*) | |
| Pronghorn (*Antilocapra americana*) | |
| American Beaver (*Castor canadensis*) | |
| River Otter (*Lontra canadensis*) | |
| Snowshoe Hare (*Lepus americanus*) | |
| Bighorn Sheep (*Ovis canadensis*) | |
| Wolverine (*Gulo gulo*) | |

Plus `empty` (no animal, vegetation trigger) and `human` (filter from wildlife data) classes.

## Architecture

```
Camera Trap SD Card
        |
        v
  Image Preprocessing          (IR normalization, blur filtering, animal cropping)
        |
        v
  MobileNetV3 Classifier       (Multi-label, temperature-scaled confidence)
        |
        v
  Ecological Analytics          (Shannon index, occupancy, activity patterns)
        |
        v
  AWS Pipeline                  (S3 storage, Lambda classification, DynamoDB results)
        |
        v
  Streamlit Dashboard           (Species ID, biodiversity metrics, sighting maps)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
cd 17-wildlife-classifier
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run the API Server

```bash
uvicorn wild_eye.api.main:app --reload --port 8000
```

API documentation available at `http://localhost:8000/docs`

### Run the Dashboard

```bash
streamlit run app.py
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"  # Skip slow tests (ONNX export)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classify` | Upload and classify a camera trap image |
| `GET` | `/species` | List all detectable species with metadata |
| `GET` | `/analytics/{camera_id}` | Biodiversity metrics for a camera station |
| `GET` | `/sightings` | Recent sightings with geospatial data |
| `GET` | `/co-occurrence` | Species co-occurrence analysis |
| `GET` | `/health` | Health check |

## Biodiversity Metrics

- **Species Richness (S)**: Count of unique species detected
- **Shannon-Wiener Index (H')**: Information-theoretic diversity measure; H' = -sum(p_i * ln(p_i))
- **Simpson's Index (1-D)**: Probability two random individuals are different species
- **Pielou's Evenness (J')**: How evenly individuals are distributed; J' = H' / ln(S)
- **Naive Occupancy**: Proportion of sites where a species was detected
- **Relative Abundance Index (RAI)**: Detections per 100 trap-nights
- **Diel Activity Patterns**: Time-of-day detection distributions (diurnal/nocturnal/crepuscular)
- **Species Co-occurrence**: Positive/negative interspecific associations

## Docker Deployment

```bash
# Build
docker build -t wildeye .

# Run API server
docker run -p 8000:8000 wildeye

# Run Streamlit dashboard
docker run -p 8501:8501 wildeye streamlit run app.py --server.port 8501
```

## AWS Deployment

Deploy the CloudFormation stack for serverless classification:

```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name wildeye-production \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides EnvironmentName=production
```

This provisions:
- **S3 bucket** with lifecycle rules (Standard -> Intelligent-Tiering -> Glacier)
- **Lambda function** triggered on image upload for automatic classification
- **DynamoDB table** for classification results with GSI for species queries
- **API Gateway** for external REST API access

## Project Structure

```
17-wildlife-classifier/
├── pyproject.toml                          # Dependencies and build config
├── Dockerfile                              # Multi-stage production image
├── app.py                                  # Streamlit dashboard
├── infrastructure/
│   └── cloudformation.yaml                 # AWS infrastructure as code
├── src/wild_eye/
│   ├── __init__.py                         # Species labels, app metadata
│   ├── models/
│   │   └── species_classifier.py           # MobileNetV3 classifier + ONNX export
│   ├── preprocessing/
│   │   └── camera_trap_processor.py        # IR handling, blur detection, cropping
│   ├── analytics/
│   │   └── biodiversity_metrics.py         # Shannon, Simpson, occupancy, trends
│   ├── pipeline/
│   │   └── s3_processor.py                 # S3 upload, DynamoDB, Lambda handler
│   └── api/
│       └── main.py                         # FastAPI REST endpoints
└── tests/
    ├── conftest.py                         # Shared fixtures
    ├── test_classifier.py                  # Model and inference tests
    └── test_analytics.py                   # Biodiversity metrics tests
```

## References

- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
- MacKenzie, D.I. et al. (2002). Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One. Ecology.
- Norouzzadeh, M.S. et al. (2018). Automatically Identifying, Counting, and Describing Wild Animals in Camera-Trap Images with Deep Learning. PNAS.
- Beery, S. et al. (2018). Recognition in Terra Incognita. ECCV.
- Tabak, M.A. et al. (2019). Machine Learning to Classify Animal Species in Camera Trap Images. Methods in Ecology and Evolution.

## License

MIT
