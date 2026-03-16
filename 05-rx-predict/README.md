# RxPredict - Real-time Drug Response Prediction API

A production-grade ML API that predicts patient drug response based on genetic profile, demographics, and medical history -- with **sub-100ms inference latency**.

## Architecture

```
Client Request
      |
  [Rate Limiter] -> [Request ID Middleware] -> [Latency Tracker]
      |
  [Cache Check] -----> HIT -> Return cached result (~1ms)
      |
    MISS
      |
  [Feature Processor]  (~2ms)
      |  - Genetic variant one-hot encoding (cached)
      |  - Demographic z-score normalization
      |  - Drug feature hashing
      |  - Medical history summarization
      |
  [ML Model Inference]  (~5-15ms)
      |  - GradientBoosting (50 trees, depth 4)
      |  - Calibrated probability outputs
      |  - Confidence intervals
      |
  [Cache Store] + [Metrics Recording]
      |
  Response (<50ms p99)
```

## Latency Optimization Strategy

| Component | Target | Technique |
|---|---|---|
| Feature processing | <2ms | Pre-computed index maps, LRU caching, vectorized numpy ops |
| Model inference | <15ms | Shallow GBT (50 trees, depth 4), pre-warmed model |
| Serialization | <1ms | orjson (3-10x faster than stdlib json) |
| Caching | ~1ms | Redis with BLAKE2b cache keys, TTL expiration |
| Total request | <50ms p99 | All above + uvloop, httptools, connection pooling |

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Run the API locally
uv run uvicorn rx_predict.api.main:app --reload --host 0.0.0.0 --port 8000

# Run with Docker Compose (includes Redis, Prometheus, Grafana)
docker compose up --build
```

## API Endpoints

| Method | Path | Description | Latency Target |
|---|---|---|---|
| POST | `/predict` | Single patient prediction | <50ms p99 |
| POST | `/batch-predict` | Batch predictions (up to 100) | <200ms p95 |
| GET | `/health` | Detailed health check | -- |
| GET | `/metrics` | Prometheus metrics | -- |
| GET | `/` | API info | -- |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "genetic_profile": {
      "CYP2D6": ["*1", "*2"],
      "CYP2C19": ["*1"]
    },
    "metabolizer_phenotype": "normal",
    "demographics": {
      "age": 45,
      "weight_kg": 75.0,
      "height_cm": 170.0,
      "sex": "male"
    },
    "drug": {
      "name": "Sertraline",
      "drug_class": "ssri",
      "dosage_mg": 50.0
    }
  }'
```

### Example Response

```json
{
  "request_id": "a1b2c3d4-...",
  "response_probability": 0.7234,
  "confidence_lower": 0.6421,
  "confidence_upper": 0.7891,
  "predicted_class": "good_response",
  "risk_level": "low_risk",
  "inference_time_ms": 8.234,
  "model_version": "1.0.0",
  "cache_hit": false
}
```

## Testing

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/ -v -m "not slow and not benchmark"

# Performance benchmarks
uv run pytest tests/test_performance.py -v -m benchmark

# Latency benchmark with SLA check
uv run python benchmarks/latency_benchmark.py --iterations 1000 --fail-threshold 100
```

## CI/CD Pipeline

The GitHub Actions workflow (`ci-cd.yml`) runs:

1. **Lint** - ruff + mypy type checking
2. **Unit tests** - pytest with coverage
3. **Integration tests** - With Redis service container
4. **Performance benchmark** - Fails if p99 > 100ms
5. **Docker build** - Multi-stage, optimized image
6. **Deploy staging** - Automatic on main
7. **Smoke tests** - Verify staging health and latency
8. **Deploy production** - Manual approval required
9. **Rollback** - Automatic on failure

## Monitoring

- **Prometheus metrics** at `/metrics` - request latency histograms, throughput, error rates, cache hit rates
- **Grafana dashboards** at `http://localhost:3000` (admin/rxpredict)
- **Production issue detection** - Memory leaks, latency spikes, model staleness, circuit breaker

## Production Issue Resolution

The system includes automated detection for:

- **Memory leaks**: RSS growth tracking with auto-GC trigger
- **Latency spikes**: Auto-diagnosis (GC pressure, cache miss, sustained degradation)
- **Model staleness**: Alerts when model exceeds configured age
- **Error cascades**: Circuit breaker pattern with automatic recovery
- **Cache degradation**: Graceful fallback when Redis is unavailable

Each detected issue includes a runbook with step-by-step resolution guidance.

## Project Structure

```
05-rx-predict/
  src/rx_predict/
    api/
      main.py           # FastAPI app with endpoints
      middleware.py      # Latency tracking, caching, rate limiting
    models/
      drug_response_model.py  # ML model with confidence intervals
      feature_processor.py    # Vectorized feature extraction
    cache/
      redis_cache.py    # Redis caching with graceful degradation
    monitoring/
      performance.py    # Prometheus metrics, SLA tracking
      production_issues.py  # Issue detection, circuit breaker, runbooks
  frontend/
    index.html          # Single-page app
    styles.css          # Clinical theme with dark/light mode
    app.js              # API integration and visualization
  tests/                # pytest test suite
  benchmarks/           # Latency benchmarking
  .github/workflows/    # CI/CD pipeline
  Dockerfile            # Multi-stage optimized build
  docker-compose.yml    # Full stack with Redis, Prometheus, Grafana
```
