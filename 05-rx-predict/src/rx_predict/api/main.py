"""FastAPI application for real-time drug response prediction.

Optimized for sub-100ms latency with:
- orjson for fast JSON serialization
- Model warm-up on startup
- Redis caching for repeated requests
- Prometheus metrics exposure
- Structured logging with correlation IDs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import orjson
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rx_predict import __version__
from rx_predict.api.middleware import (
    LatencyTracker,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from rx_predict.cache.redis_cache import RedisCache
from rx_predict.models.drug_response_model import RESPONSE_CLASSES, DrugResponseModel
from rx_predict.monitoring.performance import ACTIVE_REQUESTS, PerformanceMonitor
from rx_predict.monitoring.production_issues import ProductionIssueDetector

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("ENV") == "dev"
        else structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# --- Pydantic Models ---


class GeneticProfile(BaseModel):
    """Patient genetic profile for pharmacogenomics."""

    CYP2D6: list[str] = Field(default_factory=list, description="CYP2D6 star alleles")
    CYP2C19: list[str] = Field(default_factory=list, description="CYP2C19 star alleles")
    CYP3A4: list[str] = Field(default_factory=list, description="CYP3A4 star alleles")
    CYP2C9: list[str] = Field(default_factory=list, description="CYP2C9 star alleles")
    VKORC1: list[str] = Field(default_factory=list, description="VKORC1 variants")
    DPYD: list[str] = Field(default_factory=list, description="DPYD star alleles")
    TPMT: list[str] = Field(default_factory=list, description="TPMT star alleles")
    UGT1A1: list[str] = Field(default_factory=list, description="UGT1A1 star alleles")
    SLCO1B1: list[str] = Field(default_factory=list, description="SLCO1B1 star alleles")
    HLA_B: list[str] = Field(default_factory=list, alias="HLA-B", description="HLA-B alleles")


class Demographics(BaseModel):
    """Patient demographics."""

    age: int = Field(ge=0, le=120, description="Patient age in years")
    weight_kg: float = Field(ge=1, le=500, description="Weight in kilograms")
    height_cm: float = Field(ge=30, le=300, description="Height in centimeters")
    bmi: float | None = Field(default=None, description="BMI (calculated if not provided)")
    sex: str = Field(default="unknown", description="Biological sex")
    ethnicity: str = Field(default="unknown", description="Self-reported ethnicity")


class DrugInfo(BaseModel):
    """Drug information for prediction."""

    name: str = Field(description="Drug name")
    drug_class: str = Field(default="other", description="Drug class (e.g., ssri, statin)")
    dosage_mg: float = Field(ge=0, description="Dosage in milligrams")
    max_dosage_mg: float = Field(default=1000.0, ge=0, description="Maximum dosage")


class MedicalHistory(BaseModel):
    """Patient medical history."""

    num_current_medications: int = Field(default=0, ge=0)
    num_allergies: int = Field(default=0, ge=0)
    num_adverse_reactions: int = Field(default=0, ge=0)
    conditions: list[str] = Field(default_factory=list)
    pregnant: bool = Field(default=False)
    age: int | None = Field(default=None)


class PredictionRequest(BaseModel):
    """Full prediction request."""

    genetic_profile: GeneticProfile = Field(default_factory=GeneticProfile)
    metabolizer_phenotype: str = Field(default="normal")
    demographics: Demographics
    drug: DrugInfo
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)

    model_config = {"populate_by_name": True}


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    patients: list[PredictionRequest] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Prediction response."""

    request_id: str
    response_probability: float
    confidence_lower: float
    confidence_upper: float
    predicted_class: str
    risk_level: str
    inference_time_ms: float
    model_version: str
    feature_importance: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    model_version: str | None
    redis_connected: bool
    uptime_seconds: float
    performance: dict[str, Any]
    issues: list[dict[str, Any]]


# --- Application State ---


class AppState:
    """Singleton holding application state."""

    def __init__(self) -> None:
        self.model: DrugResponseModel | None = None
        self.cache: RedisCache | None = None
        self.monitor: PerformanceMonitor = PerformanceMonitor()
        self.issue_detector: ProductionIssueDetector = ProductionIssueDetector()
        self.start_time: float = time.time()


app_state = AppState()


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application startup and shutdown."""
    logger.info("application_starting", version=__version__)

    # Initialize model
    start = time.perf_counter()
    model = DrugResponseModel()
    model.build_default_model()
    warmup_ms = model.warm_up()
    load_time = time.perf_counter() - start

    app_state.model = model
    app_state.monitor.record_model_load(load_time)
    app_state.issue_detector.staleness_detector.record_model_update(model.model_version)

    logger.info(
        "model_ready",
        version=model.model_version,
        load_time_s=round(load_time, 3),
        warmup_ms=round(warmup_ms, 3),
    )

    # Initialize Redis cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache = RedisCache(redis_url=redis_url)
    app_state.cache = cache

    # Run benchmark
    benchmark = model.benchmark(n_iterations=100)
    logger.info(
        "startup_benchmark",
        p50_ms=benchmark["p50_ms"],
        p95_ms=benchmark["p95_ms"],
        p99_ms=benchmark["p99_ms"],
    )

    yield

    # Shutdown
    logger.info("application_shutting_down")
    if app_state.cache:
        app_state.cache.close()


# --- FastAPI App ---

app = FastAPI(
    title="RxPredict",
    description="Real-time Drug Response Prediction API",
    version=__version__,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# Middleware (order matters - outermost first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time", "X-Cache"],
)
app.add_middleware(RateLimitMiddleware, max_requests_per_minute=600)
app.add_middleware(LatencyTracker, performance_monitor=app_state.monitor)
app.add_middleware(RequestIDMiddleware)


# --- Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest) -> ORJSONResponse:
    """Predict drug response for a single patient.

    Target: <50ms p99 latency.
    """
    ACTIVE_REQUESTS.inc()
    try:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Check circuit breaker
        if not app_state.issue_detector.circuit_breaker.allow_request():
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable (circuit breaker open)",
            )

        request_id = getattr(request.state, "request_id", "unknown")

        # Convert Pydantic model to dict for the ML model
        patient_data = _request_to_dict(payload)

        # Check cache first
        if app_state.cache is not None:
            cached = app_state.cache.get_prediction(patient_data)
            if cached is not None:
                app_state.monitor.record_cache_operation("prediction", hit=True)
                cached["request_id"] = request_id
                cached["cache_hit"] = True
                return ORJSONResponse(content=cached)
            app_state.monitor.record_cache_operation("prediction", hit=False)

        # Run prediction
        result = app_state.model.predict(patient_data)

        # Record metrics
        app_state.monitor.record_inference(result.inference_time_ms)
        app_state.monitor.record_prediction(result.predicted_class, result.risk_level)
        app_state.issue_detector.record_request(result.inference_time_ms, success=True)

        response_data = {
            "request_id": request_id,
            "response_probability": result.response_probability,
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
            "predicted_class": result.predicted_class,
            "risk_level": result.risk_level,
            "inference_time_ms": result.inference_time_ms,
            "model_version": result.model_version,
            "feature_importance": result.feature_importance,
            "cache_hit": False,
        }

        # Cache the result
        if app_state.cache is not None:
            app_state.cache.set_prediction(patient_data, response_data)

        return ORJSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        app_state.issue_detector.record_request(0, success=False)
        logger.error("prediction_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/batch-predict")
async def batch_predict(request: Request, payload: BatchPredictionRequest) -> ORJSONResponse:
    """Batch prediction for multiple patients."""
    ACTIVE_REQUESTS.inc()
    try:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        request_id = getattr(request.state, "request_id", "unknown")

        patient_dicts = [_request_to_dict(p) for p in payload.patients]
        batch_result = app_state.model.predict_batch(patient_dicts)

        # Record metrics
        app_state.monitor.record_inference(batch_result.total_inference_time_ms)

        response_data = {
            "request_id": request_id,
            "predictions": [
                {
                    "response_probability": p.response_probability,
                    "confidence_lower": p.confidence_lower,
                    "confidence_upper": p.confidence_upper,
                    "predicted_class": p.predicted_class,
                    "risk_level": p.risk_level,
                    "model_version": p.model_version,
                }
                for p in batch_result.predictions
            ],
            "total_inference_time_ms": batch_result.total_inference_time_ms,
            "avg_inference_time_ms": batch_result.avg_inference_time_ms,
            "batch_size": len(payload.patients),
            "model_version": batch_result.model_version,
        }

        return ORJSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("batch_prediction_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> ORJSONResponse:
    """Detailed health check with component status."""
    model_loaded = app_state.model is not None
    model_version = app_state.model.model_version if app_state.model else None
    redis_connected = app_state.cache.is_connected if app_state.cache else False
    uptime = time.time() - app_state.start_time
    performance = app_state.monitor.get_performance_report()
    issues = app_state.issue_detector.get_active_issues()

    status = "healthy" if model_loaded else "degraded"
    if not model_loaded:
        status = "unhealthy"
    elif issues:
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            status = "degraded"

    return ORJSONResponse(
        content={
            "status": status,
            "version": __version__,
            "model_loaded": model_loaded,
            "model_version": model_version,
            "redis_connected": redis_connected,
            "uptime_seconds": round(uptime, 1),
            "performance": performance,
            "issues": issues,
        },
        status_code=200 if status != "unhealthy" else 503,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root() -> ORJSONResponse:
    """API root - basic info."""
    return ORJSONResponse(
        content={
            "name": "RxPredict",
            "version": __version__,
            "description": "Real-time Drug Response Prediction API",
            "endpoints": {
                "POST /predict": "Single patient prediction (<50ms p99)",
                "POST /batch-predict": "Batch predictions",
                "GET /health": "Detailed health check",
                "GET /metrics": "Prometheus metrics",
            },
            "latency_target_ms": 100,
            "response_classes": RESPONSE_CLASSES,
        }
    )


# --- Helpers ---


def _request_to_dict(req: PredictionRequest) -> dict[str, Any]:
    """Convert a PredictionRequest to a flat dict for the ML model."""
    genetic = req.genetic_profile.model_dump(by_alias=True)
    demographics = req.demographics.model_dump()

    # Calculate BMI if not provided
    if demographics.get("bmi") is None:
        height_m = demographics["height_cm"] / 100.0
        if height_m > 0:
            demographics["bmi"] = round(demographics["weight_kg"] / (height_m**2), 1)
        else:
            demographics["bmi"] = 25.0

    return {
        "genetic_profile": genetic,
        "metabolizer_phenotype": req.metabolizer_phenotype,
        "demographics": demographics,
        "drug": req.drug.model_dump(),
        "medical_history": req.medical_history.model_dump(),
    }
