"""FastAPI application for PharmaSentinel adverse event classification.

Provides REST endpoints for single and batch prediction of adverse
event severity, health checks, and DataDog-compatible metrics.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from pharma_sentinel import __app_name__, __version__
from pharma_sentinel.config import AppSettings, get_settings
from pharma_sentinel.models.adverse_event_classifier import (
    AdverseEventClassifier,
    SeverityLevel,
)
from pharma_sentinel.monitoring.datadog_config import MetricsCollector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request schema for adverse event classification."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Adverse event report text to classify",
        examples=["Patient experienced severe nausea and vomiting after taking Ibuprofen 400mg."],
    )
    drug_name: str | None = Field(
        default=None,
        max_length=500,
        description="Name of the drug involved",
    )
    patient_age: float | None = Field(
        default=None,
        ge=0,
        le=150,
        description="Patient age in years",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("Text must contain at least 10 non-whitespace characters")
        return stripped


class PredictResponse(BaseModel):
    """Response schema for adverse event classification."""

    severity: str = Field(description="Predicted severity level")
    confidence: float = Field(description="Prediction confidence (0-1)")
    probabilities: dict[str, float] = Field(description="Class probability distribution")
    drug_name: str | None = Field(default=None, description="Drug name from request")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    model_version: str = Field(description="Model version used for prediction")
    timestamp: str = Field(description="Prediction timestamp in ISO 8601")


class BatchPredictRequest(BaseModel):
    """Request schema for batch adverse event classification."""

    reports: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of adverse event reports to classify (max 100)",
    )


class BatchPredictResponse(BaseModel):
    """Response schema for batch classification."""

    predictions: list[PredictResponse] = Field(description="List of predictions")
    total_count: int = Field(description="Total number of predictions")
    processing_time_ms: float = Field(description="Total processing time in ms")
    severity_summary: dict[str, int] = Field(description="Count of each severity level")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    environment: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str


class MetricsResponse(BaseModel):
    """DataDog-compatible metrics response."""

    predictions_total: int
    predictions_by_severity: dict[str, int]
    average_latency_ms: float
    average_confidence: float
    error_count: int
    uptime_seconds: float


# ─────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────


class AppState:
    """Holds runtime state for the application."""

    def __init__(self) -> None:
        self.classifier: AdverseEventClassifier | None = None
        self.settings: AppSettings = get_settings()
        self.metrics: MetricsCollector = MetricsCollector(self.settings)
        self.start_time: float = time.monotonic()
        self.prediction_count: int = 0
        self.error_count: int = 0
        self.severity_counts: dict[str, int] = {s.value: 0 for s in SeverityLevel}
        self.total_latency_ms: float = 0.0
        self.total_confidence: float = 0.0


app_state = AppState()


# ─────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown tasks."""
    logger.info("Starting %s v%s", __app_name__, __version__)

    # Initialize classifier
    app_state.classifier = AdverseEventClassifier()
    try:
        app_state.classifier.load_model(app_state.settings.model.artifact_path)
        logger.info("Model loaded successfully from %s", app_state.settings.model.artifact_path)
    except FileNotFoundError:
        logger.warning(
            "Model file not found at %s. Prediction endpoints will return 503 until model is loaded.",
            app_state.settings.model.artifact_path,
        )
        app_state.classifier = None

    # Initialize metrics
    app_state.metrics.initialize()

    yield

    # Shutdown
    logger.info("Shutting down %s", __app_name__)
    app_state.metrics.flush()


# ─────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="PharmaSentinel API",
    description=(
        "Drug Adverse Event Detection Pipeline - Classifies FDA adverse "
        "event reports by severity using NLP and machine learning."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_state.settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────


@app.middleware("http")
async def datadog_apm_middleware(request: Request, call_next: Any) -> Response:
    """DataDog APM tracing middleware.

    Records request latency, status codes, and routes for every HTTP request.
    """
    start = time.monotonic()
    response: Response | None = None

    try:
        response = await call_next(request)
        return response
    except Exception:
        app_state.error_count += 1
        raise
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        status = response.status_code if response else 500
        app_state.metrics.record_request(
            method=request.method,
            path=request.url.path,
            status_code=status,
            latency_ms=elapsed_ms,
        )


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single adverse event report by severity.

    Takes an adverse event report text and returns the predicted
    severity level (mild/moderate/severe/critical) with confidence
    scores and class probability distribution.
    """
    if app_state.classifier is None or not app_state.classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready for predictions.",
        )

    start = time.monotonic()

    try:
        result = app_state.classifier.predict_single(request.text)
    except Exception as exc:
        app_state.error_count += 1
        logger.exception("Prediction failed for text (length=%d)", len(request.text))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000

    # Update metrics
    app_state.prediction_count += 1
    app_state.total_latency_ms += elapsed_ms
    app_state.total_confidence += result["confidence"]
    severity = result["severity"]
    if severity in app_state.severity_counts:
        app_state.severity_counts[severity] += 1

    app_state.metrics.record_prediction(
        severity=severity,
        confidence=result["confidence"],
        latency_ms=elapsed_ms,
    )

    return PredictResponse(
        severity=severity,
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        drug_name=request.drug_name,
        processing_time_ms=round(elapsed_ms, 2),
        model_version=__version__,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of adverse event reports.

    Accepts up to 100 reports and returns predictions for each.
    More efficient than calling /predict repeatedly due to
    vectorized inference.
    """
    if app_state.classifier is None or not app_state.classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready for predictions.",
        )

    start = time.monotonic()

    texts = [report.text for report in request.reports]

    try:
        results = app_state.classifier.predict(texts)
    except Exception as exc:
        app_state.error_count += 1
        logger.exception("Batch prediction failed for %d texts", len(texts))
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000

    predictions: list[PredictResponse] = []
    severity_summary: dict[str, int] = {s.value: 0 for s in SeverityLevel}

    for report, result in zip(request.reports, results):
        severity = result["severity"]
        if severity in severity_summary:
            severity_summary[severity] += 1

        # Update app-level metrics
        app_state.prediction_count += 1
        app_state.total_confidence += result["confidence"]
        if severity in app_state.severity_counts:
            app_state.severity_counts[severity] += 1

        predictions.append(
            PredictResponse(
                severity=severity,
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                drug_name=report.drug_name,
                processing_time_ms=round(elapsed_ms / len(texts), 2),
                model_version=__version__,
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )

    app_state.total_latency_ms += elapsed_ms

    app_state.metrics.record_batch_prediction(
        count=len(texts),
        latency_ms=elapsed_ms,
    )

    return BatchPredictResponse(
        predictions=predictions,
        total_count=len(predictions),
        processing_time_ms=round(elapsed_ms, 2),
        severity_summary=severity_summary,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and orchestrators.

    Returns service status, version, model state, and uptime.
    """
    uptime = time.monotonic() - app_state.start_time

    return HealthResponse(
        status="healthy" if app_state.classifier and app_state.classifier.is_fitted else "degraded",
        service=__app_name__,
        version=__version__,
        environment=app_state.settings.environment.value,
        model_loaded=app_state.classifier is not None and app_state.classifier.is_fitted,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """DataDog-compatible metrics endpoint.

    Returns aggregated application metrics for monitoring dashboards.
    """
    uptime = time.monotonic() - app_state.start_time
    avg_latency = (
        app_state.total_latency_ms / app_state.prediction_count
        if app_state.prediction_count > 0
        else 0.0
    )
    avg_confidence = (
        app_state.total_confidence / app_state.prediction_count
        if app_state.prediction_count > 0
        else 0.0
    )

    return MetricsResponse(
        predictions_total=app_state.prediction_count,
        predictions_by_severity=app_state.severity_counts,
        average_latency_ms=round(avg_latency, 2),
        average_confidence=round(avg_confidence, 4),
        error_count=app_state.error_count,
        uptime_seconds=round(uptime, 2),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors with a clean JSON response."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )
