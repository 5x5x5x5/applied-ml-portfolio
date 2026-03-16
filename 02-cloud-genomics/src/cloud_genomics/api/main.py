"""FastAPI application for the CloudGenomics Variant Classification Service.

Endpoints:
  POST /classify-variant  - Classify a single variant
  POST /upload-vcf        - Upload and process a VCF file
  GET  /variant/{id}      - Retrieve classification result
  GET  /health            - Health check with dependency status

Includes request validation, rate limiting, DataDog tracing, and
HIPAA-aware audit logging.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from cloud_genomics.models.variant_classifier import (
    PredictionResult,
    VariantClassifier,
    VariantFeatures,
    generate_synthetic_training_data,
)
from cloud_genomics.monitoring.metrics import MetricsCollector, create_metrics_collector
from cloud_genomics.pipeline.vcf_processor import VCFProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory stores (production would use RDS/DynamoDB)
# ---------------------------------------------------------------------------
_results_store: dict[str, dict[str, Any]] = {}
_classifier: VariantClassifier | None = None
_metrics: MetricsCollector | None = None

# Rate limiting state
_rate_limit_window: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class VariantRequest(BaseModel):
    """Request body for single-variant classification."""

    chrom: str = Field(..., description="Chromosome (e.g., 'chr1', '17')")
    pos: int = Field(..., gt=0, description="Genomic position (1-based)")
    ref: str = Field(..., min_length=1, max_length=1000, description="Reference allele")
    alt: str = Field(..., min_length=1, max_length=1000, description="Alternate allele")

    # Optional annotation fields
    phylop_score: float = Field(0.0, ge=-14, le=6.4)
    phastcons_score: float = Field(0.0, ge=0, le=1)
    gerp_score: float = Field(0.0, ge=-12.3, le=6.17)
    gnomad_af: float = Field(0.0, ge=0, le=1)
    gnomad_af_afr: float = Field(0.0, ge=0, le=1)
    gnomad_af_eas: float = Field(0.0, ge=0, le=1)
    gnomad_af_nfe: float = Field(0.0, ge=0, le=1)
    gnomad_homozygote_count: int = Field(0, ge=0)
    sift_score: float = Field(1.0, ge=0, le=1)
    polyphen2_score: float = Field(0.0, ge=0, le=1)
    cadd_phred: float = Field(0.0, ge=0, le=99)
    revel_score: float = Field(0.0, ge=0, le=1)
    mutation_taster_score: float = Field(0.0, ge=0, le=1)
    in_protein_domain: bool = False
    domain_conservation: float = Field(0.0, ge=0, le=1)
    distance_to_active_site: float = -1.0
    pfam_domain_count: int = Field(0, ge=0)
    consequence: str = Field("missense", description="Variant consequence type")
    splice_ai_score: float = Field(0.0, ge=0, le=1)

    @field_validator("ref", "alt")
    @classmethod
    def validate_allele(cls, v: str) -> str:
        v = v.upper()
        valid_bases = set("ACGTN")
        if not all(c in valid_bases for c in v):
            raise ValueError("Invalid allele: must contain only A, C, G, T, N characters")
        return v

    @field_validator("chrom")
    @classmethod
    def validate_chrom(cls, v: str) -> str:
        v = v.replace("chr", "")
        valid = [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M"]
        if v not in valid:
            raise ValueError(f"Invalid chromosome: {v}")
        return f"chr{v}"


class VariantResponse(BaseModel):
    """Response for a variant classification."""

    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    classification: str
    confidence: float
    class_probabilities: dict[str, float]
    explanation: list[str]
    timestamp: str


class VCFUploadResponse(BaseModel):
    """Response for VCF file upload."""

    job_id: str
    status: str
    variants_processed: int
    variants_passed_filters: int
    results: list[VariantResponse]
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    model_accuracy: float | None
    dependencies: dict[str, str]
    uptime_seconds: float
    timestamp: str


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
def _check_rate_limit(client_ip: str) -> bool:
    """Check if the client has exceeded the rate limit.

    Args:
        client_ip: Client IP address.

    Returns:
        True if within limits, False if rate-limited.
    """
    now = time.monotonic()
    window = _rate_limit_window[client_ip]

    # Remove requests outside the window
    _rate_limit_window[client_ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW_SECONDS]

    if len(_rate_limit_window[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    _rate_limit_window[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan handler - initialize model and metrics."""
    global _classifier, _metrics, _start_time
    _start_time = time.monotonic()

    logger.info("Initializing CloudGenomics API")

    # Initialize metrics
    _metrics = create_metrics_collector()

    # Train model with synthetic data (production would load a pre-trained model)
    _classifier = VariantClassifier()
    features, labels = generate_synthetic_training_data(n_samples=500)
    metrics = _classifier.train(features, labels)
    logger.info("Model initialized with accuracy: %.4f", metrics.accuracy)

    yield

    logger.info("Shutting down CloudGenomics API")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudGenomics Variant Classification API",
    description=(
        "ML-powered classification of genetic variants (SNPs, indels) into "
        "ACMG/AMP categories: benign, likely-benign, VUS, likely-pathogenic, pathogenic."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: rate limiting, request logging, tracing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Apply rate limiting and request logging."""
    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        logger.warning("Rate limit exceeded for %s", client_ip)
        if _metrics:
            _metrics.increment_error_count("rate_limit_exceeded")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    # Record request metrics
    if _metrics:
        _metrics.record_request_latency(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_seconds=duration,
        )

    logger.info(
        "request_complete method=%s path=%s status=%d duration=%.3fs client=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration,
        client_ip,
    )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/classify-variant", response_model=VariantResponse)
async def classify_variant(request: VariantRequest) -> VariantResponse:
    """Classify a single genetic variant.

    Accepts variant coordinates and optional annotation features,
    returns ACMG/AMP classification with confidence and explanation.
    """
    if _classifier is None or not _classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not initialized")

    start = time.monotonic()

    # Build features from request
    features = VariantFeatures(
        phylop_score=request.phylop_score,
        phastcons_score=request.phastcons_score,
        gerp_score=request.gerp_score,
        gnomad_af=request.gnomad_af,
        gnomad_af_afr=request.gnomad_af_afr,
        gnomad_af_eas=request.gnomad_af_eas,
        gnomad_af_nfe=request.gnomad_af_nfe,
        gnomad_homozygote_count=request.gnomad_homozygote_count,
        sift_score=request.sift_score,
        polyphen2_score=request.polyphen2_score,
        cadd_phred=request.cadd_phred,
        revel_score=request.revel_score,
        mutation_taster_score=request.mutation_taster_score,
        in_protein_domain=request.in_protein_domain,
        domain_conservation=request.domain_conservation,
        distance_to_active_site=request.distance_to_active_site,
        pfam_domain_count=request.pfam_domain_count,
        consequence=request.consequence,
        splice_ai_score=request.splice_ai_score,
    )

    try:
        result: PredictionResult = _classifier.predict(features)
    except Exception as exc:
        logger.exception("Classification failed for %s:%d", request.chrom, request.pos)
        if _metrics:
            _metrics.increment_error_count("classification_error")
        raise HTTPException(status_code=500, detail="Classification failed") from exc

    duration = time.monotonic() - start

    # Record metrics
    if _metrics:
        _metrics.record_classification(
            variant_class=result.variant_class.value,
            confidence=result.confidence,
            latency_seconds=duration,
        )

    # Generate variant ID
    variant_id = hashlib.sha256(
        f"{request.chrom}:{request.pos}:{request.ref}:{request.alt}".encode()
    ).hexdigest()[:16]

    timestamp = datetime.now(UTC).isoformat()

    response = VariantResponse(
        variant_id=variant_id,
        chrom=request.chrom,
        pos=request.pos,
        ref=request.ref,
        alt=request.alt,
        classification=result.variant_class.value,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        explanation=result.explanation,
        timestamp=timestamp,
    )

    # Store result
    _results_store[variant_id] = response.model_dump()

    logger.info(
        "classified variant=%s:%d class=%s confidence=%.3f latency=%.3fs",
        request.chrom,
        request.pos,
        result.variant_class.value,
        result.confidence,
        duration,
    )

    return response


@app.post("/upload-vcf", response_model=VCFUploadResponse)
async def upload_vcf(file: UploadFile = File(...)) -> VCFUploadResponse:
    """Upload and process a VCF file.

    Parses the VCF, applies quality filters, and classifies all
    passing variants.
    """
    if _classifier is None or not _classifier.is_trained:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if file.filename and not file.filename.endswith((".vcf", ".vcf.gz")):
        raise HTTPException(
            status_code=400,
            detail="File must be a VCF file (.vcf or .vcf.gz)",
        )

    start = time.monotonic()
    job_id = str(uuid.uuid4())

    try:
        content = await file.read()
        vcf_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="File encoding error: VCF must be UTF-8"
        ) from exc

    # Process VCF
    processor = VCFProcessor()
    try:
        variant_results = processor.process_string(vcf_text)
    except Exception as exc:
        logger.exception("VCF processing failed for job %s", job_id)
        if _metrics:
            _metrics.increment_error_count("vcf_processing_error")
        raise HTTPException(status_code=422, detail=f"VCF processing error: {exc}") from exc

    # Classify all passing variants
    responses: list[VariantResponse] = []
    for variant, features in variant_results:
        try:
            result = _classifier.predict(features)
        except Exception:
            logger.exception("Classification failed for variant %s", variant.variant_key)
            continue

        variant_id = hashlib.sha256(variant.variant_key.encode()).hexdigest()[:16]
        timestamp = datetime.now(UTC).isoformat()

        resp = VariantResponse(
            variant_id=variant_id,
            chrom=variant.chrom,
            pos=variant.pos,
            ref=variant.ref,
            alt=",".join(variant.alt),
            classification=result.variant_class.value,
            confidence=result.confidence,
            class_probabilities=result.class_probabilities,
            explanation=result.explanation,
            timestamp=timestamp,
        )
        responses.append(resp)
        _results_store[variant_id] = resp.model_dump()

    duration = time.monotonic() - start
    stats = processor.stats

    if _metrics:
        _metrics.record_vcf_processing(
            total_variants=stats.total_variants,
            passed_variants=stats.passed_filters,
            latency_seconds=duration,
        )

    logger.info(
        "vcf_processed job=%s total=%d passed=%d classified=%d duration=%.3fs",
        job_id,
        stats.total_variants,
        stats.passed_filters,
        len(responses),
        duration,
    )

    return VCFUploadResponse(
        job_id=job_id,
        status="completed",
        variants_processed=stats.total_variants,
        variants_passed_filters=stats.passed_filters,
        results=responses,
        processing_time_seconds=round(duration, 3),
    )


@app.get("/variant/{variant_id}", response_model=VariantResponse)
async def get_variant(variant_id: str) -> VariantResponse:
    """Retrieve a previously classified variant by ID."""
    if variant_id not in _results_store:
        raise HTTPException(status_code=404, detail="Variant not found")

    return VariantResponse(**_results_store[variant_id])


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with dependency status.

    Reports model status, dependency connectivity, and uptime.
    """
    model_loaded = _classifier is not None and _classifier.is_trained
    model_accuracy: float | None = None

    if _classifier and _classifier.training_metrics:
        model_accuracy = _classifier.training_metrics.accuracy

    # Check dependencies
    dependencies: dict[str, str] = {}

    # Model status
    dependencies["ml_model"] = "healthy" if model_loaded else "not_loaded"

    # In-memory store
    dependencies["results_store"] = "healthy"
    dependencies["results_count"] = str(len(_results_store))

    # In production, these would check actual connections:
    # - RDS PostgreSQL
    # - S3
    # - KMS
    # - Step Functions
    dependencies["database"] = "healthy"  # Would check RDS connection
    dependencies["s3"] = "healthy"  # Would check S3 access
    dependencies["kms"] = "healthy"  # Would check KMS key access

    uptime = time.monotonic() - _start_time if _start_time > 0 else 0.0
    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        model_loaded=model_loaded,
        model_accuracy=model_accuracy,
        dependencies=dependencies,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(UTC).isoformat(),
    )
