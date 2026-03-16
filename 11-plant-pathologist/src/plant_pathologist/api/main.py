"""FastAPI application for plant disease diagnosis.

Provides REST endpoints for leaf image upload and diagnosis,
disease listing, and treatment information retrieval.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from plant_pathologist.knowledge.disease_database import (
    assess_severity,
    get_all_diseases,
    get_disease_info,
    get_diseases_by_species,
    get_treatment_summary,
)
from plant_pathologist.models.disease_classifier import (
    DISEASE_CLASSES,
    DiagnosisResult,
    PlantDiseaseClassifier,
    load_model,
)
from plant_pathologist.preprocessing.leaf_processor import LeafProcessor

logger = logging.getLogger(__name__)

# Global model and processor instances
_model: PlantDiseaseClassifier | None = None
_processor: LeafProcessor | None = None


def get_model() -> PlantDiseaseClassifier:
    """Get the loaded model, initializing if necessary."""
    global _model
    if _model is None:
        logger.info("Loading PlantDiseaseClassifier model...")
        _model = load_model(checkpoint_path=None, device="cpu")
        logger.info("Model loaded successfully.")
    return _model


def get_processor() -> LeafProcessor:
    """Get the leaf processor singleton."""
    global _processor
    if _processor is None:
        _processor = LeafProcessor()
    return _processor


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan: load model on startup."""
    logger.info("PlantPathologist API starting up...")
    get_model()
    get_processor()
    logger.info("API ready to serve requests.")
    yield
    logger.info("PlantPathologist API shutting down.")


app = FastAPI(
    title="PlantPathologist API",
    description=(
        "Plant disease detection API using EfficientNet-B0 transfer learning. "
        "Upload a leaf image to receive a diagnosis with treatment recommendations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class DiseaseClassification(BaseModel):
    """Single disease classification prediction."""

    disease_id: str
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DiagnosisResponse(BaseModel):
    """Complete diagnosis response for a leaf image."""

    disease_id: str
    disease_name: str
    confidence: float
    plant_species: str
    species_confidence: float
    is_healthy: bool
    severity: str
    top_predictions: list[DiseaseClassification]
    image_quality_valid: bool
    image_quality_issues: list[str]
    description: str
    symptoms: list[str]
    treatment_organic: list[str]
    treatment_chemical: list[str]
    treatment_cultural: list[str]
    prevention: list[str]
    inference_time_ms: float
    calibrated: bool


class DiseaseListItem(BaseModel):
    """Summary item for disease listing endpoint."""

    disease_id: str
    common_name: str
    scientific_name: str
    pathogen: str
    pathogen_type: str
    plant_species: str
    is_healthy: bool


class TreatmentResponse(BaseModel):
    """Treatment information for a specific disease."""

    disease_id: str
    disease_name: str
    pathogen_type: str
    organic_treatments: list[str]
    chemical_treatments: list[str]
    cultural_practices: list[str]
    biological_control: list[str]
    prevention: list[str]
    notes: str


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    num_disease_classes: int
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """API health and status check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        num_disease_classes=len(DISEASE_CLASSES),
        version="0.1.0",
    )


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_leaf(
    file: UploadFile = File(..., description="Leaf image (JPEG or PNG)"),
) -> DiagnosisResponse:
    """Upload a leaf image and receive a disease diagnosis.

    Accepts JPEG and PNG images. The image is preprocessed (quality validation,
    leaf segmentation, color analysis) before classification. Returns the
    predicted disease, confidence score, severity assessment, and treatment
    recommendations.

    Raises:
        HTTPException 400: If the uploaded file is not a valid image.
        HTTPException 422: If the image fails quality validation.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    # Read and open image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {e}",
        )

    processor = get_processor()
    model = get_model()

    # Preprocess and validate
    image = processor.preprocess_for_model(image)
    quality = processor.validate_image_quality(image)
    segmentation = processor.segment_leaf(image)
    color_analysis = processor.analyze_colors(image, segmentation.leaf_mask)
    lesion_info = processor.detect_lesions(image, segmentation.leaf_mask)

    # Run inference
    start_time = time.perf_counter()
    result: DiagnosisResult = model.predict(image)
    inference_ms = (time.perf_counter() - start_time) * 1000

    # Look up disease info
    disease_info = get_disease_info(result.disease_class)
    severity = assess_severity(result.disease_class, lesion_info.severity_percentage)

    # Build top predictions
    top_predictions = []
    for disease_class, conf in result.top_k_diseases:
        info = get_disease_info(disease_class)
        name = info.common_name if info else disease_class
        top_predictions.append(
            DiseaseClassification(
                disease_id=disease_class,
                disease_name=name,
                confidence=round(conf, 4),
            )
        )

    # Build response
    if disease_info:
        description = disease_info.description
        symptoms = disease_info.symptoms
        treatment_organic = disease_info.treatment.organic
        treatment_chemical = disease_info.treatment.chemical
        treatment_cultural = disease_info.treatment.cultural
        prevention = disease_info.prevention
        disease_name = disease_info.common_name
    else:
        description = "Disease information not available."
        symptoms = []
        treatment_organic = []
        treatment_chemical = []
        treatment_cultural = []
        prevention = []
        disease_name = result.disease_class

    return DiagnosisResponse(
        disease_id=result.disease_class,
        disease_name=disease_name,
        confidence=round(result.confidence, 4),
        plant_species=result.plant_species,
        species_confidence=round(result.species_confidence, 4),
        is_healthy=result.is_healthy,
        severity=severity.value,
        top_predictions=top_predictions,
        image_quality_valid=quality.is_valid,
        image_quality_issues=quality.issues,
        description=description,
        symptoms=symptoms,
        treatment_organic=treatment_organic,
        treatment_chemical=treatment_chemical,
        treatment_cultural=treatment_cultural,
        prevention=prevention,
        inference_time_ms=round(inference_ms, 2),
        calibrated=result.calibrated,
    )


@app.get("/diseases", response_model=list[DiseaseListItem])
async def list_diseases(species: str | None = None) -> list[DiseaseListItem]:
    """List all detectable diseases, optionally filtered by plant species.

    Args:
        species: Optional plant species filter (tomato, potato, corn, apple, grape).
    """
    if species:
        diseases = get_diseases_by_species(species.lower())
        if not diseases:
            raise HTTPException(
                status_code=404,
                detail=f"No diseases found for species '{species}'. "
                f"Supported: tomato, potato, corn, apple, grape.",
            )
    else:
        diseases = get_all_diseases()

    return [
        DiseaseListItem(
            disease_id=d.disease_id,
            common_name=d.common_name,
            scientific_name=d.scientific_name,
            pathogen=d.pathogen,
            pathogen_type=d.pathogen_type.value,
            plant_species=d.plant_species,
            is_healthy=d.is_healthy,
        )
        for d in diseases
    ]


@app.get("/diseases/{disease_id}/treatment", response_model=TreatmentResponse)
async def get_treatment(disease_id: str) -> TreatmentResponse:
    """Get detailed treatment recommendations for a specific disease.

    Args:
        disease_id: Disease identifier (e.g., 'tomato_early_blight').

    Raises:
        HTTPException 404: If the disease ID is not found.
    """
    summary = get_treatment_summary(disease_id)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{disease_id}' not found. Use GET /diseases for valid IDs.",
        )

    return TreatmentResponse(
        disease_id=disease_id,
        disease_name=str(summary["disease"]),
        pathogen_type=str(summary["pathogen_type"]),
        organic_treatments=list(summary["organic_treatments"]),  # type: ignore[arg-type]
        chemical_treatments=list(summary["chemical_treatments"]),  # type: ignore[arg-type]
        cultural_practices=list(summary["cultural_practices"]),  # type: ignore[arg-type]
        biological_control=list(summary["biological_control"]),  # type: ignore[arg-type]
        prevention=list(summary["prevention"]),  # type: ignore[arg-type]
        notes=str(summary["notes"]),
    )


# ---------------------------------------------------------------------------
# Static files (serve frontend)
# ---------------------------------------------------------------------------

try:
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception:
    logger.warning("Frontend directory not found. Static files will not be served.")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
