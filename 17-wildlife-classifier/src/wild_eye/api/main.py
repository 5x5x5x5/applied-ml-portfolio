"""FastAPI application for the WildEye Wildlife Species Classifier.

Provides REST endpoints for:
    - Image classification (upload and classify)
    - Species catalog with ecological metadata
    - Per-camera biodiversity analytics
    - Recent sightings feed with geospatial data

Designed for integration with the Streamlit dashboard, mobile field apps,
and third-party conservation platforms (e.g., Wildlife Insights, GBIF).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from wild_eye import __version__
from wild_eye.analytics.biodiversity_metrics import (
    Sighting,
    compute_activity_pattern,
    compute_full_biodiversity_summary,
    compute_species_co_occurrence,
)
from wild_eye.models.species_classifier import (
    WildEyeClassifier,
    get_inference_transform,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="WildEye API",
    description=(
        "Wildlife Species Classifier API for camera trap images. "
        "Powered by MobileNetV3 deep learning with ecological analytics."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Pydantic response models ------------------------------------------------


class SpeciesInfo(BaseModel):
    """Metadata for a detectable species."""

    label: str = Field(description="Internal species label")
    common_name: str = Field(description="Common English name")
    scientific_name: str = Field(description="Binomial Latin name")
    taxonomic_class: str = Field(description="Taxonomic class (Mammalia, Aves, etc.)")
    activity_pattern: str = Field(
        description="Primary diel activity: diurnal, nocturnal, crepuscular, cathemeral"
    )
    conservation_status: str = Field(description="IUCN Red List status")


class ClassifyResponse(BaseModel):
    """Response from the /classify endpoint."""

    species: list[str] = Field(description="Detected species labels")
    top_species: str = Field(description="Highest-confidence species")
    top_confidence: float = Field(description="Confidence score for top species")
    is_empty: bool = Field(description="True if no animal detected")
    is_human: bool = Field(description="True if human detected")
    probabilities: dict[str, float] = Field(description="Per-species probabilities")


class SightingResponse(BaseModel):
    """A single sighting record for the feed."""

    species: str
    timestamp: str
    camera_id: str
    latitude: float
    longitude: float
    confidence: float
    image_url: str = ""


class AnalyticsResponse(BaseModel):
    """Biodiversity analytics for a camera station."""

    camera_id: str
    species_richness: int
    shannon_index: float
    simpson_index: float
    evenness: float
    total_detections: int
    species_counts: dict[str, int]
    detection_rate: float
    activity_pattern: dict[int, int]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool


# -- Species catalog (ecological reference data) ------------------------------

SPECIES_CATALOG: dict[str, SpeciesInfo] = {
    "white_tailed_deer": SpeciesInfo(
        label="white_tailed_deer",
        common_name="White-tailed Deer",
        scientific_name="Odocoileus virginianus",
        taxonomic_class="Mammalia",
        activity_pattern="crepuscular",
        conservation_status="Least Concern",
    ),
    "mule_deer": SpeciesInfo(
        label="mule_deer",
        common_name="Mule Deer",
        scientific_name="Odocoileus hemionus",
        taxonomic_class="Mammalia",
        activity_pattern="crepuscular",
        conservation_status="Least Concern",
    ),
    "elk": SpeciesInfo(
        label="elk",
        common_name="Elk",
        scientific_name="Cervus canadensis",
        taxonomic_class="Mammalia",
        activity_pattern="cathemeral",
        conservation_status="Least Concern",
    ),
    "moose": SpeciesInfo(
        label="moose",
        common_name="Moose",
        scientific_name="Alces alces",
        taxonomic_class="Mammalia",
        activity_pattern="crepuscular",
        conservation_status="Least Concern",
    ),
    "black_bear": SpeciesInfo(
        label="black_bear",
        common_name="American Black Bear",
        scientific_name="Ursus americanus",
        taxonomic_class="Mammalia",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "grizzly_bear": SpeciesInfo(
        label="grizzly_bear",
        common_name="Grizzly Bear",
        scientific_name="Ursus arctos horribilis",
        taxonomic_class="Mammalia",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "gray_wolf": SpeciesInfo(
        label="gray_wolf",
        common_name="Gray Wolf",
        scientific_name="Canis lupus",
        taxonomic_class="Mammalia",
        activity_pattern="cathemeral",
        conservation_status="Least Concern",
    ),
    "coyote": SpeciesInfo(
        label="coyote",
        common_name="Coyote",
        scientific_name="Canis latrans",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "red_fox": SpeciesInfo(
        label="red_fox",
        common_name="Red Fox",
        scientific_name="Vulpes vulpes",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "bobcat": SpeciesInfo(
        label="bobcat",
        common_name="Bobcat",
        scientific_name="Lynx rufus",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "mountain_lion": SpeciesInfo(
        label="mountain_lion",
        common_name="Mountain Lion",
        scientific_name="Puma concolor",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "raccoon": SpeciesInfo(
        label="raccoon",
        common_name="Raccoon",
        scientific_name="Procyon lotor",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "striped_skunk": SpeciesInfo(
        label="striped_skunk",
        common_name="Striped Skunk",
        scientific_name="Mephitis mephitis",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "wild_turkey": SpeciesInfo(
        label="wild_turkey",
        common_name="Wild Turkey",
        scientific_name="Meleagris gallopavo",
        taxonomic_class="Aves",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "bald_eagle": SpeciesInfo(
        label="bald_eagle",
        common_name="Bald Eagle",
        scientific_name="Haliaeetus leucocephalus",
        taxonomic_class="Aves",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "great_horned_owl": SpeciesInfo(
        label="great_horned_owl",
        common_name="Great Horned Owl",
        scientific_name="Bubo virginianus",
        taxonomic_class="Aves",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "pronghorn": SpeciesInfo(
        label="pronghorn",
        common_name="Pronghorn",
        scientific_name="Antilocapra americana",
        taxonomic_class="Mammalia",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "american_beaver": SpeciesInfo(
        label="american_beaver",
        common_name="American Beaver",
        scientific_name="Castor canadensis",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "river_otter": SpeciesInfo(
        label="river_otter",
        common_name="North American River Otter",
        scientific_name="Lontra canadensis",
        taxonomic_class="Mammalia",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "snowshoe_hare": SpeciesInfo(
        label="snowshoe_hare",
        common_name="Snowshoe Hare",
        scientific_name="Lepus americanus",
        taxonomic_class="Mammalia",
        activity_pattern="nocturnal",
        conservation_status="Least Concern",
    ),
    "bighorn_sheep": SpeciesInfo(
        label="bighorn_sheep",
        common_name="Bighorn Sheep",
        scientific_name="Ovis canadensis",
        taxonomic_class="Mammalia",
        activity_pattern="diurnal",
        conservation_status="Least Concern",
    ),
    "wolverine": SpeciesInfo(
        label="wolverine",
        common_name="Wolverine",
        scientific_name="Gulo gulo",
        taxonomic_class="Mammalia",
        activity_pattern="cathemeral",
        conservation_status="Least Concern",
    ),
}


# -- Application state --------------------------------------------------------

# In-memory sightings store for demo/development. Production uses DynamoDB.
_sightings_store: list[Sighting] = []

# Model instance (loaded on startup).
_model: WildEyeClassifier | None = None
_transform = get_inference_transform()


@app.on_event("startup")
async def load_model() -> None:
    """Load the classifier model on application startup."""
    global _model
    try:
        # In production, load from a checkpoint file.
        # For development, initialize with pretrained backbone.
        _model = WildEyeClassifier(pretrained=True)
        _model.eval()
        logger.info("WildEye model loaded successfully")
    except Exception:
        logger.exception("Failed to load model -- classification will be unavailable")
        _model = None


# -- Endpoints ----------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_model is not None,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_image(
    file: UploadFile = File(..., description="Camera trap image (JPEG/PNG)"),
    confidence_threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to report a species detection",
    ),
    camera_id: str = Query(default="unknown", description="Camera station ID"),
    latitude: float = Query(default=0.0, description="Camera latitude"),
    longitude: float = Query(default=0.0, description="Camera longitude"),
) -> ClassifyResponse:
    """Upload and classify a camera trap image.

    Accepts a single image file, runs it through the MobileNetV3
    classifier, and returns detected species with confidence scores.
    Multi-label output supports frames with multiple species.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess and classify.
    tensor = _transform(image).unsqueeze(0)
    results = _model.predict(tensor, confidence_threshold=confidence_threshold)
    result = results[0]

    # Store the sighting for analytics.
    if not result.is_empty and not result.is_human:
        for species in result.species:
            sighting = Sighting(
                species=species,
                timestamp=datetime.utcnow(),
                camera_id=camera_id,
                latitude=latitude,
                longitude=longitude,
                confidence=result.probabilities.get(species, 0.0),
            )
            _sightings_store.append(sighting)

    return ClassifyResponse(
        species=result.species,
        top_species=result.top_species,
        top_confidence=result.top_confidence,
        is_empty=result.is_empty,
        is_human=result.is_human,
        probabilities=result.probabilities,
    )


@app.get("/species", response_model=list[SpeciesInfo])
async def list_species(
    taxonomic_class: str | None = Query(
        default=None,
        description="Filter by taxonomic class (e.g., Mammalia, Aves)",
    ),
) -> list[SpeciesInfo]:
    """List all detectable wildlife species with ecological metadata.

    Returns taxonomic information, diel activity patterns, and IUCN
    conservation status for each species in the classifier vocabulary.
    """
    species_list = list(SPECIES_CATALOG.values())
    if taxonomic_class:
        species_list = [
            s for s in species_list if s.taxonomic_class.lower() == taxonomic_class.lower()
        ]
    return species_list


@app.get("/analytics/{camera_id}", response_model=AnalyticsResponse)
async def get_camera_analytics(
    camera_id: str,
) -> AnalyticsResponse:
    """Get biodiversity analytics for a specific camera station.

    Computes species richness, Shannon diversity index, Simpson index,
    Pielou's evenness, detection rates, and diel activity patterns from
    all sightings recorded at the specified camera.
    """
    camera_sightings = [s for s in _sightings_store if s.camera_id == camera_id]

    if not camera_sightings:
        raise HTTPException(
            status_code=404,
            detail=f"No sightings found for camera '{camera_id}'",
        )

    summary = compute_full_biodiversity_summary(camera_sightings, camera_id=camera_id)
    activity = compute_activity_pattern(camera_sightings)

    return AnalyticsResponse(
        camera_id=camera_id,
        species_richness=summary.species_richness,
        shannon_index=round(summary.shannon_index, 4),
        simpson_index=round(summary.simpson_index, 4),
        evenness=round(summary.evenness, 4),
        total_detections=summary.total_detections,
        species_counts=summary.species_counts,
        detection_rate=round(summary.detection_rate, 2),
        activity_pattern=activity,
    )


@app.get("/sightings", response_model=list[SightingResponse])
async def get_recent_sightings(
    limit: int = Query(default=50, ge=1, le=500, description="Max sightings to return"),
    species: str | None = Query(default=None, description="Filter by species"),
    camera_id: str | None = Query(default=None, description="Filter by camera"),
) -> list[SightingResponse]:
    """Get recent wildlife sightings with map-ready geospatial data.

    Returns the most recent detection events, optionally filtered by
    species or camera station. Each sighting includes GPS coordinates
    for map visualization.
    """
    filtered = list(_sightings_store)

    if species:
        filtered = [s for s in filtered if s.species == species]
    if camera_id:
        filtered = [s for s in filtered if s.camera_id == camera_id]

    # Sort by timestamp descending (most recent first).
    filtered.sort(key=lambda s: s.timestamp, reverse=True)
    filtered = filtered[:limit]

    return [
        SightingResponse(
            species=s.species,
            timestamp=s.timestamp.isoformat(),
            camera_id=s.camera_id,
            latitude=s.latitude,
            longitude=s.longitude,
            confidence=round(s.confidence, 4),
        )
        for s in filtered
    ]


@app.get("/co-occurrence")
async def get_species_co_occurrence(
    time_window_minutes: int = Query(
        default=30,
        ge=1,
        le=1440,
        description="Time window for co-occurrence (minutes)",
    ),
) -> list[dict[str, Any]]:
    """Analyse species co-occurrence patterns across all camera stations.

    Uses temporally proximate detections to identify positive (attraction),
    negative (avoidance), and random interspecific associations.
    """
    if not _sightings_store:
        return []

    df = compute_species_co_occurrence(
        _sightings_store,
        time_window_minutes=time_window_minutes,
    )
    return df.to_dict(orient="records")  # type: ignore[return-value]
