"""FastAPI inference server for CellVision cell classification.

Endpoints:
    POST /classify         - Classify a single microscopy image
    POST /batch-classify   - Classify multiple images in one request
    GET  /model-info       - Model architecture and performance metrics
    GET  /health           - Health check
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

from cell_vision import __version__
from cell_vision.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from cell_vision.models.cell_classifier import CellClassifier

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CellVision API",
    description=(
        "Microscopy image cell type classifier. Upload cell images from blood smears "
        "to identify red blood cells, white blood cell subtypes, and platelets."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global state ----

_classifier: CellClassifier | None = None
MODEL_PATH = Path("models/cellnet_best.pt")
IMAGE_SIZE = 224


# ---- Pydantic schemas ----


class Prediction(BaseModel):
    """Single cell type prediction."""

    cell_class: str = Field(description="Internal cell type identifier")
    label: str = Field(description="Human-readable cell type label")
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence")


class ClassificationResult(BaseModel):
    """Result for a single image classification."""

    filename: str
    predictions: list[Prediction]
    inference_time_ms: float = Field(description="Inference time in milliseconds")


class BatchClassificationResult(BaseModel):
    """Result for batch classification."""

    results: list[ClassificationResult]
    total_images: int
    total_time_ms: float


class ModelInfo(BaseModel):
    """Model architecture and metadata."""

    model_type: str
    version: str
    num_classes: int
    total_parameters: int
    trainable_parameters: int
    device: str
    cell_types: list[str]
    supported_image_formats: list[str]


# ---- Helpers ----


def get_classifier() -> CellClassifier:
    """Lazy-load the classifier model."""
    global _classifier
    if _classifier is None:
        model_type = "cellnet"
        _classifier = CellClassifier(model_type=model_type)

        if MODEL_PATH.exists():
            _classifier.load(MODEL_PATH)
            logger.info("Loaded model weights from %s", MODEL_PATH)
        else:
            logger.warning(
                "No saved model found at %s; using randomly initialized weights",
                MODEL_PATH,
            )

    return _classifier


def preprocess_upload(image_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image bytes to a model-ready tensor.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        Preprocessed tensor of shape (1, 3, H, W).

    Raises:
        HTTPException: If the image cannot be loaded or processed.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return tensor


# ---- Endpoints ----


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


@app.post("/classify", response_model=ClassificationResult)
async def classify_image(
    file: UploadFile = File(..., description="Microscopy image to classify"),
    top_k: int = 3,
) -> ClassificationResult:
    """Classify a single microscopy image.

    Upload a cell image (PNG, JPEG, TIFF) and receive top-k cell type
    predictions with confidence scores.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    tensor = preprocess_upload(image_bytes)

    classifier = get_classifier()

    start = time.perf_counter()
    predictions = classifier.predict(tensor, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return ClassificationResult(
        filename=file.filename or "unknown",
        predictions=[
            Prediction(
                cell_class=p["class"],
                label=p["label"],
                confidence=p["confidence"],
            )
            for p in predictions
        ],
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/batch-classify", response_model=BatchClassificationResult)
async def batch_classify(
    files: list[UploadFile] = File(..., description="Multiple microscopy images"),
    top_k: int = 3,
) -> BatchClassificationResult:
    """Classify multiple microscopy images in a single request.

    Returns predictions for each uploaded image along with total
    processing time.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    classifier = get_classifier()
    results: list[ClassificationResult] = []
    total_start = time.perf_counter()

    for file in files:
        image_bytes = await file.read()
        tensor = preprocess_upload(image_bytes)

        start = time.perf_counter()
        predictions = classifier.predict(tensor, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(
            ClassificationResult(
                filename=file.filename or "unknown",
                predictions=[
                    Prediction(
                        cell_class=p["class"],
                        label=p["label"],
                        confidence=p["confidence"],
                    )
                    for p in predictions
                ],
                inference_time_ms=round(elapsed_ms, 2),
            )
        )

    total_ms = (time.perf_counter() - total_start) * 1000

    return BatchClassificationResult(
        results=results,
        total_images=len(results),
        total_time_ms=round(total_ms, 2),
    )


@app.get("/model-info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return model architecture details and performance metrics."""
    classifier = get_classifier()
    info = classifier.get_model_info()

    return ModelInfo(
        model_type=info["model_type"],
        version=__version__,
        num_classes=info["num_classes"],
        total_parameters=info["total_parameters"],
        trainable_parameters=info["trainable_parameters"],
        device=info["device"],
        cell_types=info["cell_types"],
        supported_image_formats=["PNG", "JPEG", "TIFF", "BMP"],
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
