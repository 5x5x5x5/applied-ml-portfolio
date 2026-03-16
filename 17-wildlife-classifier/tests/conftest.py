"""Shared test fixtures for WildEye test suite.

Provides reusable fixtures for:
    - Model instantiation (CPU, no pretrained weights for speed)
    - Synthetic camera trap images (RGB, IR, blurry)
    - Sighting data for biodiversity analytics testing
    - FastAPI test client
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from PIL import Image

from wild_eye import NUM_SPECIES
from wild_eye.analytics.biodiversity_metrics import Sighting
from wild_eye.models.species_classifier import WildEyeClassifier


@pytest.fixture
def classifier() -> WildEyeClassifier:
    """Create a WildEye classifier with random weights (no pretrained download).

    Using pretrained=False avoids network calls in CI and keeps tests fast.
    """
    model = WildEyeClassifier(
        num_classes=NUM_SPECIES,
        pretrained=False,
        dropout_rate=0.1,
        freeze_backbone_layers=0,
    )
    model.eval()
    return model


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """A random input tensor simulating a preprocessed camera trap image.

    Shape: (1, 3, 224, 224) matching MobileNetV3 input requirements.
    """
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def batch_tensor() -> torch.Tensor:
    """A batch of 4 random input tensors for batch inference testing."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """A synthetic RGB camera trap image (640x480, daytime scene).

    Simulates a typical colour daytime camera trap capture with
    moderate brightness and contrast variation.
    """
    rng = np.random.default_rng(42)
    # Background (greenish forest floor).
    array = rng.integers(40, 120, size=(480, 640, 3), dtype=np.uint8)
    # Add a bright region simulating an animal body.
    array[150:350, 200:450, :] = rng.integers(100, 200, size=(200, 250, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def sample_ir_image() -> Image.Image:
    """A synthetic IR (infrared) camera trap image.

    Simulates a night-vision capture: near-grayscale with very low
    colour channel spread -- characteristic of IR LED illumination.
    """
    rng = np.random.default_rng(99)
    # IR images have nearly identical RGB channels.
    gray_base = rng.integers(30, 150, size=(480, 640), dtype=np.uint8)
    array = np.stack([gray_base, gray_base, gray_base + 2], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def sample_blurry_image() -> Image.Image:
    """A synthetic blurry image simulating motion blur.

    Motion blur is common when fast-moving animals (e.g., deer, fox)
    trigger the PIR sensor. The image has very low edge content.
    """
    rng = np.random.default_rng(77)
    # Low-frequency noise = no sharp edges = low Laplacian variance.
    array = rng.integers(80, 120, size=(480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    # Apply heavy Gaussian blur to simulate motion blur.
    from PIL import ImageFilter

    return image.filter(ImageFilter.GaussianBlur(radius=10))


@pytest.fixture
def sample_sightings() -> list[Sighting]:
    """Ecologically plausible sighting data for analytics testing.

    Simulates a 30-day survey at 3 camera stations in a mixed forest
    ecosystem with realistic species composition and diel activity.
    """
    rng = np.random.default_rng(42)
    base_date = datetime(2025, 7, 1)

    cameras = [
        ("CAM-001", 44.43, -110.59),
        ("CAM-002", 44.46, -110.83),
        ("CAM-003", 44.55, -110.40),
    ]

    # Species with expected detection patterns.
    species_data = [
        # (species, count, typical_hours)
        ("white_tailed_deer", 30, [6, 7, 17, 18]),
        ("elk", 25, [7, 8, 16, 17]),
        ("black_bear", 10, [10, 11, 14, 15]),
        ("gray_wolf", 8, [5, 6, 20, 21]),
        ("coyote", 15, [22, 23, 0, 1, 2, 3]),
        ("red_fox", 10, [21, 22, 23, 0, 3, 4]),
        ("raccoon", 12, [22, 23, 0, 1, 2]),
        ("bald_eagle", 5, [9, 10, 11, 13, 14]),
        ("great_horned_owl", 4, [21, 22, 23, 0, 1]),
        ("snowshoe_hare", 8, [20, 21, 4, 5]),
    ]

    sightings: list[Sighting] = []

    for species, count, hours in species_data:
        for _ in range(count):
            cam_id, lat, lon = cameras[int(rng.integers(0, len(cameras)))]
            day = int(rng.integers(0, 30))
            hour = int(rng.choice(hours))
            minute = int(rng.integers(0, 60))

            ts = base_date + timedelta(days=day, hours=hour, minutes=minute)
            confidence = float(rng.beta(8, 2))

            sightings.append(
                Sighting(
                    species=species,
                    timestamp=ts,
                    camera_id=cam_id,
                    latitude=lat + float(rng.normal(0, 0.002)),
                    longitude=lon + float(rng.normal(0, 0.002)),
                    confidence=confidence,
                )
            )

    return sightings


@pytest.fixture
def single_species_sightings() -> list[Sighting]:
    """Sightings with only one species -- edge case for diversity metrics.

    Shannon index should be 0; evenness undefined for S=1.
    """
    base_date = datetime(2025, 7, 1)
    return [
        Sighting(
            species="elk",
            timestamp=base_date + timedelta(hours=i),
            camera_id="CAM-001",
            latitude=44.43,
            longitude=-110.59,
            confidence=0.95,
        )
        for i in range(10)
    ]


@pytest.fixture
def empty_sightings() -> list[Sighting]:
    """Empty sightings list -- edge case for all metrics."""
    return []


@pytest.fixture
def api_client():
    """FastAPI test client for API endpoint testing."""
    from fastapi.testclient import TestClient

    from wild_eye.api.main import app

    return TestClient(app)
