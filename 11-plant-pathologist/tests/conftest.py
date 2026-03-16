"""Shared test fixtures for PlantPathologist tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plant_pathologist.models.disease_classifier import (
    PlantDiseaseClassifier,
)
from plant_pathologist.preprocessing.leaf_processor import LeafProcessor


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a synthetic 256x256 RGB image resembling a green leaf."""
    rng = np.random.default_rng(42)
    # Base green leaf color with slight variation
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = rng.integers(30, 80, size=(256, 256))  # R - low
    img[:, :, 1] = rng.integers(100, 200, size=(256, 256))  # G - high (green leaf)
    img[:, :, 2] = rng.integers(20, 70, size=(256, 256))  # B - low
    return Image.fromarray(img, mode="RGB")


@pytest.fixture
def sample_diseased_image() -> Image.Image:
    """Create a synthetic image with brown spots (simulating lesions)."""
    rng = np.random.default_rng(123)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Green background (healthy tissue)
    img[:, :, 0] = rng.integers(30, 70, size=(256, 256))
    img[:, :, 1] = rng.integers(120, 180, size=(256, 256))
    img[:, :, 2] = rng.integers(20, 60, size=(256, 256))

    # Add brown spots (lesions) in several locations
    lesion_positions = [(50, 50), (120, 80), (180, 150), (80, 200)]
    for cy, cx in lesion_positions:
        y, x = np.ogrid[-cy : 256 - cy, -cx : 256 - cx]
        mask = x**2 + y**2 <= 15**2
        img[mask, 0] = rng.integers(120, 160, size=int(np.sum(mask)))  # R - brown
        img[mask, 1] = rng.integers(60, 90, size=int(np.sum(mask)))  # G - low
        img[mask, 2] = rng.integers(20, 40, size=int(np.sum(mask)))  # B - low

    return Image.fromarray(img, mode="RGB")


@pytest.fixture
def tiny_image() -> Image.Image:
    """Create a very small image for quality validation testing."""
    return Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def overexposed_image() -> Image.Image:
    """Create an overexposed (very bright) image."""
    img = np.full((256, 256, 3), 250, dtype=np.uint8)
    return Image.fromarray(img, mode="RGB")


@pytest.fixture
def model() -> PlantDiseaseClassifier:
    """Create a PlantDiseaseClassifier in eval mode (random weights, no pretrained)."""
    m = PlantDiseaseClassifier(pretrained=False, freeze_backbone=False)
    m.eval()
    return m


@pytest.fixture
def processor() -> LeafProcessor:
    """Create a LeafProcessor with default settings."""
    return LeafProcessor()
