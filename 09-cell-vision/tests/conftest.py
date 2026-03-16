"""Shared test fixtures for CellVision tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cell_vision import CELL_TYPES, NUM_CLASSES
from cell_vision.models.cell_classifier import CellClassifier, CellNet, CellNetResNet


@pytest.fixture
def sample_image_tensor() -> torch.Tensor:
    """Create a random image tensor shaped like a preprocessed microscopy image.

    Shape: (1, 3, 224, 224), normalized to ImageNet statistics.
    """
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_tensor() -> torch.Tensor:
    """Create a batch of random image tensors.

    Shape: (4, 3, 224, 224).
    """
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a synthetic RGB microscopy-like image.

    Simulates a simple cell on a lighter background with H&E-like colors.
    Shape: (256, 256, 3), uint8.
    """
    image = np.full((256, 256, 3), fill_value=220, dtype=np.uint8)

    # Simulate a pink/eosin background (cytoplasm-like)
    image[:, :, 0] = 230  # R
    image[:, :, 1] = 210  # G
    image[:, :, 2] = 215  # B

    # Draw a dark circular "cell" in the center (hematoxylin-stained nucleus)
    y, x = np.ogrid[-128:128, -128:128]
    mask = x**2 + y**2 <= 40**2
    image[mask, 0] = 80  # Dark purple/blue
    image[mask, 1] = 50
    image[mask, 2] = 120

    return image


@pytest.fixture
def sample_pil_image(sample_rgb_image: np.ndarray) -> Image.Image:
    """Create a PIL Image from the synthetic RGB array."""
    return Image.fromarray(sample_rgb_image)


@pytest.fixture
def cellnet_model() -> CellNet:
    """Create a CellNet model instance for testing."""
    return CellNet(num_classes=NUM_CLASSES, dropout_rate=0.4)


@pytest.fixture
def resnet_model() -> CellNetResNet:
    """Create a CellNetResNet model instance for testing (no pretrained weights)."""
    return CellNetResNet(num_classes=NUM_CLASSES, pretrained=False)


@pytest.fixture
def classifier() -> CellClassifier:
    """Create a CellClassifier instance using CellNet."""
    return CellClassifier(model_type="cellnet", device="cpu")


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic cell images for dataset testing.

    Creates 3 images per cell type in the expected directory structure.
    """
    for cell_type in CELL_TYPES:
        class_dir = tmp_path / cell_type
        class_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(class_dir / f"cell_{i:03d}.png")
    return tmp_path


@pytest.fixture
def tmp_model_path(tmp_path: Path) -> Path:
    """Return a temporary path for saving/loading model checkpoints."""
    return tmp_path / "test_model.pt"
