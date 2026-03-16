"""Tests for CellVision image preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cell_vision.preprocessing.image_processor import (
    CellSegmenter,
    MacenkoNormalizer,
    PatchExtractor,
    preprocess_image,
    remove_background,
)


class TestMacenkoNormalizer:
    """Tests for the Macenko stain normalization."""

    def test_default_initialization(self) -> None:
        """Normalizer should initialize with default H&E reference vectors."""
        normalizer = MacenkoNormalizer()
        assert normalizer.he_ref.shape == (3, 2)
        assert normalizer.max_conc_ref.shape == (2,)

    def test_rgb_to_od_conversion(self) -> None:
        """OD conversion: white (255) -> ~0.0 OD, dark pixels -> high OD."""
        normalizer = MacenkoNormalizer()
        # White pixel
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        od = normalizer._rgb_to_od(white)
        np.testing.assert_allclose(od, 0.0, atol=1e-6)

        # Dark pixel (near-zero) -> high OD
        dark = np.array([[[1, 1, 1]]], dtype=np.uint8)
        od = normalizer._rgb_to_od(dark)
        assert np.all(od > 2.0)

    def test_od_to_rgb_roundtrip(self) -> None:
        """RGB -> OD -> RGB should approximately recover original values."""
        normalizer = MacenkoNormalizer()
        original = np.array([[[128, 64, 200]]], dtype=np.uint8)
        od = normalizer._rgb_to_od(original)
        recovered = normalizer._od_to_rgb(od)
        np.testing.assert_allclose(recovered, original, atol=1)

    def test_normalize_output_shape(self, sample_rgb_image: np.ndarray) -> None:
        """Normalized image should have the same shape as input."""
        normalizer = MacenkoNormalizer()
        result = normalizer.normalize(sample_rgb_image)
        assert result.shape == sample_rgb_image.shape

    def test_normalize_output_dtype(self, sample_rgb_image: np.ndarray) -> None:
        """Normalized image should be uint8."""
        normalizer = MacenkoNormalizer()
        result = normalizer.normalize(sample_rgb_image)
        assert result.dtype == np.uint8

    def test_normalize_output_range(self, sample_rgb_image: np.ndarray) -> None:
        """Normalized image pixel values should be in [0, 255]."""
        normalizer = MacenkoNormalizer()
        result = normalizer.normalize(sample_rgb_image)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_normalize_with_reference_image(self, sample_rgb_image: np.ndarray) -> None:
        """Normalizer fitted to a reference should produce valid output."""
        normalizer = MacenkoNormalizer(reference_image=sample_rgb_image)
        result = normalizer.normalize(sample_rgb_image)
        assert result.shape == sample_rgb_image.shape
        assert result.dtype == np.uint8


class TestCellSegmenter:
    """Tests for Otsu thresholding and cell segmentation."""

    def test_otsu_threshold_range(self) -> None:
        """Otsu threshold should be between 0 and 255."""
        segmenter = CellSegmenter()
        # Bimodal histogram
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 50  # Dark region
        image[50:, :] = 200  # Light region
        threshold = segmenter.otsu_threshold(image)
        assert 0 <= threshold <= 255
        # Should be somewhere between the two modes
        assert 50 < threshold < 200

    def test_otsu_uniform_image(self) -> None:
        """Otsu on a uniform image should return a valid threshold."""
        segmenter = CellSegmenter()
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        threshold = segmenter.otsu_threshold(uniform)
        assert 0 <= threshold <= 255

    def test_segment_returns_binary_mask(self, sample_rgb_image: np.ndarray) -> None:
        """Segment should return a binary mask and list of cell regions."""
        segmenter = CellSegmenter(min_cell_area=50)
        binary, cells = segmenter.segment(sample_rgb_image)
        assert binary.shape == sample_rgb_image.shape[:2]
        assert set(np.unique(binary)).issubset({0, 1})
        assert isinstance(cells, list)

    def test_segment_finds_cells(self, sample_rgb_image: np.ndarray) -> None:
        """The synthetic image with a dark circle should yield at least one cell."""
        segmenter = CellSegmenter(min_cell_area=50, morphology_kernel_size=3)
        _, cells = segmenter.segment(sample_rgb_image)
        # The synthetic image has a dark circle (simulated cell)
        assert len(cells) >= 1

    def test_cell_region_structure(self, sample_rgb_image: np.ndarray) -> None:
        """Each detected cell region should have expected keys."""
        segmenter = CellSegmenter(min_cell_area=50, morphology_kernel_size=3)
        _, cells = segmenter.segment(sample_rgb_image)
        if cells:
            cell = cells[0]
            assert "bbox" in cell
            assert "area" in cell
            assert "centroid" in cell
            assert "crop" in cell
            assert len(cell["bbox"]) == 4
            assert cell["area"] > 0


class TestRemoveBackground:
    """Tests for background removal."""

    def test_output_shape(self, sample_rgb_image: np.ndarray) -> None:
        """Background removal should preserve image shape."""
        result = remove_background(sample_rgb_image)
        assert result.shape == sample_rgb_image.shape

    def test_background_pixels_white(self, sample_rgb_image: np.ndarray) -> None:
        """Background pixels should be set to white (255)."""
        result = remove_background(sample_rgb_image)
        # Check that some pixels are white (background)
        white_mask = np.all(result == 255, axis=2)
        assert white_mask.any(), "No background pixels were removed"

    def test_manual_threshold(self, sample_rgb_image: np.ndarray) -> None:
        """Manual threshold should be respected."""
        # Very low threshold -> most pixels classified as background
        result = remove_background(sample_rgb_image, threshold=10)
        white_mask = np.all(result == 255, axis=2)
        assert white_mask.sum() > sample_rgb_image.shape[0] * sample_rgb_image.shape[1] * 0.9


class TestPatchExtractor:
    """Tests for whole slide image patch extraction."""

    def test_extract_patches(self) -> None:
        """Should extract non-overlapping patches from a large image."""
        image = np.random.randint(0, 180, (512, 512, 3), dtype=np.uint8)
        extractor = PatchExtractor(patch_size=128, tissue_threshold=0.0)
        patches = extractor.extract(image)
        # (512 / 128)^2 = 16 patches
        assert len(patches) == 16
        assert patches[0]["patch"].shape == (128, 128, 3)

    def test_tissue_threshold_filters_patches(self) -> None:
        """Patches with insufficient tissue should be filtered out."""
        # Create image with tissue (dark) on left, background (white) on right
        image = np.full((256, 256, 3), 240, dtype=np.uint8)
        image[:, :128, :] = 50  # Dark "tissue" on left half
        extractor = PatchExtractor(patch_size=128, tissue_threshold=0.5)
        patches = extractor.extract(image)
        # Only left-half patches should pass the tissue threshold
        assert len(patches) == 2  # Two rows, left column

    def test_overlapping_patches(self) -> None:
        """Stride < patch_size should produce overlapping patches."""
        image = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        extractor = PatchExtractor(patch_size=128, stride=64, tissue_threshold=0.0)
        patches = extractor.extract(image)
        # ((256 - 128) / 64 + 1)^2 = 3^2 = 9
        assert len(patches) == 9

    def test_patch_position_info(self) -> None:
        """Each patch should include its (y, x) position in the original image."""
        image = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        extractor = PatchExtractor(patch_size=256, tissue_threshold=0.0)
        patches = extractor.extract(image)
        assert len(patches) == 1
        assert patches[0]["position"] == (0, 0)

    def test_save_patches(self, tmp_path: Path) -> None:
        """Patches should be saved to disk when output_dir is provided."""
        image = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        extractor = PatchExtractor(patch_size=128, tissue_threshold=0.0)
        patches = extractor.extract(image, output_dir=tmp_path / "patches")
        saved_files = list((tmp_path / "patches").glob("*.png"))
        assert len(saved_files) == len(patches)


class TestPreprocessImage:
    """Tests for the full preprocessing pipeline."""

    def test_preprocess_from_file(self, tmp_path: Path) -> None:
        """Full pipeline should produce a correctly-sized image from a file."""
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_path = tmp_path / "test_cell.png"
        img.save(img_path)

        result = preprocess_image(
            img_path, normalize_stain=False, remove_bg=False, target_size=(64, 64)
        )
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_preprocess_with_stain_normalization(self, tmp_path: Path) -> None:
        """Pipeline with stain normalization enabled should still produce valid output."""
        # Create an image with some color variation
        img_array = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_he.png"
        img.save(img_path)

        result = preprocess_image(
            img_path, normalize_stain=True, remove_bg=False, target_size=(224, 224)
        )
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8
