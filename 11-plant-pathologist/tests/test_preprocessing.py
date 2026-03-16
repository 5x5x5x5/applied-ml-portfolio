"""Tests for the leaf image preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from plant_pathologist.preprocessing.leaf_processor import (
    ColorAnalysis,
    ImageQualityReport,
    LeafProcessor,
    LeafSegmentationResult,
    LesionInfo,
    _rgb_to_hsv_array,
)


class TestImageQualityValidation:
    """Tests for image quality checks."""

    def test_valid_image_passes(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        report = processor.validate_image_quality(sample_rgb_image)
        assert isinstance(report, ImageQualityReport)
        assert report.width == 256
        assert report.height == 256

    def test_tiny_image_fails(self, processor: LeafProcessor, tiny_image: Image.Image) -> None:
        report = processor.validate_image_quality(tiny_image)
        assert not report.is_valid
        assert any("too small" in issue.lower() for issue in report.issues)

    def test_overexposed_image_detected(
        self, processor: LeafProcessor, overexposed_image: Image.Image
    ) -> None:
        report = processor.validate_image_quality(overexposed_image)
        assert report.is_overexposed
        assert report.mean_brightness > 240

    def test_blur_score_is_numeric(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        report = processor.validate_image_quality(sample_rgb_image)
        assert isinstance(report.blur_score, float)
        assert report.blur_score >= 0

    def test_quality_report_fields(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        report = processor.validate_image_quality(sample_rgb_image)
        assert isinstance(report.is_blurry, bool)
        assert isinstance(report.is_overexposed, bool)
        assert isinstance(report.is_underexposed, bool)
        assert isinstance(report.has_sufficient_contrast, bool)


class TestLeafSegmentation:
    """Tests for leaf segmentation from background."""

    def test_segmentation_returns_result(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.segment_leaf(sample_rgb_image)
        assert isinstance(result, LeafSegmentationResult)

    def test_segmentation_mask_shape(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.segment_leaf(sample_rgb_image)
        assert result.leaf_mask.shape == (256, 256)
        assert result.leaf_mask.dtype == bool

    def test_green_image_has_high_coverage(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.segment_leaf(sample_rgb_image)
        # Predominantly green image should have high leaf coverage
        assert result.leaf_coverage > 0.3
        assert result.background_removed

    def test_segmented_image_is_pil(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.segment_leaf(sample_rgb_image)
        assert isinstance(result.leaf_image, Image.Image)
        assert result.leaf_image.size == (256, 256)


class TestColorAnalysis:
    """Tests for color space discoloration detection."""

    def test_green_image_has_high_green_ratio(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        analysis = processor.analyze_colors(sample_rgb_image)
        assert isinstance(analysis, ColorAnalysis)
        # A predominantly green image should have high green ratio
        assert analysis.green_ratio > 0.3

    def test_color_analysis_with_mask(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        mask = np.ones((256, 256), dtype=bool)
        analysis = processor.analyze_colors(sample_rgb_image, mask=mask)
        assert analysis.green_ratio > 0

    def test_empty_mask_returns_zeros(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        mask = np.zeros((256, 256), dtype=bool)
        analysis = processor.analyze_colors(sample_rgb_image, mask=mask)
        assert analysis.green_ratio == 0.0
        assert analysis.dominant_hue == "unknown"

    def test_chlorosis_score_range(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        analysis = processor.analyze_colors(sample_rgb_image)
        assert 0.0 <= analysis.chlorosis_score <= 1.0

    def test_necrosis_score_range(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        analysis = processor.analyze_colors(sample_rgb_image)
        assert 0.0 <= analysis.necrosis_score <= 1.0

    def test_mean_rgb_values(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        analysis = processor.analyze_colors(sample_rgb_image)
        r, g, b = analysis.mean_rgb
        # Green image: green channel should dominate
        assert g > r
        assert g > b


class TestLesionDetection:
    """Tests for lesion detection and measurement."""

    def test_lesion_detection_returns_info(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.detect_lesions(sample_rgb_image)
        assert isinstance(result, LesionInfo)

    def test_healthy_leaf_low_lesion_area(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        segmentation = processor.segment_leaf(sample_rgb_image)
        result = processor.detect_lesions(sample_rgb_image, mask=segmentation.leaf_mask)
        # Healthy green image should have low lesion ratio
        assert result.severity_percentage < 50

    def test_diseased_image_detects_lesions(
        self, processor: LeafProcessor, sample_diseased_image: Image.Image
    ) -> None:
        result = processor.detect_lesions(sample_diseased_image)
        assert isinstance(result.lesion_count, int)
        assert result.lesion_count >= 0

    def test_lesion_color_profile(
        self, processor: LeafProcessor, sample_diseased_image: Image.Image
    ) -> None:
        result = processor.detect_lesions(sample_diseased_image)
        assert result.lesion_color_profile in (
            "dark_necrotic",
            "reddish_brown",
            "tan_brown",
            "mixed",
            "none",
        )

    def test_lesion_boolean_flags(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.detect_lesions(sample_rgb_image)
        assert isinstance(result.has_concentric_rings, bool)
        assert isinstance(result.has_halo, bool)

    def test_severity_percentage_range(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        result = processor.detect_lesions(sample_rgb_image)
        assert 0.0 <= result.severity_percentage <= 100.0


class TestRgbToHsv:
    """Tests for the pure-numpy RGB to HSV conversion."""

    def test_pure_red(self) -> None:
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        hsv = _rgb_to_hsv_array(rgb)
        # Red: H=0, S=255, V=255
        assert hsv[0, 0, 0] == 0  # Hue 0 (red)
        assert hsv[0, 0, 1] == 255  # Full saturation
        assert hsv[0, 0, 2] == 255  # Full value

    def test_pure_green(self) -> None:
        rgb = np.array([[[0, 255, 0]]], dtype=np.uint8)
        hsv = _rgb_to_hsv_array(rgb)
        # Green: H=60 (in 0-180 range)
        assert hsv[0, 0, 0] == 60
        assert hsv[0, 0, 1] == 255
        assert hsv[0, 0, 2] == 255

    def test_pure_black(self) -> None:
        rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)
        hsv = _rgb_to_hsv_array(rgb)
        assert hsv[0, 0, 2] == 0  # Value = 0

    def test_output_shape_matches_input(self) -> None:
        rgb = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        hsv = _rgb_to_hsv_array(rgb)
        assert hsv.shape == (100, 80, 3)
        assert hsv.dtype == np.uint8


class TestPreprocessForModel:
    """Tests for the model preprocessing utility."""

    def test_rgb_conversion(self, processor: LeafProcessor) -> None:
        # Create a grayscale image
        gray = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")
        result = processor.preprocess_for_model(gray)
        assert result.mode == "RGB"

    def test_large_image_downscaled(self, processor: LeafProcessor) -> None:
        large = Image.fromarray(np.zeros((5000, 5000, 3), dtype=np.uint8), mode="RGB")
        result = processor.preprocess_for_model(large)
        w, h = result.size
        assert max(w, h) <= 4096


class TestFullAnalysis:
    """Tests for the complete preprocessing pipeline."""

    def test_full_analysis_returns_tuple(
        self, processor: LeafProcessor, sample_rgb_image: Image.Image
    ) -> None:
        quality, segmentation, colors, lesions = processor.full_analysis(sample_rgb_image)
        assert isinstance(quality, ImageQualityReport)
        assert isinstance(segmentation, LeafSegmentationResult)
        assert isinstance(colors, ColorAnalysis)
        assert isinstance(lesions, LesionInfo)

    def test_full_analysis_diseased_image(
        self, processor: LeafProcessor, sample_diseased_image: Image.Image
    ) -> None:
        quality, segmentation, colors, lesions = processor.full_analysis(sample_diseased_image)
        assert quality.width == 256
        assert quality.height == 256
