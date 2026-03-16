"""Leaf image preprocessing pipeline for plant disease detection.

Handles leaf segmentation, color space analysis for discoloration detection,
lesion identification and measurement, and image quality validation. These
preprocessing steps improve classifier accuracy by isolating the leaf from
background noise and extracting handcrafted features that complement the
deep learning model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)

# Minimum acceptable image dimensions (pixels)
MIN_IMAGE_SIZE = 224
# Maximum dimension before downscaling
MAX_IMAGE_SIZE = 4096
# Minimum fraction of image that should be leaf (vs background)
MIN_LEAF_COVERAGE = 0.05
# Blur detection threshold (variance of Laplacian)
BLUR_THRESHOLD = 50.0
# Minimum brightness (0-255) to avoid underexposed images
MIN_BRIGHTNESS = 30
# Maximum brightness to avoid overexposed images
MAX_BRIGHTNESS = 240


@dataclass
class ImageQualityReport:
    """Results of image quality validation checks."""

    is_valid: bool
    width: int
    height: int
    is_blurry: bool
    blur_score: float
    mean_brightness: float
    is_overexposed: bool
    is_underexposed: bool
    has_sufficient_contrast: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class ColorAnalysis:
    """Color space analysis results for discoloration detection."""

    mean_rgb: tuple[float, float, float]
    mean_hsv: tuple[float, float, float]
    green_ratio: float
    brown_ratio: float
    yellow_ratio: float
    chlorosis_score: float
    necrosis_score: float
    dominant_hue: str


@dataclass
class LesionInfo:
    """Information about detected lesions on the leaf surface."""

    lesion_count: int
    total_lesion_area_ratio: float
    mean_lesion_size: float
    lesion_color_profile: str
    severity_percentage: float
    has_concentric_rings: bool
    has_halo: bool


@dataclass
class LeafSegmentationResult:
    """Result of leaf segmentation from background."""

    leaf_mask: np.ndarray
    leaf_image: Image.Image
    leaf_coverage: float
    background_removed: bool


class LeafProcessor:
    """Complete leaf image preprocessing pipeline.

    Processes raw photographs of plant leaves through segmentation,
    quality validation, color analysis, and lesion detection stages
    to prepare them for the disease classification model.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        green_range_hsv: tuple[tuple[int, int, int], tuple[int, int, int]] = (
            (25, 30, 30),
            (95, 255, 255),
        ),
    ) -> None:
        self.target_size = target_size
        self.green_lower = np.array(green_range_hsv[0], dtype=np.uint8)
        self.green_upper = np.array(green_range_hsv[1], dtype=np.uint8)

    def validate_image_quality(self, image: Image.Image) -> ImageQualityReport:
        """Check image quality for classification suitability.

        Evaluates resolution, blur, exposure, and contrast to determine
        whether the image meets minimum standards for reliable disease
        classification.

        Args:
            image: Input PIL Image.

        Returns:
            ImageQualityReport with detailed quality metrics.
        """
        issues: list[str] = []
        width, height = image.size

        # Resolution check
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            issues.append(
                f"Image too small ({width}x{height}). "
                f"Minimum {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} required."
            )

        # Convert to grayscale for blur and exposure analysis
        gray = image.convert("L")
        gray_array = np.array(gray, dtype=np.float64)

        # Blur detection via variance of the Laplacian approximation
        # Uses the difference between a blurred and original image
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
        blurred_array = np.array(blurred, dtype=np.float64)
        laplacian_var = float(np.var(gray_array - blurred_array))
        is_blurry = laplacian_var < BLUR_THRESHOLD
        if is_blurry:
            issues.append(
                f"Image appears blurry (sharpness score: {laplacian_var:.1f}, "
                f"threshold: {BLUR_THRESHOLD})."
            )

        # Exposure analysis
        stat = ImageStat.Stat(gray)
        mean_brightness = stat.mean[0]
        is_underexposed = mean_brightness < MIN_BRIGHTNESS
        is_overexposed = mean_brightness > MAX_BRIGHTNESS

        if is_underexposed:
            issues.append(
                f"Image is underexposed (brightness: {mean_brightness:.0f}). Try better lighting."
            )
        if is_overexposed:
            issues.append(
                f"Image is overexposed (brightness: {mean_brightness:.0f}). Reduce direct light."
            )

        # Contrast check via standard deviation
        contrast = stat.stddev[0]
        has_sufficient_contrast = contrast > 20.0
        if not has_sufficient_contrast:
            issues.append(
                f"Low contrast image (stddev: {contrast:.1f}). Ensure leaf is clearly visible."
            )

        is_valid = len(issues) == 0

        return ImageQualityReport(
            is_valid=is_valid,
            width=width,
            height=height,
            is_blurry=is_blurry,
            blur_score=laplacian_var,
            mean_brightness=mean_brightness,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            has_sufficient_contrast=has_sufficient_contrast,
            issues=issues,
        )

    def segment_leaf(self, image: Image.Image) -> LeafSegmentationResult:
        """Segment leaf from background using color-based thresholding.

        Uses HSV color space to identify green leaf tissue and create
        a binary mask. The mask is cleaned with morphological operations
        (approximated via PIL filters) to remove noise.

        Args:
            image: Input RGB PIL Image.

        Returns:
            LeafSegmentationResult with mask, cropped image, and coverage.
        """
        rgb_array = np.array(image.convert("RGB"))

        # Convert RGB to HSV manually (PIL doesn't have direct HSV conversion
        # that returns arrays, and we avoid opencv dependency)
        hsv_array = _rgb_to_hsv_array(rgb_array)

        # Create mask for green-ish pixels (leaf tissue)
        h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]

        # Broad green range to capture healthy and diseased tissue
        green_mask = (
            (h >= self.green_lower[0])
            & (h <= self.green_upper[0])
            & (s >= self.green_lower[1])
            & (v >= self.green_lower[2])
        )

        # Also include brown/yellow tissue (diseased areas)
        brown_mask = (h >= 5) & (h <= 30) & (s >= 40) & (v >= 30)
        yellow_mask = (h >= 20) & (h <= 45) & (s >= 50) & (v >= 80)

        combined_mask = green_mask | brown_mask | yellow_mask

        # Clean mask with morphological-like operations using PIL
        mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")
        # Dilate then erode to fill small gaps (closing operation)
        mask_image = mask_image.filter(ImageFilter.MaxFilter(5))
        mask_image = mask_image.filter(ImageFilter.MinFilter(5))
        # Remove small noise (opening operation)
        mask_image = mask_image.filter(ImageFilter.MinFilter(3))
        mask_image = mask_image.filter(ImageFilter.MaxFilter(3))

        final_mask = np.array(mask_image) > 127

        # Calculate leaf coverage
        leaf_coverage = float(np.sum(final_mask)) / final_mask.size

        # Apply mask to original image
        result_array = rgb_array.copy()
        result_array[~final_mask] = [0, 0, 0]  # Black background
        leaf_image = Image.fromarray(result_array)

        background_removed = leaf_coverage >= MIN_LEAF_COVERAGE

        if leaf_coverage < MIN_LEAF_COVERAGE:
            logger.warning(
                "Low leaf coverage (%.1f%%). Segmentation may have failed.",
                leaf_coverage * 100,
            )

        return LeafSegmentationResult(
            leaf_mask=final_mask,
            leaf_image=leaf_image,
            leaf_coverage=leaf_coverage,
            background_removed=background_removed,
        )

    def analyze_colors(self, image: Image.Image, mask: np.ndarray | None = None) -> ColorAnalysis:
        """Analyze color distribution to detect discoloration patterns.

        Examines color ratios in the leaf region to identify symptoms like:
        - Chlorosis (yellowing from chlorophyll loss)
        - Necrosis (browning from cell death)
        - Unusual pigmentation from viral or nutrient deficiency

        Args:
            image: Input RGB PIL Image.
            mask: Optional binary mask of leaf region. If None, uses entire image.

        Returns:
            ColorAnalysis with color ratios and symptom scores.
        """
        rgb_array = np.array(image.convert("RGB"), dtype=np.float64)
        hsv_array = _rgb_to_hsv_array(np.array(image.convert("RGB")))

        if mask is not None:
            # Only analyze pixels within the leaf mask
            leaf_pixels_rgb = rgb_array[mask]
            leaf_pixels_hsv = hsv_array[mask]
        else:
            leaf_pixels_rgb = rgb_array.reshape(-1, 3)
            leaf_pixels_hsv = hsv_array.reshape(-1, 3)

        if len(leaf_pixels_rgb) == 0:
            return ColorAnalysis(
                mean_rgb=(0.0, 0.0, 0.0),
                mean_hsv=(0.0, 0.0, 0.0),
                green_ratio=0.0,
                brown_ratio=0.0,
                yellow_ratio=0.0,
                chlorosis_score=0.0,
                necrosis_score=0.0,
                dominant_hue="unknown",
            )

        mean_r, mean_g, mean_b = (
            float(np.mean(leaf_pixels_rgb[:, 0])),
            float(np.mean(leaf_pixels_rgb[:, 1])),
            float(np.mean(leaf_pixels_rgb[:, 2])),
        )
        mean_h, mean_s, mean_v = (
            float(np.mean(leaf_pixels_hsv[:, 0])),
            float(np.mean(leaf_pixels_hsv[:, 1])),
            float(np.mean(leaf_pixels_hsv[:, 2])),
        )

        hues = leaf_pixels_hsv[:, 0]
        sats = leaf_pixels_hsv[:, 1]
        vals = leaf_pixels_hsv[:, 2]

        total_pixels = len(hues)

        # Green pixels: hue 35-85, moderate-high saturation
        green_count = int(np.sum((hues >= 35) & (hues <= 85) & (sats >= 40)))
        green_ratio = green_count / total_pixels

        # Brown pixels: hue 5-25, lower saturation
        brown_count = int(np.sum((hues >= 5) & (hues <= 25) & (sats >= 30) & (vals >= 20)))
        brown_ratio = brown_count / total_pixels

        # Yellow pixels: hue 20-40, high saturation and value
        yellow_count = int(np.sum((hues >= 20) & (hues <= 40) & (sats >= 60) & (vals >= 100)))
        yellow_ratio = yellow_count / total_pixels

        # Chlorosis score: proportion of yellowing (loss of chlorophyll)
        # High yellow ratio + low green ratio indicates chlorosis
        chlorosis_score = min(1.0, yellow_ratio * 2.0 + max(0, 0.5 - green_ratio))

        # Necrosis score: proportion of dead (brown/black) tissue
        dark_count = int(np.sum(vals < 50))
        dark_ratio = dark_count / total_pixels
        necrosis_score = min(1.0, brown_ratio + dark_ratio)

        # Determine dominant hue category
        if green_ratio > 0.5:
            dominant_hue = "green"
        elif yellow_ratio > brown_ratio and yellow_ratio > green_ratio:
            dominant_hue = "yellow"
        elif brown_ratio > green_ratio:
            dominant_hue = "brown"
        else:
            dominant_hue = "mixed"

        return ColorAnalysis(
            mean_rgb=(mean_r, mean_g, mean_b),
            mean_hsv=(mean_h, mean_s, mean_v),
            green_ratio=green_ratio,
            brown_ratio=brown_ratio,
            yellow_ratio=yellow_ratio,
            chlorosis_score=chlorosis_score,
            necrosis_score=necrosis_score,
            dominant_hue=dominant_hue,
        )

    def detect_lesions(self, image: Image.Image, mask: np.ndarray | None = None) -> LesionInfo:
        """Detect and measure lesions (diseased spots) on the leaf.

        Identifies areas of non-green discoloration that indicate disease
        lesions. Analyzes lesion characteristics including size, color,
        presence of concentric rings (target spots), and halos.

        Args:
            image: Input RGB PIL Image.
            mask: Optional binary mask of leaf region.

        Returns:
            LesionInfo with lesion count, sizes, and characteristics.
        """
        rgb_array = np.array(image.convert("RGB"))
        hsv_array = _rgb_to_hsv_array(rgb_array)

        h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]

        # Lesions are typically brown, dark brown, or black spots on the leaf
        # that fall outside the healthy green hue range
        lesion_mask = (
            ((h < 25) | (h > 85))  # Non-green hue
            & (s > 20)  # Some saturation (not pure gray)
            & (v > 15)  # Not completely black
            & (v < 200)  # Not white/specular highlight
        )

        if mask is not None:
            lesion_mask = lesion_mask & mask

        # Simple connected-component-like analysis using region scanning
        # (avoiding scipy/opencv dependency)
        total_leaf_pixels = (
            int(np.sum(mask)) if mask is not None else rgb_array.shape[0] * rgb_array.shape[1]
        )
        total_lesion_pixels = int(np.sum(lesion_mask))

        if total_leaf_pixels == 0:
            total_leaf_pixels = 1  # Avoid division by zero

        lesion_area_ratio = total_lesion_pixels / total_leaf_pixels

        # Estimate lesion count using a grid-based approach
        # Divide image into blocks and count blocks with significant lesion presence
        block_size = 32
        h_blocks = max(1, rgb_array.shape[0] // block_size)
        w_blocks = max(1, rgb_array.shape[1] // block_size)
        lesion_block_count = 0

        for bi in range(h_blocks):
            for bj in range(w_blocks):
                block = lesion_mask[
                    bi * block_size : (bi + 1) * block_size,
                    bj * block_size : (bj + 1) * block_size,
                ]
                if block.size > 0 and np.mean(block) > 0.3:
                    lesion_block_count += 1

        # Approximate individual lesion count (blocks can be parts of same lesion)
        estimated_lesion_count = max(0, lesion_block_count)

        mean_lesion_size = total_lesion_pixels / max(1, estimated_lesion_count)

        # Determine lesion color profile
        if total_lesion_pixels > 0:
            lesion_hues = h[lesion_mask]
            lesion_vals = v[lesion_mask]
            mean_lesion_hue = float(np.mean(lesion_hues))
            mean_lesion_val = float(np.mean(lesion_vals))

            if mean_lesion_val < 60:
                lesion_color_profile = "dark_necrotic"
            elif mean_lesion_hue < 15:
                lesion_color_profile = "reddish_brown"
            elif mean_lesion_hue < 30:
                lesion_color_profile = "tan_brown"
            else:
                lesion_color_profile = "mixed"
        else:
            lesion_color_profile = "none"

        # Severity based on affected area percentage (Horsfall-Barratt scale inspired)
        severity_percentage = min(100.0, lesion_area_ratio * 100)

        # Detect concentric rings (characteristic of target spot / early blight)
        has_concentric_rings = _detect_ring_pattern(v, lesion_mask)

        # Detect yellow halo around lesions (bacterial spots often have halos)
        has_halo = _detect_halo(hsv_array, lesion_mask)

        return LesionInfo(
            lesion_count=estimated_lesion_count,
            total_lesion_area_ratio=lesion_area_ratio,
            mean_lesion_size=mean_lesion_size,
            lesion_color_profile=lesion_color_profile,
            severity_percentage=severity_percentage,
            has_concentric_rings=has_concentric_rings,
            has_halo=has_halo,
        )

    def preprocess_for_model(self, image: Image.Image) -> Image.Image:
        """Prepare an image for model inference.

        Performs downscaling if needed and ensures RGB mode.

        Args:
            image: Raw input image.

        Returns:
            Preprocessed PIL Image ready for the inference transform.
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale if too large
        w, h = image.size
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info("Downscaled image from %dx%d to %dx%d", w, h, *new_size)

        return image

    def full_analysis(
        self, image: Image.Image
    ) -> tuple[ImageQualityReport, LeafSegmentationResult, ColorAnalysis, LesionInfo]:
        """Run the complete preprocessing and analysis pipeline.

        Args:
            image: Raw input photograph of a plant leaf.

        Returns:
            Tuple of (quality_report, segmentation, color_analysis, lesion_info).
        """
        image = self.preprocess_for_model(image)
        quality = self.validate_image_quality(image)
        segmentation = self.segment_leaf(image)
        colors = self.analyze_colors(image, segmentation.leaf_mask)
        lesions = self.detect_lesions(image, segmentation.leaf_mask)

        logger.info(
            "Full analysis: quality_valid=%s, leaf_coverage=%.1f%%, "
            "green_ratio=%.2f, lesion_count=%d, severity=%.1f%%",
            quality.is_valid,
            segmentation.leaf_coverage * 100,
            colors.green_ratio,
            lesions.lesion_count,
            lesions.severity_percentage,
        )

        return quality, segmentation, colors, lesions


def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB numpy array (uint8) to HSV.

    H is in range [0, 180] (OpenCV convention), S and V in [0, 255].
    Pure numpy implementation to avoid opencv dependency.

    Args:
        rgb: Array of shape (H, W, 3) with dtype uint8.

    Returns:
        HSV array of same shape and dtype.
    """
    rgb_float = rgb.astype(np.float64) / 255.0
    r, g, b = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue calculation
    h = np.zeros_like(cmax)
    # When cmax == r
    mask_r = (cmax == r) & (diff > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    # When cmax == g
    mask_g = (cmax == g) & (diff > 0)
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    # When cmax == b
    mask_b = (cmax == b) & (diff > 0)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # Scale hue to [0, 180] like OpenCV
    h = h / 2.0

    # Saturation
    s = np.zeros_like(cmax)
    s[cmax > 0] = (diff[cmax > 0] / cmax[cmax > 0]) * 255.0

    # Value
    v = cmax * 255.0

    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    return hsv


def _detect_ring_pattern(value_channel: np.ndarray, lesion_mask: np.ndarray) -> bool:
    """Detect concentric ring patterns within lesions.

    Target spot (Corynespora cassiicola) and early blight (Alternaria solani)
    produce characteristic concentric rings of alternating light and dark tissue.

    This is a simplified detection using radial intensity variation analysis.
    """
    if np.sum(lesion_mask) < 100:
        return False

    # Sample intensity values along rows within lesion regions
    lesion_rows = np.any(lesion_mask, axis=1)
    row_indices = np.where(lesion_rows)[0]

    if len(row_indices) < 10:
        return False

    # Check for alternating intensity patterns in lesion rows
    alternation_count = 0
    for row_idx in row_indices[::3]:  # Sample every 3rd row
        row_vals = value_channel[row_idx][lesion_mask[row_idx]]
        if len(row_vals) < 5:
            continue
        # Count sign changes in the gradient
        gradient = np.diff(row_vals.astype(np.float64))
        sign_changes = np.sum(np.abs(np.diff(np.sign(gradient))) > 0)
        if sign_changes > len(row_vals) * 0.3:
            alternation_count += 1

    # If many rows show alternating pattern, likely concentric rings
    return alternation_count > len(row_indices) * 0.15


def _detect_halo(hsv_array: np.ndarray, lesion_mask: np.ndarray) -> bool:
    """Detect yellow/light halos surrounding lesions.

    Bacterial diseases like bacterial spot often produce a chlorotic
    (yellow) halo around the necrotic lesion center.
    """
    if np.sum(lesion_mask) < 50:
        return False

    # Dilate the lesion mask to get the border region
    lesion_img = Image.fromarray((lesion_mask * 255).astype(np.uint8), mode="L")
    dilated = lesion_img.filter(ImageFilter.MaxFilter(7))
    dilated_mask = np.array(dilated) > 127

    # Border region = dilated minus original
    border_mask = dilated_mask & ~lesion_mask

    if np.sum(border_mask) < 20:
        return False

    # Check if border region has yellow-ish hue
    border_hues = hsv_array[:, :, 0][border_mask]
    border_sats = hsv_array[:, :, 1][border_mask]

    # Yellow hue range: 20-40 in our HSV scale
    yellow_border = np.sum((border_hues >= 20) & (border_hues <= 40) & (border_sats >= 50))
    yellow_ratio = yellow_border / len(border_hues)

    return yellow_ratio > 0.3
