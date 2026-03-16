"""Camera trap image processing pipeline.

Handles the unique challenges of camera trap imagery:
- Night-vision / infrared (IR) illumination with single-channel grayscale
- Motion blur from fast-moving animals triggering PIR sensors
- Wide-angle lens distortion at frame edges
- EXIF metadata extraction (timestamp, GPS, camera serial)
- Batch processing for SD cards containing thousands of images

Camera trap images differ significantly from standard photographic datasets.
IR images lack colour information, timestamps are embedded in EXIF or
burned into pixel data, and animals may occupy a small portion of the frame.
This module preprocesses raw trap images into classifier-ready tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)

# Common camera trap image extensions.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Motion blur detection threshold (variance of Laplacian).
BLUR_THRESHOLD = 100.0

# Minimum animal crop size as fraction of frame dimensions.
MIN_CROP_FRACTION = 0.05


@dataclass
class TrapImageMetadata:
    """Metadata extracted from a camera trap image file.

    Attributes:
        file_path: Original file path on disk.
        timestamp: Capture time from EXIF or file modification time.
        camera_id: Camera serial number or station identifier.
        latitude: GPS latitude in decimal degrees (WGS84).
        longitude: GPS longitude in decimal degrees (WGS84).
        altitude_m: GPS altitude in metres above sea level.
        is_infrared: Whether the image was captured with IR illumination.
        is_blurry: Whether motion blur was detected above threshold.
        blur_score: Variance of Laplacian (higher = sharper).
        width: Image width in pixels.
        height: Image height in pixels.
        file_size_bytes: File size on disk.
        extra: Additional EXIF tags of interest.
    """

    file_path: str = ""
    timestamp: datetime | None = None
    camera_id: str = "unknown"
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    is_infrared: bool = False
    is_blurry: bool = False
    blur_score: float = 0.0
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _dms_to_decimal(degrees: float, minutes: float, seconds: float, ref: str) -> float:
    """Convert GPS coordinates from DMS (degrees/minutes/seconds) to decimal degrees."""
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_exif_metadata(image_path: Path) -> TrapImageMetadata:
    """Extract EXIF metadata from a camera trap image.

    Attempts to read standard EXIF tags including GPS coordinates,
    capture timestamp, and camera make/model. Falls back to filesystem
    metadata when EXIF is unavailable (common with budget trail cameras).

    Args:
        image_path: Path to the image file.

    Returns:
        Populated TrapImageMetadata dataclass.
    """
    metadata = TrapImageMetadata(
        file_path=str(image_path),
        file_size_bytes=image_path.stat().st_size,
    )

    try:
        import exifread

        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Timestamp extraction.
        date_tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if date_tag:
            try:
                metadata.timestamp = datetime.strptime(str(date_tag), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                logger.warning("Could not parse EXIF date: %s", date_tag)

        # Camera identification.
        make = str(tags.get("Image Make", ""))
        model = str(tags.get("Image Model", ""))
        serial = str(tags.get("EXIF BodySerialNumber", ""))
        if serial:
            metadata.camera_id = serial
        elif make or model:
            metadata.camera_id = f"{make} {model}".strip()

        # GPS coordinate extraction.
        gps_lat = tags.get("GPS GPSLatitude")
        gps_lat_ref = tags.get("GPS GPSLatitudeRef")
        gps_lon = tags.get("GPS GPSLongitude")
        gps_lon_ref = tags.get("GPS GPSLongitudeRef")

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat_vals = [float(v.num) / float(v.den) for v in gps_lat.values]
            lon_vals = [float(v.num) / float(v.den) for v in gps_lon.values]
            metadata.latitude = _dms_to_decimal(*lat_vals, str(gps_lat_ref))
            metadata.longitude = _dms_to_decimal(*lon_vals, str(gps_lon_ref))

        gps_alt = tags.get("GPS GPSAltitude")
        if gps_alt:
            metadata.altitude_m = float(gps_alt.values[0].num) / float(gps_alt.values[0].den)

    except ImportError:
        logger.warning("exifread not installed; skipping EXIF extraction")
    except Exception:
        logger.exception("Failed to extract EXIF from %s", image_path)

    # Fall back to file modification time if EXIF timestamp missing.
    if metadata.timestamp is None:
        metadata.timestamp = datetime.fromtimestamp(image_path.stat().st_mtime)

    return metadata


def detect_infrared(image: Image.Image) -> bool:
    """Detect whether an image was captured with infrared illumination.

    IR camera trap images typically have very low colour saturation because
    the IR LEDs produce near-monochromatic illumination. We check the
    standard deviation of the per-channel means: a low spread indicates
    grayscale-like imagery characteristic of IR capture.

    Args:
        image: PIL Image in RGB mode.

    Returns:
        True if the image appears to be captured under IR illumination.
    """
    if image.mode != "RGB":
        return image.mode in ("L", "LA")

    stat = ImageStat.Stat(image)
    channel_means = stat.mean[:3]  # R, G, B means
    spread = np.std(channel_means)

    # IR images have very similar channel means (nearly grayscale).
    # Threshold determined empirically from Reconyx/Bushnell IR datasets.
    return float(spread) < 10.0


def compute_blur_score(image: Image.Image) -> float:
    """Compute a sharpness score using variance of the Laplacian.

    The Laplacian highlights edges; its variance measures overall edge
    strength. Blurry images (from animal motion or camera vibration)
    have low Laplacian variance. This metric is widely used in camera
    trap quality filtering pipelines (e.g., Norouzzadeh et al., 2018).

    Args:
        image: PIL Image (any mode).

    Returns:
        Laplacian variance score. Higher = sharper.
    """
    gray = image.convert("L")
    laplacian = gray.filter(
        ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1,
            offset=128,
        )
    )
    stat = ImageStat.Stat(laplacian)
    variance = stat.var[0]
    return float(variance)


def detect_and_crop_animal(
    image: Image.Image,
    padding_fraction: float = 0.15,
) -> Image.Image:
    """Localize the animal subject and crop to a tight bounding box.

    Uses a simple difference-from-background heuristic: camera trap images
    have a relatively static background, and the animal creates a region
    of high local contrast. We find the bounding box of the high-contrast
    region and add padding to ensure the full animal is captured.

    For production use, replace this with a dedicated object detector
    (e.g., MegaDetector v5 from Microsoft AI for Earth).

    Args:
        image: PIL Image in RGB mode.
        padding_fraction: Fraction of crop dimensions to add as padding.

    Returns:
        Cropped PIL Image containing the detected animal region.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    height, width = gray.shape

    # Edge detection to find high-contrast regions (animal body).
    from PIL import ImageFilter as _IF

    edges = image.convert("L").filter(_IF.FIND_EDGES)
    edge_array = np.array(edges, dtype=np.float32)

    # Threshold edges to find significant contours.
    threshold = np.percentile(edge_array, 90)
    mask = edge_array > threshold

    # Find bounding box of the thresholded region.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        logger.debug("No high-contrast region found; returning full image")
        return image

    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    # Check minimum crop size to avoid tiny noise regions.
    crop_h = y_max - y_min
    crop_w = x_max - x_min
    if crop_h < height * MIN_CROP_FRACTION or crop_w < width * MIN_CROP_FRACTION:
        logger.debug("Detected region too small; returning full image")
        return image

    # Add padding around the detected region.
    pad_y = int(crop_h * padding_fraction)
    pad_x = int(crop_w * padding_fraction)
    y_min = max(0, y_min - pad_y)
    y_max = min(height, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(width, x_max + pad_x)

    return image.crop((x_min, y_min, x_max, y_max))


def normalize_ir_image(image: Image.Image) -> Image.Image:
    """Convert an IR (infrared) image to a pseudo-RGB representation.

    Camera trap IR images are effectively single-channel. We perform
    histogram equalization for contrast enhancement (IR images are often
    low-contrast) and then replicate to 3 channels for model input
    compatibility.

    Args:
        image: PIL Image (may be grayscale or pseudo-colour IR).

    Returns:
        3-channel PIL Image with enhanced contrast.
    """
    gray = image.convert("L")

    # Histogram equalization to enhance contrast in dark IR images.
    hist = gray.histogram()
    nonzero = [h for h in hist if h > 0]
    if not nonzero:
        return image.convert("RGB")

    total_pixels = sum(hist)
    cdf = []
    cumulative = 0
    for count in hist:
        cumulative += count
        cdf.append(cumulative)

    cdf_min = min(c for c in cdf if c > 0)
    denominator = total_pixels - cdf_min
    if denominator <= 0:
        return image.convert("RGB")

    lut = [int(((c - cdf_min) / denominator) * 255.0) for c in cdf]

    equalized = gray.point(lut)
    return equalized.convert("RGB")


def process_single_image(
    image_path: Path,
    crop_animal: bool = True,
    filter_blurry: bool = True,
    blur_threshold: float = BLUR_THRESHOLD,
) -> tuple[Image.Image | None, TrapImageMetadata]:
    """Process a single camera trap image through the full pipeline.

    Steps:
        1. Load image and extract EXIF metadata
        2. Detect IR illumination and normalize if needed
        3. Assess motion blur quality
        4. Optionally detect and crop the animal subject
        5. Return processed image and metadata

    Args:
        image_path: Path to the image file.
        crop_animal: Whether to attempt animal detection and cropping.
        filter_blurry: Whether to reject blurry images.
        blur_threshold: Laplacian variance threshold for blur rejection.

    Returns:
        Tuple of (processed_image, metadata). Image is None if rejected.
    """
    metadata = extract_exif_metadata(image_path)

    try:
        image = Image.open(image_path)
        image.load()
    except Exception:
        logger.exception("Failed to load image: %s", image_path)
        return None, metadata

    metadata.width = image.width
    metadata.height = image.height

    # Detect and handle IR images.
    image_rgb = image.convert("RGB")
    metadata.is_infrared = detect_infrared(image_rgb)
    if metadata.is_infrared:
        image_rgb = normalize_ir_image(image)
        logger.debug("IR image detected and normalized: %s", image_path)

    # Motion blur assessment.
    metadata.blur_score = compute_blur_score(image_rgb)
    metadata.is_blurry = metadata.blur_score < blur_threshold

    if filter_blurry and metadata.is_blurry:
        logger.info(
            "Rejected blurry image (score=%.1f): %s",
            metadata.blur_score,
            image_path,
        )
        return None, metadata

    # Animal detection and cropping.
    if crop_animal:
        image_rgb = detect_and_crop_animal(image_rgb)

    return image_rgb, metadata


def batch_process_sd_card(
    input_directory: Path,
    output_directory: Path | None = None,
    crop_animal: bool = True,
    filter_blurry: bool = True,
    blur_threshold: float = BLUR_THRESHOLD,
) -> list[tuple[Path | None, TrapImageMetadata]]:
    """Batch process all images from a camera trap SD card.

    Scans the input directory recursively for supported image files,
    processes each through the full pipeline, and optionally saves
    processed images to an output directory preserving the folder structure.

    Trail cameras typically organize images in folders by date or
    camera event (e.g., DCIM/100RECNX/).

    Args:
        input_directory: Root directory of the SD card or image folder.
        output_directory: Where to save processed images. None = skip saving.
        crop_animal: Whether to crop to detected animal region.
        filter_blurry: Whether to filter out blurry images.
        blur_threshold: Laplacian variance threshold for blur detection.

    Returns:
        List of (output_path_or_None, metadata) tuples for each image found.
    """
    input_directory = Path(input_directory)
    results: list[tuple[Path | None, TrapImageMetadata]] = []

    image_files = sorted(
        f for f in input_directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Found %d images in %s", len(image_files), input_directory)

    for image_path in image_files:
        processed_image, metadata = process_single_image(
            image_path,
            crop_animal=crop_animal,
            filter_blurry=filter_blurry,
            blur_threshold=blur_threshold,
        )

        output_path: Path | None = None

        if processed_image is not None and output_directory is not None:
            relative = image_path.relative_to(input_directory)
            output_path = Path(output_directory) / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_image.save(str(output_path), quality=95)
            logger.debug("Saved processed image: %s", output_path)

        results.append((output_path, metadata))

    accepted = sum(1 for path, _ in results if path is not None)
    logger.info(
        "Batch processing complete: %d/%d images accepted",
        accepted,
        len(results),
    )
    return results
