"""Image preprocessing for microscopy cell images.

Implements standard histopathology preprocessing methods:
    - Macenko stain normalization for H&E stained slides
    - Otsu thresholding for cell segmentation
    - Background removal
    - Patch extraction from whole slide images (WSI)

References:
    Macenko et al., "A method for normalizing histology slides for
    quantitative analysis", ISBI 2009.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MacenkoNormalizer:
    """Stain normalization using the Macenko method for H&E images.

    H&E (Hematoxylin and Eosin) staining is the most common stain in
    histopathology. Hematoxylin stains nuclei blue/purple; Eosin stains
    cytoplasm and extracellular matrix pink. Stain intensity varies
    significantly between labs, scanners, and slide preparation batches.

    The Macenko method decomposes the image into stain vectors in optical
    density (OD) space and normalizes them to a reference standard, making
    models robust to stain variation.

    Args:
        reference_image: Optional reference image to fit the target stain
            vectors. If None, uses default H&E reference vectors.
        luminosity_threshold: Minimum OD value to distinguish tissue from
            background in optical density space.
        percentile: Percentile for robust estimation of stain vectors
            (avoids outlier influence).
    """

    # Default H&E reference stain vectors (OD space)
    # These represent typical hematoxylin and eosin stain directions
    DEFAULT_HE_REF = np.array(
        [
            [0.5626, 0.2159],  # Hematoxylin (R, G, B in OD)
            [0.7201, 0.8012],
            [0.4062, 0.5581],
        ]
    )

    # Default maximum stain concentrations for normalization target
    DEFAULT_MAX_CONC = np.array([1.9705, 1.0308])

    def __init__(
        self,
        reference_image: np.ndarray | None = None,
        luminosity_threshold: float = 0.15,
        percentile: float = 99.0,
    ) -> None:
        self.luminosity_threshold = luminosity_threshold
        self.percentile = percentile

        if reference_image is not None:
            self.he_ref, self.max_conc_ref = self._extract_stain_vectors(reference_image)
            logger.info("Fitted Macenko normalizer to reference image")
        else:
            self.he_ref = self.DEFAULT_HE_REF.copy()
            self.max_conc_ref = self.DEFAULT_MAX_CONC.copy()
            logger.info("Using default H&E reference stain vectors")

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to Optical Density (OD) space.

        OD = -log10(I / I_0), where I_0 = 255 (max intensity).
        OD is the physically meaningful representation of stain absorption.

        Args:
            image: RGB image array, shape (H, W, 3), values in [0, 255].

        Returns:
            OD image array, same shape.
        """
        # Avoid log(0) by clamping minimum to 1
        image = np.maximum(image, 1).astype(np.float64)
        return -np.log10(image / 255.0)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert Optical Density back to RGB.

        Args:
            od: OD image array, shape (H, W, 3).

        Returns:
            RGB image array, uint8 in [0, 255].
        """
        rgb = 255.0 * np.power(10, -od)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _extract_stain_vectors(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract H&E stain vectors using SVD decomposition.

        The Macenko method:
        1. Convert to OD space
        2. Remove background (low OD) pixels
        3. Compute SVD of the OD data
        4. Project onto the plane of the two largest singular vectors
        5. Find the two extreme angles in this plane -> stain vectors

        Args:
            image: RGB image, shape (H, W, 3), uint8.

        Returns:
            Tuple of (stain_vectors, max_concentrations).
        """
        od = self._rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Threshold: keep tissue pixels (sufficient OD)
        od_mag = np.sqrt(np.sum(od_flat**2, axis=1))
        tissue_mask = od_mag > self.luminosity_threshold
        od_tissue = od_flat[tissue_mask]

        if od_tissue.shape[0] < 10:
            logger.warning(
                "Too few tissue pixels (%d); returning default stain vectors",
                od_tissue.shape[0],
            )
            return self.DEFAULT_HE_REF.copy(), self.DEFAULT_MAX_CONC.copy()

        # SVD to find the plane of maximal variance
        _, _, vt = np.linalg.svd(od_tissue, full_matrices=False)
        plane = vt[:2, :]  # Top 2 singular vectors

        # Project tissue OD values onto this plane
        projections = od_tissue @ plane.T

        # Find angles of each projected point
        angles = np.arctan2(projections[:, 1], projections[:, 0])

        # Robust estimation: use percentiles to find extreme angles
        min_angle = np.percentile(angles, 100 - self.percentile)
        max_angle = np.percentile(angles, self.percentile)

        # Reconstruct stain vectors from extreme angles
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Ensure hematoxylin is first (it absorbs more in the blue channel)
        if v1[0] > v2[0]:
            he_vectors = np.array([v1, v2]).T
        else:
            he_vectors = np.array([v2, v1]).T

        # Normalize stain vectors
        he_vectors = he_vectors / np.linalg.norm(he_vectors, axis=0)

        # Get concentrations and max values
        concentrations = od_tissue @ np.linalg.pinv(he_vectors)
        max_conc = np.percentile(concentrations, self.percentile, axis=0)

        return he_vectors, max_conc

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize the stain appearance of an H&E image.

        Maps the image's stain vectors and concentrations to the reference,
        producing consistent coloring regardless of the original staining.

        Args:
            image: Input RGB image, shape (H, W, 3), uint8.

        Returns:
            Stain-normalized RGB image, shape (H, W, 3), uint8.
        """
        h, w, _ = image.shape
        od = self._rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Extract source stain vectors
        he_source, max_conc_source = self._extract_stain_vectors(image)

        # Get source concentrations
        concentrations = od_flat @ np.linalg.pinv(he_source)

        # Normalize concentrations to reference range
        max_conc_source = np.maximum(max_conc_source, 1e-6)
        concentrations *= self.max_conc_ref / max_conc_source

        # Reconstruct OD with reference stain vectors
        od_normalized = concentrations @ self.he_ref.T
        od_normalized = od_normalized.reshape(h, w, 3)

        return self._od_to_rgb(od_normalized)


class CellSegmenter:
    """Segment individual cells from microscopy images using Otsu thresholding.

    Converts the image to grayscale, applies Otsu's method to find an optimal
    threshold separating foreground (cells) from background, and extracts
    connected components as individual cell regions.

    Args:
        min_cell_area: Minimum area in pixels for a valid cell region.
        max_cell_area: Maximum area in pixels for a valid cell region.
        morphology_kernel_size: Size of the kernel for morphological operations
            (erosion/dilation for noise removal).
    """

    def __init__(
        self,
        min_cell_area: int = 100,
        max_cell_area: int = 50000,
        morphology_kernel_size: int = 5,
    ) -> None:
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.morphology_kernel_size = morphology_kernel_size

    def otsu_threshold(self, grayscale: np.ndarray) -> int:
        """Compute Otsu's optimal threshold for bimodal histogram separation.

        Otsu's method finds the threshold that minimizes intra-class variance
        (equivalently, maximizes inter-class variance) between foreground
        and background.

        Args:
            grayscale: 2D grayscale image, values in [0, 255].

        Returns:
            Optimal threshold value.
        """
        histogram = np.zeros(256, dtype=np.int64)
        for val in grayscale.ravel():
            histogram[int(val)] += 1

        total_pixels = grayscale.size
        total_sum = np.sum(np.arange(256) * histogram)

        sum_background = 0.0
        weight_background = 0
        max_variance = 0.0
        best_threshold = 0

        for t in range(256):
            weight_background += histogram[t]
            if weight_background == 0:
                continue

            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * histogram[t]
            mean_background = sum_background / weight_background
            mean_foreground = (total_sum - sum_background) / weight_foreground

            # Inter-class variance
            variance = (
                weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            )

            if variance > max_variance:
                max_variance = variance
                best_threshold = t

        logger.debug("Otsu threshold: %d", best_threshold)
        return best_threshold

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Segment cells from a microscopy image.

        Steps:
        1. Convert to grayscale
        2. Apply Otsu thresholding
        3. Morphological cleanup (erosion then dilation)
        4. Connected component labeling
        5. Filter by area constraints

        Args:
            image: RGB image, shape (H, W, 3), uint8.

        Returns:
            Tuple of (binary mask, list of cell region dicts with bbox info).
        """
        # Convert to grayscale using luminance formula
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Otsu thresholding (invert because cells are darker than background)
        threshold = self.otsu_threshold(grayscale)
        binary = (grayscale < threshold).astype(np.uint8)

        # Morphological operations for noise cleanup
        binary = self._morphological_cleanup(binary)

        # Connected component labeling
        labels = self._connected_components(binary)

        # Extract cell regions
        cells = self._extract_regions(labels, image)

        logger.info("Segmented %d cells from image", len(cells))
        return binary, cells

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Apply erosion then dilation to remove small noise and smooth edges.

        Args:
            binary: Binary mask (0 or 1).

        Returns:
            Cleaned binary mask.
        """
        k = self.morphology_kernel_size

        # Erosion: shrink regions (removes noise)
        eroded = np.zeros_like(binary)
        pad = k // 2
        for i in range(pad, binary.shape[0] - pad):
            for j in range(pad, binary.shape[1] - pad):
                window = binary[i - pad : i + pad + 1, j - pad : j + pad + 1]
                eroded[i, j] = 1 if np.all(window) else 0

        # Dilation: grow regions back (restores cell boundaries)
        dilated = np.zeros_like(eroded)
        for i in range(pad, eroded.shape[0] - pad):
            for j in range(pad, eroded.shape[1] - pad):
                window = eroded[i - pad : i + pad + 1, j - pad : j + pad + 1]
                dilated[i, j] = 1 if np.any(window) else 0

        return dilated

    def _connected_components(self, binary: np.ndarray) -> np.ndarray:
        """Label connected components via flood-fill (4-connectivity).

        Args:
            binary: Binary mask.

        Returns:
            Label image where each connected component has a unique integer ID.
        """
        labels = np.zeros_like(binary, dtype=np.int32)
        current_label = 0

        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] == 1 and labels[i, j] == 0:
                    current_label += 1
                    # BFS flood fill
                    stack = [(i, j)]
                    while stack:
                        y, x = stack.pop()
                        if (
                            0 <= y < binary.shape[0]
                            and 0 <= x < binary.shape[1]
                            and binary[y, x] == 1
                            and labels[y, x] == 0
                        ):
                            labels[y, x] = current_label
                            stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])

        return labels

    def _extract_regions(self, labels: np.ndarray, image: np.ndarray) -> list[dict]:
        """Extract bounding boxes and crops for each labeled region.

        Args:
            labels: Connected component label image.
            image: Original RGB image.

        Returns:
            List of dicts with 'bbox', 'area', 'centroid', and 'crop' keys.
        """
        regions = []
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue

            mask = labels == label_id
            area = int(np.sum(mask))

            if area < self.min_cell_area or area > self.max_cell_area:
                continue

            # Bounding box
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y_min, y_max = int(rows[0]), int(rows[-1])
            x_min, x_max = int(cols[0]), int(cols[-1])

            # Centroid
            ys, xs = np.where(mask)
            centroid = (float(np.mean(ys)), float(np.mean(xs)))

            # Crop cell from image
            crop = image[y_min : y_max + 1, x_min : x_max + 1].copy()

            regions.append(
                {
                    "bbox": (y_min, x_min, y_max, x_max),
                    "area": area,
                    "centroid": centroid,
                    "crop": crop,
                }
            )

        return regions


def remove_background(image: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Remove background from a microscopy image.

    Sets background pixels to white (255, 255, 255), keeping only foreground
    cells. Useful for standardizing input images.

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        threshold: Manual threshold. If None, uses Otsu's method.

    Returns:
        Image with background replaced by white pixels.
    """
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    if threshold is None:
        segmenter = CellSegmenter()
        threshold = segmenter.otsu_threshold(grayscale)

    # Cells are darker than background
    foreground_mask = grayscale < threshold

    result = np.full_like(image, 255)
    result[foreground_mask] = image[foreground_mask]

    return result


class PatchExtractor:
    """Extract fixed-size patches from whole slide images (WSI).

    Whole slide images are extremely large (often >100,000 x 100,000 pixels),
    so they must be processed as smaller patches. This extractor generates
    non-overlapping or overlapping patches, filtering out patches that are
    predominantly background.

    Args:
        patch_size: Width and height of each square patch.
        stride: Step size between patches. If < patch_size, patches overlap.
        tissue_threshold: Minimum fraction of tissue pixels required for a
            patch to be kept (filters out mostly-blank patches).
        background_intensity: Pixels brighter than this are considered background.
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int | None = None,
        tissue_threshold: float = 0.3,
        background_intensity: int = 220,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.tissue_threshold = tissue_threshold
        self.background_intensity = background_intensity

    def extract(
        self,
        image: np.ndarray,
        output_dir: str | Path | None = None,
    ) -> list[dict]:
        """Extract patches from a large image.

        Args:
            image: RGB image array (potentially very large).
            output_dir: If provided, save patches as image files.

        Returns:
            List of dicts with 'patch' (array), 'position' (y, x), and
            'tissue_fraction' for each qualifying patch.
        """
        h, w = image.shape[:2]
        patches = []
        patch_idx = 0

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y : y + self.patch_size, x : x + self.patch_size]

                # Compute tissue fraction
                grayscale = np.dot(patch[..., :3], [0.2989, 0.5870, 0.1140])
                tissue_pixels = np.sum(grayscale < self.background_intensity)
                tissue_fraction = tissue_pixels / (self.patch_size**2)

                if tissue_fraction < self.tissue_threshold:
                    continue

                patch_info = {
                    "patch": patch,
                    "position": (y, x),
                    "tissue_fraction": float(tissue_fraction),
                }
                patches.append(patch_info)

                if output_dir is not None:
                    patch_img = Image.fromarray(patch)
                    patch_img.save(output_dir / f"patch_{patch_idx:05d}_y{y}_x{x}.png")
                    patch_idx += 1

        logger.info(
            "Extracted %d tissue patches from image (%d x %d)",
            len(patches),
            w,
            h,
        )
        return patches


def preprocess_image(
    image_path: str | Path,
    normalize_stain: bool = True,
    remove_bg: bool = True,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Full preprocessing pipeline for a single cell image.

    Steps:
    1. Load image
    2. Optionally normalize H&E stain (Macenko)
    3. Optionally remove background
    4. Resize to target dimensions

    Args:
        image_path: Path to input image.
        normalize_stain: Whether to apply Macenko stain normalization.
        remove_bg: Whether to remove background pixels.
        target_size: Output image dimensions (height, width).

    Returns:
        Preprocessed RGB image as uint8 numpy array.
    """
    image = np.array(Image.open(image_path).convert("RGB"))

    if normalize_stain:
        normalizer = MacenkoNormalizer()
        image = normalizer.normalize(image)
        logger.debug("Applied Macenko stain normalization")

    if remove_bg:
        image = remove_background(image)
        logger.debug("Removed background")

    # Resize
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    image = np.array(pil_image)

    return image
