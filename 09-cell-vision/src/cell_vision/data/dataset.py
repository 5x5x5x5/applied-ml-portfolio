"""Custom PyTorch Dataset for microscopy cell images.

Supports loading cell images organized in class-based directory structures,
with augmentation pipelines tuned for microscopy data (rotation-invariant
cells, stain variation, etc.) and class-balanced weighted sampling.

Expected directory layout:
    data_root/
        red_blood_cell/
            img_001.png
            img_002.png
        neutrophil/
            img_003.png
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from cell_vision import CELL_TYPES

logger = logging.getLogger(__name__)

# ImageNet normalization (used for transfer learning compatibility)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ElasticDeformation:
    """Apply random elastic deformation to simulate tissue distortion.

    Microscopy samples often have slight deformations from slide preparation.
    This augmentation simulates those effects by applying a random displacement
    field to the image.

    Args:
        alpha: Intensity of the displacement field.
        sigma: Smoothness of the displacement field (Gaussian kernel sigma).
        p: Probability of applying the transform.
    """

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.3) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img

        img_array = np.array(img, dtype=np.float32)
        shape = img_array.shape[:2]

        # Generate random displacement fields
        dx = np.random.randn(*shape).astype(np.float32)
        dy = np.random.randn(*shape).astype(np.float32)

        # Smooth with Gaussian filter (approximated via repeated box blur)
        from PIL import ImageFilter

        for field in [dx, dy]:
            field_img = Image.fromarray(field, mode="F")
            field_img = field_img.filter(ImageFilter.GaussianBlur(radius=self.sigma))
            field[:] = np.array(field_img) * self.alpha

        # Create coordinate grids
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        map_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.intp)
        map_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.intp)

        # Apply displacement
        if img_array.ndim == 3:
            result = img_array[map_y, map_x, :]
        else:
            result = img_array[map_y, map_x]

        return Image.fromarray(result.astype(np.uint8))


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentation pipeline for training microscopy images.

    Includes:
        - Random rotation (cells are orientation-invariant)
        - Random horizontal and vertical flips
        - Color jitter (simulates stain variation between labs/batches)
        - Elastic deformation (simulates tissue distortion)
        - Normalization to ImageNet statistics

    Args:
        image_size: Target image dimensions (square).

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=360),  # Cells have no preferred orientation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.3,  # Higher for H&E stain variation
                hue=0.05,
            ),
            ElasticDeformation(alpha=50.0, sigma=5.0, p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation/test (no augmentation).

    Args:
        image_size: Target image dimensions.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CellDataset(Dataset):
    """PyTorch Dataset for microscopy cell images.

    Loads images from a directory structure where each subdirectory is named
    after the cell type it contains. Supports train/val/test splits and
    class-balanced sampling.

    Args:
        root_dir: Path to dataset root directory.
        transform: Optional torchvision transforms to apply.
        cell_types: List of cell type names (subdirectory names).
        extensions: Allowed image file extensions.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
        cell_types: list[str] | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cell_types = cell_types or CELL_TYPES
        self.extensions = extensions

        self.samples: list[tuple[Path, int]] = []
        self.class_counts: dict[int, int] = {}
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory tree and build list of (image_path, class_index) pairs."""
        for class_idx, cell_type in enumerate(self.cell_types):
            class_dir = self.root_dir / cell_type
            if not class_dir.is_dir():
                logger.warning("Directory not found for class '%s': %s", cell_type, class_dir)
                continue

            count = 0
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, class_idx))
                    count += 1

            self.class_counts[class_idx] = count
            logger.info("Class '%s': %d images", cell_type, count)

        logger.info(
            "Total samples: %d across %d classes", len(self.samples), len(self.class_counts)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load as RGB (microscopy images may be RGBA or grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing.

        Useful for handling class imbalance common in blood cell datasets
        (e.g., RBCs vastly outnumber basophils in typical samples).

        Returns:
            Tensor of per-class weights, inversely proportional to frequency.
        """
        total = sum(self.class_counts.values())
        weights = []
        for i in range(len(self.cell_types)):
            count = self.class_counts.get(i, 1)  # avoid division by zero
            weights.append(total / (len(self.cell_types) * count))
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> list[float]:
        """Compute per-sample weights for WeightedRandomSampler.

        Each sample receives a weight inversely proportional to its class
        frequency, ensuring balanced mini-batches during training.

        Returns:
            List of float weights, one per sample.
        """
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for _, label in self.samples]


def create_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Create train/val/test data loaders with class-balanced sampling.

    Splits the full dataset into train, validation, and test subsets, then
    wraps each in a DataLoader. The training loader uses WeightedRandomSampler
    for class-balanced batches.

    Args:
        data_dir: Root directory containing per-class image subdirectories.
        batch_size: Number of images per batch.
        image_size: Target image size (square).
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        num_workers: Parallel data loading workers.
        seed: Random seed for reproducible splits.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and 'class_weights'.
    """
    # Load full dataset without transforms first (for splitting)
    full_dataset = CellDataset(root_dir=data_dir)

    # Compute split sizes
    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply appropriate transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Wrap subsets with transforms
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)
    test_dataset = TransformSubset(test_subset, val_transform)

    # Weighted sampler for class-balanced training
    sample_weights = full_dataset.get_sample_weights()
    train_weights = [sample_weights[i] for i in train_subset.indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = full_dataset.get_class_weights()
    logger.info(
        "Data loaders created: train=%d, val=%d, test=%d",
        train_size,
        val_size,
        test_size,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "num_classes": len(full_dataset.cell_types),
    }


class TransformSubset(Dataset):
    """Wraps a torch Subset with a specific transform pipeline.

    Since random_split returns Subsets that share the parent dataset's
    transform, this wrapper allows applying different transforms to
    train vs. val/test subsets.

    Args:
        subset: A torch Subset from random_split.
        transform: Transform pipeline to apply to images.
    """

    def __init__(
        self,
        subset: torch.utils.data.Subset,
        transform: transforms.Compose,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
