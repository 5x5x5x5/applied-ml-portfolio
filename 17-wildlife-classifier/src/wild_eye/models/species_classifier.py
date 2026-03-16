"""Wildlife species classifier built on MobileNetV3 for edge deployment.

Uses MobileNetV3-Large as the backbone for efficient inference on edge devices
(e.g., NVIDIA Jetson, Raspberry Pi with Coral TPU) while maintaining accuracy
sufficient for camera trap species identification.

The model supports multi-label classification to handle frames containing
multiple species (e.g., a deer and a coyote in the same trap image) and
includes temperature scaling for well-calibrated confidence outputs -- critical
for downstream ecological analyses that rely on detection probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from wild_eye import NUM_SPECIES, SPECIES_LABELS

logger = logging.getLogger(__name__)

# Standard ImageNet normalization used by MobileNetV3 pre-training.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Input resolution for MobileNetV3 (maintains pre-training resolution).
INPUT_SIZE = 224


@dataclass
class ClassificationResult:
    """Result of classifying a single camera trap image.

    Attributes:
        species: List of detected species labels above the confidence threshold.
        probabilities: Per-species sigmoid probabilities (multi-label).
        top_species: Highest-probability species label.
        top_confidence: Confidence score for the top species.
        is_empty: True if the frame is classified as empty (no animal).
        is_human: True if a human is detected (should be filtered).
        raw_logits: Raw model logits before temperature scaling.
    """

    species: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    top_species: str = "empty"
    top_confidence: float = 0.0
    is_empty: bool = True
    is_human: bool = False
    raw_logits: list[float] = field(default_factory=list)


class TemperatureScaler(nn.Module):
    """Learned temperature scaling for post-hoc calibration.

    Temperature scaling (Guo et al., 2017) is a single-parameter calibration
    method that divides logits by a learned temperature T before applying
    the sigmoid. This preserves the ranking of predictions while improving
    the reliability of the reported confidence values -- essential when
    occupancy models downstream treat detection probabilities as inputs.
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature parameter."""
        return logits / self.temperature


class WildEyeClassifier(nn.Module):
    """MobileNetV3-Large based multi-label wildlife species classifier.

    Architecture overview:
        1. MobileNetV3-Large backbone (pretrained on ImageNet)
        2. Global average pooling (from backbone)
        3. Custom classification head with dropout for regularization
        4. Temperature scaling layer for calibrated confidence
        5. Sigmoid activation for multi-label output

    The model is designed for camera trap images which present unique
    challenges: low light / IR illumination, motion blur, partial
    occlusion, and variable distance to subject.

    Args:
        num_classes: Number of species classes (including empty/human).
        dropout_rate: Dropout probability in the classification head.
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone_layers: Number of backbone layers to freeze for
            transfer learning. Set to -1 to freeze all except the head.
    """

    def __init__(
        self,
        num_classes: int = NUM_SPECIES,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        freeze_backbone_layers: int = -1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load MobileNetV3-Large backbone.
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        # Extract feature layers (everything before the classifier).
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-Large produces 960-dimensional features after pooling.
        backbone_out_features = 960

        # Custom classification head for wildlife species.
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Temperature scaling for calibrated confidence outputs.
        self.temperature_scaler = TemperatureScaler(initial_temperature=1.5)

        # Optionally freeze backbone layers for transfer learning.
        if freeze_backbone_layers != 0:
            self._freeze_backbone(freeze_backbone_layers)

        logger.info(
            "WildEyeClassifier initialized: %d classes, dropout=%.2f, pretrained=%s",
            num_classes,
            dropout_rate,
            pretrained,
        )

    def _freeze_backbone(self, num_layers: int) -> None:
        """Freeze backbone layers for transfer learning.

        Args:
            num_layers: Number of feature blocks to freeze.
                        -1 freezes all backbone layers.
        """
        layers_to_freeze = (
            list(self.features.children())
            if num_layers == -1
            else list(self.features.children())[:num_layers]
        )
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        frozen_count = sum(1 for p in self.features.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in self.features.parameters())
        logger.info("Froze %d/%d backbone parameters", frozen_count, total_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning temperature-scaled logits.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        features = self.features(x)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        scaled_logits = self.temperature_scaler(logits)
        return scaled_logits

    def predict(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.5,
    ) -> list[ClassificationResult]:
        """Run inference and return structured classification results.

        Uses sigmoid activation for multi-label prediction -- a single frame
        may contain multiple species (e.g., predator-prey interaction).

        Args:
            x: Preprocessed input tensor (batch, 3, 224, 224).
            confidence_threshold: Minimum sigmoid probability to consider
                a species as detected. Default 0.5 balances precision/recall;
                lower for higher recall in occupancy studies.

        Returns:
            List of ClassificationResult, one per image in the batch.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)

        results: list[ClassificationResult] = []
        for i in range(x.shape[0]):
            probs = probabilities[i].cpu().numpy()
            raw = logits[i].cpu().numpy()

            prob_dict = {SPECIES_LABELS[j]: float(probs[j]) for j in range(self.num_classes)}

            detected = [
                SPECIES_LABELS[j]
                for j in range(self.num_classes)
                if probs[j] >= confidence_threshold and SPECIES_LABELS[j] not in ("empty", "human")
            ]

            top_idx = int(np.argmax(probs))
            top_label = SPECIES_LABELS[top_idx]
            top_conf = float(probs[top_idx])

            result = ClassificationResult(
                species=detected,
                probabilities=prob_dict,
                top_species=top_label,
                top_confidence=top_conf,
                is_empty=prob_dict.get("empty", 0.0) >= confidence_threshold,
                is_human=prob_dict.get("human", 0.0) >= confidence_threshold,
                raw_logits=raw.tolist(),
            )
            results.append(result)

        return results


def get_inference_transform() -> transforms.Compose:
    """Standard preprocessing transform for inference.

    Matches the MobileNetV3 pre-training pipeline: resize, center crop,
    normalize to ImageNet statistics. Camera trap images are typically
    higher resolution, so we resize down to 256 then crop to 224.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_training_transform() -> transforms.Compose:
    """Data augmentation transform for training.

    Augmentations are chosen to simulate real camera trap variability:
    - RandomResizedCrop: animals at different distances
    - HorizontalFlip: cameras face both directions on trails
    - ColorJitter: varying light conditions (dawn/dusk/overcast)
    - RandomRotation: slight camera tilt from mounting
    - RandomGrayscale: simulate IR/night-vision images
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def export_to_onnx(
    model: WildEyeClassifier,
    output_path: str | Path,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export the classifier to ONNX format for edge deployment.

    ONNX export enables inference on devices without PyTorch, such as
    NVIDIA Jetson (TensorRT), Raspberry Pi (ONNX Runtime), or cloud
    Lambda functions where a full PyTorch installation is too heavy.

    Args:
        model: Trained WildEyeClassifier instance.
        output_path: Destination file path for the .onnx model.
        opset_version: ONNX opset version (17 recommended for MobileNetV3).
        dynamic_batch: Allow variable batch size at inference time.

    Returns:
        Path to the exported ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info("Exported ONNX model to %s", output_path)
    return output_path


def load_classifier(
    checkpoint_path: str | Path,
    device: str = "cpu",
    num_classes: int = NUM_SPECIES,
) -> WildEyeClassifier:
    """Load a trained classifier from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt or .pth checkpoint file.
        device: Target device ('cpu', 'cuda', 'mps').
        num_classes: Must match the checkpoint's class count.

    Returns:
        WildEyeClassifier with loaded weights in eval mode.
    """
    model = WildEyeClassifier(num_classes=num_classes, pretrained=False)
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info("Loaded classifier from %s on %s", checkpoint_path, device)
    return model
