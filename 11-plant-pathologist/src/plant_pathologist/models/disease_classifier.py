"""EfficientNet-B0 based plant disease classifier with multi-task learning.

Performs joint classification of plant species and disease type using
transfer learning. Includes temperature-scaled confidence calibration
for reliable probability estimates in agricultural decision-making.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class PlantSpecies(str, Enum):
    """Supported plant species for disease detection."""

    TOMATO = "tomato"
    POTATO = "potato"
    CORN = "corn"
    APPLE = "apple"
    GRAPE = "grape"


# Complete mapping of disease classes used by the classifier.
# Index order matches the model output logits.
DISEASE_CLASSES: list[str] = [
    # Tomato diseases
    "tomato_healthy",
    "tomato_early_blight",  # Alternaria solani
    "tomato_late_blight",  # Phytophthora infestans
    "tomato_bacterial_spot",  # Xanthomonas campestris pv. vesicatoria
    "tomato_septoria_leaf_spot",  # Septoria lycopersici
    "tomato_leaf_mold",  # Passalora fulva (syn. Cladosporium fulvum)
    "tomato_yellow_leaf_curl_virus",  # Begomovirus (TYLCV)
    "tomato_target_spot",  # Corynespora cassiicola
    # Potato diseases
    "potato_healthy",
    "potato_early_blight",  # Alternaria solani
    "potato_late_blight",  # Phytophthora infestans
    # Corn diseases
    "corn_healthy",
    "corn_northern_leaf_blight",  # Exserohilum turcicum
    "corn_common_rust",  # Puccinia sorghi
    "corn_gray_leaf_spot",  # Cercospora zeae-maydis
    # Apple diseases
    "apple_healthy",
    "apple_scab",  # Venturia inaequalis
    "apple_black_rot",  # Botryosphaeria obtusa
    "apple_cedar_apple_rust",  # Gymnosporangium juniperi-virginianae
    # Grape diseases
    "grape_healthy",
    "grape_black_rot",  # Guignardia bidwellii
    "grape_esca",  # Phaeomoniella chlamydospora complex
    "grape_leaf_blight",  # Pseudocercospora vitis (Isariopsis leaf spot)
]

# Maps each disease class to its plant species for the multi-task head.
DISEASE_TO_SPECIES: dict[str, PlantSpecies] = {
    name: PlantSpecies(name.split("_")[0]) for name in DISEASE_CLASSES
}

SPECIES_CLASSES: list[str] = [s.value for s in PlantSpecies]

NUM_DISEASE_CLASSES: int = len(DISEASE_CLASSES)
NUM_SPECIES_CLASSES: int = len(SPECIES_CLASSES)


@dataclass
class DiagnosisResult:
    """Result of a single plant disease diagnosis inference."""

    disease_class: str
    disease_index: int
    confidence: float
    plant_species: str
    species_confidence: float
    top_k_diseases: list[tuple[str, float]]
    is_healthy: bool
    calibrated: bool


class TemperatureScaler(nn.Module):
    """Learned temperature scaling for post-hoc confidence calibration.

    Applies a single scalar temperature parameter to logits before
    softmax, reducing overconfidence common in deep neural networks.
    Calibrated on a held-out validation set after training.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class PlantDiseaseClassifier(nn.Module):
    """Multi-task EfficientNet-B0 classifier for plant disease detection.

    Architecture:
        - Backbone: EfficientNet-B0 pretrained on ImageNet
        - Shared feature extractor with frozen early layers
        - Disease classification head (23 classes)
        - Plant species classification head (5 classes)
        - Temperature scaling for confidence calibration

    The multi-task design encourages the model to learn species-specific
    features that improve disease classification, especially for diseases
    that present differently across species (e.g., early blight on tomato
    vs. potato).
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()

        # Load EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Extract feature layers (everything before the final classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # EfficientNet-B0 outputs 1280-dimensional features
        feature_dim = 1280

        # Optionally freeze early convolutional blocks to preserve
        # low-level features learned on ImageNet
        if freeze_backbone:
            for i, block in enumerate(self.features):
                if i < 5:  # Freeze first 5 of 9 blocks
                    for param in block.parameters():
                        param.requires_grad = False

        # Shared projection layer
        self.shared_fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, NUM_DISEASE_CLASSES),
        )

        # Species classification head
        self.species_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, NUM_SPECIES_CLASSES),
        )

        # Confidence calibration
        self.temperature_scaler = TemperatureScaler()
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from input image tensor."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.shared_fc(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning disease and species logits.

        Args:
            x: Batch of preprocessed leaf images [B, 3, 224, 224].

        Returns:
            Tuple of (disease_logits, species_logits).
        """
        features = self.extract_features(x)
        disease_logits = self.disease_head(features)
        species_logits = self.species_head(features)
        return disease_logits, species_logits

    def predict(self, image: Image.Image, top_k: int = 5) -> DiagnosisResult:
        """Run inference on a single PIL image and return structured diagnosis.

        Args:
            image: RGB PIL Image of a plant leaf.
            top_k: Number of top predictions to include.

        Returns:
            DiagnosisResult with disease classification and confidence.
        """
        self.eval()
        tensor = get_inference_transform()(image).unsqueeze(0)

        device = next(self.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            disease_logits, species_logits = self.forward(tensor)

            # Apply temperature scaling if calibrated
            if self._calibrated:
                disease_logits = self.temperature_scaler(disease_logits)
                species_logits = self.temperature_scaler(species_logits)

            disease_probs = F.softmax(disease_logits, dim=1).squeeze(0)
            species_probs = F.softmax(species_logits, dim=1).squeeze(0)

        # Top disease prediction
        disease_conf, disease_idx = torch.max(disease_probs, dim=0)
        disease_idx_int = disease_idx.item()
        disease_name = DISEASE_CLASSES[disease_idx_int]

        # Top species prediction
        species_conf, species_idx = torch.max(species_probs, dim=0)
        species_name = SPECIES_CLASSES[species_idx.item()]

        # Top-k disease predictions
        topk_values, topk_indices = torch.topk(disease_probs, min(top_k, NUM_DISEASE_CLASSES))
        top_k_diseases = [
            (DISEASE_CLASSES[idx.item()], val.item())
            for val, idx in zip(topk_values, topk_indices)
        ]

        is_healthy = disease_name.endswith("_healthy")

        result = DiagnosisResult(
            disease_class=disease_name,
            disease_index=disease_idx_int,
            confidence=disease_conf.item(),
            plant_species=species_name,
            species_confidence=species_conf.item(),
            top_k_diseases=top_k_diseases,
            is_healthy=is_healthy,
            calibrated=self._calibrated,
        )

        logger.info(
            "Diagnosis: %s (%.2f%% confidence), species: %s",
            disease_name,
            disease_conf.item() * 100,
            species_name,
        )
        return result

    def calibrate(
        self,
        val_loader: Any,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Calibrate confidence using temperature scaling on validation data.

        Args:
            val_loader: DataLoader yielding (images, disease_labels) batches.
            lr: Learning rate for temperature optimization.
            max_iter: Maximum LBFGS iterations.

        Returns:
            Optimal temperature value.
        """
        self.eval()
        device = next(self.parameters()).device

        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                disease_logits, _ = self.forward(images)
                logits_list.append(disease_logits)
                labels_list.append(labels)

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature_scaler.temperature], lr=lr, max_iter=max_iter
        )

        def _eval_closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.temperature_scaler(all_logits)
            loss = nll_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(_eval_closure)
        self._calibrated = True

        optimal_temp = self.temperature_scaler.temperature.item()
        logger.info("Calibration complete. Optimal temperature: %.4f", optimal_temp)
        return optimal_temp


class MultiTaskLoss(nn.Module):
    """Combined loss for joint disease and species classification.

    Uses learned task weights (uncertainty weighting) to balance the
    disease classification loss against the species classification loss.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses", CVPR 2018.
    """

    def __init__(self) -> None:
        super().__init__()
        # Log variance parameters for learned weighting
        self.log_var_disease = nn.Parameter(torch.zeros(1))
        self.log_var_species = nn.Parameter(torch.zeros(1))
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        disease_logits: torch.Tensor,
        species_logits: torch.Tensor,
        disease_labels: torch.Tensor,
        species_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        disease_loss = self.ce_loss(disease_logits, disease_labels)
        species_loss = self.ce_loss(species_logits, species_labels)

        # Uncertainty weighting: L_total = sum(1/(2*sigma^2) * L_i + log(sigma))
        precision_disease = torch.exp(-self.log_var_disease)
        precision_species = torch.exp(-self.log_var_species)

        total_loss = (
            precision_disease * disease_loss
            + self.log_var_disease
            + precision_species * species_loss
            + self.log_var_species
        )

        components = {
            "disease_loss": disease_loss.item(),
            "species_loss": species_loss.item(),
            "total_loss": total_loss.item(),
            "disease_weight": precision_disease.item(),
            "species_weight": precision_species.item(),
        }
        return total_loss, components


def get_inference_transform() -> transforms.Compose:
    """Standard inference-time image transform matching EfficientNet-B0 input.

    Returns:
        torchvision transforms pipeline: resize, center crop, normalize.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_training_transform() -> transforms.Compose:
    """Training-time augmentation pipeline for leaf images.

    Applies random spatial and color augmentations that simulate
    real-world variation in field photography conditions.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> PlantDiseaseClassifier:
    """Load a PlantDiseaseClassifier, optionally from a saved checkpoint.

    Args:
        checkpoint_path: Path to a .pth checkpoint file, or None for fresh model.
        device: Target device ('cpu', 'cuda', 'mps').

    Returns:
        Model loaded onto the specified device in eval mode.
    """
    model = PlantDiseaseClassifier(pretrained=checkpoint_path is None)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "temperature" in checkpoint:
                model.temperature_scaler.temperature.data = torch.tensor(
                    [checkpoint["temperature"]]
                )
                model._calibrated = True
            logger.info(
                "Loaded checkpoint from %s (epoch %s)",
                checkpoint_path,
                checkpoint.get("epoch", "unknown"),
            )
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded state dict from %s", checkpoint_path)

    model = model.to(device)
    model.eval()
    return model
