"""CellNet - CNN architecture for microscopy cell type classification.

Provides both a custom CNN (CellNet) and a ResNet18-based transfer learning model
for classifying cell types from microscopy images of blood smears.

Supported cell types:
    - Red blood cells (erythrocytes)
    - White blood cells: neutrophils, lymphocytes, monocytes, eosinophils, basophils
    - Platelets (thrombocytes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cell_vision import CELL_TYPES, NUM_CLASSES
from cell_vision.visualization.gradcam import GradCAM

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm -> ReLU -> optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CellNet(nn.Module):
    """Custom CNN architecture for cell type classification.

    Architecture:
        - 4 convolutional blocks with increasing channels (32->64->128->256)
        - BatchNorm after every conv layer for training stability
        - MaxPool after each block for spatial down-sampling
        - Global average pooling before the classifier head
        - Dropout for regularization
        - Fully connected classifier: 256 -> 128 -> num_classes

    Args:
        num_classes: Number of cell type categories to classify.
        dropout_rate: Dropout probability for regularization.
        in_channels: Number of input image channels (3 for RGB microscopy images).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.4,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor: progressive channel expansion
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, padding=1, pool=True),
            ConvBlock(32, 64, kernel_size=3, padding=1, pool=True),
            ConvBlock(64, 128, kernel_size=3, padding=1, pool=True),
            ConvBlock(128, 256, kernel_size=3, padding=1, pool=True),
        )

        # Global average pooling collapses spatial dims to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv layers, constant init for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from the last convolutional block.

        Used for GradCAM visualization.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Feature maps from the final conv block.
        """
        return self.features(x)


class CellNetResNet(nn.Module):
    """Transfer learning model based on ResNet18 for cell classification.

    Replaces the final fully-connected layer of a pretrained ResNet18 with
    a custom classifier head suited for cell type classification. Optionally
    freezes the backbone for feature-extraction-only training.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all backbone layers for fine-tuning
            only the classifier head.
        dropout_rate: Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen - only classifier head will be trained")

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet backbone and custom head."""
        return self.backbone(x)

    def unfreeze_backbone(self, from_layer: int = 6) -> None:
        """Progressively unfreeze backbone layers for fine-tuning.

        ResNet18 has layers: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc.
        Unfreezing from layer N means layers >= N will be trainable.

        Args:
            from_layer: Index of the first child module to unfreeze (0-indexed).
        """
        children = list(self.backbone.children())
        for i, child in enumerate(children):
            if i >= from_layer:
                for param in child.parameters():
                    param.requires_grad = True
                logger.info("Unfreezing backbone layer %d: %s", i, child.__class__.__name__)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from the last convolutional layer (layer4).

        Used by GradCAM for visualization of model attention.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


class CellClassifier:
    """High-level classifier wrapping model selection, training, and inference.

    Provides a unified interface for training, predicting, and explaining
    cell type classifications from microscopy images.

    Args:
        model_type: Architecture to use ("cellnet" or "resnet18").
        num_classes: Number of cell type categories.
        device: Torch device for computation.
        pretrained: Whether to use pretrained weights (ResNet only).
    """

    def __init__(
        self,
        model_type: str = "cellnet",
        num_classes: int = NUM_CLASSES,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_type = model_type
        self.num_classes = num_classes

        if model_type == "cellnet":
            self.model = CellNet(num_classes=num_classes).to(self.device)
        elif model_type == "resnet18":
            self.model = CellNetResNet(num_classes=num_classes, pretrained=pretrained).to(
                self.device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'cellnet' or 'resnet18'.")

        self.criterion = nn.CrossEntropyLoss()
        self._gradcam: GradCAM | None = None

        logger.info(
            "Initialized %s on %s with %d classes",
            model_type,
            self.device,
            num_classes,
        )

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train the model on a dataset.

        Args:
            train_loader: DataLoader for training samples.
            val_loader: Optional DataLoader for validation.
            epochs: Number of training epochs.
            lr: Initial learning rate.
            weight_decay: L2 regularization factor.

        Returns:
            Dictionary with training history (losses, accuracies).
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                logger.info(
                    "Epoch %d/%d - Loss: %.4f, Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    epoch_acc,
                    val_loss,
                    val_acc,
                )
            else:
                logger.info(
                    "Epoch %d/%d - Loss: %.4f, Acc: %.4f",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    epoch_acc,
                )

        return history

    def _validate(self, val_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """Run validation and return loss and accuracy."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        self.model.train()
        return running_loss / total, correct / total

    def predict(self, image: torch.Tensor, top_k: int = 3) -> list[dict[str, Any]]:
        """Classify a microscopy image.

        Args:
            image: Preprocessed image tensor of shape (1, C, H, W) or (C, H, W).
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with 'class', 'label', and 'confidence' for top-k predictions.
        """
        self.model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probabilities = F.softmax(logits, dim=1)

        top_probs, top_indices = probabilities.topk(top_k, dim=1)
        results: list[dict[str, Any]] = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = CELL_TYPES[idx.item()]
            results.append(
                {
                    "class": class_name,
                    "label": class_name.replace("_", " ").title(),
                    "confidence": round(prob.item(), 4),
                }
            )

        return results

    def explain(self, image: torch.Tensor, target_class: int | None = None) -> dict[str, Any]:
        """Generate GradCAM explanation for a prediction.

        Produces a heatmap showing which regions of the image most influenced
        the classification decision, useful for verifying the model focuses
        on biologically relevant features (e.g., granules, nucleus shape).

        Args:
            image: Input image tensor (1, C, H, W) or (C, H, W).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            Dict with 'prediction', 'heatmap' (numpy array), and 'confidence'.
        """
        if self._gradcam is None:
            target_layer = self._get_target_layer()
            self._gradcam = GradCAM(self.model, target_layer)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        heatmap = self._gradcam.generate(image, target_class=target_class)

        # Also get prediction
        predictions = self.predict(image)
        return {
            "prediction": predictions[0],
            "heatmap": heatmap,
            "all_predictions": predictions,
        }

    def _get_target_layer(self) -> nn.Module:
        """Return the target layer for GradCAM based on model type."""
        if self.model_type == "cellnet":
            # Last conv block in features
            return self.model.features[-1].block[0]
        else:
            # Last conv layer in ResNet layer4
            return self.model.backbone.layer4[-1].conv2

    def save(self, path: str | Path) -> None:
        """Save model state and metadata to disk.

        Args:
            path: File path for the saved checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the saved checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model loaded from %s", path)

    def get_model_info(self) -> dict[str, Any]:
        """Return model architecture summary and parameter counts."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "cell_types": CELL_TYPES,
        }
