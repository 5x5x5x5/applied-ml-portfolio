"""Tests for CellVision model architectures and classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cell_vision import CELL_TYPES, NUM_CLASSES
from cell_vision.models.cell_classifier import CellClassifier, CellNet, CellNetResNet
from cell_vision.visualization.gradcam import GradCAM


class TestCellNet:
    """Tests for the custom CellNet architecture."""

    def test_output_shape(self, cellnet_model: CellNet, sample_image_tensor: torch.Tensor) -> None:
        """Model output should have shape (batch_size, num_classes)."""
        cellnet_model.eval()
        with torch.no_grad():
            output = cellnet_model(sample_image_tensor)
        assert output.shape == (1, NUM_CLASSES)

    def test_batch_output_shape(
        self, cellnet_model: CellNet, sample_batch_tensor: torch.Tensor
    ) -> None:
        """Model should handle batch inputs correctly."""
        cellnet_model.eval()
        with torch.no_grad():
            output = cellnet_model(sample_batch_tensor)
        assert output.shape == (4, NUM_CLASSES)

    def test_feature_maps(self, cellnet_model: CellNet, sample_image_tensor: torch.Tensor) -> None:
        """Feature maps should have expected spatial dimensions."""
        cellnet_model.eval()
        with torch.no_grad():
            features = cellnet_model.get_feature_maps(sample_image_tensor)
        # After 4 MaxPool layers: 224 -> 112 -> 56 -> 28 -> 14
        assert features.shape[0] == 1
        assert features.shape[1] == 256  # Last conv block output channels
        assert features.shape[2] == 14
        assert features.shape[3] == 14

    def test_parameter_count(self, cellnet_model: CellNet) -> None:
        """Model should have a reasonable number of parameters."""
        total_params = sum(p.numel() for p in cellnet_model.parameters())
        assert total_params > 0
        # CellNet should be relatively lightweight
        assert total_params < 5_000_000

    def test_all_parameters_require_grad(self, cellnet_model: CellNet) -> None:
        """All CellNet parameters should be trainable."""
        for name, param in cellnet_model.named_parameters():
            assert param.requires_grad, f"Parameter {name} has requires_grad=False"

    def test_different_input_sizes(self, cellnet_model: CellNet) -> None:
        """CellNet should handle various input sizes due to adaptive pooling."""
        cellnet_model.eval()
        for size in [128, 224, 256, 320]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = cellnet_model(x)
            assert output.shape == (1, NUM_CLASSES), f"Failed for input size {size}"


class TestCellNetResNet:
    """Tests for the ResNet18-based transfer learning model."""

    def test_output_shape(
        self, resnet_model: CellNetResNet, sample_image_tensor: torch.Tensor
    ) -> None:
        """ResNet model output should have shape (batch_size, num_classes)."""
        resnet_model.eval()
        with torch.no_grad():
            output = resnet_model(sample_image_tensor)
        assert output.shape == (1, NUM_CLASSES)

    def test_freeze_backbone(self) -> None:
        """Freezing backbone should make backbone params non-trainable."""
        model = CellNetResNet(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
        # Check that backbone conv/bn layers are frozen
        frozen_count = 0
        trainable_count = 0
        for param in model.parameters():
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1
        assert frozen_count > 0, "No parameters were frozen"
        assert trainable_count > 0, "All parameters were frozen (classifier should be trainable)"

    def test_unfreeze_backbone(self) -> None:
        """Unfreezing should make previously frozen parameters trainable."""
        model = CellNetResNet(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
        # Count initially frozen
        frozen_before = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_before > 0

        # Unfreeze from layer 6
        model.unfreeze_backbone(from_layer=6)
        frozen_after = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_after < frozen_before

    def test_feature_maps(
        self, resnet_model: CellNetResNet, sample_image_tensor: torch.Tensor
    ) -> None:
        """Feature maps from layer4 should have expected shape."""
        resnet_model.eval()
        with torch.no_grad():
            features = resnet_model.get_feature_maps(sample_image_tensor)
        assert features.shape[1] == 512  # ResNet18 layer4 output channels
        assert features.shape[2] == 7  # 224 / 32
        assert features.shape[3] == 7


class TestCellClassifier:
    """Tests for the high-level CellClassifier interface."""

    def test_predict_returns_top_k(
        self, classifier: CellClassifier, sample_image_tensor: torch.Tensor
    ) -> None:
        """Predict should return the requested number of top predictions."""
        predictions = classifier.predict(sample_image_tensor, top_k=3)
        assert len(predictions) == 3

    def test_predict_structure(
        self, classifier: CellClassifier, sample_image_tensor: torch.Tensor
    ) -> None:
        """Each prediction should have class, label, and confidence fields."""
        predictions = classifier.predict(sample_image_tensor, top_k=1)
        pred = predictions[0]
        assert "class" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert pred["class"] in CELL_TYPES
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_predict_confidences_sum(
        self, classifier: CellClassifier, sample_image_tensor: torch.Tensor
    ) -> None:
        """Top-k confidences should not exceed 1.0 total."""
        predictions = classifier.predict(sample_image_tensor, top_k=NUM_CLASSES)
        total_confidence = sum(p["confidence"] for p in predictions)
        assert abs(total_confidence - 1.0) < 0.01

    def test_predict_3d_input(self, classifier: CellClassifier) -> None:
        """Predict should handle unbatched 3D tensors by adding batch dim."""
        tensor_3d = torch.randn(3, 224, 224)
        predictions = classifier.predict(tensor_3d, top_k=1)
        assert len(predictions) == 1

    def test_explain(self, classifier: CellClassifier, sample_image_tensor: torch.Tensor) -> None:
        """Explain should return heatmap and prediction info."""
        result = classifier.explain(sample_image_tensor)
        assert "prediction" in result
        assert "heatmap" in result
        assert "all_predictions" in result
        assert isinstance(result["heatmap"], np.ndarray)
        assert result["heatmap"].shape == (224, 224)

    def test_save_and_load(
        self,
        classifier: CellClassifier,
        sample_image_tensor: torch.Tensor,
        tmp_model_path: Path,
    ) -> None:
        """Saving and loading should preserve model predictions."""
        # Get predictions before save
        pred_before = classifier.predict(sample_image_tensor, top_k=1)

        # Save
        classifier.save(tmp_model_path)
        assert tmp_model_path.exists()

        # Load into a new classifier
        new_classifier = CellClassifier(model_type="cellnet", device="cpu")
        new_classifier.load(tmp_model_path)

        # Predictions should match
        pred_after = new_classifier.predict(sample_image_tensor, top_k=1)
        assert pred_before[0]["class"] == pred_after[0]["class"]
        assert abs(pred_before[0]["confidence"] - pred_after[0]["confidence"]) < 1e-5

    def test_get_model_info(self, classifier: CellClassifier) -> None:
        """Model info should contain expected fields."""
        info = classifier.get_model_info()
        assert info["model_type"] == "cellnet"
        assert info["num_classes"] == NUM_CLASSES
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["cell_types"] == CELL_TYPES

    def test_invalid_model_type(self) -> None:
        """Should raise ValueError for unknown model types."""
        with pytest.raises(ValueError, match="Unknown model type"):
            CellClassifier(model_type="invalid_model")


class TestGradCAM:
    """Tests for GradCAM visualization."""

    def test_gradcam_output_shape(
        self, cellnet_model: CellNet, sample_image_tensor: torch.Tensor
    ) -> None:
        """GradCAM heatmap should match input spatial dimensions."""
        target_layer = cellnet_model.features[-1].block[0]
        gradcam = GradCAM(cellnet_model, target_layer)

        heatmap = gradcam.generate(sample_image_tensor)
        assert heatmap.shape == (224, 224)
        gradcam.remove_hooks()

    def test_gradcam_values_in_range(
        self, cellnet_model: CellNet, sample_image_tensor: torch.Tensor
    ) -> None:
        """GradCAM values should be normalized to [0, 1]."""
        target_layer = cellnet_model.features[-1].block[0]
        gradcam = GradCAM(cellnet_model, target_layer)

        heatmap = gradcam.generate(sample_image_tensor)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        gradcam.remove_hooks()

    def test_gradcam_specific_class(
        self, cellnet_model: CellNet, sample_image_tensor: torch.Tensor
    ) -> None:
        """GradCAM should accept a specific target class."""
        target_layer = cellnet_model.features[-1].block[0]
        gradcam = GradCAM(cellnet_model, target_layer)

        # Generate for class 0 (red blood cell)
        heatmap = gradcam.generate(sample_image_tensor, target_class=0)
        assert heatmap.shape == (224, 224)
        gradcam.remove_hooks()
