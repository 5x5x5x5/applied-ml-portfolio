"""Tests for the WildEye species classifier model.

Covers model architecture, forward pass, prediction output structure,
temperature scaling, training/inference transforms, and ONNX export.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from wild_eye import NUM_SPECIES, SPECIES_LABELS
from wild_eye.models.species_classifier import (
    INPUT_SIZE,
    ClassificationResult,
    TemperatureScaler,
    WildEyeClassifier,
    export_to_onnx,
    get_inference_transform,
    get_training_transform,
)


class TestWildEyeClassifier:
    """Tests for the WildEyeClassifier model architecture and inference."""

    def test_model_initializes(self, classifier: WildEyeClassifier) -> None:
        """Model should initialize with correct number of output classes."""
        assert classifier.num_classes == NUM_SPECIES

    def test_forward_pass_shape(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """Forward pass should produce logits of shape (batch, num_classes)."""
        logits = classifier(sample_tensor)
        assert logits.shape == (1, NUM_SPECIES)

    def test_batch_forward_pass(
        self, classifier: WildEyeClassifier, batch_tensor: torch.Tensor
    ) -> None:
        """Forward pass should handle batches correctly."""
        logits = classifier(batch_tensor)
        assert logits.shape == (4, NUM_SPECIES)

    def test_predict_returns_classification_results(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """predict() should return a list of ClassificationResult objects."""
        results = classifier.predict(sample_tensor)
        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)

    def test_predict_result_structure(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """ClassificationResult should have all required fields populated."""
        result = classifier.predict(sample_tensor)[0]

        assert isinstance(result.species, list)
        assert isinstance(result.probabilities, dict)
        assert result.top_species in SPECIES_LABELS
        assert 0.0 <= result.top_confidence <= 1.0
        assert isinstance(result.is_empty, bool)
        assert isinstance(result.is_human, bool)
        assert len(result.raw_logits) == NUM_SPECIES

    def test_predict_probabilities_sum(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """Sigmoid probabilities should each be in [0, 1] (multi-label, so no sum constraint)."""
        result = classifier.predict(sample_tensor)[0]
        for label, prob in result.probabilities.items():
            assert 0.0 <= prob <= 1.0, f"{label} probability {prob} out of range"

    def test_predict_all_species_covered(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """Probabilities dict should contain an entry for every species label."""
        result = classifier.predict(sample_tensor)[0]
        for label in SPECIES_LABELS:
            assert label in result.probabilities

    def test_predict_confidence_threshold_high(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """High threshold should produce fewer (or zero) detected species."""
        results_high = classifier.predict(sample_tensor, confidence_threshold=0.99)
        results_low = classifier.predict(sample_tensor, confidence_threshold=0.01)
        assert len(results_high[0].species) <= len(results_low[0].species)

    def test_predict_batch_results(
        self, classifier: WildEyeClassifier, batch_tensor: torch.Tensor
    ) -> None:
        """Batch predict should return one result per image."""
        results = classifier.predict(batch_tensor)
        assert len(results) == 4
        for result in results:
            assert isinstance(result, ClassificationResult)

    def test_model_eval_mode(self, classifier: WildEyeClassifier) -> None:
        """Classifier should be in eval mode for deterministic inference."""
        assert not classifier.training

    def test_freeze_backbone(self) -> None:
        """Freezing backbone should make feature params non-trainable."""
        model = WildEyeClassifier(pretrained=False, freeze_backbone_layers=-1)
        frozen = sum(1 for p in model.features.parameters() if not p.requires_grad)
        total = sum(1 for _ in model.features.parameters())
        assert frozen == total
        # Classifier head should still be trainable.
        trainable_head = sum(1 for p in model.classifier.parameters() if p.requires_grad)
        assert trainable_head > 0

    def test_no_freeze_backbone(self) -> None:
        """With freeze=0, all backbone params should be trainable."""
        model = WildEyeClassifier(pretrained=False, freeze_backbone_layers=0)
        frozen = sum(1 for p in model.features.parameters() if not p.requires_grad)
        assert frozen == 0


class TestTemperatureScaler:
    """Tests for the temperature scaling calibration layer."""

    def test_default_temperature(self) -> None:
        """Default temperature should be 1.5."""
        scaler = TemperatureScaler()
        assert float(scaler.temperature) == pytest.approx(1.5)

    def test_custom_temperature(self) -> None:
        """Custom initial temperature should be respected."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        assert float(scaler.temperature) == pytest.approx(2.0)

    def test_scaling_reduces_magnitude(self) -> None:
        """Temperature > 1 should reduce logit magnitudes (soften predictions)."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.tensor([3.0, -1.0, 0.5])
        scaled = scaler(logits)
        assert torch.all(torch.abs(scaled) < torch.abs(logits))

    def test_temperature_is_learnable(self) -> None:
        """Temperature should be a learnable parameter."""
        scaler = TemperatureScaler()
        assert scaler.temperature.requires_grad


class TestTransforms:
    """Tests for image preprocessing transforms."""

    def test_inference_transform_output_shape(self) -> None:
        """Inference transform should produce (3, 224, 224) tensor."""
        from PIL import Image

        transform = get_inference_transform()
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        tensor = transform(image)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_training_transform_output_shape(self) -> None:
        """Training transform should produce (3, 224, 224) tensor."""
        from PIL import Image

        transform = get_training_transform()
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        tensor = transform(image)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_training_transform_augments(self) -> None:
        """Training transform should produce different outputs (stochastic)."""
        from PIL import Image

        transform = get_training_transform()
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        tensors = [transform(image) for _ in range(5)]
        # At least some pairs should differ due to random augmentation.
        any_different = any(
            not torch.equal(tensors[i], tensors[j])
            for i in range(len(tensors))
            for j in range(i + 1, len(tensors))
        )
        assert any_different


class TestONNXExport:
    """Tests for ONNX model export."""

    def test_export_creates_file(self, classifier: WildEyeClassifier) -> None:
        """ONNX export should create a valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"
            result = export_to_onnx(classifier, output_path)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_export_returns_correct_path(self, classifier: WildEyeClassifier) -> None:
        """Export function should return the output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            result = export_to_onnx(classifier, output_path)
            assert result == output_path

    @pytest.mark.slow
    def test_onnx_inference_matches_pytorch(
        self, classifier: WildEyeClassifier, sample_tensor: torch.Tensor
    ) -> None:
        """ONNX model output should closely match PyTorch output."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(classifier, onnx_path)

            # PyTorch inference.
            classifier.eval()
            with torch.no_grad():
                pytorch_output = classifier(sample_tensor).numpy()

            # ONNX inference.
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: sample_tensor.numpy()})[0]

            np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
