"""Tests for the plant disease classification model."""

from __future__ import annotations

import torch
from PIL import Image

from plant_pathologist.models.disease_classifier import (
    DISEASE_CLASSES,
    DISEASE_TO_SPECIES,
    NUM_DISEASE_CLASSES,
    NUM_SPECIES_CLASSES,
    SPECIES_CLASSES,
    DiagnosisResult,
    MultiTaskLoss,
    PlantDiseaseClassifier,
    TemperatureScaler,
    get_inference_transform,
    get_training_transform,
    load_model,
)


class TestDiseaseConstants:
    """Verify disease class definitions are consistent."""

    def test_disease_class_count(self) -> None:
        assert NUM_DISEASE_CLASSES == 23

    def test_species_class_count(self) -> None:
        assert NUM_SPECIES_CLASSES == 5

    def test_all_diseases_have_species_mapping(self) -> None:
        for disease in DISEASE_CLASSES:
            assert disease in DISEASE_TO_SPECIES, f"Missing species mapping for {disease}"

    def test_species_values(self) -> None:
        expected = {"tomato", "potato", "corn", "apple", "grape"}
        assert set(SPECIES_CLASSES) == expected

    def test_healthy_classes_exist(self) -> None:
        healthy_classes = [c for c in DISEASE_CLASSES if c.endswith("_healthy")]
        assert len(healthy_classes) == 5
        for species in SPECIES_CLASSES:
            assert f"{species}_healthy" in DISEASE_CLASSES

    def test_disease_names_follow_convention(self) -> None:
        """Each disease name should start with a valid species prefix."""
        valid_prefixes = set(SPECIES_CLASSES)
        for disease in DISEASE_CLASSES:
            prefix = disease.split("_")[0]
            assert prefix in valid_prefixes, f"Invalid prefix in '{disease}'"


class TestPlantDiseaseClassifier:
    """Tests for the EfficientNet-B0 classifier model."""

    def test_model_instantiation(self, model: PlantDiseaseClassifier) -> None:
        assert isinstance(model, PlantDiseaseClassifier)

    def test_forward_output_shapes(self, model: PlantDiseaseClassifier) -> None:
        batch = torch.randn(2, 3, 224, 224)
        disease_logits, species_logits = model(batch)
        assert disease_logits.shape == (2, NUM_DISEASE_CLASSES)
        assert species_logits.shape == (2, NUM_SPECIES_CLASSES)

    def test_forward_single_image(self, model: PlantDiseaseClassifier) -> None:
        single = torch.randn(1, 3, 224, 224)
        disease_logits, species_logits = model(single)
        assert disease_logits.shape == (1, NUM_DISEASE_CLASSES)
        assert species_logits.shape == (1, NUM_SPECIES_CLASSES)

    def test_predict_returns_diagnosis_result(
        self, model: PlantDiseaseClassifier, sample_rgb_image: Image.Image
    ) -> None:
        result = model.predict(sample_rgb_image)
        assert isinstance(result, DiagnosisResult)
        assert result.disease_class in DISEASE_CLASSES
        assert result.plant_species in SPECIES_CLASSES
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.species_confidence <= 1.0
        assert isinstance(result.is_healthy, bool)

    def test_predict_top_k_diseases(
        self, model: PlantDiseaseClassifier, sample_rgb_image: Image.Image
    ) -> None:
        result = model.predict(sample_rgb_image, top_k=3)
        assert len(result.top_k_diseases) == 3
        # Top predictions should be sorted by confidence (descending)
        confidences = [conf for _, conf in result.top_k_diseases]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_top_k_sum_approximately_one(
        self, model: PlantDiseaseClassifier, sample_rgb_image: Image.Image
    ) -> None:
        result = model.predict(sample_rgb_image, top_k=NUM_DISEASE_CLASSES)
        total = sum(conf for _, conf in result.top_k_diseases)
        assert abs(total - 1.0) < 1e-4, f"Top-k probabilities sum to {total}, expected ~1.0"

    def test_model_eval_mode_no_grad(
        self, model: PlantDiseaseClassifier, sample_rgb_image: Image.Image
    ) -> None:
        result = model.predict(sample_rgb_image)
        # Model should be in eval mode after predict
        assert not model.training

    def test_extract_features(self, model: PlantDiseaseClassifier) -> None:
        batch = torch.randn(1, 3, 224, 224)
        features = model.extract_features(batch)
        assert features.shape == (1, 512)  # shared_fc output dim

    def test_calibrated_flag(self, model: PlantDiseaseClassifier) -> None:
        assert not model.calibrated


class TestTemperatureScaler:
    """Tests for the confidence calibration module."""

    def test_initial_temperature(self) -> None:
        scaler = TemperatureScaler()
        assert scaler.temperature.item() == pytest.approx(1.5)

    def test_scaling_reduces_confidence(self) -> None:
        scaler = TemperatureScaler()
        scaler.temperature.data = torch.tensor([2.0])
        logits = torch.tensor([[3.0, 1.0, 0.5]])
        scaled = scaler(logits)
        # Higher temperature -> more uniform distribution -> lower max prob
        original_max_prob = torch.softmax(logits, dim=1).max().item()
        scaled_max_prob = torch.softmax(scaled, dim=1).max().item()
        assert scaled_max_prob < original_max_prob

    def test_temperature_one_is_identity(self) -> None:
        scaler = TemperatureScaler()
        scaler.temperature.data = torch.tensor([1.0])
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        scaled = scaler(logits)
        assert torch.allclose(logits, scaled)


class TestMultiTaskLoss:
    """Tests for the multi-task loss with uncertainty weighting."""

    def test_loss_returns_positive_value(self) -> None:
        loss_fn = MultiTaskLoss()
        disease_logits = torch.randn(4, NUM_DISEASE_CLASSES)
        species_logits = torch.randn(4, NUM_SPECIES_CLASSES)
        disease_labels = torch.randint(0, NUM_DISEASE_CLASSES, (4,))
        species_labels = torch.randint(0, NUM_SPECIES_CLASSES, (4,))

        total_loss, components = loss_fn(
            disease_logits, species_logits, disease_labels, species_labels
        )
        assert total_loss.item() > 0
        assert "disease_loss" in components
        assert "species_loss" in components
        assert "total_loss" in components

    def test_loss_components_are_positive(self) -> None:
        loss_fn = MultiTaskLoss()
        disease_logits = torch.randn(8, NUM_DISEASE_CLASSES)
        species_logits = torch.randn(8, NUM_SPECIES_CLASSES)
        disease_labels = torch.randint(0, NUM_DISEASE_CLASSES, (8,))
        species_labels = torch.randint(0, NUM_SPECIES_CLASSES, (8,))

        _, components = loss_fn(disease_logits, species_logits, disease_labels, species_labels)
        assert components["disease_loss"] > 0
        assert components["species_loss"] > 0

    def test_loss_is_differentiable(self) -> None:
        loss_fn = MultiTaskLoss()
        disease_logits = torch.randn(4, NUM_DISEASE_CLASSES, requires_grad=True)
        species_logits = torch.randn(4, NUM_SPECIES_CLASSES, requires_grad=True)
        disease_labels = torch.randint(0, NUM_DISEASE_CLASSES, (4,))
        species_labels = torch.randint(0, NUM_SPECIES_CLASSES, (4,))

        total_loss, _ = loss_fn(disease_logits, species_logits, disease_labels, species_labels)
        total_loss.backward()
        assert disease_logits.grad is not None
        assert species_logits.grad is not None


class TestTransforms:
    """Tests for inference and training transforms."""

    def test_inference_transform_output_shape(self, sample_rgb_image: Image.Image) -> None:
        transform = get_inference_transform()
        tensor = transform(sample_rgb_image)
        assert tensor.shape == (3, 224, 224)

    def test_training_transform_output_shape(self, sample_rgb_image: Image.Image) -> None:
        transform = get_training_transform()
        tensor = transform(sample_rgb_image)
        assert tensor.shape == (3, 224, 224)

    def test_inference_transform_normalization(self, sample_rgb_image: Image.Image) -> None:
        transform = get_inference_transform()
        tensor = transform(sample_rgb_image)
        # After ImageNet normalization, values should not be in [0, 1]
        assert tensor.min() < 0 or tensor.max() > 1


class TestLoadModel:
    """Tests for the model loading utility."""

    def test_load_fresh_model(self) -> None:
        model = load_model(checkpoint_path=None, device="cpu")
        assert isinstance(model, PlantDiseaseClassifier)
        assert not model.training  # Should be in eval mode

    def test_load_model_on_cpu(self) -> None:
        model = load_model(device="cpu")
        param_device = next(model.parameters()).device
        assert param_device == torch.device("cpu")
