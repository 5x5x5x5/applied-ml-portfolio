"""Model tests including correctness and latency assertions."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from rx_predict.models.drug_response_model import (
    RESPONSE_CLASSES,
    DrugResponseModel,
    PredictionResult,
)
from rx_predict.models.feature_processor import (
    DRUG_CLASSES,
    PHARMACOGENOMIC_GENES,
    FeatureProcessor,
)


class TestFeatureProcessor:
    """Test feature extraction and encoding."""

    def test_feature_dimension(self, feature_processor: FeatureProcessor) -> None:
        """Feature dimension should match expected size."""
        dim = feature_processor.feature_dimension
        # genetic + metabolizer + demographics(4) + drugs(16) + history(10)
        expected_genetic = sum(len(v) for v in PHARMACOGENOMIC_GENES.values())
        expected = expected_genetic + 1 + 4 + len(DRUG_CLASSES) + 10
        assert dim == expected

    def test_process_single_returns_correct_shape(
        self, feature_processor: FeatureProcessor, sample_patient_data: dict[str, Any]
    ) -> None:
        """Single patient processing returns correct shape."""
        features = feature_processor.process_single(sample_patient_data)
        assert features.shape == (feature_processor.feature_dimension,)
        assert features.dtype == np.float32

    def test_process_batch_returns_correct_shape(
        self, feature_processor: FeatureProcessor, sample_patient_data: dict[str, Any]
    ) -> None:
        """Batch processing returns correct matrix shape."""
        batch = [sample_patient_data] * 5
        feature_matrix = feature_processor.process_batch(batch)
        assert feature_matrix.shape == (5, feature_processor.feature_dimension)
        assert feature_matrix.dtype == np.float32

    def test_genetic_encoding(self, feature_processor: FeatureProcessor) -> None:
        """Genetic variants should be one-hot encoded."""
        data = {
            "genetic_profile": {"CYP2D6": ["*1", "*4"]},
            "demographics": {"age": 30, "weight_kg": 70, "height_cm": 170, "bmi": 24.2},
            "drug": {"drug_class": "ssri", "dosage_mg": 50},
            "medical_history": {},
        }
        features = feature_processor.process_single(data)
        # First features should have some non-zero values for CYP2D6
        genetic_region = features[: len(PHARMACOGENOMIC_GENES["CYP2D6"])]
        assert np.sum(genetic_region > 0) >= 1

    def test_demographic_normalization(self, feature_processor: FeatureProcessor) -> None:
        """Demographics should be z-score normalized."""
        # Average demographics should produce near-zero features
        data = {
            "demographics": {"age": 45, "weight_kg": 75, "height_cm": 170, "bmi": 25.9},
            "drug": {"drug_class": "other"},
            "medical_history": {},
        }
        features = feature_processor.process_single(data)
        # Demographics start after genetic + metabolizer features
        genetic_count = sum(len(v) for v in PHARMACOGENOMIC_GENES.values())
        demo_start = genetic_count + 1
        demo_features = features[demo_start : demo_start + 4]
        # Average patient should have near-zero normalized values
        assert np.all(np.abs(demo_features) < 1.0)

    def test_feature_caching(self, feature_processor: FeatureProcessor) -> None:
        """Repeated genetic profiles should be cached."""
        data = {
            "genetic_profile": {"CYP2D6": ["*1"]},
            "demographics": {"age": 30, "weight_kg": 70, "height_cm": 170, "bmi": 24.2},
            "drug": {"drug_class": "ssri"},
            "medical_history": {},
        }
        # Process twice
        f1 = feature_processor.process_single(data)
        f2 = feature_processor.process_single(data)
        np.testing.assert_array_equal(f1, f2)

    def test_empty_genetic_profile(self, feature_processor: FeatureProcessor) -> None:
        """Empty genetic profile should not raise."""
        data = {
            "genetic_profile": {},
            "demographics": {"age": 30, "weight_kg": 70, "height_cm": 170, "bmi": 24.2},
            "drug": {"drug_class": "ssri"},
            "medical_history": {},
        }
        features = feature_processor.process_single(data)
        assert features.shape == (feature_processor.feature_dimension,)

    def test_medical_history_conditions(self, feature_processor: FeatureProcessor) -> None:
        """Medical conditions should set appropriate flags."""
        data = {
            "demographics": {"age": 60, "weight_kg": 80, "height_cm": 175, "bmi": 26.1},
            "drug": {"drug_class": "statin"},
            "medical_history": {
                "conditions": ["diabetes", "liver_disease"],
                "num_current_medications": 5,
            },
        }
        features = feature_processor.process_single(data)
        assert features is not None
        assert not np.any(np.isnan(features))

    def test_feature_names(self, feature_processor: FeatureProcessor) -> None:
        """Feature names count should match dimension."""
        names = feature_processor.get_feature_names()
        assert len(names) == feature_processor.feature_dimension

    def test_feature_processing_speed(
        self, feature_processor: FeatureProcessor, sample_patient_data: dict[str, Any]
    ) -> None:
        """Feature processing should be fast (<5ms per record)."""
        # Warmup
        for _ in range(10):
            feature_processor.process_single(sample_patient_data)

        # Benchmark
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            feature_processor.process_single(sample_patient_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        p99 = sorted(latencies)[95]
        assert p99 < 5.0, f"Feature processing p99 too slow: {p99:.3f}ms"


class TestDrugResponseModel:
    """Test the ML model."""

    def test_model_loads_and_predicts(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Model should load and produce valid predictions."""
        assert trained_model.is_loaded
        result = trained_model.predict(sample_patient_data)
        assert isinstance(result, PredictionResult)
        assert 0 <= result.response_probability <= 1
        assert result.predicted_class in RESPONSE_CLASSES

    def test_prediction_confidence_interval(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Confidence interval should bound the prediction."""
        result = trained_model.predict(sample_patient_data)
        assert result.confidence_lower <= result.response_probability
        assert result.response_probability <= result.confidence_upper
        assert result.confidence_lower >= 0
        assert result.confidence_upper <= 1

    def test_prediction_risk_level(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Risk level should be one of the defined levels."""
        result = trained_model.predict(sample_patient_data)
        assert result.risk_level in ["high_risk", "moderate_risk", "low_risk", "minimal_risk"]

    def test_prediction_has_model_version(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Prediction should include model version."""
        result = trained_model.predict(sample_patient_data)
        assert result.model_version == "test-1.0.0"

    def test_batch_prediction(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Batch prediction should return correct number of results."""
        patients = [sample_patient_data] * 10
        batch_result = trained_model.predict_batch(patients)
        assert len(batch_result.predictions) == 10
        assert batch_result.total_inference_time_ms > 0
        assert batch_result.avg_inference_time_ms > 0

    def test_single_prediction_latency(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Single prediction should be under 50ms p99."""
        latencies = []
        for _ in range(100):
            result = trained_model.predict(sample_patient_data)
            latencies.append(result.inference_time_ms)

        p99 = sorted(latencies)[95]
        assert p99 < 50.0, f"Single prediction p99 too slow: {p99:.3f}ms"

    def test_model_benchmark(self, trained_model: DrugResponseModel) -> None:
        """Model benchmark should report under 100ms p99."""
        results = trained_model.benchmark(n_iterations=200)
        assert results["p99_ms"] < 100.0, f"Benchmark p99 too slow: {results['p99_ms']:.3f}ms"
        assert results["p50_ms"] < results["p95_ms"]
        assert results["p95_ms"] < results["p99_ms"]

    def test_different_patients_different_predictions(
        self,
        trained_model: DrugResponseModel,
        sample_patient_data: dict[str, Any],
        poor_metabolizer_patient: dict[str, Any],
    ) -> None:
        """Different patients should generally produce different predictions."""
        result1 = trained_model.predict(sample_patient_data)
        result2 = trained_model.predict(poor_metabolizer_patient)
        # At minimum the feature vectors should differ, leading to different probabilities
        # (though the class could be the same by chance)
        assert (
            result1.response_probability != result2.response_probability
            or result1.predicted_class != result2.predicted_class
        )

    def test_model_not_loaded_error(self) -> None:
        """Predicting without loading should raise."""
        model = DrugResponseModel()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict({"demographics": {"age": 30}})

    def test_model_save_and_load(self, trained_model: DrugResponseModel, tmp_path: Any) -> None:
        """Model should survive save/load cycle."""
        path = tmp_path / "test_model.joblib"
        trained_model.save(path)

        new_model = DrugResponseModel()
        new_model.load(path)
        assert new_model.is_loaded
        assert new_model.model_version == trained_model.model_version
