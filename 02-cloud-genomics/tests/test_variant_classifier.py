"""Tests for the genomic variant classifier ML model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cloud_genomics.models.variant_classifier import (
    ModelMetrics,
    PredictionResult,
    VariantClass,
    VariantClassifier,
    VariantFeatures,
    generate_synthetic_training_data,
)


class TestVariantFeatures:
    """Tests for VariantFeatures data class."""

    def test_default_features(self) -> None:
        """Default features should have sane values."""
        features = VariantFeatures()
        assert features.phylop_score == 0.0
        assert features.gnomad_af == 0.0
        assert features.sift_score == 1.0  # tolerated by default
        assert features.variant_type == "SNV"
        assert features.consequence == "missense"

    def test_to_array_shape(self) -> None:
        """to_array should produce the correct number of features."""
        features = VariantFeatures()
        arr = features.to_array()
        assert arr.shape == (VariantClassifier.N_FEATURES,)
        assert arr.dtype == np.float64

    def test_to_array_values(self) -> None:
        """to_array should encode values correctly."""
        features = VariantFeatures(
            phylop_score=3.5,
            gnomad_af=0.01,
            sift_score=0.05,
            in_protein_domain=True,
        )
        arr = features.to_array()
        assert arr[0] == 3.5  # phylop_score is first
        assert arr[3] == 0.01  # gnomad_af
        assert arr[8] == 0.05  # sift_score
        assert arr[13] == 1.0  # in_protein_domain (bool -> float)

    def test_feature_names_count(self) -> None:
        """feature_names should match the array length."""
        names = VariantFeatures.feature_names()
        features = VariantFeatures()
        arr = features.to_array()
        assert len(names) == len(arr)

    def test_consequence_encoding(self) -> None:
        """Different consequences should produce different encoded values."""
        missense = VariantFeatures(consequence="missense")
        nonsense = VariantFeatures(consequence="nonsense")
        assert missense.to_array()[18] != nonsense.to_array()[18]

    def test_variant_type_encoding(self) -> None:
        """Different variant types should produce different encoded values."""
        snv = VariantFeatures(variant_type="SNV")
        ins = VariantFeatures(variant_type="insertion")
        assert snv.to_array()[17] != ins.to_array()[17]


class TestVariantClassifier:
    """Tests for the VariantClassifier."""

    def test_untrained_prediction_raises(self) -> None:
        """Predicting with untrained model should raise RuntimeError."""
        classifier = VariantClassifier()
        features = VariantFeatures()
        with pytest.raises(RuntimeError, match="trained"):
            classifier.predict(features)

    def test_untrained_save_raises(self) -> None:
        """Saving untrained model should raise RuntimeError."""
        classifier = VariantClassifier()
        with pytest.raises(RuntimeError, match="untrained"):
            classifier.save("/tmp/test_model.joblib")

    def test_train_insufficient_samples(self) -> None:
        """Training with too few samples should raise ValueError."""
        classifier = VariantClassifier()
        features = [VariantFeatures() for _ in range(5)]
        labels = [VariantClass.BENIGN] * 5
        with pytest.raises(ValueError, match="at least 10"):
            classifier.train(features, labels)

    def test_train_mismatched_lengths(self) -> None:
        """Training with different-length features and labels should raise."""
        classifier = VariantClassifier()
        features = [VariantFeatures() for _ in range(20)]
        labels = [VariantClass.BENIGN] * 10
        with pytest.raises(ValueError, match="same length"):
            classifier.train(features, labels)

    def test_train_returns_metrics(
        self,
        synthetic_training_data: tuple[list[VariantFeatures], list[VariantClass]],
    ) -> None:
        """Training should return ModelMetrics with valid values."""
        features, labels = synthetic_training_data
        classifier = VariantClassifier(n_estimators=20, random_state=42)
        metrics = classifier.train(features, labels, calibrate=False)

        assert isinstance(metrics, ModelMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.cross_val_mean <= 1.0
        assert metrics.cross_val_std >= 0.0
        assert len(metrics.feature_importances) > 0
        assert metrics.classification_report  # non-empty string
        assert metrics.confusion_matrix.shape[0] > 0

    def test_prediction_returns_result(
        self,
        trained_classifier: VariantClassifier,
        benign_variant_features: VariantFeatures,
    ) -> None:
        """Prediction should return a valid PredictionResult."""
        result = trained_classifier.predict(benign_variant_features)

        assert isinstance(result, PredictionResult)
        assert isinstance(result.variant_class, VariantClass)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.class_probabilities) == 5
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 0.01
        assert len(result.explanation) > 0
        assert len(result.feature_importances) > 0

    def test_benign_variant_classification(
        self,
        trained_classifier: VariantClassifier,
        benign_variant_features: VariantFeatures,
    ) -> None:
        """Benign-like features should lean toward benign classification."""
        result = trained_classifier.predict(benign_variant_features)
        benign_prob = result.class_probabilities.get("benign", 0) + result.class_probabilities.get(
            "likely_benign", 0
        )
        pathogenic_prob = result.class_probabilities.get(
            "pathogenic", 0
        ) + result.class_probabilities.get("likely_pathogenic", 0)
        # Benign probability should be higher than pathogenic
        assert benign_prob > pathogenic_prob

    def test_pathogenic_variant_classification(
        self,
        trained_classifier: VariantClassifier,
        pathogenic_variant_features: VariantFeatures,
    ) -> None:
        """Pathogenic-like features should lean toward pathogenic classification."""
        result = trained_classifier.predict(pathogenic_variant_features)
        benign_prob = result.class_probabilities.get("benign", 0) + result.class_probabilities.get(
            "likely_benign", 0
        )
        pathogenic_prob = result.class_probabilities.get(
            "pathogenic", 0
        ) + result.class_probabilities.get("likely_pathogenic", 0)
        # Pathogenic probability should be higher than benign
        assert pathogenic_prob > benign_prob

    def test_batch_prediction(
        self,
        trained_classifier: VariantClassifier,
        benign_variant_features: VariantFeatures,
        pathogenic_variant_features: VariantFeatures,
    ) -> None:
        """Batch prediction should return results for all variants."""
        results = trained_classifier.predict_batch(
            [benign_variant_features, pathogenic_variant_features]
        )
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_save_and_load(
        self,
        trained_classifier: VariantClassifier,
        benign_variant_features: VariantFeatures,
    ) -> None:
        """Saved and loaded model should produce same predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            trained_classifier.save(model_path)

            assert model_path.exists()

            loaded = VariantClassifier()
            loaded.load(model_path)

            assert loaded.is_trained

            original_result = trained_classifier.predict(benign_variant_features)
            loaded_result = loaded.predict(benign_variant_features)

            assert original_result.variant_class == loaded_result.variant_class
            assert abs(original_result.confidence - loaded_result.confidence) < 0.01

    def test_load_nonexistent_file(self) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        classifier = VariantClassifier()
        with pytest.raises(FileNotFoundError):
            classifier.load("/tmp/nonexistent_model.joblib")

    def test_explanation_content(
        self,
        trained_classifier: VariantClassifier,
        pathogenic_variant_features: VariantFeatures,
    ) -> None:
        """Explanation should contain relevant clinical information."""
        result = trained_classifier.predict(pathogenic_variant_features)

        explanation_text = " ".join(result.explanation)
        # Should mention the predicted class
        assert "Predicted class" in explanation_text
        # Should mention population frequency
        assert "gnomAD" in explanation_text or "Absent" in explanation_text
        # Should mention in-silico predictors
        assert (
            "SIFT" in explanation_text or "REVEL" in explanation_text or "CADD" in explanation_text
        )

    def test_is_trained_property(self) -> None:
        """is_trained should reflect training state."""
        classifier = VariantClassifier()
        assert not classifier.is_trained

        features, labels = generate_synthetic_training_data(n_samples=50)
        classifier.train(features, labels, calibrate=False)
        assert classifier.is_trained


class TestSyntheticDataGeneration:
    """Tests for the synthetic data generator."""

    def test_generates_correct_count(self) -> None:
        """Should generate the requested number of samples."""
        features, labels = generate_synthetic_training_data(n_samples=100)
        assert len(features) == 100
        assert len(labels) == 100

    def test_all_classes_represented(self) -> None:
        """All five ACMG classes should appear in the generated data."""
        _, labels = generate_synthetic_training_data(n_samples=500)
        classes = set(labels)
        assert len(classes) == 5
        for vc in VariantClass:
            assert vc in classes

    def test_reproducibility(self) -> None:
        """Same random_state should produce identical data."""
        f1, l1 = generate_synthetic_training_data(n_samples=50, random_state=123)
        f2, l2 = generate_synthetic_training_data(n_samples=50, random_state=123)

        for feat1, feat2 in zip(f1, f2, strict=True):
            np.testing.assert_array_equal(feat1.to_array(), feat2.to_array())
        assert l1 == l2

    def test_feature_ranges(self) -> None:
        """Generated features should be within valid ranges."""
        features, _ = generate_synthetic_training_data(n_samples=200)
        for f in features:
            assert -14 <= f.phylop_score <= 6.4
            assert 0 <= f.phastcons_score <= 1
            assert 0 <= f.sift_score <= 1
            assert 0 <= f.polyphen2_score <= 1
            assert f.gnomad_af >= 0
            assert f.cadd_phred >= 0
