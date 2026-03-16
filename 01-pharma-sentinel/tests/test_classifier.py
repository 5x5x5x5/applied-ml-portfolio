"""Tests for the AdverseEventClassifier model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pharma_sentinel.models.adverse_event_classifier import (
    SEVERITY_LABELS,
    AdverseEventClassifier,
    TextPreprocessor,
)

# ─────────────────────────────────────────────────────────────────────
# TextPreprocessor tests
# ─────────────────────────────────────────────────────────────────────


class TestTextPreprocessor:
    """Tests for the TextPreprocessor transformer."""

    def test_basic_preprocessing(self) -> None:
        """Test basic text cleaning and normalization."""
        preprocessor = TextPreprocessor()
        result = preprocessor.transform(["  Patient had NAUSEA  and  vomiting  "])
        assert len(result) == 1
        assert "nausea" in result[0]
        assert "vomiting" in result[0]
        # Should not have excessive whitespace
        assert "  " not in result[0]

    def test_empty_text(self) -> None:
        """Test handling of empty strings."""
        preprocessor = TextPreprocessor()
        result = preprocessor.transform(["", "   ", "valid text here"])
        assert result[0] == ""
        assert result[1] == ""
        assert "valid text here" in result[2]

    def test_severity_token_extraction(self) -> None:
        """Test that severity indicator tokens are appended."""
        preprocessor = TextPreprocessor()
        result = preprocessor.transform(
            ["Patient died from cardiac arrest and respiratory failure"]
        )
        assert "SEV_CRITICAL" in result[0]

    def test_phi_removal(self) -> None:
        """Test that PHI-like patterns are replaced."""
        preprocessor = TextPreprocessor()
        result = preprocessor.transform(["Patient SSN 123-45-6789 DOB 01/15/1990 had nausea"])
        assert "123-45-6789" not in result[0]
        assert "01/15/1990" not in result[0]
        assert "PHI_SSN" in result[0]
        assert "PHI_DATE" in result[0]

    def test_fit_is_noop(self) -> None:
        """Test that fit returns self without modification."""
        preprocessor = TextPreprocessor()
        result = preprocessor.fit(["some text"])
        assert result is preprocessor

    def test_number_removal_option(self) -> None:
        """Test number removal when configured."""
        preprocessor = TextPreprocessor(remove_numbers=True)
        result = preprocessor.transform(["Patient took 400mg dose"])
        assert "400" not in result[0]
        assert "NUM" in result[0]


# ─────────────────────────────────────────────────────────────────────
# AdverseEventClassifier tests
# ─────────────────────────────────────────────────────────────────────


class TestAdverseEventClassifier:
    """Tests for the main classifier model."""

    def test_initialization(self) -> None:
        """Test classifier initializes with correct defaults."""
        classifier = AdverseEventClassifier()
        assert classifier.pipeline is not None
        assert not classifier.is_fitted
        assert classifier.max_features == 15000
        assert classifier.ngram_range == (1, 3)

    def test_initialization_custom_params(self) -> None:
        """Test classifier with custom hyperparameters."""
        classifier = AdverseEventClassifier(
            max_features=5000,
            ngram_range=(1, 2),
            c_param=0.5,
            max_iter=500,
        )
        assert classifier.max_features == 5000
        assert classifier.ngram_range == (1, 2)
        assert classifier.c_param == 0.5
        assert classifier.max_iter == 500

    def test_train_basic(
        self,
        sample_training_texts: list[str],
        sample_training_labels: list[str],
    ) -> None:
        """Test basic model training succeeds."""
        classifier = AdverseEventClassifier(max_features=5000, ngram_range=(1, 2))
        metrics = classifier.train(
            texts=sample_training_texts,
            labels=sample_training_labels,
            validate=False,
        )
        assert classifier.is_fitted
        assert "train_f1_weighted" in metrics
        assert metrics["train_f1_weighted"] > 0.0
        assert metrics["n_samples"] == len(sample_training_texts)

    def test_train_with_validation(
        self,
        sample_training_texts: list[str],
        sample_training_labels: list[str],
    ) -> None:
        """Test training with cross-validation."""
        classifier = AdverseEventClassifier(max_features=5000, ngram_range=(1, 2))
        metrics = classifier.train(
            texts=sample_training_texts,
            labels=sample_training_labels,
            validate=True,
            cv_folds=2,
        )
        assert "cv_f1_mean" in metrics
        assert "cv_f1_std" in metrics
        assert len(metrics["cv_scores"]) == 2

    def test_train_empty_dataset(self) -> None:
        """Test training raises on empty dataset."""
        classifier = AdverseEventClassifier()
        with pytest.raises(ValueError, match="Cannot train on empty dataset"):
            classifier.train(texts=[], labels=[])

    def test_train_mismatched_lengths(self) -> None:
        """Test training raises on mismatched text/label lengths."""
        classifier = AdverseEventClassifier()
        with pytest.raises(ValueError, match="Mismatched lengths"):
            classifier.train(
                texts=["text1", "text2"],
                labels=["mild"],
            )

    def test_train_invalid_labels(self) -> None:
        """Test training raises on invalid severity labels."""
        classifier = AdverseEventClassifier()
        with pytest.raises(ValueError, match="Invalid severity labels"):
            classifier.train(
                texts=["some text", "other text"],
                labels=["mild", "invalid_label"],
            )

    def test_predict_single(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test single prediction returns correct structure."""
        result = trained_classifier.predict_single("Patient experienced mild headache and fatigue")
        assert "severity" in result
        assert result["severity"] in SEVERITY_LABELS
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "probabilities" in result
        assert len(result["probabilities"]) == 4

    def test_predict_batch(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test batch prediction returns correct number of results."""
        texts = [
            "Mild headache and nausea reported",
            "Patient died from cardiac arrest",
            "Hospitalized with severe liver failure",
        ]
        results = trained_classifier.predict(texts)
        assert len(results) == 3
        for result in results:
            assert result["severity"] in SEVERITY_LABELS
            assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(
        self, trained_classifier: AdverseEventClassifier
    ) -> None:
        """Test that class probabilities sum to approximately 1."""
        result = trained_classifier.predict_single("Patient had moderate vomiting")
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_predict_unfitted_raises(self) -> None:
        """Test prediction on unfitted model raises RuntimeError."""
        classifier = AdverseEventClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            classifier.predict(["some text"])

    def test_predict_empty_input_raises(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test prediction on empty input raises ValueError."""
        with pytest.raises(ValueError, match="Cannot predict on empty input"):
            trained_classifier.predict([])

    def test_save_and_load_model(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test model save/load roundtrip preserves predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"

            # Save
            trained_classifier.save_model(model_path)
            assert model_path.exists()
            assert model_path.stat().st_size > 0

            # Load into fresh classifier
            loaded = AdverseEventClassifier()
            loaded.load_model(model_path)
            assert loaded.is_fitted

            # Verify predictions match
            test_text = "Patient experienced severe seizure and hospitalization"
            original_result = trained_classifier.predict_single(test_text)
            loaded_result = loaded.predict_single(test_text)

            assert original_result["severity"] == loaded_result["severity"]
            assert abs(original_result["confidence"] - loaded_result["confidence"]) < 0.001

    def test_save_unfitted_raises(self) -> None:
        """Test saving unfitted model raises RuntimeError."""
        classifier = AdverseEventClassifier()
        with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
            classifier.save_model("/tmp/model.joblib")

    def test_load_nonexistent_raises(self) -> None:
        """Test loading from non-existent path raises FileNotFoundError."""
        classifier = AdverseEventClassifier()
        with pytest.raises(FileNotFoundError):
            classifier.load_model("/nonexistent/path/model.joblib")

    def test_load_invalid_artifact_raises(self) -> None:
        """Test loading invalid artifact raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            import joblib

            joblib.dump({"invalid": "data"}, f.name)
            classifier = AdverseEventClassifier()
            with pytest.raises(ValueError, match="Invalid model artifact"):
                classifier.load_model(f.name)

    def test_feature_importance(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test feature importance extraction."""
        importance = trained_classifier.get_feature_importance(top_n=5)
        assert set(importance.keys()) == set(SEVERITY_LABELS)
        for severity, features in importance.items():
            assert len(features) <= 5
            for feat in features:
                assert "feature" in feat
                assert "weight" in feat
                assert isinstance(feat["weight"], float)

    def test_feature_importance_unfitted_raises(self) -> None:
        """Test feature importance on unfitted model raises."""
        classifier = AdverseEventClassifier()
        with pytest.raises(RuntimeError):
            classifier.get_feature_importance()

    def test_metadata_populated_after_training(
        self,
        sample_training_texts: list[str],
        sample_training_labels: list[str],
    ) -> None:
        """Test that metadata is populated after training."""
        classifier = AdverseEventClassifier(max_features=5000, ngram_range=(1, 2))
        classifier.train(
            texts=sample_training_texts,
            labels=sample_training_labels,
            validate=False,
        )
        assert "training_metrics" in classifier.metadata
        assert "hyperparameters" in classifier.metadata
        assert classifier.metadata["hyperparameters"]["max_features"] == 5000

    def test_severity_keyword_detection(self, trained_classifier: AdverseEventClassifier) -> None:
        """Test that severity keywords influence predictions."""
        critical_text = "Patient died from fatal cardiac arrest and anaphylaxis"
        mild_text = "Patient reported mild headache and dry mouth"

        critical_result = trained_classifier.predict_single(critical_text)
        mild_result = trained_classifier.predict_single(mild_text)

        # Critical text should have higher critical probability
        assert critical_result["probabilities"].get("critical", 0) > mild_result[
            "probabilities"
        ].get("critical", 0)
