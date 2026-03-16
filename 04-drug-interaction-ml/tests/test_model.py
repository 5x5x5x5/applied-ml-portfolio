"""Tests for the drug interaction predictor model."""

from __future__ import annotations

import pandas as pd
import pytest

from drug_interaction.models.interaction_predictor import (
    DrugInteractionPredictor,
    HyperparameterConfig,
    InteractionPrediction,
    InteractionType,
    ModelMetrics,
    SeverityLevel,
)

# ---------------------------------------------------------------------------
# InteractionPredictor training tests
# ---------------------------------------------------------------------------


class TestDrugInteractionPredictor:
    """Tests for DrugInteractionPredictor."""

    def test_train_returns_metrics(
        self,
        sample_feature_df: pd.DataFrame,
        sample_severity_labels: pd.Series,
        sample_type_labels: pd.Series,
    ) -> None:
        """Training produces valid ModelMetrics."""
        predictor = DrugInteractionPredictor(
            hyperparams=HyperparameterConfig(n_estimators=10, max_depth=3),
            n_cv_folds=3,
        )
        metrics = predictor.train(sample_feature_df, sample_severity_labels, sample_type_labels)

        assert isinstance(metrics, ModelMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0
        assert 0.0 <= metrics.precision_macro <= 1.0
        assert 0.0 <= metrics.recall_macro <= 1.0
        assert len(metrics.feature_importances) > 0

    def test_train_with_eval_set(
        self,
        sample_feature_df: pd.DataFrame,
        sample_severity_labels: pd.Series,
        sample_type_labels: pd.Series,
    ) -> None:
        """Training with eval_set succeeds."""
        n_train = 150
        X_train = sample_feature_df.iloc[:n_train]
        X_val = sample_feature_df.iloc[n_train:]
        y_sev_train = sample_severity_labels.iloc[:n_train]
        y_sev_val = sample_severity_labels.iloc[n_train:]
        y_type_train = sample_type_labels.iloc[:n_train]
        y_type_val = sample_type_labels.iloc[n_train:]

        predictor = DrugInteractionPredictor(
            hyperparams=HyperparameterConfig(n_estimators=10, max_depth=3),
        )
        metrics = predictor.train(
            X_train,
            y_sev_train,
            y_type_train,
            eval_set=(X_val, y_sev_val, y_type_val),
        )
        assert isinstance(metrics, ModelMetrics)

    def test_cross_validate_returns_cv_metrics(
        self,
        sample_feature_df: pd.DataFrame,
        sample_severity_labels: pd.Series,
    ) -> None:
        """Cross-validation returns metrics with CV mean and std."""
        predictor = DrugInteractionPredictor(
            hyperparams=HyperparameterConfig(n_estimators=10, max_depth=3),
            n_cv_folds=3,
        )
        metrics = predictor.cross_validate(sample_feature_df, sample_severity_labels)

        assert metrics.cv_f1_mean is not None
        assert metrics.cv_f1_std is not None
        assert metrics.cv_f1_mean >= 0.0
        assert metrics.cv_f1_std >= 0.0

    def test_predict_returns_predictions(
        self,
        sample_feature_df: pd.DataFrame,
        sample_severity_labels: pd.Series,
        sample_type_labels: pd.Series,
    ) -> None:
        """Prediction returns list of InteractionPrediction objects."""
        predictor = DrugInteractionPredictor(
            hyperparams=HyperparameterConfig(n_estimators=10, max_depth=3),
        )
        predictor.train(sample_feature_df, sample_severity_labels, sample_type_labels)

        test_df = sample_feature_df.iloc[:5]
        drug_a_ids = [f"DRUG_A_{i}" for i in range(5)]
        drug_b_ids = [f"DRUG_B_{i}" for i in range(5)]

        predictions = predictor.predict(test_df, drug_a_ids, drug_b_ids)

        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, InteractionPrediction)
            assert 0.0 <= pred.interaction_probability <= 1.0
            assert 0.0 <= pred.confidence <= 1.0
            assert isinstance(pred.severity, SeverityLevel)
            assert isinstance(pred.interaction_type, InteractionType)

    def test_predict_without_training_raises(self) -> None:
        """Predicting without training raises RuntimeError."""
        predictor = DrugInteractionPredictor()
        dummy_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(dummy_df, ["A"], ["B"])

    def test_save_and_load(
        self,
        sample_feature_df: pd.DataFrame,
        sample_severity_labels: pd.Series,
        sample_type_labels: pd.Series,
        tmp_path: str,
    ) -> None:
        """Model can be saved and loaded."""
        predictor = DrugInteractionPredictor(
            hyperparams=HyperparameterConfig(n_estimators=10, max_depth=3),
        )
        predictor.train(sample_feature_df, sample_severity_labels, sample_type_labels)
        predictor.save(tmp_path)

        new_predictor = DrugInteractionPredictor()
        new_predictor.load(tmp_path)
        # Verify the loaded model can make predictions
        assert new_predictor._severity_model is not None
        assert new_predictor._type_model is not None


# ---------------------------------------------------------------------------
# HyperparameterConfig tests
# ---------------------------------------------------------------------------


class TestHyperparameterConfig:
    """Tests for HyperparameterConfig validation."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = HyperparameterConfig()
        assert config.n_estimators == 500
        assert config.max_depth == 6
        assert config.learning_rate == 0.05
        assert config.eval_metric == "mlogloss"

    def test_custom_config(self) -> None:
        """Custom config values are accepted."""
        config = HyperparameterConfig(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
        )
        assert config.n_estimators == 100
        assert config.max_depth == 4


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSeverityLevel:
    """Tests for SeverityLevel enum."""

    def test_all_levels_defined(self) -> None:
        """All expected severity levels exist."""
        expected = {"none", "mild", "moderate", "severe", "contraindicated"}
        actual = {level.value for level in SeverityLevel}
        assert actual == expected


class TestInteractionType:
    """Tests for InteractionType enum."""

    def test_all_types_defined(self) -> None:
        """All expected interaction types exist."""
        expected = {"pharmacokinetic", "pharmacodynamic", "unknown"}
        actual = {t.value for t in InteractionType}
        assert actual == expected


class TestInteractionPrediction:
    """Tests for InteractionPrediction model."""

    def test_valid_prediction(self) -> None:
        """Valid prediction is created successfully."""
        pred = InteractionPrediction(
            drug_a_id="DRUG_001",
            drug_b_id="DRUG_002",
            interaction_probability=0.85,
            severity=SeverityLevel.MODERATE,
            interaction_type=InteractionType.PHARMACOKINETIC,
            confidence=0.78,
        )
        assert pred.drug_a_id == "DRUG_001"
        assert pred.interaction_probability == 0.85

    def test_probability_bounds(self) -> None:
        """Probability out of [0,1] is rejected."""
        with pytest.raises(Exception):
            InteractionPrediction(
                drug_a_id="A",
                drug_b_id="B",
                interaction_probability=1.5,
                severity=SeverityLevel.NONE,
                interaction_type=InteractionType.UNKNOWN,
                confidence=0.5,
            )
