"""Tests for feature and model drift detection."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from feature_forge.drift.feature_drift_detector import (
    DriftReport,
    DriftResult,
    DriftSeverity,
    DriftThresholds,
    FeatureDriftDetector,
)
from feature_forge.drift.model_drift_detector import (
    DriftType,
    ModelDriftDetector,
    ModelDriftResult,
    ModelMonitorConfig,
)
from feature_forge.extractors.structured_extractor import SnowflakeConfig


class TestPSI:
    """Tests for Population Stability Index computation."""

    def test_identical_distributions_zero_psi(self) -> None:
        """PSI of identical distributions is approximately zero."""
        dist = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        psi = FeatureDriftDetector.compute_psi(dist, dist)
        assert psi < 0.01

    def test_shifted_distribution_positive_psi(self) -> None:
        """PSI of shifted distributions is positive."""
        baseline = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        current = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        psi = FeatureDriftDetector.compute_psi(baseline, current)
        assert psi > 0.1

    def test_psi_handles_zero_bins(self) -> None:
        """PSI handles bins with zero counts via epsilon clipping."""
        baseline = np.array([0.5, 0.5, 0.0, 0.0])
        current = np.array([0.0, 0.0, 0.5, 0.5])
        psi = FeatureDriftDetector.compute_psi(baseline, current)
        assert np.isfinite(psi)
        assert psi > 0

    def test_psi_symmetry_approximate(self) -> None:
        """PSI is approximately symmetric (not exactly due to KL divergence)."""
        a = np.array([0.4, 0.3, 0.2, 0.1])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        psi_ab = FeatureDriftDetector.compute_psi(a, b)
        psi_ba = FeatureDriftDetector.compute_psi(b, a)
        # PSI is symmetric by definition (sum of both KL terms)
        assert abs(psi_ab - psi_ba) < 0.01

    def test_psi_moderate_shift(self) -> None:
        """Moderate distribution shift gives PSI in expected range."""
        baseline = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.array([0.30, 0.30, 0.20, 0.20])
        psi = FeatureDriftDetector.compute_psi(baseline, current)
        assert 0.0 < psi < 0.1  # Small shift


class TestKSTest:
    """Tests for Kolmogorov-Smirnov test."""

    def test_same_distribution_high_p_value(self) -> None:
        """Samples from the same distribution produce high p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        stat, p_value = FeatureDriftDetector.ks_test(a, b)
        assert p_value > 0.05

    def test_different_distributions_low_p_value(self) -> None:
        """Samples from different distributions produce low p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(3, 1, 500)
        stat, p_value = FeatureDriftDetector.ks_test(a, b)
        assert p_value < 0.01

    def test_ks_statistic_range(self) -> None:
        """KS statistic is between 0 and 1."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 1, 100)
        stat, _ = FeatureDriftDetector.ks_test(a, b)
        assert 0 <= stat <= 1


class TestChiSquaredTest:
    """Tests for chi-squared categorical drift test."""

    def test_identical_counts_high_p_value(self) -> None:
        """Identical category counts give high p-value."""
        counts = {"A": 100, "B": 200, "C": 100}
        chi2, p_value = FeatureDriftDetector.chi_squared_test(counts, counts)
        assert p_value > 0.99

    def test_shifted_counts_low_p_value(self) -> None:
        """Significantly different category counts give low p-value."""
        baseline = {"A": 100, "B": 200, "C": 100}
        current = {"A": 300, "B": 50, "C": 50}
        chi2, p_value = FeatureDriftDetector.chi_squared_test(baseline, current)
        assert p_value < 0.05

    def test_new_categories(self) -> None:
        """Handles categories present in only one distribution."""
        baseline = {"A": 100, "B": 200}
        current = {"A": 100, "B": 200, "C": 50}
        chi2, p_value = FeatureDriftDetector.chi_squared_test(baseline, current)
        assert chi2 > 0

    def test_single_category(self) -> None:
        """Single category returns no drift."""
        baseline = {"A": 100}
        current = {"A": 150}
        chi2, p_value = FeatureDriftDetector.chi_squared_test(baseline, current)
        assert chi2 == 0.0
        assert p_value == 1.0


class TestDriftSeverityClassification:
    """Tests for severity classification."""

    def test_no_drift(self, snowflake_config: SnowflakeConfig) -> None:
        detector = FeatureDriftDetector(snowflake_config)
        assert detector.classify_psi_severity(0.05) == DriftSeverity.NONE

    def test_low_drift(self, snowflake_config: SnowflakeConfig) -> None:
        detector = FeatureDriftDetector(snowflake_config)
        assert detector.classify_psi_severity(0.15) == DriftSeverity.LOW

    def test_medium_drift(self, snowflake_config: SnowflakeConfig) -> None:
        detector = FeatureDriftDetector(snowflake_config)
        assert detector.classify_psi_severity(0.25) == DriftSeverity.MEDIUM

    def test_high_drift(self, snowflake_config: SnowflakeConfig) -> None:
        detector = FeatureDriftDetector(snowflake_config)
        assert detector.classify_psi_severity(0.35) == DriftSeverity.HIGH

    def test_critical_drift(self, snowflake_config: SnowflakeConfig) -> None:
        detector = FeatureDriftDetector(snowflake_config)
        assert detector.classify_psi_severity(0.6) == DriftSeverity.CRITICAL

    def test_custom_thresholds(self, snowflake_config: SnowflakeConfig) -> None:
        thresholds = DriftThresholds(psi_low=0.05, psi_medium=0.1, psi_high=0.15, psi_critical=0.2)
        detector = FeatureDriftDetector(snowflake_config, thresholds=thresholds)
        assert detector.classify_psi_severity(0.08) == DriftSeverity.LOW
        assert detector.classify_psi_severity(0.12) == DriftSeverity.MEDIUM


class TestDriftReport:
    """Tests for drift report generation."""

    def test_report_generation(self, snowflake_config: SnowflakeConfig) -> None:
        """Report correctly aggregates multiple results."""
        detector = FeatureDriftDetector(snowflake_config)

        results = [
            DriftResult(
                feature_name="feature_a",
                test_name="PSI",
                statistic=0.05,
                p_value=None,
                threshold=0.1,
                is_drifted=False,
                severity=DriftSeverity.NONE,
                baseline_period="2025-01-01 - 2025-01-31",
                current_period="2025-02-01 - 2025-02-28",
            ),
            DriftResult(
                feature_name="feature_b",
                test_name="PSI",
                statistic=0.35,
                p_value=None,
                threshold=0.1,
                is_drifted=True,
                severity=DriftSeverity.HIGH,
                baseline_period="2025-01-01 - 2025-01-31",
                current_period="2025-02-01 - 2025-02-28",
            ),
        ]

        report = detector.generate_drift_report(results, report_id="test-001")

        assert report.report_id == "test-001"
        assert report.total_features_checked == 2
        assert report.drifted_features_count == 1
        assert report.overall_severity == DriftSeverity.HIGH
        assert report.severity_counts["HIGH"] == 1
        assert report.severity_counts["NONE"] == 1

    def test_report_to_dict(self) -> None:
        """Report serialises to a dict."""
        report = DriftReport(
            report_id="test",
            generated_at=datetime(2025, 3, 1),
            total_features_checked=1,
            drifted_features_count=0,
            severity_counts={"NONE": 1},
            results=[],
            overall_severity=DriftSeverity.NONE,
        )
        d = report.to_dict()
        assert d["report_id"] == "test"
        assert d["overall_severity"] == "NONE"


class TestModelDriftDetector:
    """Tests for ModelDriftDetector."""

    @pytest.fixture()
    def monitor_config(self) -> ModelMonitorConfig:
        return ModelMonitorConfig(
            model_name="readmission_model",
            model_version="1.0",
            predictions_table="MODEL_PREDICTIONS",
            ground_truth_table="GROUND_TRUTH_LABELS",
        )

    def test_classify_prediction_severity(
        self,
        snowflake_config: SnowflakeConfig,
        monitor_config: ModelMonitorConfig,
    ) -> None:
        detector = ModelDriftDetector(snowflake_config, monitor_config)
        assert detector._classify_prediction_severity(0.05) == DriftSeverity.NONE
        assert detector._classify_prediction_severity(0.15) == DriftSeverity.LOW
        assert detector._classify_prediction_severity(0.25) == DriftSeverity.MEDIUM
        assert detector._classify_prediction_severity(0.35) == DriftSeverity.HIGH
        assert detector._classify_prediction_severity(0.55) == DriftSeverity.CRITICAL

    def test_classify_accuracy_severity(self) -> None:
        assert ModelDriftDetector._classify_accuracy_severity(1.0) == DriftSeverity.NONE
        assert ModelDriftDetector._classify_accuracy_severity(3.0) == DriftSeverity.LOW
        assert ModelDriftDetector._classify_accuracy_severity(7.0) == DriftSeverity.MEDIUM
        assert ModelDriftDetector._classify_accuracy_severity(15.0) == DriftSeverity.HIGH
        assert ModelDriftDetector._classify_accuracy_severity(25.0) == DriftSeverity.CRITICAL

    def test_detect_prediction_drift_empty_data(
        self,
        snowflake_config: SnowflakeConfig,
        monitor_config: ModelMonitorConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Returns no-drift result when prediction data is empty."""
        cursor = mock_snowflake_connection.cursor.return_value
        cursor.fetchall.return_value = []

        detector = ModelDriftDetector(snowflake_config, monitor_config)
        result = detector.detect_prediction_drift()

        assert not result.is_drifted
        assert result.severity == DriftSeverity.NONE
        assert result.details.get("reason") == "insufficient_data"

    def test_model_drift_result_dataclass(self) -> None:
        """ModelDriftResult fields are set correctly."""
        result = ModelDriftResult(
            model_name="test_model",
            model_version="2.0",
            drift_type=DriftType.PREDICTION_DISTRIBUTION,
            metric_name="PSI",
            baseline_value=0.5,
            current_value=0.6,
            change_pct=20.0,
            severity=DriftSeverity.MEDIUM,
            is_drifted=True,
            should_retrain=False,
        )
        assert result.model_name == "test_model"
        assert result.drift_type == DriftType.PREDICTION_DISTRIBUTION
        assert result.is_drifted is True
        assert result.should_retrain is False


class TestHistogramToProportions:
    """Tests for _histogram_to_proportions helper."""

    def test_dict_histogram(self) -> None:
        hist = [{"count": 10}, {"count": 20}, {"count": 30}]
        result = FeatureDriftDetector._histogram_to_proportions(hist)
        np.testing.assert_allclose(result, [10 / 60, 20 / 60, 30 / 60])

    def test_json_string_histogram(self) -> None:
        import json

        hist = json.dumps([{"count": 5}, {"count": 5}])
        result = FeatureDriftDetector._histogram_to_proportions(hist)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_empty_histogram(self) -> None:
        result = FeatureDriftDetector._histogram_to_proportions([])
        assert len(result) == 1
        assert result[0] == 1.0
