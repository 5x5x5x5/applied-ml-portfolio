"""Tests for the anomaly detection module."""

from __future__ import annotations

import numpy as np

from biomarker_dash.models.anomaly_detector import (
    MIN_SAMPLES_FOR_BASELINE,
    MIN_SAMPLES_FOR_ISOLATION_FOREST,
    AnomalyDetector,
    PatientBaseline,
)
from biomarker_dash.schemas import (
    AlertSeverity,
    BiomarkerType,
    PatientContext,
)
from tests.conftest import make_reading


class TestPatientBaseline:
    """Tests for the PatientBaseline statistics tracker."""

    def test_empty_baseline(self) -> None:
        baseline = PatientBaseline()
        assert baseline.count == 0
        assert baseline.mean == 0.0

    def test_single_value(self) -> None:
        baseline = PatientBaseline()
        baseline.add(85.0, 1000.0)
        assert baseline.count == 1
        assert baseline.mean == 85.0

    def test_multiple_values_statistics(self) -> None:
        baseline = PatientBaseline()
        for v in [80.0, 85.0, 90.0]:
            baseline.add(v, 1000.0)
        assert baseline.count == 3
        assert abs(baseline.mean - 85.0) < 0.01
        assert baseline.std > 0

    def test_rolling_window_truncation(self) -> None:
        baseline = PatientBaseline()
        for i in range(600):
            baseline.add(float(i), float(i))
        assert baseline.count == 500


class TestRangeBasedDetection:
    """Tests for range-based (normal range) anomaly detection."""

    def test_normal_glucose(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=85.0, biomarker_type=BiomarkerType.GLUCOSE)
        result = anomaly_detector.detect(reading)
        assert not result.is_anomaly
        assert result.severity == AlertSeverity.INFO

    def test_high_glucose(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=250.0, biomarker_type=BiomarkerType.GLUCOSE)
        result = anomaly_detector.detect(reading)
        assert result.is_anomaly
        assert result.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)

    def test_low_glucose(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=40.0, biomarker_type=BiomarkerType.GLUCOSE)
        result = anomaly_detector.detect(reading)
        assert result.is_anomaly

    def test_borderline_value(self, anomaly_detector: AnomalyDetector) -> None:
        # Just barely above normal range for glucose (100)
        reading = make_reading(value=105.0, biomarker_type=BiomarkerType.GLUCOSE)
        result = anomaly_detector.detect(reading)
        assert result.is_anomaly
        assert result.severity == AlertSeverity.LOW

    def test_critical_heart_rate(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=160.0, biomarker_type=BiomarkerType.HEART_RATE, unit="bpm")
        result = anomaly_detector.detect(reading)
        assert result.is_anomaly
        assert result.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)

    def test_normal_heart_rate(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=72.0, biomarker_type=BiomarkerType.HEART_RATE, unit="bpm")
        result = anomaly_detector.detect(reading)
        assert not result.is_anomaly

    def test_sex_adjusted_hemoglobin_male(
        self,
        anomaly_detector: AnomalyDetector,
        patient_context: PatientContext,
    ) -> None:
        """Male patient with hemoglobin in male-normal but below female range."""
        reading = make_reading(
            patient_id=patient_context.patient_id,
            value=14.0,
            biomarker_type=BiomarkerType.HEMOGLOBIN,
            unit="g/dL",
        )
        result = anomaly_detector.detect(reading, patient_context)
        assert not result.is_anomaly

    def test_sex_adjusted_hemoglobin_female(
        self,
        anomaly_detector: AnomalyDetector,
        patient_context_female: PatientContext,
    ) -> None:
        """Female patient with hemoglobin above female normal range."""
        reading = make_reading(
            patient_id=patient_context_female.patient_id,
            value=16.0,
            biomarker_type=BiomarkerType.HEMOGLOBIN,
            unit="g/dL",
        )
        result = anomaly_detector.detect(reading, patient_context_female)
        assert result.is_anomaly

    def test_anomaly_score_increases_with_deviation(
        self, anomaly_detector: AnomalyDetector
    ) -> None:
        """More extreme values should produce higher anomaly scores."""
        reading_mild = make_reading(value=110.0, biomarker_type=BiomarkerType.GLUCOSE)
        reading_extreme = make_reading(value=300.0, biomarker_type=BiomarkerType.GLUCOSE)
        result_mild = anomaly_detector.detect(reading_mild)
        result_extreme = anomaly_detector.detect(reading_extreme)
        assert result_extreme.anomaly_score > result_mild.anomaly_score

    def test_normal_range_in_result(self, anomaly_detector: AnomalyDetector) -> None:
        reading = make_reading(value=85.0, biomarker_type=BiomarkerType.GLUCOSE)
        result = anomaly_detector.detect(reading)
        assert result.normal_range == (70.0, 100.0)


class TestZScoreDetection:
    """Tests for Z-score based anomaly detection."""

    def test_zscore_not_active_with_few_samples(self, anomaly_detector: AnomalyDetector) -> None:
        """Z-score detection requires MIN_SAMPLES_FOR_BASELINE readings."""
        for i in range(MIN_SAMPLES_FOR_BASELINE - 1):
            reading = make_reading(value=85.0 + i * 0.1)
            result = anomaly_detector.detect(reading)
        # Detection method should still be range_based since we don't
        # have enough for Z-score
        assert result.detection_method == "range_based"

    def test_zscore_detects_deviation_from_baseline(
        self, anomaly_detector: AnomalyDetector
    ) -> None:
        """After building baseline, a large deviation should be caught."""
        # Build a stable baseline
        for i in range(MIN_SAMPLES_FOR_BASELINE + 5):
            reading = make_reading(value=85.0 + np.random.normal(0, 1))
            anomaly_detector.detect(reading)

        # Now submit a significantly deviant value
        anomalous = make_reading(value=140.0)  # ~55 units from mean of ~85
        result = anomaly_detector.detect(anomalous)
        # Should be flagged either by range or zscore
        assert result.is_anomaly

    def test_zscore_normal_after_baseline(self, anomaly_detector: AnomalyDetector) -> None:
        """Values within normal variation should not trigger Z-score."""
        for i in range(MIN_SAMPLES_FOR_BASELINE + 5):
            reading = make_reading(value=85.0 + np.random.normal(0, 2))
            anomaly_detector.detect(reading)

        normal = make_reading(value=87.0)
        result = anomaly_detector.detect(normal)
        assert not result.is_anomaly


class TestIsolationForest:
    """Tests for the Isolation Forest multivariate detection."""

    def test_isolation_forest_requires_sufficient_data(
        self, anomaly_detector: AnomalyDetector
    ) -> None:
        """Isolation Forest should not fire with too few samples."""
        reading = make_reading(value=85.0)
        result = anomaly_detector.detect(reading)
        # With only 1 sample, IF should not contribute
        assert result.detection_method in ("range_based", "zscore")

    def test_isolation_forest_trains_with_multivariate_data(
        self, anomaly_detector: AnomalyDetector
    ) -> None:
        """After sufficient multivariate data, IF should contribute."""
        biomarkers = [
            BiomarkerType.GLUCOSE,
            BiomarkerType.HEART_RATE,
            BiomarkerType.POTASSIUM,
        ]

        for i in range(MIN_SAMPLES_FOR_ISOLATION_FOREST + 10):
            for bt in biomarkers:
                reading = make_reading(
                    patient_id="TEST001",
                    biomarker_type=bt,
                    value=float(
                        np.random.normal(85, 5)
                        if bt == BiomarkerType.GLUCOSE
                        else np.random.normal(75, 5)
                        if bt == BiomarkerType.HEART_RATE
                        else np.random.normal(4.2, 0.2)
                    ),
                )
                anomaly_detector.detect(reading)

        # The IF model should now exist
        assert "TEST001" in anomaly_detector._isolation_forests


class TestAnomalyDetectorUtility:
    """Tests for utility methods."""

    def test_get_baseline_stats_empty(self, anomaly_detector: AnomalyDetector) -> None:
        stats = anomaly_detector.get_baseline_stats("NONE", "glucose")
        assert stats == {"count": 0}

    def test_get_baseline_stats_populated(self, anomaly_detector: AnomalyDetector) -> None:
        for i in range(5):
            reading = make_reading(value=80.0 + i)
            anomaly_detector.detect(reading)

        stats = anomaly_detector.get_baseline_stats("TEST001", "glucose")
        assert stats["count"] == 5
        assert "mean" in stats
        assert "std" in stats

    def test_reset_patient(self, anomaly_detector: AnomalyDetector) -> None:
        for _ in range(5):
            reading = make_reading(value=85.0)
            anomaly_detector.detect(reading)

        anomaly_detector.reset_patient("TEST001")
        stats = anomaly_detector.get_baseline_stats("TEST001", "glucose")
        assert stats == {"count": 0}
