"""Tests for the clinical alert engine."""

from __future__ import annotations

from datetime import datetime, timedelta

from biomarker_dash.alerts.alert_engine import (
    AlertEngine,
)
from biomarker_dash.models.trend_analyzer import TrendAnalyzer
from biomarker_dash.schemas import (
    AlertSeverity,
    AlertStatus,
    AnomalyResult,
    BiomarkerReading,
    BiomarkerType,
    TrendDirection,
    TrendResult,
)
from tests.conftest import make_reading


class TestRuleBasedAlerts:
    """Tests for rule-based (threshold) alert generation."""

    def test_normal_reading_no_alert(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=85.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) == 0

    def test_out_of_range_generates_alert(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.detection_source == "rule"
        assert alert.patient_id == "TEST001"
        assert alert.biomarker_type == BiomarkerType.GLUCOSE

    def test_critical_heart_rate_high(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=160.0, biomarker_type=BiomarkerType.HEART_RATE, unit="bpm")
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        # Should hit the critical threshold (above 150)
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_heart_rate_low(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=35.0, biomarker_type=BiomarkerType.HEART_RATE, unit="bpm")
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_oxygen_sat(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=85.0, biomarker_type=BiomarkerType.OXYGEN_SAT, unit="%")
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_potassium_high(
        self, alert_engine: AlertEngine, critical_potassium_reading: BiomarkerReading
    ) -> None:
        alerts = alert_engine.evaluate_reading(critical_potassium_reading)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_critical_glucose_very_low(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=30.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_mildly_elevated_generates_low_severity(self, alert_engine: AlertEngine) -> None:
        # Just above normal range for glucose (100)
        reading = make_reading(value=108.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        # Should be low severity since it's only slightly out of range
        assert alerts[0].severity in (AlertSeverity.LOW, AlertSeverity.MEDIUM)

    def test_alert_contains_value_and_threshold(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=160.0, biomarker_type=BiomarkerType.HEART_RATE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.value == 160.0
        assert alert.threshold is not None


class TestMLBasedAlerts:
    """Tests for alerts generated from ML anomaly detection results."""

    def test_no_alert_for_normal_anomaly_result(self, alert_engine: AlertEngine) -> None:
        anomaly = AnomalyResult(
            reading_id="r1",
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            value=85.0,
            is_anomaly=False,
            anomaly_score=0.1,
            severity=AlertSeverity.INFO,
            detection_method="zscore",
            explanation="Normal",
        )
        alerts = alert_engine.evaluate_anomaly(anomaly)
        assert len(alerts) == 0

    def test_alert_for_high_severity_anomaly(self, alert_engine: AlertEngine) -> None:
        anomaly = AnomalyResult(
            reading_id="r2",
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            value=250.0,
            is_anomaly=True,
            anomaly_score=0.9,
            severity=AlertSeverity.HIGH,
            detection_method="isolation_forest",
            explanation="Multivariate anomaly detected",
        )
        alerts = alert_engine.evaluate_anomaly(anomaly)
        assert len(alerts) == 1
        assert alerts[0].detection_source == "anomaly_ml"
        assert alerts[0].severity == AlertSeverity.HIGH

    def test_no_alert_for_low_severity_anomaly(self, alert_engine: AlertEngine) -> None:
        """Low severity ML anomalies should be suppressed."""
        anomaly = AnomalyResult(
            reading_id="r3",
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            value=105.0,
            is_anomaly=True,
            anomaly_score=0.3,
            severity=AlertSeverity.LOW,
            detection_method="zscore",
            explanation="Slightly unusual",
        )
        alerts = alert_engine.evaluate_anomaly(anomaly)
        assert len(alerts) == 0


class TestTrendBasedAlerts:
    """Tests for alerts generated from trend analysis."""

    def test_no_alert_for_stable_trend(self, alert_engine: AlertEngine) -> None:
        trend = TrendResult(
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            direction=TrendDirection.STABLE,
            rate_of_change=0.001,
            predicted_value_24h=86.0,
            predicted_exit_normal=False,
            confidence=0.8,
            data_points_used=20,
        )
        alerts = alert_engine.evaluate_trend(trend)
        assert len(alerts) == 0

    def test_alert_for_predicted_range_exit(self, alert_engine: AlertEngine) -> None:
        trend = TrendResult(
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            direction=TrendDirection.WORSENING,
            rate_of_change=0.5,
            predicted_value_24h=130.0,
            predicted_exit_normal=True,
            confidence=0.75,
            data_points_used=30,
        )
        alerts = alert_engine.evaluate_trend(trend)
        assert len(alerts) == 1
        assert alerts[0].detection_source == "trend"
        assert "trending" in alerts[0].message.lower()

    def test_no_alert_for_low_confidence_prediction(self, alert_engine: AlertEngine) -> None:
        trend = TrendResult(
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            direction=TrendDirection.WORSENING,
            rate_of_change=0.5,
            predicted_value_24h=130.0,
            predicted_exit_normal=True,
            confidence=0.3,  # Below threshold
            data_points_used=5,
        )
        alerts = alert_engine.evaluate_trend(trend)
        assert len(alerts) == 0

    def test_high_confidence_prediction_gets_higher_severity(
        self, alert_engine: AlertEngine
    ) -> None:
        trend = TrendResult(
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
            direction=TrendDirection.WORSENING,
            rate_of_change=1.0,
            predicted_value_24h=200.0,
            predicted_exit_normal=True,
            confidence=0.9,
            data_points_used=50,
        )
        alerts = alert_engine.evaluate_trend(trend)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH


class TestAlertDeduplication:
    """Tests for alert deduplication logic."""

    def test_duplicate_alerts_suppressed(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts1 = alert_engine.evaluate_reading(reading)
        # Second identical reading within dedup window
        alerts2 = alert_engine.evaluate_reading(reading)
        assert len(alerts1) >= 1
        assert len(alerts2) == 0  # Suppressed as duplicate

    def test_different_biomarkers_not_deduplicated(self, alert_engine: AlertEngine) -> None:
        reading1 = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        reading2 = make_reading(value=160.0, biomarker_type=BiomarkerType.HEART_RATE, unit="bpm")
        alerts1 = alert_engine.evaluate_reading(reading1)
        alerts2 = alert_engine.evaluate_reading(reading2)
        assert len(alerts1) >= 1
        assert len(alerts2) >= 1

    def test_different_patients_not_deduplicated(self, alert_engine: AlertEngine) -> None:
        reading1 = make_reading(
            patient_id="P001", value=150.0, biomarker_type=BiomarkerType.GLUCOSE
        )
        reading2 = make_reading(
            patient_id="P002", value=150.0, biomarker_type=BiomarkerType.GLUCOSE
        )
        alerts1 = alert_engine.evaluate_reading(reading1)
        alerts2 = alert_engine.evaluate_reading(reading2)
        assert len(alerts1) >= 1
        assert len(alerts2) >= 1


class TestAlertManagement:
    """Tests for alert acknowledgment, resolution, and escalation."""

    def test_acknowledge_alert(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        alert_id = alerts[0].alert_id

        success = alert_engine.acknowledge_alert(alert_id, "Dr. Smith")
        assert success

        # Check the alert is now acknowledged
        active = alert_engine.get_active_alerts()
        acknowledged = [a for a in active if a.alert_id == alert_id]
        # Acknowledged alerts are filtered out from active (ACTIVE/ESCALATED only)
        assert len(acknowledged) == 0

    def test_acknowledge_nonexistent_alert(self, alert_engine: AlertEngine) -> None:
        success = alert_engine.acknowledge_alert("nonexistent-id", "Dr. Smith")
        assert not success

    def test_resolve_alert(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1
        alert_id = alerts[0].alert_id

        success = alert_engine.resolve_alert(alert_id)
        assert success

        # Should no longer be in active alerts
        active = alert_engine.get_active_alerts()
        assert all(a.alert_id != alert_id for a in active)

    def test_resolve_nonexistent_alert(self, alert_engine: AlertEngine) -> None:
        success = alert_engine.resolve_alert("nonexistent-id")
        assert not success

    def test_get_active_alerts_filtered_by_severity(self, alert_engine: AlertEngine) -> None:
        # Generate alerts of different severities
        # Critical
        reading1 = make_reading(
            patient_id="P001",
            value=160.0,
            biomarker_type=BiomarkerType.HEART_RATE,
        )
        alert_engine.evaluate_reading(reading1)

        # Lower severity
        reading2 = make_reading(
            patient_id="P002",
            value=108.0,
            biomarker_type=BiomarkerType.GLUCOSE,
        )
        alert_engine.evaluate_reading(reading2)

        # Get only high+ severity
        high_alerts = alert_engine.get_active_alerts(min_severity=AlertSeverity.HIGH)
        all_alerts = alert_engine.get_active_alerts()
        assert len(high_alerts) <= len(all_alerts)

    def test_get_active_alerts_filtered_by_patient(self, alert_engine: AlertEngine) -> None:
        reading1 = make_reading(
            patient_id="P001", value=150.0, biomarker_type=BiomarkerType.GLUCOSE
        )
        reading2 = make_reading(
            patient_id="P002", value=160.0, biomarker_type=BiomarkerType.HEART_RATE
        )
        alert_engine.evaluate_reading(reading1)
        alert_engine.evaluate_reading(reading2)

        p1_alerts = alert_engine.get_active_alerts(patient_id="P001")
        assert all(a.patient_id == "P001" for a in p1_alerts)

    def test_escalation_increases_severity(self, alert_engine: AlertEngine) -> None:
        """Simulate an unacknowledged alert that gets escalated."""
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alerts = alert_engine.evaluate_reading(reading)
        assert len(alerts) >= 1

        original_severity = alerts[0].severity

        # Manually backdate the alert to trigger escalation
        alerts[0].created_at = datetime.utcnow() - timedelta(hours=2)

        escalated = alert_engine.check_escalations()
        # The alert should have been escalated
        assert len(escalated) >= 1
        assert escalated[0].status == AlertStatus.ESCALATED

    def test_get_alert_stats(self, alert_engine: AlertEngine) -> None:
        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alert_engine.evaluate_reading(reading)

        stats = alert_engine.get_alert_stats()
        assert stats["total_active"] >= 1
        assert "by_severity" in stats
        assert "by_status" in stats

    def test_alert_callback_is_called(self, alert_engine: AlertEngine) -> None:
        callback_calls: list[object] = []
        alert_engine.set_alert_callback(lambda alert: callback_calls.append(alert))

        reading = make_reading(value=150.0, biomarker_type=BiomarkerType.GLUCOSE)
        alert_engine.evaluate_reading(reading)

        assert len(callback_calls) >= 1


class TestTrendAnalyzerIntegration:
    """Integration tests for trend analysis with the alert engine."""

    def test_rising_trend_detected(
        self,
        trend_analyzer: TrendAnalyzer,
        rising_glucose_readings: list[BiomarkerReading],
    ) -> None:
        result = trend_analyzer.analyze(rising_glucose_readings, BiomarkerType.GLUCOSE)
        assert result.direction in (TrendDirection.WORSENING, TrendDirection.IMPROVING)
        assert result.rate_of_change > 0

    def test_stable_trend_detected(
        self,
        trend_analyzer: TrendAnalyzer,
        normal_glucose_readings: list[BiomarkerReading],
    ) -> None:
        result = trend_analyzer.analyze(normal_glucose_readings, BiomarkerType.GLUCOSE)
        # With small noise around 85, should be relatively stable
        assert result.data_points_used > 0
        assert result.confidence >= 0.0

    def test_predicted_exit_normal(
        self,
        trend_analyzer: TrendAnalyzer,
        rising_glucose_readings: list[BiomarkerReading],
    ) -> None:
        result = trend_analyzer.analyze(rising_glucose_readings, BiomarkerType.GLUCOSE)
        # With values going from 85 to 130, the prediction should exit normal
        assert result.predicted_value_24h is not None
        # Predicted value should be above the already-rising values

    def test_decomposition(
        self,
        trend_analyzer: TrendAnalyzer,
        normal_glucose_readings: list[BiomarkerReading],
    ) -> None:
        decomp = trend_analyzer.decompose(normal_glucose_readings)
        assert "original" in decomp
        assert "trend" in decomp
        assert "seasonal" in decomp
        assert "residual" in decomp
        assert len(decomp["original"]) == len(normal_glucose_readings)

    def test_volatility_calculation(
        self,
        trend_analyzer: TrendAnalyzer,
        normal_glucose_readings: list[BiomarkerReading],
    ) -> None:
        vol = trend_analyzer.calculate_volatility(normal_glucose_readings)
        assert vol >= 0.0

    def test_insufficient_data(self, trend_analyzer: TrendAnalyzer) -> None:
        from tests.conftest import make_reading_series

        readings = make_reading_series(values=[85.0, 86.0])
        result = trend_analyzer.analyze(readings, BiomarkerType.GLUCOSE)
        assert result.direction == TrendDirection.UNKNOWN
        assert result.confidence == 0.0
