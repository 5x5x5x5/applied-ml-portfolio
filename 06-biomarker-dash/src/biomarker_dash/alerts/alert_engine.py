"""Clinical alert engine with rule-based and ML-based alerting.

Handles alert generation, prioritization, deduplication, escalation,
and acknowledgment tracking for clinical biomarker monitoring.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from biomarker_dash.schemas import (
    NORMAL_RANGES,
    AlertSeverity,
    AlertStatus,
    AnomalyResult,
    BiomarkerReading,
    BiomarkerType,
    ClinicalAlert,
    TrendResult,
)

logger = logging.getLogger(__name__)

# Deduplication window: suppress duplicate alerts within this period
DEDUP_WINDOW_SECONDS = 300  # 5 minutes

# Escalation thresholds: unacknowledged alerts escalate after these durations
ESCALATION_TIMES: dict[AlertSeverity, timedelta] = {
    AlertSeverity.CRITICAL: timedelta(minutes=5),
    AlertSeverity.HIGH: timedelta(minutes=15),
    AlertSeverity.MEDIUM: timedelta(minutes=60),
    AlertSeverity.LOW: timedelta(hours=4),
}

# Critical thresholds that trigger immediate alerts regardless of context
CRITICAL_THRESHOLDS: dict[BiomarkerType, list[tuple[str, float, AlertSeverity]]] = {
    BiomarkerType.HEART_RATE: [
        ("below", 40.0, AlertSeverity.CRITICAL),
        ("above", 150.0, AlertSeverity.CRITICAL),
        ("below", 50.0, AlertSeverity.HIGH),
        ("above", 130.0, AlertSeverity.HIGH),
    ],
    BiomarkerType.OXYGEN_SAT: [
        ("below", 88.0, AlertSeverity.CRITICAL),
        ("below", 92.0, AlertSeverity.HIGH),
    ],
    BiomarkerType.POTASSIUM: [
        ("below", 2.5, AlertSeverity.CRITICAL),
        ("above", 6.5, AlertSeverity.CRITICAL),
        ("below", 3.0, AlertSeverity.HIGH),
        ("above", 6.0, AlertSeverity.HIGH),
    ],
    BiomarkerType.GLUCOSE: [
        ("below", 40.0, AlertSeverity.CRITICAL),
        ("above", 500.0, AlertSeverity.CRITICAL),
        ("below", 54.0, AlertSeverity.HIGH),
        ("above", 400.0, AlertSeverity.HIGH),
    ],
    BiomarkerType.TROPONIN: [
        ("above", 0.4, AlertSeverity.CRITICAL),
        ("above", 0.1, AlertSeverity.HIGH),
    ],
    BiomarkerType.TEMPERATURE: [
        ("above", 104.0, AlertSeverity.CRITICAL),
        ("below", 95.0, AlertSeverity.CRITICAL),
        ("above", 102.0, AlertSeverity.HIGH),
    ],
    BiomarkerType.SODIUM: [
        ("below", 120.0, AlertSeverity.CRITICAL),
        ("above", 160.0, AlertSeverity.CRITICAL),
    ],
}


class AlertEngine:
    """Manages clinical alert lifecycle.

    Responsibilities:
    - Generate rule-based alerts from biomarker readings
    - Generate ML-based alerts from anomaly detection results
    - Generate trend-based alerts from trend analysis
    - Deduplicate alerts within a configurable window
    - Track alert acknowledgment
    - Escalate unacknowledged alerts based on severity timers
    """

    def __init__(self) -> None:
        # Deduplication: (patient_id, biomarker_type, source) -> last alert time
        self._dedup_cache: dict[str, datetime] = {}
        # Active alerts by alert_id
        self._active_alerts: dict[str, ClinicalAlert] = {}
        # Alert history per patient
        self._patient_alert_history: dict[str, list[ClinicalAlert]] = defaultdict(list)
        # Callback for alert notifications (set by the application layer)
        self._on_alert_callback: Any = None

    def set_alert_callback(self, callback: Any) -> None:
        """Register a callback to be called when new alerts are generated."""
        self._on_alert_callback = callback

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def evaluate_reading(self, reading: BiomarkerReading) -> list[ClinicalAlert]:
        """Apply rule-based checks to a raw biomarker reading.

        Returns list of alerts generated (may be empty).
        """
        alerts: list[ClinicalAlert] = []

        # Check critical thresholds first
        critical_alert = self._check_critical_thresholds(reading)
        if critical_alert:
            alerts.append(critical_alert)

        # Check normal range boundaries
        range_alert = self._check_normal_range(reading)
        if range_alert and not critical_alert:
            alerts.append(range_alert)

        # Deduplicate and register
        final_alerts = []
        for alert in alerts:
            if not self._is_duplicate(alert):
                self._register_alert(alert)
                final_alerts.append(alert)

        return final_alerts

    def evaluate_anomaly(self, anomaly: AnomalyResult) -> list[ClinicalAlert]:
        """Generate alerts from ML anomaly detection results."""
        if not anomaly.is_anomaly:
            return []

        # Only alert on medium severity and above for ML detections
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1,
            AlertSeverity.INFO: 0,
        }
        if severity_order.get(anomaly.severity, 0) < 2:
            return []

        alert = ClinicalAlert(
            alert_id=str(uuid.uuid4()),
            patient_id=anomaly.patient_id,
            biomarker_type=anomaly.biomarker_type,
            severity=anomaly.severity,
            title=f"ML Anomaly: {anomaly.biomarker_type.value}",
            message=(f"Anomaly detected via {anomaly.detection_method}: {anomaly.explanation}"),
            value=anomaly.value,
            detection_source="anomaly_ml",
            related_reading_id=anomaly.reading_id,
        )

        if self._is_duplicate(alert):
            return []

        self._register_alert(alert)
        return [alert]

    def evaluate_trend(self, trend: TrendResult) -> list[ClinicalAlert]:
        """Generate alerts from trend analysis results."""
        alerts: list[ClinicalAlert] = []

        # Alert if predicted to exit normal range with reasonable confidence
        if trend.predicted_exit_normal and trend.confidence > 0.5:
            severity = AlertSeverity.MEDIUM
            if trend.confidence > 0.8:
                severity = AlertSeverity.HIGH

            predicted_str = (
                f"{trend.predicted_value_24h:.2f}"
                if trend.predicted_value_24h is not None
                else "N/A"
            )

            alert = ClinicalAlert(
                alert_id=str(uuid.uuid4()),
                patient_id=trend.patient_id,
                biomarker_type=trend.biomarker_type,
                severity=severity,
                title=f"Trend Alert: {trend.biomarker_type.value} {trend.direction.value}",
                message=(
                    f"{trend.biomarker_type.value} trending {trend.direction.value} "
                    f"(rate: {trend.rate_of_change:+.4f}/hr). "
                    f"Predicted to exit normal range within 24h "
                    f"(predicted value: {predicted_str}, "
                    f"confidence: {trend.confidence:.0%})"
                ),
                value=trend.predicted_value_24h or 0.0,
                detection_source="trend",
            )

            if not self._is_duplicate(alert):
                self._register_alert(alert)
                alerts.append(alert)

        return alerts

    # ------------------------------------------------------------------
    # Alert management
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Mark an alert as acknowledged by a user."""
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            logger.warning("Attempt to acknowledge unknown alert %s", alert_id)
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user
        logger.info("Alert %s acknowledged by %s", alert_id, user)
        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve and deactivate an alert."""
        alert = self._active_alerts.pop(alert_id, None)
        if alert is None:
            return False
        alert.status = AlertStatus.RESOLVED
        logger.info("Alert %s resolved", alert_id)
        return True

    def get_active_alerts(
        self,
        patient_id: str | None = None,
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ) -> list[ClinicalAlert]:
        """Get all active alerts, optionally filtered by patient and severity."""
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1,
            AlertSeverity.INFO: 0,
        }
        min_level = severity_order.get(min_severity, 0)

        alerts = []
        for alert in self._active_alerts.values():
            if alert.status not in (AlertStatus.ACTIVE, AlertStatus.ESCALATED):
                continue
            if patient_id and alert.patient_id != patient_id:
                continue
            if severity_order.get(alert.severity, 0) < min_level:
                continue
            alerts.append(alert)

        # Sort: critical first, then by time
        alerts.sort(
            key=lambda a: (
                -severity_order.get(a.severity, 0),
                a.created_at,
            )
        )
        return alerts

    def check_escalations(self) -> list[ClinicalAlert]:
        """Check for alerts that need escalation and escalate them.

        Returns list of newly escalated alerts.
        """
        now = datetime.utcnow()
        escalated: list[ClinicalAlert] = []

        for alert in self._active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue

            escalation_time = ESCALATION_TIMES.get(alert.severity)
            if escalation_time is None:
                continue

            if now - alert.created_at > escalation_time:
                alert.status = AlertStatus.ESCALATED
                alert.escalated_at = now
                # Increase severity by one level
                alert.severity = self._escalate_severity(alert.severity)
                escalated.append(alert)
                logger.warning(
                    "Alert %s escalated to %s for patient %s (unacknowledged for %s)",
                    alert.alert_id,
                    alert.severity.value,
                    alert.patient_id,
                    str(now - alert.created_at),
                )

        return escalated

    def get_alert_stats(self) -> dict[str, Any]:
        """Return summary statistics about current alerts."""
        severity_counts: dict[str, int] = defaultdict(int)
        status_counts: dict[str, int] = defaultdict(int)
        for alert in self._active_alerts.values():
            severity_counts[alert.severity.value] += 1
            status_counts[alert.status.value] += 1

        return {
            "total_active": len(self._active_alerts),
            "by_severity": dict(severity_counts),
            "by_status": dict(status_counts),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_critical_thresholds(self, reading: BiomarkerReading) -> ClinicalAlert | None:
        """Check hardcoded critical thresholds."""
        thresholds = CRITICAL_THRESHOLDS.get(reading.biomarker_type, [])
        for direction, threshold_value, severity in thresholds:
            triggered = False
            if direction == "below" and reading.value < threshold_value or direction == "above" and reading.value > threshold_value:
                triggered = True

            if triggered:
                return ClinicalAlert(
                    alert_id=str(uuid.uuid4()),
                    patient_id=reading.patient_id,
                    biomarker_type=reading.biomarker_type,
                    severity=severity,
                    title=(
                        f"CRITICAL: {reading.biomarker_type.value} {direction} {threshold_value}"
                    ),
                    message=(
                        f"Patient {reading.patient_id}: "
                        f"{reading.biomarker_type.value} = {reading.value:.2f} "
                        f"(threshold: {direction} {threshold_value}). "
                        f"Immediate clinical attention required."
                    ),
                    value=reading.value,
                    threshold=threshold_value,
                    detection_source="rule",
                    related_reading_id=reading.reading_id,
                )
        return None

    def _check_normal_range(self, reading: BiomarkerReading) -> ClinicalAlert | None:
        """Check if value is outside normal range."""
        normal = NORMAL_RANGES.get(reading.biomarker_type)
        if normal is None:
            return None

        low, high = normal
        if low <= reading.value <= high:
            return None

        # Determine severity based on how far outside range
        range_width = high - low
        if range_width <= 0:
            range_width = 1.0

        if reading.value < low:
            deviation = (low - reading.value) / range_width
            direction_word = "below"
        else:
            deviation = (reading.value - high) / range_width
            direction_word = "above"

        if deviation > 0.5:
            severity = AlertSeverity.HIGH
        elif deviation > 0.25:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        return ClinicalAlert(
            alert_id=str(uuid.uuid4()),
            patient_id=reading.patient_id,
            biomarker_type=reading.biomarker_type,
            severity=severity,
            title=f"Out of Range: {reading.biomarker_type.value}",
            message=(
                f"{reading.biomarker_type.value} = {reading.value:.2f} is "
                f"{direction_word} normal range [{low:.2f}, {high:.2f}]"
            ),
            value=reading.value,
            threshold=low if reading.value < low else high,
            detection_source="rule",
            related_reading_id=reading.reading_id,
        )

    def _is_duplicate(self, alert: ClinicalAlert) -> bool:
        """Check if an alert is a duplicate within the dedup window."""
        key = f"{alert.patient_id}:{alert.biomarker_type.value}:{alert.detection_source}"
        last_time = self._dedup_cache.get(key)
        now = datetime.utcnow()

        if last_time and (now - last_time).total_seconds() < DEDUP_WINDOW_SECONDS:
            logger.debug("Suppressed duplicate alert: %s", key)
            return True

        self._dedup_cache[key] = now
        return False

    def _register_alert(self, alert: ClinicalAlert) -> None:
        """Register an alert as active and store in history."""
        self._active_alerts[alert.alert_id] = alert
        self._patient_alert_history[alert.patient_id].append(alert)

        # Trim history per patient
        if len(self._patient_alert_history[alert.patient_id]) > 500:
            self._patient_alert_history[alert.patient_id] = self._patient_alert_history[
                alert.patient_id
            ][-500:]

        logger.info(
            "New alert [%s/%s] for patient %s: %s",
            alert.severity.value,
            alert.detection_source,
            alert.patient_id,
            alert.title,
        )

        if self._on_alert_callback:
            try:
                self._on_alert_callback(alert)
            except Exception:
                logger.exception("Alert callback failed for %s", alert.alert_id)

    @staticmethod
    def _escalate_severity(current: AlertSeverity) -> AlertSeverity:
        """Increase severity by one level."""
        order = [
            AlertSeverity.INFO,
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        ]
        idx = order.index(current) if current in order else 0
        return order[min(idx + 1, len(order) - 1)]
