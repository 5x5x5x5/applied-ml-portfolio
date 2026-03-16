"""ML-based anomaly detection for biomarker readings.

Combines Isolation Forest for multivariate anomaly detection with
Z-score methods for individual biomarkers. Supports patient-specific
baselines and contextual (age/sex/condition) adjustments.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from biomarker_dash.schemas import (
    NORMAL_RANGES,
    SEX_ADJUSTED_RANGES,
    AlertSeverity,
    AnomalyResult,
    BiomarkerReading,
    BiomarkerType,
    PatientContext,
)

logger = logging.getLogger(__name__)

# Minimum samples needed to train patient-specific models
MIN_SAMPLES_FOR_BASELINE = 10
MIN_SAMPLES_FOR_ISOLATION_FOREST = 30

# Z-score thresholds mapped to severity
ZSCORE_THRESHOLDS: list[tuple[float, AlertSeverity]] = [
    (4.0, AlertSeverity.CRITICAL),
    (3.0, AlertSeverity.HIGH),
    (2.5, AlertSeverity.MEDIUM),
    (2.0, AlertSeverity.LOW),
]


class PatientBaseline:
    """Tracks rolling statistics for a single patient's biomarker."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self.timestamps: list[float] = []

    def add(self, value: float, ts: float) -> None:
        self.values.append(value)
        self.timestamps.append(ts)
        # Keep last 500 observations
        if len(self.values) > 500:
            self.values = self.values[-500:]
            self.timestamps = self.timestamps[-500:]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 1.0

    @property
    def count(self) -> int:
        return len(self.values)


class AnomalyDetector:
    """Multi-method anomaly detection engine for biomarker data.

    Detection methods:
    1. Rule-based: value vs. normal range (population or sex-adjusted)
    2. Z-score: deviation from patient-specific baseline
    3. Isolation Forest: multivariate anomaly detection across biomarkers
    """

    def __init__(self) -> None:
        # patient_id -> biomarker_type -> PatientBaseline
        self._baselines: dict[str, dict[str, PatientBaseline]] = defaultdict(
            lambda: defaultdict(PatientBaseline)
        )
        # patient_id -> trained IsolationForest
        self._isolation_forests: dict[str, IsolationForest] = {}
        # patient_id -> feature matrix for retraining
        self._multivariate_data: dict[str, list[dict[str, float]]] = defaultdict(list)
        # Track which biomarker types a patient has data for
        self._patient_biomarker_types: dict[str, set[str]] = defaultdict(set)

    def detect(
        self,
        reading: BiomarkerReading,
        patient_context: PatientContext | None = None,
    ) -> AnomalyResult:
        """Run all anomaly detection methods and return combined result.

        The most severe finding across all methods is returned.
        """
        results: list[AnomalyResult] = []

        # Update baseline
        baseline = self._baselines[reading.patient_id][reading.biomarker_type.value]
        baseline.add(reading.value, reading.timestamp.timestamp())

        # Update multivariate tracking
        self._patient_biomarker_types[reading.patient_id].add(reading.biomarker_type.value)

        # Method 1: Range-based detection
        range_result = self._range_based_detection(reading, patient_context)
        results.append(range_result)

        # Method 2: Z-score detection (needs sufficient baseline)
        if baseline.count >= MIN_SAMPLES_FOR_BASELINE:
            zscore_result = self._zscore_detection(reading, baseline)
            results.append(zscore_result)

        # Method 3: Isolation Forest (needs multivariate data)
        self._update_multivariate_data(reading)
        if_result = self._isolation_forest_detection(reading)
        if if_result is not None:
            results.append(if_result)

        # Pick the most severe anomaly
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1,
            AlertSeverity.INFO: 0,
        }
        results.sort(key=lambda r: severity_order.get(r.severity, -1), reverse=True)
        best = results[0]

        if best.is_anomaly:
            logger.info(
                "Anomaly detected for patient %s, %s=%.2f [%s] via %s: %s",
                reading.patient_id,
                reading.biomarker_type.value,
                reading.value,
                best.severity.value,
                best.detection_method,
                best.explanation,
            )

        return best

    # ------------------------------------------------------------------
    # Range-based detection
    # ------------------------------------------------------------------

    def _range_based_detection(
        self,
        reading: BiomarkerReading,
        patient_context: PatientContext | None,
    ) -> AnomalyResult:
        """Detect anomalies based on normal ranges (sex/age adjusted)."""
        low, high = self._get_normal_range(reading.biomarker_type, patient_context)
        value = reading.value
        is_anomaly = value < low or value > high
        severity = AlertSeverity.INFO

        if is_anomaly:
            range_width = high - low
            if range_width <= 0:
                range_width = 1.0
            # How far outside the range (as a fraction of range width)
            if value < low:
                deviation_pct = (low - value) / range_width
            else:
                deviation_pct = (value - high) / range_width

            if deviation_pct > 0.5:
                severity = AlertSeverity.CRITICAL
            elif deviation_pct > 0.3:
                severity = AlertSeverity.HIGH
            elif deviation_pct > 0.15:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

        explanation = ""
        if is_anomaly:
            if value < low:
                explanation = (
                    f"{reading.biomarker_type.value} value {value:.2f} is below "
                    f"normal range [{low:.2f}, {high:.2f}]"
                )
            else:
                explanation = (
                    f"{reading.biomarker_type.value} value {value:.2f} is above "
                    f"normal range [{low:.2f}, {high:.2f}]"
                )
        else:
            explanation = f"{reading.biomarker_type.value} value {value:.2f} is within normal range"

        return AnomalyResult(
            reading_id=reading.reading_id,
            patient_id=reading.patient_id,
            biomarker_type=reading.biomarker_type,
            value=value,
            is_anomaly=is_anomaly,
            anomaly_score=min(1.0, abs(value - (low + high) / 2) / ((high - low) / 2))
            if (high - low) > 0
            else 0.0,
            severity=severity,
            detection_method="range_based",
            explanation=explanation,
            timestamp=reading.timestamp,
            normal_range=(low, high),
        )

    def _get_normal_range(
        self,
        biomarker_type: BiomarkerType,
        patient_context: PatientContext | None,
    ) -> tuple[float, float]:
        """Get the appropriate normal range, adjusting for sex if available."""
        # Try sex-adjusted first
        if patient_context and biomarker_type in SEX_ADJUSTED_RANGES:
            sex_ranges = SEX_ADJUSTED_RANGES[biomarker_type]
            if patient_context.sex in sex_ranges:
                return sex_ranges[patient_context.sex]

        # Fall back to population range
        return NORMAL_RANGES.get(biomarker_type, (0.0, 1000.0))

    # ------------------------------------------------------------------
    # Z-score detection
    # ------------------------------------------------------------------

    def _zscore_detection(
        self,
        reading: BiomarkerReading,
        baseline: PatientBaseline,
    ) -> AnomalyResult:
        """Detect anomalies via Z-score against patient-specific baseline."""
        z = abs(reading.value - baseline.mean) / max(baseline.std, 1e-6)

        is_anomaly = False
        severity = AlertSeverity.INFO
        for threshold, sev in ZSCORE_THRESHOLDS:
            if z >= threshold:
                is_anomaly = True
                severity = sev
                break

        explanation = (
            f"Z-score {z:.2f} (patient baseline: mean={baseline.mean:.2f}, "
            f"std={baseline.std:.2f}, n={baseline.count})"
        )
        if is_anomaly:
            explanation = f"Unusual value detected - {explanation}"

        low, high = NORMAL_RANGES.get(reading.biomarker_type, (0.0, 1000.0))

        return AnomalyResult(
            reading_id=reading.reading_id,
            patient_id=reading.patient_id,
            biomarker_type=reading.biomarker_type,
            value=reading.value,
            is_anomaly=is_anomaly,
            anomaly_score=min(1.0, z / 5.0),
            severity=severity,
            detection_method="zscore",
            explanation=explanation,
            timestamp=reading.timestamp,
            normal_range=(low, high),
        )

    # ------------------------------------------------------------------
    # Isolation Forest detection
    # ------------------------------------------------------------------

    def _update_multivariate_data(self, reading: BiomarkerReading) -> None:
        """Track latest values per biomarker for multivariate analysis."""
        patient_data = self._multivariate_data[reading.patient_id]
        # Each entry is a snapshot of the most recent values at a point in time
        latest: dict[str, float] = {}
        if patient_data:
            latest = dict(patient_data[-1])
        latest[reading.biomarker_type.value] = reading.value
        patient_data.append(latest)
        # Keep bounded
        if len(patient_data) > 1000:
            self._multivariate_data[reading.patient_id] = patient_data[-1000:]

    def _isolation_forest_detection(self, reading: BiomarkerReading) -> AnomalyResult | None:
        """Run Isolation Forest on multivariate biomarker data."""
        patient_id = reading.patient_id
        data = self._multivariate_data.get(patient_id, [])
        if len(data) < MIN_SAMPLES_FOR_ISOLATION_FOREST:
            return None

        # Build feature matrix from snapshots that have at least 3 biomarkers
        feature_names = sorted(self._patient_biomarker_types.get(patient_id, set()))
        if len(feature_names) < 2:
            return None

        rows: list[list[float]] = []
        for snapshot in data:
            if all(fn in snapshot for fn in feature_names):
                rows.append([snapshot[fn] for fn in feature_names])

        if len(rows) < MIN_SAMPLES_FOR_ISOLATION_FOREST:
            return None

        X: NDArray[np.float64] = np.array(rows, dtype=np.float64)

        # Retrain periodically (every 50 new samples or on first call)
        should_retrain = patient_id not in self._isolation_forests or len(rows) % 50 == 0

        if should_retrain:
            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X)
            self._isolation_forests[patient_id] = model
            logger.debug(
                "Retrained Isolation Forest for patient %s with %d samples, %d features",
                patient_id,
                len(rows),
                len(feature_names),
            )

        model = self._isolation_forests[patient_id]

        # Score the latest point
        current_point = [data[-1].get(fn, 0.0) for fn in feature_names]
        point_array: NDArray[np.float64] = np.array([current_point], dtype=np.float64)
        prediction = model.predict(point_array)[0]
        raw_score = model.decision_function(point_array)[0]

        # Isolation Forest: -1 = anomaly, 1 = normal
        is_anomaly = prediction == -1
        # Convert raw_score to 0-1 range (more negative = more anomalous)
        anomaly_score = max(0.0, min(1.0, 0.5 - float(raw_score)))

        severity = AlertSeverity.INFO
        if is_anomaly:
            if anomaly_score > 0.8:
                severity = AlertSeverity.HIGH
            elif anomaly_score > 0.6:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

        low, high = NORMAL_RANGES.get(reading.biomarker_type, (0.0, 1000.0))

        return AnomalyResult(
            reading_id=reading.reading_id,
            patient_id=reading.patient_id,
            biomarker_type=reading.biomarker_type,
            value=reading.value,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            severity=severity,
            detection_method="isolation_forest",
            explanation=(
                f"Multivariate anomaly score: {anomaly_score:.3f} "
                f"(features: {', '.join(feature_names)})"
            ),
            timestamp=reading.timestamp,
            normal_range=(low, high),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_baseline_stats(self, patient_id: str, biomarker_type: str) -> dict[str, Any]:
        """Return baseline statistics for debugging/display."""
        baseline = self._baselines.get(patient_id, {}).get(biomarker_type)
        if baseline is None or baseline.count == 0:
            return {"count": 0}
        return {
            "count": baseline.count,
            "mean": round(baseline.mean, 4),
            "std": round(baseline.std, 4),
            "min": round(min(baseline.values), 4),
            "max": round(max(baseline.values), 4),
        }

    def reset_patient(self, patient_id: str) -> None:
        """Clear all learned data for a patient."""
        self._baselines.pop(patient_id, None)
        self._isolation_forests.pop(patient_id, None)
        self._multivariate_data.pop(patient_id, None)
        self._patient_biomarker_types.pop(patient_id, None)
        logger.info("Reset anomaly detector state for patient %s", patient_id)
