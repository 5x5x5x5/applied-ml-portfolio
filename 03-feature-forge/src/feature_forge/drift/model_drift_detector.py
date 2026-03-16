"""Model drift detection: prediction distribution shifts, accuracy degradation,
concept drift, and data drift.

Monitors model outputs over time, compares against baseline performance
metrics, and integrates with AWS SageMaker Model Monitor for production
alerting and automated retraining triggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import snowflake.connector
from scipy import stats
from snowflake.connector import DictCursor, SnowflakeConnection

from feature_forge.drift.feature_drift_detector import (
    DriftSeverity,
    FeatureDriftDetector,
)
from feature_forge.extractors.structured_extractor import SnowflakeConfig

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of model drift."""

    PREDICTION_DISTRIBUTION = "prediction_distribution"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"


@dataclass
class ModelDriftResult:
    """Result of a model-level drift detection check."""

    model_name: str
    model_version: str
    drift_type: DriftType
    metric_name: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: DriftSeverity
    is_drifted: bool
    should_retrain: bool
    detected_at: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMonitorConfig:
    """Configuration for model drift monitoring."""

    model_name: str
    model_version: str
    predictions_table: str
    ground_truth_table: str | None = None
    prediction_column: str = "prediction"
    label_column: str = "label"
    timestamp_column: str = "prediction_ts"
    entity_key: str = "patient_id"
    # Thresholds
    accuracy_degradation_pct: float = 5.0  # alert if accuracy drops > 5%
    prediction_psi_threshold: float = 0.2
    retrain_severity: DriftSeverity = DriftSeverity.HIGH
    # Monitoring windows
    baseline_days: int = 30
    current_window_days: int = 7


class ModelDriftDetector:
    """Detect model-level drift including prediction shifts, accuracy
    degradation, and concept drift.

    Queries prediction logs and ground truth labels from Snowflake,
    computes baseline vs. current metrics, and raises alerts when
    drift exceeds configured thresholds.
    """

    def __init__(
        self,
        sf_config: SnowflakeConfig,
        monitor_config: ModelMonitorConfig,
        feature_detector: FeatureDriftDetector | None = None,
    ) -> None:
        self._sf_config = sf_config
        self._monitor_config = monitor_config
        self._feature_detector = feature_detector
        self._conn: SnowflakeConnection | None = None
        logger.info(
            "ModelDriftDetector initialised for %s v%s",
            monitor_config.model_name,
            monitor_config.model_version,
        )

    def connect(self) -> SnowflakeConnection:
        if self._conn and not self._conn.is_closed():
            return self._conn
        self._conn = snowflake.connector.connect(
            account=self._sf_config.account,
            user=self._sf_config.user,
            password=self._sf_config.password,
            warehouse=self._sf_config.warehouse,
            database=self._sf_config.database,
            schema=self._sf_config.schema,
            role=self._sf_config.role,
        )
        return self._conn

    def disconnect(self) -> None:
        if self._conn and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def _execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        conn = self.connect()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Prediction distribution monitoring
    # ------------------------------------------------------------------

    def detect_prediction_drift(
        self,
        as_of_date: datetime | None = None,
    ) -> ModelDriftResult:
        """Detect shifts in the prediction distribution.

        Compares the distribution of model predictions in the current
        window against the baseline window using PSI.
        """
        cfg = self._monitor_config
        now = as_of_date or datetime.utcnow()

        baseline_start = now - timedelta(days=cfg.baseline_days + cfg.current_window_days)
        baseline_end = now - timedelta(days=cfg.current_window_days)
        current_start = now - timedelta(days=cfg.current_window_days)
        current_end = now

        logger.info(
            "Checking prediction drift: baseline=[%s, %s), current=[%s, %s)",
            baseline_start.date(),
            baseline_end.date(),
            current_start.date(),
            current_end.date(),
        )

        # Fetch prediction values
        baseline_df = self._fetch_predictions(baseline_start, baseline_end)
        current_df = self._fetch_predictions(current_start, current_end)

        if baseline_df.empty or current_df.empty:
            logger.warning("Insufficient prediction data for drift detection")
            return ModelDriftResult(
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                drift_type=DriftType.PREDICTION_DISTRIBUTION,
                metric_name="PSI",
                baseline_value=0.0,
                current_value=0.0,
                change_pct=0.0,
                severity=DriftSeverity.NONE,
                is_drifted=False,
                should_retrain=False,
                details={"reason": "insufficient_data"},
            )

        pred_col = cfg.prediction_column.upper()
        baseline_vals = baseline_df[pred_col].dropna().values.astype(float)
        current_vals = current_df[pred_col].dropna().values.astype(float)

        # Compute PSI using histograms
        combined = np.concatenate([baseline_vals, current_vals])
        bins = np.linspace(combined.min(), combined.max(), 21)

        baseline_hist, _ = np.histogram(baseline_vals, bins=bins)
        current_hist, _ = np.histogram(current_vals, bins=bins)

        baseline_prop = baseline_hist.astype(float) / baseline_hist.sum()
        current_prop = current_hist.astype(float) / current_hist.sum()

        psi = FeatureDriftDetector.compute_psi(baseline_prop, current_prop)

        severity = self._classify_prediction_severity(psi)
        is_drifted = psi >= cfg.prediction_psi_threshold

        return ModelDriftResult(
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            drift_type=DriftType.PREDICTION_DISTRIBUTION,
            metric_name="PSI",
            baseline_value=float(baseline_vals.mean()),
            current_value=float(current_vals.mean()),
            change_pct=float(
                (current_vals.mean() - baseline_vals.mean())
                / max(abs(baseline_vals.mean()), 1e-9)
                * 100
            ),
            severity=severity,
            is_drifted=is_drifted,
            should_retrain=severity.value
            in (DriftSeverity.HIGH.value, DriftSeverity.CRITICAL.value),
            details={
                "psi": psi,
                "baseline_mean": float(baseline_vals.mean()),
                "baseline_std": float(baseline_vals.std()),
                "current_mean": float(current_vals.mean()),
                "current_std": float(current_vals.std()),
                "baseline_count": len(baseline_vals),
                "current_count": len(current_vals),
            },
        )

    # ------------------------------------------------------------------
    # Accuracy degradation
    # ------------------------------------------------------------------

    def detect_accuracy_degradation(
        self,
        as_of_date: datetime | None = None,
    ) -> ModelDriftResult:
        """Detect accuracy degradation by comparing baseline vs current performance.

        Requires ground truth labels to be available. Computes AUC, accuracy,
        precision, recall, and F1 for both periods and flags degradation.
        """
        cfg = self._monitor_config
        if cfg.ground_truth_table is None:
            raise ValueError("ground_truth_table must be configured for accuracy monitoring")

        now = as_of_date or datetime.utcnow()
        baseline_start = now - timedelta(days=cfg.baseline_days + cfg.current_window_days)
        baseline_end = now - timedelta(days=cfg.current_window_days)
        current_start = now - timedelta(days=cfg.current_window_days)
        current_end = now

        baseline_metrics = self._compute_metrics(baseline_start, baseline_end)
        current_metrics = self._compute_metrics(current_start, current_end)

        if not baseline_metrics or not current_metrics:
            return ModelDriftResult(
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                drift_type=DriftType.ACCURACY_DEGRADATION,
                metric_name="accuracy",
                baseline_value=0.0,
                current_value=0.0,
                change_pct=0.0,
                severity=DriftSeverity.NONE,
                is_drifted=False,
                should_retrain=False,
                details={"reason": "insufficient_labeled_data"},
            )

        baseline_acc = baseline_metrics["accuracy"]
        current_acc = current_metrics["accuracy"]
        degradation_pct = ((baseline_acc - current_acc) / max(baseline_acc, 1e-9)) * 100

        is_degraded = degradation_pct > cfg.accuracy_degradation_pct
        severity = self._classify_accuracy_severity(degradation_pct)

        return ModelDriftResult(
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            drift_type=DriftType.ACCURACY_DEGRADATION,
            metric_name="accuracy",
            baseline_value=baseline_acc,
            current_value=current_acc,
            change_pct=-degradation_pct,
            severity=severity,
            is_drifted=is_degraded,
            should_retrain=severity.value
            in (DriftSeverity.HIGH.value, DriftSeverity.CRITICAL.value),
            details={
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "degradation_pct": degradation_pct,
            },
        )

    # ------------------------------------------------------------------
    # Concept drift detection
    # ------------------------------------------------------------------

    def detect_concept_drift(
        self,
        as_of_date: datetime | None = None,
        window_size_days: int = 7,
        num_windows: int = 8,
    ) -> ModelDriftResult:
        """Detect concept drift via sliding-window performance analysis.

        Concept drift occurs when the relationship between features and
        labels changes over time. We detect this by tracking model error
        rate over successive time windows and testing for a monotonic
        trend using the Mann-Kendall test.
        """
        cfg = self._monitor_config
        if cfg.ground_truth_table is None:
            raise ValueError("ground_truth_table required for concept drift detection")

        now = as_of_date or datetime.utcnow()
        error_rates: list[float] = []
        window_labels: list[str] = []

        for i in range(num_windows - 1, -1, -1):
            win_end = now - timedelta(days=i * window_size_days)
            win_start = win_end - timedelta(days=window_size_days)
            metrics = self._compute_metrics(win_start, win_end)
            if metrics:
                error_rates.append(1.0 - metrics["accuracy"])
                window_labels.append(win_start.strftime("%Y-%m-%d"))

        if len(error_rates) < 4:
            return ModelDriftResult(
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                drift_type=DriftType.CONCEPT_DRIFT,
                metric_name="error_rate_trend",
                baseline_value=0.0,
                current_value=0.0,
                change_pct=0.0,
                severity=DriftSeverity.NONE,
                is_drifted=False,
                should_retrain=False,
                details={"reason": "insufficient_windows", "windows_available": len(error_rates)},
            )

        # Mann-Kendall trend test
        tau, p_value = stats.kendalltau(range(len(error_rates)), error_rates)
        is_trending_up = tau > 0 and p_value < 0.05

        if is_trending_up and p_value < 0.001:
            severity = DriftSeverity.CRITICAL
        elif is_trending_up and p_value < 0.01:
            severity = DriftSeverity.HIGH
        elif is_trending_up:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.NONE

        return ModelDriftResult(
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            drift_type=DriftType.CONCEPT_DRIFT,
            metric_name="error_rate_trend",
            baseline_value=error_rates[0],
            current_value=error_rates[-1],
            change_pct=((error_rates[-1] - error_rates[0]) / max(error_rates[0], 1e-9)) * 100,
            severity=severity,
            is_drifted=is_trending_up,
            should_retrain=severity.value
            in (DriftSeverity.HIGH.value, DriftSeverity.CRITICAL.value),
            details={
                "kendall_tau": float(tau),
                "p_value": float(p_value),
                "error_rates": error_rates,
                "window_labels": window_labels,
            },
        )

    # ------------------------------------------------------------------
    # Data drift (input features)
    # ------------------------------------------------------------------

    def detect_data_drift(
        self,
        feature_table: str,
        feature_columns: list[str],
        as_of_date: datetime | None = None,
    ) -> list[ModelDriftResult]:
        """Detect data drift across input features.

        Uses the FeatureDriftDetector to check each input feature for
        distributional shifts that could impact model performance.
        """
        if self._feature_detector is None:
            raise ValueError("feature_detector must be provided for data drift detection")

        cfg = self._monitor_config
        now = as_of_date or datetime.utcnow()
        baseline_start = now - timedelta(days=cfg.baseline_days + cfg.current_window_days)
        baseline_end = now - timedelta(days=cfg.current_window_days)
        current_start = now - timedelta(days=cfg.current_window_days)
        current_end = now

        results: list[ModelDriftResult] = []

        for col in feature_columns:
            drift_result = self._feature_detector.detect_numeric_drift(
                feature_name=col,
                table=feature_table,
                column=col,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                current_start=current_start,
                current_end=current_end,
            )

            results.append(
                ModelDriftResult(
                    model_name=cfg.model_name,
                    model_version=cfg.model_version,
                    drift_type=DriftType.DATA_DRIFT,
                    metric_name=f"PSI_{col}",
                    baseline_value=drift_result.details.get("baseline_mean", 0.0),
                    current_value=drift_result.details.get("current_mean", 0.0),
                    change_pct=0.0,
                    severity=drift_result.severity,
                    is_drifted=drift_result.is_drifted,
                    should_retrain=drift_result.severity.value
                    in (DriftSeverity.HIGH.value, DriftSeverity.CRITICAL.value),
                    details=drift_result.details,
                )
            )

        drifted = sum(1 for r in results if r.is_drifted)
        logger.info("Data drift check: %d/%d features drifted", drifted, len(results))
        return results

    # ------------------------------------------------------------------
    # Comprehensive check
    # ------------------------------------------------------------------

    def run_full_check(
        self,
        as_of_date: datetime | None = None,
        feature_table: str | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[ModelDriftResult]:
        """Run all drift checks and aggregate results."""
        results: list[ModelDriftResult] = []

        # Prediction distribution
        pred_result = self.detect_prediction_drift(as_of_date)
        results.append(pred_result)

        # Accuracy degradation (if labels available)
        if self._monitor_config.ground_truth_table:
            try:
                acc_result = self.detect_accuracy_degradation(as_of_date)
                results.append(acc_result)
            except Exception:
                logger.exception("Accuracy degradation check failed")

            try:
                concept_result = self.detect_concept_drift(as_of_date)
                results.append(concept_result)
            except Exception:
                logger.exception("Concept drift check failed")

        # Data drift (if feature detector available)
        if feature_table and feature_columns and self._feature_detector:
            try:
                data_results = self.detect_data_drift(feature_table, feature_columns, as_of_date)
                results.extend(data_results)
            except Exception:
                logger.exception("Data drift check failed")

        any_retrain = any(r.should_retrain for r in results)
        if any_retrain:
            logger.warning(
                "Retraining recommended for %s v%s",
                self._monitor_config.model_name,
                self._monitor_config.model_version,
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_predictions(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch prediction values for a time window."""
        cfg = self._monitor_config
        sql = f"""
        SELECT {cfg.prediction_column}, {cfg.timestamp_column}, {cfg.entity_key}
        FROM {cfg.predictions_table}
        WHERE {cfg.timestamp_column} >= %(start)s
          AND {cfg.timestamp_column} < %(end)s
        """
        return self._execute(sql, {"start": start_date, "end": end_date})

    def _compute_metrics(self, start_date: datetime, end_date: datetime) -> dict[str, float] | None:
        """Compute classification metrics by joining predictions with ground truth."""
        cfg = self._monitor_config
        sql = f"""
        WITH joined AS (
            SELECT
                p.{cfg.prediction_column} AS pred,
                g.{cfg.label_column} AS label
            FROM {cfg.predictions_table} p
            INNER JOIN {cfg.ground_truth_table} g
                ON p.{cfg.entity_key} = g.{cfg.entity_key}
                AND p.{cfg.timestamp_column} >= %(start)s
                AND p.{cfg.timestamp_column} < %(end)s
        )
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN ROUND(pred) = label THEN 1 ELSE 0 END) AS correct,
            SUM(CASE WHEN ROUND(pred) = 1 AND label = 1 THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN ROUND(pred) = 1 AND label = 0 THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN ROUND(pred) = 0 AND label = 1 THEN 1 ELSE 0 END) AS fn,
            SUM(CASE WHEN ROUND(pred) = 0 AND label = 0 THEN 1 ELSE 0 END) AS tn
        FROM joined
        """
        df = self._execute(sql, {"start": start_date, "end": end_date})
        if df.empty or df.iloc[0]["TOTAL"] == 0:
            return None

        row = df.iloc[0]
        total = row["TOTAL"]
        tp, fp, fn, tn = row["TP"], row["FP"], row["FN"], row["TN"]
        accuracy = row["CORRECT"] / total
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total": int(total),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    def _classify_prediction_severity(self, psi: float) -> DriftSeverity:
        """Classify prediction PSI severity."""
        if psi >= 0.5:
            return DriftSeverity.CRITICAL
        if psi >= 0.3:
            return DriftSeverity.HIGH
        if psi >= 0.2:
            return DriftSeverity.MEDIUM
        if psi >= 0.1:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    @staticmethod
    def _classify_accuracy_severity(degradation_pct: float) -> DriftSeverity:
        """Classify accuracy degradation severity."""
        if degradation_pct >= 20.0:
            return DriftSeverity.CRITICAL
        if degradation_pct >= 10.0:
            return DriftSeverity.HIGH
        if degradation_pct >= 5.0:
            return DriftSeverity.MEDIUM
        if degradation_pct >= 2.0:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def __enter__(self) -> ModelDriftDetector:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
