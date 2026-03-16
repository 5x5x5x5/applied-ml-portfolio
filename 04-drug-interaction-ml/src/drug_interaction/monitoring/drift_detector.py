"""Comprehensive drift detection for drug interaction models.

Detects feature drift (PSI, KS test), prediction drift, and label drift.
Queries Snowflake for production feature distributions, compares against
training baselines, and integrates with SageMaker Model Monitor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DriftSeverity(str, Enum):
    """Severity classification for detected drift."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Type of drift detected."""

    FEATURE = "feature"
    PREDICTION = "prediction"
    LABEL = "label"
    CONCEPT = "concept"


class FeatureDriftResult(BaseModel):
    """Drift analysis result for a single feature."""

    feature_name: str
    drift_type: DriftType = DriftType.FEATURE
    psi: float = Field(..., ge=0.0, description="Population Stability Index")
    ks_statistic: float = Field(..., ge=0.0, le=1.0, description="Kolmogorov-Smirnov statistic")
    ks_pvalue: float = Field(..., ge=0.0, le=1.0)
    mean_shift: float = Field(..., description="Absolute shift in mean")
    std_shift: float = Field(..., description="Absolute shift in standard deviation")
    severity: DriftSeverity
    is_drifted: bool


class PredictionDriftResult(BaseModel):
    """Drift analysis for model predictions."""

    drift_type: DriftType = DriftType.PREDICTION
    psi: float
    ks_statistic: float
    ks_pvalue: float
    mean_prediction_baseline: float
    mean_prediction_current: float
    prediction_shift: float
    severity: DriftSeverity
    is_drifted: bool
    class_distribution_baseline: dict[str, float] = Field(default_factory=dict)
    class_distribution_current: dict[str, float] = Field(default_factory=dict)


class LabelDriftResult(BaseModel):
    """Drift analysis for ground-truth labels."""

    drift_type: DriftType = DriftType.LABEL
    chi2_statistic: float
    chi2_pvalue: float
    severity: DriftSeverity
    is_drifted: bool
    label_distribution_baseline: dict[str, float] = Field(default_factory=dict)
    label_distribution_current: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    """Complete drift detection report."""

    report_date: str
    feature_drift: list[FeatureDriftResult] = Field(default_factory=list)
    prediction_drift: PredictionDriftResult | None = None
    label_drift: LabelDriftResult | None = None
    total_features_analyzed: int = 0
    features_drifted: int = 0
    overall_severity: DriftSeverity = DriftSeverity.NONE
    requires_retraining: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# PSI thresholds
# ---------------------------------------------------------------------------

PSI_THRESHOLDS = {
    DriftSeverity.NONE: 0.0,
    DriftSeverity.LOW: 0.1,
    DriftSeverity.MODERATE: 0.2,
    DriftSeverity.HIGH: 0.3,
    DriftSeverity.CRITICAL: 0.5,
}


# ---------------------------------------------------------------------------
# Drift detection functions
# ---------------------------------------------------------------------------


def compute_psi(
    baseline: NDArray[np.float64],
    current: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    baseline : array
        Reference distribution values.
    current : array
        Current (production) distribution values.
    n_bins : int
        Number of bins for histogram discretisation.

    Returns
    -------
    float
        PSI value. 0 = identical distributions.
    """
    # Use baseline percentiles as bin edges for consistency
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(baseline, percentiles)
    edges[-1] = np.inf
    edges[0] = -np.inf
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    baseline_counts = np.histogram(baseline, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    # Normalise to proportions
    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    # Add small epsilon to avoid log(0)
    eps = 1e-6
    baseline_pct = np.clip(baseline_pct, eps, None)
    current_pct = np.clip(current_pct, eps, None)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return max(psi, 0.0)


def classify_severity(psi_value: float) -> DriftSeverity:
    """Map a PSI value to a drift severity level."""
    if psi_value >= PSI_THRESHOLDS[DriftSeverity.CRITICAL]:
        return DriftSeverity.CRITICAL
    if psi_value >= PSI_THRESHOLDS[DriftSeverity.HIGH]:
        return DriftSeverity.HIGH
    if psi_value >= PSI_THRESHOLDS[DriftSeverity.MODERATE]:
        return DriftSeverity.MODERATE
    if psi_value >= PSI_THRESHOLDS[DriftSeverity.LOW]:
        return DriftSeverity.LOW
    return DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Snowflake drift queries
# ---------------------------------------------------------------------------

FEATURE_DISTRIBUTION_SQL = """
SELECT
    feature_name,
    COUNT(*) AS sample_count,
    AVG(feature_value) AS mean_val,
    STDDEV(feature_value) AS std_val,
    MIN(feature_value) AS min_val,
    MAX(feature_value) AS max_val,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) AS p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) AS p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY feature_value) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY feature_value) AS p99,
    -- Histogram bins (10 equal-width bins)
    WIDTH_BUCKET(feature_value, min_val_global, max_val_global, 10) AS bin_id,
    COUNT(*) AS bin_count
FROM (
    SELECT
        f.feature_name,
        f.feature_value,
        MIN(f.feature_value) OVER (PARTITION BY f.feature_name) AS min_val_global,
        MAX(f.feature_value) OVER (PARTITION BY f.feature_name) AS max_val_global
    FROM {database}.{schema}.production_features f
    WHERE f.prediction_date >= %(start_date)s
      AND f.prediction_date < %(end_date)s
)
GROUP BY feature_name, bin_id
ORDER BY feature_name, bin_id
"""

PREDICTION_DISTRIBUTION_SQL = """
SELECT
    prediction_class,
    COUNT(*) AS count,
    AVG(prediction_probability) AS avg_probability,
    STDDEV(prediction_probability) AS std_probability,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prediction_probability) AS median_probability
FROM {database}.{schema}.production_predictions
WHERE prediction_date >= %(start_date)s
  AND prediction_date < %(end_date)s
GROUP BY prediction_class
ORDER BY prediction_class
"""


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


@dataclass
class DriftDetector:
    """Detect feature, prediction, and label drift.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold above which a feature is flagged as drifted.
    ks_alpha : float
        Significance level for the KS test.
    max_drifted_features_pct : float
        If more than this fraction of features drift, recommend retraining.
    snowflake_config : dict, optional
        Snowflake connection config for production feature queries.
    """

    psi_threshold: float = 0.2
    ks_alpha: float = 0.05
    max_drifted_features_pct: float = 0.30
    snowflake_config: dict[str, Any] = field(default_factory=dict)

    # -- Feature drift ------------------------------------------------------

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> list[FeatureDriftResult]:
        """Detect drift for each numeric feature.

        Parameters
        ----------
        baseline_df : pd.DataFrame
            Training-time feature distribution.
        current_df : pd.DataFrame
            Production feature distribution.
        feature_columns : list[str], optional
            Columns to check. If None, all numeric columns are used.

        Returns
        -------
        list[FeatureDriftResult]
        """
        if feature_columns is None:
            feature_columns = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        results: list[FeatureDriftResult] = []
        for col in feature_columns:
            if col not in current_df.columns:
                logger.warning("Feature %s not found in current data, skipping", col)
                continue

            baseline_vals = baseline_df[col].dropna().values.astype(float)
            current_vals = current_df[col].dropna().values.astype(float)

            if len(baseline_vals) < 10 or len(current_vals) < 10:
                logger.warning("Insufficient samples for %s, skipping", col)
                continue

            psi = compute_psi(baseline_vals, current_vals)
            ks_stat, ks_p = stats.ks_2samp(baseline_vals, current_vals)
            mean_shift = abs(float(np.mean(current_vals) - np.mean(baseline_vals)))
            std_shift = abs(float(np.std(current_vals) - np.std(baseline_vals)))

            severity = classify_severity(psi)
            is_drifted = psi >= self.psi_threshold or ks_p < self.ks_alpha

            results.append(
                FeatureDriftResult(
                    feature_name=col,
                    psi=round(psi, 6),
                    ks_statistic=round(float(ks_stat), 6),
                    ks_pvalue=round(float(ks_p), 6),
                    mean_shift=round(mean_shift, 6),
                    std_shift=round(std_shift, 6),
                    severity=severity,
                    is_drifted=is_drifted,
                )
            )

        drifted = sum(1 for r in results if r.is_drifted)
        logger.info(
            "Feature drift: %d / %d features drifted",
            drifted,
            len(results),
        )
        return results

    # -- Prediction drift ---------------------------------------------------

    def detect_prediction_drift(
        self,
        baseline_predictions: NDArray[np.float64],
        current_predictions: NDArray[np.float64],
        baseline_classes: NDArray[np.int64] | None = None,
        current_classes: NDArray[np.int64] | None = None,
    ) -> PredictionDriftResult:
        """Detect drift in model prediction distributions.

        Parameters
        ----------
        baseline_predictions : array
            Prediction probabilities from the training/validation set.
        current_predictions : array
            Prediction probabilities from production.
        baseline_classes, current_classes : array, optional
            Predicted class labels for distribution comparison.

        Returns
        -------
        PredictionDriftResult
        """
        psi = compute_psi(baseline_predictions, current_predictions)
        ks_stat, ks_p = stats.ks_2samp(baseline_predictions, current_predictions)

        class_dist_baseline: dict[str, float] = {}
        class_dist_current: dict[str, float] = {}
        if baseline_classes is not None and current_classes is not None:
            for cls in np.unique(np.concatenate([baseline_classes, current_classes])):
                class_dist_baseline[str(cls)] = float(np.mean(baseline_classes == cls))
                class_dist_current[str(cls)] = float(np.mean(current_classes == cls))

        severity = classify_severity(psi)
        result = PredictionDriftResult(
            psi=round(psi, 6),
            ks_statistic=round(float(ks_stat), 6),
            ks_pvalue=round(float(ks_p), 6),
            mean_prediction_baseline=round(float(np.mean(baseline_predictions)), 6),
            mean_prediction_current=round(float(np.mean(current_predictions)), 6),
            prediction_shift=round(
                float(np.mean(current_predictions) - np.mean(baseline_predictions)), 6
            ),
            severity=severity,
            is_drifted=psi >= self.psi_threshold,
            class_distribution_baseline=class_dist_baseline,
            class_distribution_current=class_dist_current,
        )
        logger.info("Prediction drift: PSI=%.4f, severity=%s", psi, severity.value)
        return result

    # -- Label drift --------------------------------------------------------

    def detect_label_drift(
        self,
        baseline_labels: NDArray[np.int64],
        current_labels: NDArray[np.int64],
    ) -> LabelDriftResult:
        """Detect drift in ground-truth label distributions using chi-squared test.

        Parameters
        ----------
        baseline_labels : array
            Labels from the training set.
        current_labels : array
            Labels from recent production data with ground truth.

        Returns
        -------
        LabelDriftResult
        """
        all_classes = np.unique(np.concatenate([baseline_labels, current_labels]))

        baseline_counts = np.array([np.sum(baseline_labels == c) for c in all_classes], dtype=float)
        current_counts = np.array([np.sum(current_labels == c) for c in all_classes], dtype=float)

        # Normalise current to have same total as baseline for chi2
        scale_factor = baseline_counts.sum() / max(current_counts.sum(), 1)
        expected = current_counts * scale_factor
        expected = np.clip(expected, 1e-6, None)

        chi2, p_value = stats.chisquare(baseline_counts, f_exp=expected)

        baseline_dist = {
            str(c): round(float(n / baseline_counts.sum()), 4)
            for c, n in zip(all_classes, baseline_counts)
        }
        current_dist = {
            str(c): round(float(n / current_counts.sum()), 4)
            for c, n in zip(all_classes, current_counts)
        }

        is_drifted = p_value < self.ks_alpha
        severity = (
            DriftSeverity.HIGH
            if p_value < 0.001
            else (
                DriftSeverity.MODERATE
                if p_value < 0.01
                else (DriftSeverity.LOW if p_value < 0.05 else DriftSeverity.NONE)
            )
        )

        result = LabelDriftResult(
            chi2_statistic=round(float(chi2), 6),
            chi2_pvalue=round(float(p_value), 6),
            severity=severity,
            is_drifted=is_drifted,
            label_distribution_baseline=baseline_dist,
            label_distribution_current=current_dist,
        )
        logger.info("Label drift: chi2=%.4f, p=%.4f, severity=%s", chi2, p_value, severity.value)
        return result

    # -- Full report --------------------------------------------------------

    def generate_drift_report(
        self,
        baseline_features: pd.DataFrame,
        current_features: pd.DataFrame,
        baseline_predictions: NDArray[np.float64] | None = None,
        current_predictions: NDArray[np.float64] | None = None,
        baseline_labels: NDArray[np.int64] | None = None,
        current_labels: NDArray[np.int64] | None = None,
        feature_columns: list[str] | None = None,
    ) -> DriftReport:
        """Generate a comprehensive drift detection report.

        Parameters
        ----------
        baseline_features : pd.DataFrame
            Training-time feature data.
        current_features : pd.DataFrame
            Current production feature data.
        baseline_predictions, current_predictions : array, optional
            Prediction arrays for prediction drift detection.
        baseline_labels, current_labels : array, optional
            Label arrays for label drift detection.
        feature_columns : list[str], optional
            Feature columns to check.

        Returns
        -------
        DriftReport
        """
        report_date = date.today().isoformat()

        # Feature drift
        feature_results = self.detect_feature_drift(
            baseline_features, current_features, feature_columns
        )
        n_drifted = sum(1 for r in feature_results if r.is_drifted)

        # Prediction drift
        pred_drift = None
        if baseline_predictions is not None and current_predictions is not None:
            pred_drift = self.detect_prediction_drift(baseline_predictions, current_predictions)

        # Label drift
        label_drift = None
        if baseline_labels is not None and current_labels is not None:
            label_drift = self.detect_label_drift(baseline_labels, current_labels)

        # Overall severity
        severities = [r.severity for r in feature_results]
        if pred_drift:
            severities.append(pred_drift.severity)
        if label_drift:
            severities.append(label_drift.severity)

        severity_order = list(DriftSeverity)
        overall = (
            max(severities, key=lambda s: severity_order.index(s))
            if severities
            else DriftSeverity.NONE
        )

        # Retraining recommendation
        drifted_pct = n_drifted / max(len(feature_results), 1)
        requires_retraining = (
            drifted_pct >= self.max_drifted_features_pct
            or overall in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
            or (label_drift is not None and label_drift.is_drifted)
        )

        summary_parts = [
            f"Analyzed {len(feature_results)} features: {n_drifted} drifted ({drifted_pct:.0%}).",
        ]
        if pred_drift:
            summary_parts.append(f"Prediction drift PSI={pred_drift.psi:.4f}.")
        if label_drift:
            summary_parts.append(
                f"Label drift chi2={label_drift.chi2_statistic:.4f} (p={label_drift.chi2_pvalue:.4f})."
            )
        if requires_retraining:
            summary_parts.append("RECOMMENDATION: Retrain the model.")

        report = DriftReport(
            report_date=report_date,
            feature_drift=feature_results,
            prediction_drift=pred_drift,
            label_drift=label_drift,
            total_features_analyzed=len(feature_results),
            features_drifted=n_drifted,
            overall_severity=overall,
            requires_retraining=requires_retraining,
            summary=" ".join(summary_parts),
        )
        logger.info("Drift report: %s", report.summary)
        return report

    # -- Snowflake integration ----------------------------------------------

    def fetch_production_features_from_snowflake(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Query Snowflake for production feature distributions.

        Requires ``snowflake_config`` to be set.
        """
        if not self.snowflake_config:
            raise ValueError("Snowflake config not provided")

        import snowflake.connector  # type: ignore[import-untyped]

        conn = snowflake.connector.connect(**self.snowflake_config)
        try:
            sql = FEATURE_DISTRIBUTION_SQL.format(
                database=self.snowflake_config.get("database", ""),
                schema=self.snowflake_config.get("schema", "PUBLIC"),
            )
            cursor = conn.cursor()
            cursor.execute(
                sql,
                {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    # -- SageMaker Model Monitor integration --------------------------------

    def check_sagemaker_monitor_violations(
        self,
        monitoring_schedule_name: str,
        region: str = "us-east-1",
    ) -> list[dict[str, Any]]:
        """Check recent SageMaker Model Monitor violations.

        Parameters
        ----------
        monitoring_schedule_name : str
            Name of the monitoring schedule.
        region : str
            AWS region.

        Returns
        -------
        list[dict]
            List of constraint violations.
        """
        import boto3

        client = boto3.client("sagemaker", region_name=region)
        response = client.list_monitoring_executions(
            MonitoringScheduleName=monitoring_schedule_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=5,
        )

        violations: list[dict[str, Any]] = []
        for execution in response.get("MonitoringExecutionSummaries", []):
            if execution.get("MonitoringExecutionStatus") == "CompletedWithViolations":
                violations.append(
                    {
                        "execution_time": str(execution["CreationTime"]),
                        "status": execution["MonitoringExecutionStatus"],
                        "failure_reason": execution.get("FailureReason", ""),
                    }
                )
        logger.info(
            "Found %d monitoring violations for %s",
            len(violations),
            monitoring_schedule_name,
        )
        return violations
