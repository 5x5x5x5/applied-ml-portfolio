"""Alerting system for drift and performance degradation.

Sends notifications via SNS, publishes custom CloudWatch metrics,
and triggers automatic retraining workflows when thresholds are breached.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import boto3
from pydantic import BaseModel, Field

from drug_interaction.monitoring.drift_detector import DriftReport, DriftSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertConfig(BaseModel):
    """Configuration for the alerting system."""

    sns_topic_arn: str
    cloudwatch_namespace: str = "DrugInteractionML/Monitoring"
    step_functions_arn: str = Field(
        default="", description="Step Functions ARN for retraining trigger"
    )
    enable_auto_retrain: bool = True
    critical_drift_threshold: float = 0.3
    performance_degradation_threshold: float = 0.05
    min_hours_between_alerts: int = 4
    region: str = "us-east-1"


class Alert(BaseModel):
    """A single alert record."""

    alert_id: str
    level: AlertLevel
    source: str
    title: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    action_taken: str = ""


# ---------------------------------------------------------------------------
# Alerting system
# ---------------------------------------------------------------------------


@dataclass
class DriftAlertingSystem:
    """Monitor drift reports and send alerts when thresholds are breached.

    Parameters
    ----------
    config : AlertConfig
        Alerting configuration.
    """

    config: AlertConfig
    _sns_client: Any = field(default=None, init=False, repr=False)
    _cw_client: Any = field(default=None, init=False, repr=False)
    _sfn_client: Any = field(default=None, init=False, repr=False)
    _last_alert_time: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sns_client = boto3.client("sns", region_name=self.config.region)
        self._cw_client = boto3.client("cloudwatch", region_name=self.config.region)
        if self.config.step_functions_arn:
            self._sfn_client = boto3.client("stepfunctions", region_name=self.config.region)

    # -- Alert processing ---------------------------------------------------

    def process_drift_report(self, report: DriftReport) -> list[Alert]:
        """Analyse a drift report and generate/send appropriate alerts.

        Parameters
        ----------
        report : DriftReport
            Drift detection report to process.

        Returns
        -------
        list[Alert]
            All alerts generated from this report.
        """
        alerts: list[Alert] = []

        # Publish CloudWatch metrics regardless of alert thresholds
        self._publish_drift_metrics(report)

        # Feature drift alerts
        for feature_result in report.feature_drift:
            if feature_result.is_drifted:
                level = self._severity_to_alert_level(feature_result.severity)
                if level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
                    alert = self._create_alert(
                        level=level,
                        source="feature_drift",
                        title=f"Feature Drift: {feature_result.feature_name}",
                        message=(
                            f"Feature '{feature_result.feature_name}' has drifted. "
                            f"PSI={feature_result.psi:.4f}, "
                            f"KS stat={feature_result.ks_statistic:.4f} "
                            f"(p={feature_result.ks_pvalue:.4f}). "
                            f"Severity: {feature_result.severity.value}."
                        ),
                        metadata={
                            "feature_name": feature_result.feature_name,
                            "psi": feature_result.psi,
                            "ks_statistic": feature_result.ks_statistic,
                            "ks_pvalue": feature_result.ks_pvalue,
                            "severity": feature_result.severity.value,
                        },
                    )
                    alerts.append(alert)

        # Prediction drift alert
        if report.prediction_drift and report.prediction_drift.is_drifted:
            level = self._severity_to_alert_level(report.prediction_drift.severity)
            alert = self._create_alert(
                level=level,
                source="prediction_drift",
                title="Prediction Distribution Drift Detected",
                message=(
                    f"Model predictions are drifting. "
                    f"PSI={report.prediction_drift.psi:.4f}, "
                    f"prediction shift={report.prediction_drift.prediction_shift:.4f}. "
                    f"Severity: {report.prediction_drift.severity.value}."
                ),
                metadata={
                    "psi": report.prediction_drift.psi,
                    "prediction_shift": report.prediction_drift.prediction_shift,
                },
            )
            alerts.append(alert)

        # Label drift alert
        if report.label_drift and report.label_drift.is_drifted:
            level = self._severity_to_alert_level(report.label_drift.severity)
            alert = self._create_alert(
                level=level,
                source="label_drift",
                title="Label Distribution Drift Detected",
                message=(
                    f"Ground-truth labels are drifting. "
                    f"Chi2={report.label_drift.chi2_statistic:.4f} "
                    f"(p={report.label_drift.chi2_pvalue:.4f}). "
                    f"This may indicate a concept drift."
                ),
                metadata={
                    "chi2": report.label_drift.chi2_statistic,
                    "chi2_pvalue": report.label_drift.chi2_pvalue,
                },
            )
            alerts.append(alert)

        # Overall severity alert
        if report.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            alert = self._create_alert(
                level=AlertLevel.CRITICAL,
                source="overall_drift",
                title="Critical Overall Drift Detected",
                message=(
                    f"Overall drift severity: {report.overall_severity.value}. "
                    f"{report.features_drifted}/{report.total_features_analyzed} features drifted. "
                    f"{report.summary}"
                ),
                metadata={
                    "features_drifted": report.features_drifted,
                    "total_features": report.total_features_analyzed,
                    "overall_severity": report.overall_severity.value,
                },
            )
            alerts.append(alert)

        # Send all alerts via SNS
        for alert in alerts:
            self._send_sns_alert(alert)

        # Trigger retraining if needed
        if report.requires_retraining and self.config.enable_auto_retrain:
            self._trigger_retraining(report)
            for alert in alerts:
                alert.action_taken = "Automatic retraining triggered"

        logger.info("Processed drift report: %d alerts generated", len(alerts))
        return alerts

    # -- Performance alerting -----------------------------------------------

    def alert_on_performance_degradation(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        primary_metric: str = "f1_macro",
    ) -> Alert | None:
        """Send alert if model performance has degraded.

        Parameters
        ----------
        current_metrics : dict
            Current production metrics.
        baseline_metrics : dict
            Metrics at deployment time.
        primary_metric : str
            The metric to check for degradation.

        Returns
        -------
        Alert or None
            Alert if degradation detected, else None.
        """
        current_val = current_metrics.get(primary_metric, 0.0)
        baseline_val = baseline_metrics.get(primary_metric, 0.0)
        degradation = baseline_val - current_val

        # Publish metric
        self._cw_client.put_metric_data(
            Namespace=self.config.cloudwatch_namespace,
            MetricData=[
                {
                    "MetricName": f"production_{primary_metric}",
                    "Value": current_val,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Model", "Value": "drug-interaction-xgboost"},
                    ],
                },
                {
                    "MetricName": "performance_degradation",
                    "Value": degradation,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Model", "Value": "drug-interaction-xgboost"},
                        {"Name": "Metric", "Value": primary_metric},
                    ],
                },
            ],
        )

        if degradation > self.config.performance_degradation_threshold:
            alert = self._create_alert(
                level=AlertLevel.CRITICAL,
                source="performance_degradation",
                title=f"Model Performance Degradation: {primary_metric}",
                message=(
                    f"Production {primary_metric} has dropped from "
                    f"{baseline_val:.4f} to {current_val:.4f} "
                    f"(degradation: {degradation:.4f}). "
                    f"Threshold: {self.config.performance_degradation_threshold:.4f}."
                ),
                metadata={
                    "current_value": current_val,
                    "baseline_value": baseline_val,
                    "degradation": degradation,
                    **current_metrics,
                },
            )
            self._send_sns_alert(alert)
            return alert

        return None

    # -- CloudWatch metrics -------------------------------------------------

    def _publish_drift_metrics(self, report: DriftReport) -> None:
        """Publish drift metrics to CloudWatch."""
        metric_data: list[dict[str, Any]] = [
            {
                "MetricName": "features_drifted_count",
                "Value": float(report.features_drifted),
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "Model", "Value": "drug-interaction-xgboost"},
                ],
            },
            {
                "MetricName": "features_drifted_pct",
                "Value": (report.features_drifted / max(report.total_features_analyzed, 1)) * 100,
                "Unit": "Percent",
                "Dimensions": [
                    {"Name": "Model", "Value": "drug-interaction-xgboost"},
                ],
            },
        ]

        # Add per-feature PSI metrics for top drifted features
        drifted = sorted(
            [f for f in report.feature_drift if f.is_drifted],
            key=lambda f: f.psi,
            reverse=True,
        )[:10]
        for feat in drifted:
            metric_data.append(
                {
                    "MetricName": "feature_psi",
                    "Value": feat.psi,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Model", "Value": "drug-interaction-xgboost"},
                        {"Name": "Feature", "Value": feat.feature_name},
                    ],
                }
            )

        if report.prediction_drift:
            metric_data.append(
                {
                    "MetricName": "prediction_drift_psi",
                    "Value": report.prediction_drift.psi,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Model", "Value": "drug-interaction-xgboost"},
                    ],
                }
            )

        # CloudWatch supports max 25 metrics per put call
        for i in range(0, len(metric_data), 25):
            batch = metric_data[i : i + 25]
            self._cw_client.put_metric_data(
                Namespace=self.config.cloudwatch_namespace,
                MetricData=batch,
            )
        logger.info("Published %d CloudWatch metrics", len(metric_data))

    # -- SNS notifications --------------------------------------------------

    def _send_sns_alert(self, alert: Alert) -> None:
        """Send an alert via SNS."""
        # Throttle: skip if we sent the same source alert recently
        source_key = f"{alert.source}:{alert.level.value}"
        now = datetime.now(tz=UTC)
        if source_key in self._last_alert_time:
            hours_since = (now - self._last_alert_time[source_key]).total_seconds() / 3600
            if hours_since < self.config.min_hours_between_alerts:
                logger.info("Throttling alert %s (sent %.1f hours ago)", source_key, hours_since)
                return

        message_body = json.dumps(
            {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "source": alert.source,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata,
            },
            indent=2,
        )

        self._sns_client.publish(
            TopicArn=self.config.sns_topic_arn,
            Subject=f"[{alert.level.value}] DrugInteractionML: {alert.title}"[:100],
            Message=message_body,
            MessageAttributes={
                "alert_level": {
                    "DataType": "String",
                    "StringValue": alert.level.value,
                },
                "source": {
                    "DataType": "String",
                    "StringValue": alert.source,
                },
            },
        )
        self._last_alert_time[source_key] = now
        logger.info("Sent SNS alert: %s", alert.title)

    # -- Retraining trigger -------------------------------------------------

    def _trigger_retraining(self, report: DriftReport) -> None:
        """Trigger a retraining workflow via Step Functions."""
        if not self._sfn_client or not self.config.step_functions_arn:
            logger.warning("Step Functions not configured, skipping retraining trigger")
            return

        execution_name = f"auto-retrain-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
        input_data = {
            "trigger": "auto_drift_detection",
            "report_date": report.report_date,
            "features_drifted": report.features_drifted,
            "overall_severity": report.overall_severity.value,
            "summary": report.summary,
        }

        self._sfn_client.start_execution(
            stateMachineArn=self.config.step_functions_arn,
            name=execution_name,
            input=json.dumps(input_data),
        )
        logger.info("Triggered retraining: %s", execution_name)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _severity_to_alert_level(severity: DriftSeverity) -> AlertLevel:
        """Map drift severity to alert level."""
        if severity in (DriftSeverity.CRITICAL, DriftSeverity.HIGH):
            return AlertLevel.CRITICAL
        if severity == DriftSeverity.MODERATE:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    @staticmethod
    def _create_alert(
        *,
        level: AlertLevel,
        source: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """Create an Alert instance."""
        import uuid

        return Alert(
            alert_id=str(uuid.uuid4()),
            level=level,
            source=source,
            title=title,
            message=message,
            timestamp=datetime.now(tz=UTC).isoformat(),
            metadata=metadata or {},
        )
