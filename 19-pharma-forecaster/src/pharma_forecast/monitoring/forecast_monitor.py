"""Forecast monitoring: accuracy tracking, bias detection, drift detection, and alerting.

Tracks forecast accuracy over time using MAE, MAPE, RMSE, and SMAPE. Detects
forecast bias and distribution drift in the underlying time series. Triggers
automatic retraining when accuracy degrades below configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of forecast monitoring alerts."""

    ACCURACY_DEGRADATION = "accuracy_degradation"
    FORECAST_BIAS = "forecast_bias"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MISSING_DATA = "missing_data"
    RETRAINING_TRIGGERED = "retraining_triggered"


@dataclass
class AccuracyMetrics:
    """Container for forecast accuracy measurements."""

    mae: float
    rmse: float
    mape: float
    smape: float
    median_ae: float
    max_ae: float
    r_squared: float
    n_observations: int
    evaluation_period: tuple[str, str] | None = None


@dataclass
class Alert:
    """A monitoring alert for a forecast series."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    series_id: str
    message: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    timestamp: str = ""
    resolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class DriftResult:
    """Result of a distribution drift test."""

    is_drifted: bool
    test_statistic: float
    p_value: float
    test_name: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    window_size: int


class MonitorConfig:
    """Configuration for the forecast monitor."""

    def __init__(
        self,
        mae_threshold: float = 100.0,
        mape_threshold: float = 0.15,
        rmse_threshold: float = 150.0,
        bias_threshold: float = 0.10,
        drift_p_value_threshold: float = 0.01,
        drift_window_size: int = 30,
        accuracy_lookback_periods: int = 90,
        retraining_cooldown_hours: int = 24,
        min_evaluation_samples: int = 10,
    ) -> None:
        self.mae_threshold = mae_threshold
        self.mape_threshold = mape_threshold
        self.rmse_threshold = rmse_threshold
        self.bias_threshold = bias_threshold
        self.drift_p_value_threshold = drift_p_value_threshold
        self.drift_window_size = drift_window_size
        self.accuracy_lookback_periods = accuracy_lookback_periods
        self.retraining_cooldown_hours = retraining_cooldown_hours
        self.min_evaluation_samples = min_evaluation_samples


class ForecastMonitor:
    """Monitors forecast accuracy and data integrity over time.

    Tracks MAE, MAPE, RMSE, SMAPE for each series, detects systematic
    bias (over/under forecasting), identifies distribution drift in the
    underlying time series, and triggers retraining when needed.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._accuracy_history: dict[str, list[AccuracyMetrics]] = {}
        self._alerts: list[Alert] = []
        self._alert_counter: int = 0
        self._last_retrain: dict[str, datetime] = {}
        self._reference_distributions: dict[str, pd.Series] = {}

    def _next_alert_id(self) -> str:
        """Generate a unique alert ID."""
        self._alert_counter += 1
        return f"alert_{self._alert_counter:06d}"

    def compute_accuracy(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        series_id: str | None = None,
    ) -> AccuracyMetrics:
        """Compute comprehensive forecast accuracy metrics.

        Args:
            actual: Actual observed values.
            predicted: Forecasted values.
            series_id: Optional series ID for tracking history.

        Returns:
            AccuracyMetrics with all computed measures.
        """
        # Align series
        common_idx = actual.index.intersection(predicted.index)
        if len(common_idx) < self.config.min_evaluation_samples:
            logger.warning(
                "insufficient_samples_for_accuracy",
                available=len(common_idx),
                required=self.config.min_evaluation_samples,
            )

        a = actual.loc[common_idx].values.astype(float)
        p = predicted.loc[common_idx].values.astype(float)

        errors = a - p
        abs_errors = np.abs(errors)

        # MAE
        mae = float(np.mean(abs_errors))

        # RMSE
        rmse = float(np.sqrt(np.mean(errors**2)))

        # MAPE (guarded against zero actuals)
        non_zero_mask = np.abs(a) > 1e-8
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs(errors[non_zero_mask] / a[non_zero_mask])))
        else:
            mape = 0.0

        # SMAPE
        denominator = np.abs(a) + np.abs(p)
        smape_mask = denominator > 0
        if np.any(smape_mask):
            smape = float(np.mean(2.0 * abs_errors[smape_mask] / denominator[smape_mask]))
        else:
            smape = 0.0

        # Median Absolute Error
        median_ae = float(np.median(abs_errors))

        # Max Absolute Error
        max_ae = float(np.max(abs_errors))

        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-8) if ss_tot > 0 else 0.0

        # Evaluation period
        eval_period = None
        if isinstance(actual.index, pd.DatetimeIndex) and len(common_idx) > 0:
            eval_period = (str(common_idx.min()), str(common_idx.max()))

        metrics = AccuracyMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            smape=smape,
            median_ae=median_ae,
            max_ae=max_ae,
            r_squared=float(r_squared),
            n_observations=len(common_idx),
            evaluation_period=eval_period,
        )

        # Track history
        if series_id:
            self._accuracy_history.setdefault(series_id, []).append(metrics)

        logger.info(
            "accuracy_computed",
            series_id=series_id,
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mape=round(mape, 4),
            smape=round(smape, 4),
            n_observations=len(common_idx),
        )

        return metrics

    def detect_bias(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        series_id: str,
    ) -> Alert | None:
        """Detect systematic forecast bias (over/under forecasting).

        Uses a one-sample t-test on forecast errors to determine if the
        mean error is significantly different from zero.

        Args:
            actual: Actual observed values.
            predicted: Forecasted values.
            series_id: Series identifier.

        Returns:
            Alert if bias detected, None otherwise.
        """
        common_idx = actual.index.intersection(predicted.index)
        if len(common_idx) < self.config.min_evaluation_samples:
            return None

        errors = (predicted.loc[common_idx] - actual.loc[common_idx]).values

        # One-sample t-test: H0 is that mean error = 0
        t_stat, p_value = stats.ttest_1samp(errors, 0)
        mean_error = float(np.mean(errors))
        mean_actual = float(np.mean(actual.loc[common_idx].values))

        # Relative bias
        relative_bias = abs(mean_error) / (abs(mean_actual) + 1e-8)

        if relative_bias > self.config.bias_threshold and p_value < 0.05:
            direction = "over" if mean_error > 0 else "under"
            severity = (
                AlertSeverity.CRITICAL
                if relative_bias > 2 * self.config.bias_threshold
                else AlertSeverity.WARNING
            )

            alert = Alert(
                alert_id=self._next_alert_id(),
                alert_type=AlertType.FORECAST_BIAS,
                severity=severity,
                series_id=series_id,
                message=(
                    f"Systematic {direction}-forecasting detected for {series_id}. "
                    f"Relative bias: {relative_bias:.2%}, t-stat: {t_stat:.2f}, p={p_value:.4f}"
                ),
                metric_name="relative_bias",
                metric_value=relative_bias,
                threshold=self.config.bias_threshold,
                metadata={
                    "direction": direction,
                    "mean_error": mean_error,
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                },
            )
            self._alerts.append(alert)
            logger.warning(
                "forecast_bias_detected",
                series_id=series_id,
                direction=direction,
                relative_bias=round(relative_bias, 4),
            )
            return alert

        return None

    def check_accuracy_threshold(
        self,
        metrics: AccuracyMetrics,
        series_id: str,
    ) -> list[Alert]:
        """Check if accuracy metrics exceed alert thresholds.

        Args:
            metrics: Computed accuracy metrics.
            series_id: Series identifier.

        Returns:
            List of triggered alerts.
        """
        alerts: list[Alert] = []

        checks = [
            ("mae", metrics.mae, self.config.mae_threshold),
            ("mape", metrics.mape, self.config.mape_threshold),
            ("rmse", metrics.rmse, self.config.rmse_threshold),
        ]

        for metric_name, value, threshold in checks:
            if value > threshold:
                severity = (
                    AlertSeverity.CRITICAL if value > 2 * threshold else AlertSeverity.WARNING
                )

                alert = Alert(
                    alert_id=self._next_alert_id(),
                    alert_type=AlertType.ACCURACY_DEGRADATION,
                    severity=severity,
                    series_id=series_id,
                    message=(
                        f"{metric_name.upper()} for {series_id} ({value:.4f}) "
                        f"exceeds threshold ({threshold:.4f})"
                    ),
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=threshold,
                )
                alerts.append(alert)
                self._alerts.append(alert)

        if alerts:
            logger.warning(
                "accuracy_threshold_breached",
                series_id=series_id,
                n_alerts=len(alerts),
            )

        return alerts

    def detect_drift(
        self,
        series: pd.Series,
        series_id: str,
        window_size: int | None = None,
    ) -> DriftResult:
        """Detect distribution drift in the underlying time series.

        Compares the distribution of recent observations against a reference
        window using the Kolmogorov-Smirnov two-sample test.

        Args:
            series: Full time series.
            series_id: Series identifier.
            window_size: Size of comparison windows. Defaults to config value.

        Returns:
            DriftResult with test statistics and drift determination.
        """
        ws = window_size or self.config.drift_window_size

        if len(series) < 2 * ws:
            logger.warning("insufficient_data_for_drift_test", length=len(series), required=2 * ws)
            return DriftResult(
                is_drifted=False,
                test_statistic=0.0,
                p_value=1.0,
                test_name="ks_2samp",
                reference_mean=float(series.mean()),
                current_mean=float(series.mean()),
                reference_std=float(series.std()),
                current_std=float(series.std()),
                window_size=ws,
            )

        reference = series.iloc[-2 * ws : -ws].values
        current = series.iloc[-ws:].values

        ks_stat, p_value = stats.ks_2samp(reference, current)
        is_drifted = p_value < self.config.drift_p_value_threshold

        result = DriftResult(
            is_drifted=is_drifted,
            test_statistic=float(ks_stat),
            p_value=float(p_value),
            test_name="ks_2samp",
            reference_mean=float(np.mean(reference)),
            current_mean=float(np.mean(current)),
            reference_std=float(np.std(reference)),
            current_std=float(np.std(current)),
            window_size=ws,
        )

        if is_drifted:
            alert = Alert(
                alert_id=self._next_alert_id(),
                alert_type=AlertType.DATA_DRIFT,
                severity=AlertSeverity.WARNING,
                series_id=series_id,
                message=(
                    f"Distribution drift detected for {series_id}. "
                    f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.6f}. "
                    f"Mean shifted from {result.reference_mean:.2f} to {result.current_mean:.2f}"
                ),
                metric_name="ks_statistic",
                metric_value=float(ks_stat),
                threshold=self.config.drift_p_value_threshold,
                metadata={
                    "reference_mean": result.reference_mean,
                    "current_mean": result.current_mean,
                    "mean_shift": result.current_mean - result.reference_mean,
                },
            )
            self._alerts.append(alert)
            logger.warning(
                "data_drift_detected",
                series_id=series_id,
                ks_statistic=round(ks_stat, 4),
                p_value=round(p_value, 6),
            )

        return result

    def should_retrain(self, series_id: str) -> bool:
        """Determine if a model should be retrained.

        Checks for accuracy degradation, drift, and bias alerts. Enforces
        a cooldown period to prevent excessive retraining.

        Args:
            series_id: Series identifier.

        Returns:
            True if retraining is recommended.
        """
        # Check cooldown
        if series_id in self._last_retrain:
            hours_since = (
                datetime.now(UTC) - self._last_retrain[series_id]
            ).total_seconds() / 3600
            if hours_since < self.config.retraining_cooldown_hours:
                logger.debug(
                    "retraining_cooldown_active",
                    series_id=series_id,
                    hours_remaining=round(self.config.retraining_cooldown_hours - hours_since, 1),
                )
                return False

        # Check for active (unresolved) alerts for this series
        active_alerts = [
            a
            for a in self._alerts
            if a.series_id == series_id
            and not a.resolved
            and a.alert_type
            in (
                AlertType.ACCURACY_DEGRADATION,
                AlertType.DATA_DRIFT,
                AlertType.CONCEPT_DRIFT,
            )
        ]

        # Need at least one critical or two warnings to trigger retraining
        n_critical = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        n_warning = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)

        should = n_critical >= 1 or n_warning >= 2

        if should:
            self._last_retrain[series_id] = datetime.now(UTC)
            retrain_alert = Alert(
                alert_id=self._next_alert_id(),
                alert_type=AlertType.RETRAINING_TRIGGERED,
                severity=AlertSeverity.INFO,
                series_id=series_id,
                message=f"Automatic retraining triggered for {series_id}",
                metadata={
                    "n_critical_alerts": n_critical,
                    "n_warning_alerts": n_warning,
                },
            )
            self._alerts.append(retrain_alert)
            logger.info(
                "retraining_triggered",
                series_id=series_id,
                n_critical=n_critical,
                n_warning=n_warning,
            )

        return should

    def get_active_alerts(
        self,
        series_id: str | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Retrieve active (unresolved) alerts.

        Args:
            series_id: Filter by series ID (optional).
            severity: Filter by severity level (optional).

        Returns:
            List of matching active alerts.
        """
        alerts = [a for a in self._alerts if not a.resolved]

        if series_id:
            alerts = [a for a in alerts if a.series_id == series_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved.

        Args:
            alert_id: ID of the alert to resolve.

        Returns:
            True if alert was found and resolved.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info("alert_resolved", alert_id=alert_id)
                return True
        return False

    def get_accuracy_history(self, series_id: str) -> list[AccuracyMetrics]:
        """Get the accuracy tracking history for a series.

        Args:
            series_id: Series identifier.

        Returns:
            List of AccuracyMetrics in chronological order.
        """
        return self._accuracy_history.get(series_id, [])

    def get_accuracy_trend(self, series_id: str) -> dict[str, Any]:
        """Analyze the trend in forecast accuracy over time.

        Args:
            series_id: Series identifier.

        Returns:
            Dictionary with trend direction and statistics.
        """
        history = self._accuracy_history.get(series_id, [])
        if len(history) < 3:
            return {"trend": "insufficient_data", "n_observations": len(history)}

        mae_values = [m.mae for m in history]
        x = np.arange(len(mae_values))
        slope, intercept = np.polyfit(x, mae_values, 1)

        if slope > 0.01:
            trend = "degrading"
        elif slope < -0.01:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "mae_slope": float(slope),
            "latest_mae": mae_values[-1],
            "best_mae": min(mae_values),
            "worst_mae": max(mae_values),
            "n_observations": len(mae_values),
        }
