"""Experiment health monitoring with SRM detection, data quality, and guardrails.

Monitors running experiments for sample ratio mismatch, data quality issues,
early stopping opportunities, metric degradation, and guardrail violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog
from scipy import stats

from model_lab.experiments.experiment_manager import Experiment, ExperimentState

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity level for monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Type of monitoring alert."""

    SRM = "sample_ratio_mismatch"
    DATA_QUALITY = "data_quality"
    EARLY_STOPPING = "early_stopping"
    METRIC_DEGRADATION = "metric_degradation"
    GUARDRAIL_VIOLATION = "guardrail_violation"


@dataclass
class Alert:
    """A monitoring alert for an experiment."""

    alert_type: AlertType
    severity: AlertSeverity
    experiment_id: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    actionable: bool = True


@dataclass
class HealthReport:
    """Comprehensive health report for an experiment."""

    experiment_id: str
    is_healthy: bool
    alerts: list[Alert] = field(default_factory=list)
    srm_check: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    guardrail_status: dict[str, Any] = field(default_factory=dict)
    early_stopping_recommendation: str | None = None


@dataclass
class GuardrailConfig:
    """Configuration for a guardrail metric."""

    metric_name: str
    max_degradation_pct: float = 5.0  # Maximum allowed degradation percentage
    direction: str = "lower_is_better"  # "lower_is_better" or "higher_is_better"
    absolute_threshold: float | None = None


class ExperimentMonitor:
    """Monitors experiment health and generates alerts.

    Checks for sample ratio mismatch, data quality issues, early stopping
    opportunities, metric degradation, and guardrail violations.
    """

    def __init__(
        self,
        srm_alpha: float = 0.001,
        early_stopping_threshold: float = 0.99,
        guardrails: list[GuardrailConfig] | None = None,
    ) -> None:
        self.srm_alpha = srm_alpha
        self.early_stopping_threshold = early_stopping_threshold
        self.guardrails = guardrails or []
        self._alert_history: list[Alert] = []

    def check_health(
        self,
        experiment: Experiment,
        variant_data: dict[str, dict[str, Any]],
    ) -> HealthReport:
        """Run all health checks and generate a comprehensive report.

        Args:
            experiment: The experiment to check.
            variant_data: Dict mapping variant_id to data dict containing:
                - sample_size: int
                - conversions: int (optional)
                - metric_values: list[float] (optional)
                - latency_values: list[float] (optional)
                - error_count: int (optional)

        Returns:
            HealthReport with all check results and alerts.
        """
        alerts: list[Alert] = []

        # SRM check
        srm_result = self.check_srm(experiment, variant_data)
        if srm_result.get("is_mismatch"):
            alerts.append(
                Alert(
                    alert_type=AlertType.SRM,
                    severity=AlertSeverity.CRITICAL,
                    experiment_id=experiment.id,
                    message=(
                        f"Sample Ratio Mismatch detected (p={srm_result['p_value']:.6f}). "
                        "Traffic allocation may be biased."
                    ),
                    details=srm_result,
                )
            )

        # Data quality checks
        dq_result = self.check_data_quality(experiment, variant_data)
        for issue in dq_result.get("issues", []):
            alerts.append(
                Alert(
                    alert_type=AlertType.DATA_QUALITY,
                    severity=AlertSeverity.WARNING,
                    experiment_id=experiment.id,
                    message=issue["message"],
                    details=issue,
                )
            )

        # Guardrail checks
        guardrail_result = self.check_guardrails(experiment, variant_data)
        for violation in guardrail_result.get("violations", []):
            alerts.append(
                Alert(
                    alert_type=AlertType.GUARDRAIL_VIOLATION,
                    severity=AlertSeverity.CRITICAL,
                    experiment_id=experiment.id,
                    message=violation["message"],
                    details=violation,
                    actionable=True,
                )
            )

        # Early stopping check
        early_stop = self.check_early_stopping(experiment, variant_data)
        if early_stop.get("should_stop"):
            severity = (
                AlertSeverity.CRITICAL
                if early_stop.get("reason") == "clear_loser"
                else AlertSeverity.INFO
            )
            alerts.append(
                Alert(
                    alert_type=AlertType.EARLY_STOPPING,
                    severity=severity,
                    experiment_id=experiment.id,
                    message=early_stop["message"],
                    details=early_stop,
                )
            )

        self._alert_history.extend(alerts)

        is_healthy = not any(a.severity == AlertSeverity.CRITICAL for a in alerts)

        return HealthReport(
            experiment_id=experiment.id,
            is_healthy=is_healthy,
            alerts=alerts,
            srm_check=srm_result,
            data_quality=dq_result,
            guardrail_status=guardrail_result,
            early_stopping_recommendation=early_stop.get("recommendation"),
        )

    def check_srm(
        self,
        experiment: Experiment,
        variant_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Check for Sample Ratio Mismatch using chi-squared goodness of fit.

        SRM occurs when the observed traffic split differs significantly from
        the expected split, indicating a bug in the randomization.

        Uses a chi-squared test comparing observed counts to expected counts
        based on the configured traffic percentages.
        """
        observed = []
        expected_ratios = []

        for variant in experiment.config.variants:
            data = variant_data.get(variant.id, {})
            observed.append(data.get("sample_size", 0))
            expected_ratios.append(variant.traffic_percentage / 100.0)

        total_observed = sum(observed)
        if total_observed == 0:
            return {"is_mismatch": False, "p_value": 1.0, "message": "No data yet"}

        expected = [total_observed * r for r in expected_ratios]

        # Chi-squared goodness of fit
        chi2 = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected) if exp > 0)
        df = len(observed) - 1
        p_value = 1.0 - float(stats.chi2.cdf(chi2, df)) if df > 0 else 1.0

        is_mismatch = p_value < self.srm_alpha

        result = {
            "is_mismatch": is_mismatch,
            "p_value": p_value,
            "chi2_statistic": chi2,
            "observed": observed,
            "expected": [round(e, 1) for e in expected],
            "total_samples": total_observed,
        }

        if is_mismatch:
            logger.warning(
                "srm_detected",
                experiment_id=experiment.id,
                p_value=p_value,
                observed=observed,
                expected=expected,
            )

        return result

    def check_data_quality(
        self,
        experiment: Experiment,
        variant_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Check data quality across variants.

        Checks for:
        - Missing data (variant with no observations)
        - Extreme outliers in metric values
        - Suspiciously low variance (possible data pipeline issue)
        - Conversion rate outside plausible range
        """
        issues: list[dict[str, Any]] = []

        for variant in experiment.config.variants:
            vid = variant.id
            data = variant_data.get(vid, {})
            sample_size = data.get("sample_size", 0)

            # Check for missing data
            if sample_size == 0:
                issues.append(
                    {
                        "variant_id": vid,
                        "check": "missing_data",
                        "message": f"Variant '{variant.name}' has no observations",
                        "severity": "warning",
                    }
                )
                continue

            # Check conversion rate plausibility
            conversions = data.get("conversions")
            if conversions is not None and sample_size > 0:
                conv_rate = conversions / sample_size
                if conv_rate > 0.99 or (conv_rate < 0.001 and sample_size > 100):
                    issues.append(
                        {
                            "variant_id": vid,
                            "check": "implausible_conversion",
                            "message": (
                                f"Variant '{variant.name}' has conversion rate {conv_rate:.4f}, "
                                "which may indicate a data issue"
                            ),
                            "severity": "warning",
                            "conversion_rate": conv_rate,
                        }
                    )

            # Check metric values for outliers and zero variance
            metric_values = data.get("metric_values")
            if metric_values is not None and len(metric_values) > 10:
                values = np.asarray(metric_values, dtype=np.float64)
                std = float(np.std(values, ddof=1))
                mean = float(np.mean(values))

                # Zero or near-zero variance
                if std < 1e-10:
                    issues.append(
                        {
                            "variant_id": vid,
                            "check": "zero_variance",
                            "message": (
                                f"Variant '{variant.name}' metric values have near-zero variance"
                            ),
                            "severity": "warning",
                        }
                    )

                # Check for extreme outliers (beyond 5 sigma)
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    extreme_count = int(np.sum(z_scores > 5))
                    if extreme_count > 0:
                        issues.append(
                            {
                                "variant_id": vid,
                                "check": "extreme_outliers",
                                "message": (
                                    f"Variant '{variant.name}' has {extreme_count} "
                                    "observations beyond 5 standard deviations"
                                ),
                                "severity": "info",
                                "outlier_count": extreme_count,
                            }
                        )

            # Check for NaN or infinite values
            if metric_values is not None:
                values = np.asarray(metric_values, dtype=np.float64)
                nan_count = int(np.sum(np.isnan(values)))
                inf_count = int(np.sum(np.isinf(values)))
                if nan_count > 0 or inf_count > 0:
                    issues.append(
                        {
                            "variant_id": vid,
                            "check": "invalid_values",
                            "message": (
                                f"Variant '{variant.name}' has {nan_count} NaN and "
                                f"{inf_count} infinite values"
                            ),
                            "severity": "critical",
                        }
                    )

        return {
            "is_clean": len(issues) == 0,
            "issues": issues,
            "variants_checked": len(experiment.config.variants),
        }

    def check_guardrails(
        self,
        experiment: Experiment,
        variant_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Check guardrail metrics to ensure no variant causes unacceptable harm.

        Guardrails are metrics that must not regress beyond a threshold,
        regardless of the primary metric outcome. Common guardrails include
        latency, error rate, and crash rate.
        """
        violations: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []

        # Find control variant
        control_variant = next((v for v in experiment.config.variants if v.is_control), None)
        if not control_variant:
            return {"violations": [], "checks": [], "message": "No control variant found"}

        control_data = variant_data.get(control_variant.id, {})

        for guardrail in self.guardrails:
            for variant in experiment.config.variants:
                if variant.is_control:
                    continue

                treatment_data = variant_data.get(variant.id, {})
                check_result = self._evaluate_guardrail(
                    guardrail, control_data, treatment_data, variant.name
                )
                checks.append(check_result)

                if check_result.get("violated"):
                    violations.append(check_result)

        return {
            "violations": violations,
            "checks": checks,
            "all_passed": len(violations) == 0,
        }

    def check_early_stopping(
        self,
        experiment: Experiment,
        variant_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Check if the experiment can be stopped early.

        Uses Bayesian probability thresholds to determine if one variant
        is a clear winner or loser, making continued experimentation wasteful.
        """
        if experiment.state != ExperimentState.RUNNING:
            return {"should_stop": False, "message": "Experiment not running"}

        # Get control and treatment data
        control_variant = next((v for v in experiment.config.variants if v.is_control), None)
        if not control_variant:
            return {"should_stop": False, "message": "No control variant"}

        control_data = variant_data.get(control_variant.id, {})
        control_conversions = control_data.get("conversions", 0)
        control_n = control_data.get("sample_size", 0)

        if control_n < 100:
            return {
                "should_stop": False,
                "message": "Insufficient data for early stopping evaluation",
            }

        results = []
        for variant in experiment.config.variants:
            if variant.is_control:
                continue

            treatment_data = variant_data.get(variant.id, {})
            treatment_conversions = treatment_data.get("conversions", 0)
            treatment_n = treatment_data.get("sample_size", 0)

            if treatment_n < 100:
                continue

            # Use Beta-Binomial model to compute P(treatment > control)
            alpha_c = 1 + control_conversions
            beta_c = 1 + (control_n - control_conversions)
            alpha_t = 1 + treatment_conversions
            beta_t = 1 + (treatment_n - treatment_conversions)

            # Monte Carlo estimate of P(treatment > control)
            rng = np.random.default_rng(42)
            samples_c = rng.beta(alpha_c, beta_c, size=50000)
            samples_t = rng.beta(alpha_t, beta_t, size=50000)
            prob_treatment_better = float(np.mean(samples_t > samples_c))

            results.append(
                {
                    "variant_id": variant.id,
                    "variant_name": variant.name,
                    "prob_better": prob_treatment_better,
                    "treatment_rate": treatment_conversions / treatment_n if treatment_n > 0 else 0,
                    "control_rate": control_conversions / control_n if control_n > 0 else 0,
                }
            )

        # Check thresholds
        for result in results:
            if result["prob_better"] >= self.early_stopping_threshold:
                return {
                    "should_stop": True,
                    "reason": "clear_winner",
                    "message": (
                        f"Variant '{result['variant_name']}' is a clear winner "
                        f"(P(better)={result['prob_better']:.4f})"
                    ),
                    "recommendation": "stop_and_promote",
                    "details": result,
                }
            elif result["prob_better"] <= (1 - self.early_stopping_threshold):
                return {
                    "should_stop": True,
                    "reason": "clear_loser",
                    "message": (
                        f"Variant '{result['variant_name']}' is a clear loser "
                        f"(P(better)={result['prob_better']:.4f})"
                    ),
                    "recommendation": "stop_and_revert",
                    "details": result,
                }

        return {
            "should_stop": False,
            "message": "No clear winner/loser yet",
            "recommendation": "continue",
            "variant_results": results,
        }

    def _evaluate_guardrail(
        self,
        guardrail: GuardrailConfig,
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
        variant_name: str,
    ) -> dict[str, Any]:
        """Evaluate a single guardrail metric for a variant."""
        metric_name = guardrail.metric_name

        # Try to get metric from latency or error data
        control_value = self._extract_guardrail_value(metric_name, control_data)
        treatment_value = self._extract_guardrail_value(metric_name, treatment_data)

        if control_value is None or treatment_value is None:
            return {
                "metric": metric_name,
                "variant": variant_name,
                "violated": False,
                "message": f"Insufficient data for guardrail '{metric_name}'",
            }

        # Compute degradation
        if guardrail.direction == "lower_is_better":
            degradation_pct = (
                ((treatment_value - control_value) / control_value * 100)
                if control_value != 0
                else 0.0
            )
            violated = degradation_pct > guardrail.max_degradation_pct
        else:
            degradation_pct = (
                ((control_value - treatment_value) / control_value * 100)
                if control_value != 0
                else 0.0
            )
            violated = degradation_pct > guardrail.max_degradation_pct

        # Also check absolute threshold
        if guardrail.absolute_threshold is not None:
            if guardrail.direction == "lower_is_better":
                violated = violated or (treatment_value > guardrail.absolute_threshold)
            else:
                violated = violated or (treatment_value < guardrail.absolute_threshold)

        result = {
            "metric": metric_name,
            "variant": variant_name,
            "control_value": control_value,
            "treatment_value": treatment_value,
            "degradation_pct": degradation_pct,
            "max_allowed_pct": guardrail.max_degradation_pct,
            "violated": violated,
        }

        if violated:
            result["message"] = (
                f"Guardrail '{metric_name}' violated for variant '{variant_name}': "
                f"{degradation_pct:.1f}% degradation (max allowed: {guardrail.max_degradation_pct}%)"
            )
            logger.warning("guardrail_violated", **result)
        else:
            result["message"] = f"Guardrail '{metric_name}' passed for variant '{variant_name}'"

        return result

    def _extract_guardrail_value(
        self,
        metric_name: str,
        data: dict[str, Any],
    ) -> float | None:
        """Extract a guardrail metric value from variant data."""
        # Direct metric value
        if metric_name in data:
            return float(data[metric_name])

        # Compute from latency values
        if metric_name.startswith("latency_p") and "latency_values" in data:
            values = np.asarray(data["latency_values"], dtype=np.float64)
            if len(values) == 0:
                return None
            percentile = int(metric_name.replace("latency_p", ""))
            return float(np.percentile(values, percentile))

        # Compute error rate
        if metric_name == "error_rate" and "error_count" in data and "sample_size" in data:
            n = data["sample_size"]
            return data["error_count"] / n if n > 0 else None

        return None

    def get_alert_history(
        self,
        experiment_id: str | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Retrieve alert history with optional filtering."""
        alerts = self._alert_history
        if experiment_id:
            alerts = [a for a in alerts if a.experiment_id == experiment_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
