"""Metrics computation for A/B test experiments.

Computes conversion rates, latency percentiles, model quality metrics,
revenue impact, and CUPED variance reduction for experiment analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricResult:
    """Result of a metric computation for one variant."""

    metric_name: str
    variant_id: str
    value: float
    ci_lower: float
    ci_upper: float
    sample_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantileMetrics:
    """Quantile-based latency metrics."""

    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float
    min_val: float
    max_val: float
    sample_size: int


@dataclass
class ModelQualityMetrics:
    """Classification model quality metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_size: int


@dataclass
class RevenueImpact:
    """Estimated revenue impact of a variant."""

    mean_revenue_per_user_control: float
    mean_revenue_per_user_treatment: float
    absolute_lift: float
    relative_lift: float
    projected_annual_impact: float
    ci_lower_annual: float
    ci_upper_annual: float
    sample_size_control: int
    sample_size_treatment: int


class MetricsCalculator:
    """Compute experiment metrics with proper statistical treatment.

    Supports conversion rates, latency quantiles, model quality metrics,
    revenue estimation, and CUPED variance reduction.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def conversion_rate(
        self,
        events: np.ndarray,
        variant_id: str,
        metric_name: str = "conversion_rate",
    ) -> MetricResult:
        """Compute conversion rate with Wilson score confidence interval.

        Args:
            events: Binary array (1 = conversion, 0 = no conversion).
            variant_id: Identifier for the variant.
            metric_name: Name for this metric.

        Returns:
            MetricResult with conversion rate and CI.
        """
        events = np.asarray(events, dtype=np.float64)
        n = len(events)
        successes = int(np.sum(events))
        rate = successes / n if n > 0 else 0.0

        # Wilson score interval
        from scipy import stats

        z = float(stats.norm.ppf(1 - self.alpha / 2))
        denominator = 1 + z**2 / n if n > 0 else 1.0
        centre = (rate + z**2 / (2 * n)) / denominator if n > 0 else 0.0
        import math

        margin = (
            z * math.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denominator if n > 0 else 0.0
        )

        return MetricResult(
            metric_name=metric_name,
            variant_id=variant_id,
            value=rate,
            ci_lower=max(0, centre - margin),
            ci_upper=min(1, centre + margin),
            sample_size=n,
            metadata={"successes": successes, "method": "wilson_score"},
        )

    def mean_latency(
        self,
        latencies_ms: np.ndarray,
        variant_id: str,
        metric_name: str = "mean_latency_ms",
    ) -> MetricResult:
        """Compute mean prediction latency with t-distribution CI.

        Args:
            latencies_ms: Array of latency measurements in milliseconds.
            variant_id: Variant identifier.
            metric_name: Metric name.

        Returns:
            MetricResult with mean latency and CI.
        """
        from scipy import stats

        latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
        n = len(latencies_ms)
        mean = float(np.mean(latencies_ms))
        se = float(stats.sem(latencies_ms)) if n > 1 else 0.0
        t_crit = float(stats.t.ppf(1 - self.alpha / 2, max(n - 1, 1)))

        return MetricResult(
            metric_name=metric_name,
            variant_id=variant_id,
            value=mean,
            ci_lower=mean - t_crit * se,
            ci_upper=mean + t_crit * se,
            sample_size=n,
            metadata={"std": float(np.std(latencies_ms, ddof=1)) if n > 1 else 0.0},
        )

    def quantile_metrics(
        self,
        latencies_ms: np.ndarray,
        variant_id: str,
    ) -> QuantileMetrics:
        """Compute quantile-based latency metrics (p50, p75, p90, p95, p99).

        Args:
            latencies_ms: Array of latency measurements.
            variant_id: Variant identifier (logged for tracing).

        Returns:
            QuantileMetrics with all standard quantiles.
        """
        latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
        n = len(latencies_ms)

        return QuantileMetrics(
            p50=float(np.percentile(latencies_ms, 50)),
            p75=float(np.percentile(latencies_ms, 75)),
            p90=float(np.percentile(latencies_ms, 90)),
            p95=float(np.percentile(latencies_ms, 95)),
            p99=float(np.percentile(latencies_ms, 99)),
            mean=float(np.mean(latencies_ms)),
            std=float(np.std(latencies_ms, ddof=1)) if n > 1 else 0.0,
            min_val=float(np.min(latencies_ms)),
            max_val=float(np.max(latencies_ms)),
            sample_size=n,
        )

    def model_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        variant_id: str,
    ) -> ModelQualityMetrics:
        """Compute classification model quality metrics.

        Computes accuracy, precision, recall, and F1 for binary classification.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            variant_id: Variant identifier (logged for tracing).

        Returns:
            ModelQualityMetrics with accuracy, precision, recall, F1.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        n = len(y_true)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(
            "model_quality_computed",
            variant_id=variant_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )

        return ModelQualityMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sample_size=n,
        )

    def revenue_impact(
        self,
        control_revenue: np.ndarray,
        treatment_revenue: np.ndarray,
        daily_users: int,
    ) -> RevenueImpact:
        """Estimate revenue impact of switching from control to treatment.

        Computes per-user revenue lift and projects annual impact based on
        the observed effect size and daily traffic.

        Args:
            control_revenue: Per-user revenue in control.
            treatment_revenue: Per-user revenue in treatment.
            daily_users: Expected daily user traffic for projection.

        Returns:
            RevenueImpact with absolute/relative lift and annual projection.
        """
        import math

        from scipy import stats

        control_revenue = np.asarray(control_revenue, dtype=np.float64)
        treatment_revenue = np.asarray(treatment_revenue, dtype=np.float64)

        mean_c = float(np.mean(control_revenue))
        mean_t = float(np.mean(treatment_revenue))
        n_c = len(control_revenue)
        n_t = len(treatment_revenue)

        absolute_lift = mean_t - mean_c
        relative_lift = absolute_lift / mean_c if mean_c != 0 else 0.0

        # CI for the difference using Welch's approximation
        var_c = float(np.var(control_revenue, ddof=1))
        var_t = float(np.var(treatment_revenue, ddof=1))
        se = math.sqrt(var_c / n_c + var_t / n_t)
        z_crit = float(stats.norm.ppf(1 - self.alpha / 2))

        annual_days = 365
        projected_annual = absolute_lift * daily_users * annual_days
        ci_lower = (absolute_lift - z_crit * se) * daily_users * annual_days
        ci_upper = (absolute_lift + z_crit * se) * daily_users * annual_days

        return RevenueImpact(
            mean_revenue_per_user_control=mean_c,
            mean_revenue_per_user_treatment=mean_t,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            projected_annual_impact=projected_annual,
            ci_lower_annual=ci_lower,
            ci_upper_annual=ci_upper,
            sample_size_control=n_c,
            sample_size_treatment=n_t,
        )

    def cuped_variance_reduction(
        self,
        metric_values: np.ndarray,
        covariate_values: np.ndarray,
        variant_assignments: np.ndarray,
    ) -> dict[str, MetricResult]:
        """CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

        Reduces variance by regressing out pre-experiment behavior (the covariate).
        The adjusted metric is:
            Y_adj = Y - theta * (X - E[X])
        where theta = Cov(Y, X) / Var(X).

        Args:
            metric_values: Post-experiment metric values (Y).
            covariate_values: Pre-experiment covariate values (X).
            variant_assignments: Binary array (0=control, 1=treatment).

        Returns:
            Dict mapping variant label to MetricResult with adjusted values.
        """
        from scipy import stats

        metric_values = np.asarray(metric_values, dtype=np.float64)
        covariate_values = np.asarray(covariate_values, dtype=np.float64)
        variant_assignments = np.asarray(variant_assignments, dtype=np.int64)

        # Compute CUPED adjustment coefficient theta
        cov_xy = float(np.cov(metric_values, covariate_values)[0, 1])
        var_x = float(np.var(covariate_values, ddof=1))
        theta = cov_xy / var_x if var_x > 0 else 0.0

        # Adjusted metric
        x_mean = float(np.mean(covariate_values))
        adjusted = metric_values - theta * (covariate_values - x_mean)

        # Compute variance reduction ratio
        original_var = float(np.var(metric_values, ddof=1))
        adjusted_var = float(np.var(adjusted, ddof=1))
        reduction_ratio = 1.0 - (adjusted_var / original_var) if original_var > 0 else 0.0

        logger.info(
            "cuped_applied",
            theta=theta,
            variance_reduction=f"{reduction_ratio:.2%}",
        )

        results: dict[str, MetricResult] = {}
        for variant_label, variant_code in [("control", 0), ("treatment", 1)]:
            mask = variant_assignments == variant_code
            adj_subset = adjusted[mask]
            n = len(adj_subset)
            mean_val = float(np.mean(adj_subset)) if n > 0 else 0.0
            se = float(stats.sem(adj_subset)) if n > 1 else 0.0
            t_crit = float(stats.t.ppf(1 - self.alpha / 2, max(n - 1, 1)))

            results[variant_label] = MetricResult(
                metric_name="cuped_adjusted",
                variant_id=variant_label,
                value=mean_val,
                ci_lower=mean_val - t_crit * se,
                ci_upper=mean_val + t_crit * se,
                sample_size=n,
                metadata={
                    "theta": theta,
                    "variance_reduction": reduction_ratio,
                    "original_variance": original_var,
                    "adjusted_variance": adjusted_var,
                },
            )

        return results

    def compute_all_metrics(
        self,
        experiment_data: pd.DataFrame,
        variant_column: str = "variant_id",
        conversion_column: str | None = "converted",
        latency_column: str | None = "latency_ms",
        revenue_column: str | None = None,
    ) -> dict[str, list[MetricResult]]:
        """Compute all applicable metrics for each variant from a DataFrame.

        Args:
            experiment_data: DataFrame with experiment event data.
            variant_column: Column identifying variant assignment.
            conversion_column: Column with binary conversion indicator.
            latency_column: Column with latency measurements.
            revenue_column: Column with per-event revenue.

        Returns:
            Dict mapping metric name to list of MetricResult per variant.
        """
        results: dict[str, list[MetricResult]] = {}
        variants = experiment_data[variant_column].unique()

        for variant_id in variants:
            variant_data = experiment_data[experiment_data[variant_column] == variant_id]

            if conversion_column and conversion_column in variant_data.columns:
                metric = self.conversion_rate(
                    variant_data[conversion_column].values,
                    str(variant_id),
                )
                results.setdefault("conversion_rate", []).append(metric)

            if latency_column and latency_column in variant_data.columns:
                metric = self.mean_latency(
                    variant_data[latency_column].values,
                    str(variant_id),
                )
                results.setdefault("mean_latency", []).append(metric)

            if revenue_column and revenue_column in variant_data.columns:
                rev_data = variant_data[revenue_column].values
                n = len(rev_data)
                mean_val = float(np.mean(rev_data))
                from scipy import stats as _stats

                se = float(_stats.sem(rev_data)) if n > 1 else 0.0
                t_crit = float(_stats.t.ppf(1 - self.alpha / 2, max(n - 1, 1)))
                metric = MetricResult(
                    metric_name="mean_revenue",
                    variant_id=str(variant_id),
                    value=mean_val,
                    ci_lower=mean_val - t_crit * se,
                    ci_upper=mean_val + t_crit * se,
                    sample_size=n,
                )
                results.setdefault("mean_revenue", []).append(metric)

        return results
