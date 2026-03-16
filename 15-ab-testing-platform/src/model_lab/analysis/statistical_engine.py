"""Statistical analysis engine for A/B test experiments.

Implements both frequentist and Bayesian approaches, sequential testing
to handle the peeking problem, multiple comparison corrections, power
analysis, and confidence interval computation.

All implementations follow standard statistical methodology with proper
handling of edge cases and numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


class TestType(str, Enum):
    """Type of statistical test to perform."""

    TTEST = "ttest"
    CHI_SQUARED = "chi_squared"
    Z_TEST_PROPORTIONS = "z_test_proportions"
    BAYESIAN_BETA_BINOMIAL = "bayesian_beta_binomial"
    BAYESIAN_NORMAL = "bayesian_normal"


@dataclass
class FrequentistResult:
    """Result of a frequentist statistical test."""

    test_type: TestType
    test_statistic: float
    p_value: float
    confidence_level: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    effect_size: float
    relative_effect: float
    control_mean: float
    treatment_mean: float
    control_n: int
    treatment_n: int
    power: float = 0.0
    adjusted_p_value: float | None = None


@dataclass
class BayesianResult:
    """Result of a Bayesian analysis."""

    test_type: TestType
    probability_b_better: float
    expected_loss_b: float
    expected_loss_a: float
    credible_interval: tuple[float, float]
    posterior_a: dict[str, float] = field(default_factory=dict)
    posterior_b: dict[str, float] = field(default_factory=dict)
    risk_threshold_met: bool = False
    hdi_95: tuple[float, float] = (0.0, 0.0)


@dataclass
class SequentialTestResult:
    """Result of a sequential (group sequential) test."""

    test_statistic: float
    upper_boundary: float
    lower_boundary: float
    decision: str  # "continue", "reject_null", "accept_null"
    information_fraction: float
    adjusted_alpha: float
    p_value: float


@dataclass
class PowerAnalysis:
    """Result of a statistical power analysis."""

    required_sample_size_per_group: int
    power: float
    alpha: float
    minimum_detectable_effect: float
    baseline_rate: float | None = None


class StatisticalEngine:
    """Core statistical analysis engine for experiment evaluation.

    Provides frequentist tests (t-test, chi-squared, z-test for proportions),
    Bayesian analysis (Beta-Binomial for conversion, Normal-Normal for continuous),
    sequential testing, multiple comparison corrections, and power analysis.
    """

    def __init__(self, default_alpha: float = 0.05, default_num_simulations: int = 100_000) -> None:
        self.default_alpha = default_alpha
        self.default_num_simulations = default_num_simulations

    # ----------------------------------------------------------------
    # Frequentist Tests
    # ----------------------------------------------------------------

    def two_sample_ttest(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float | None = None,
        equal_var: bool = False,
    ) -> FrequentistResult:
        """Welch's two-sample t-test for comparing continuous metrics.

        Tests H0: mu_treatment - mu_control = 0 against H1: mu_treatment != mu_control.
        Uses Welch's t-test by default (does not assume equal variances).

        Args:
            control: Observations from the control group.
            treatment: Observations from the treatment group.
            alpha: Significance level (default: instance default).
            equal_var: If True, use Student's t-test instead of Welch's.

        Returns:
            FrequentistResult with test statistics and confidence interval.
        """
        alpha = alpha or self.default_alpha
        control = np.asarray(control, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=np.float64)

        n_c, n_t = len(control), len(treatment)
        mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
        var_c, var_t = float(np.var(control, ddof=1)), float(np.var(treatment, ddof=1))

        t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=equal_var)
        t_stat = float(t_stat)
        p_value = float(p_value)

        # Confidence interval for the difference (treatment - control)
        diff = mean_t - mean_c
        se = math.sqrt(var_t / n_t + var_c / n_c)

        # Welch-Satterthwaite degrees of freedom
        if not equal_var:
            num = (var_t / n_t + var_c / n_c) ** 2
            denom = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
            df = num / denom if denom > 0 else n_c + n_t - 2
        else:
            df = n_c + n_t - 2

        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci = (diff - t_crit * se, diff + t_crit * se)

        relative_effect = diff / abs(mean_c) if mean_c != 0 else 0.0

        # Compute observed power
        noncentrality = abs(diff) / se if se > 0 else 0.0
        power = (
            1.0
            - float(stats.nct.cdf(t_crit, df, noncentrality))
            + float(stats.nct.cdf(-t_crit, df, noncentrality))
        )

        return FrequentistResult(
            test_type=TestType.TTEST,
            test_statistic=t_stat,
            p_value=p_value,
            confidence_level=1 - alpha,
            is_significant=p_value < alpha,
            confidence_interval=ci,
            effect_size=diff,
            relative_effect=relative_effect,
            control_mean=mean_c,
            treatment_mean=mean_t,
            control_n=n_c,
            treatment_n=n_t,
            power=power,
        )

    def chi_squared_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        alpha: float | None = None,
    ) -> FrequentistResult:
        """Chi-squared test for independence on 2x2 contingency table.

        Tests whether treatment and control have different success rates.

        Args:
            control_successes: Number of successes in control.
            control_total: Total observations in control.
            treatment_successes: Number of successes in treatment.
            treatment_total: Total observations in treatment.
            alpha: Significance level.

        Returns:
            FrequentistResult with chi-squared statistic and confidence interval.
        """
        alpha = alpha or self.default_alpha

        control_failures = control_total - control_successes
        treatment_failures = treatment_total - treatment_successes

        contingency = np.array(
            [
                [control_successes, control_failures],
                [treatment_successes, treatment_failures],
            ]
        )

        chi2, p_value, dof, _ = stats.chi2_contingency(contingency, correction=True)

        p_c = control_successes / control_total
        p_t = treatment_successes / treatment_total
        diff = p_t - p_c

        # Confidence interval for difference in proportions
        se = math.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        ci = (diff - z_crit * se, diff + z_crit * se)

        relative_effect = diff / p_c if p_c > 0 else 0.0

        return FrequentistResult(
            test_type=TestType.CHI_SQUARED,
            test_statistic=float(chi2),
            p_value=float(p_value),
            confidence_level=1 - alpha,
            is_significant=float(p_value) < alpha,
            confidence_interval=ci,
            effect_size=diff,
            relative_effect=relative_effect,
            control_mean=p_c,
            treatment_mean=p_t,
            control_n=control_total,
            treatment_n=treatment_total,
        )

    def z_test_proportions(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        alpha: float | None = None,
    ) -> FrequentistResult:
        """Two-proportion z-test.

        Tests H0: p_treatment = p_control against H1: p_treatment != p_control.
        Uses pooled proportion under H0 for the standard error.

        Args:
            control_successes: Successes in control.
            control_total: Total in control.
            treatment_successes: Successes in treatment.
            treatment_total: Total in treatment.
            alpha: Significance level.

        Returns:
            FrequentistResult with z-statistic and confidence interval.
        """
        alpha = alpha or self.default_alpha

        p_c = control_successes / control_total
        p_t = treatment_successes / treatment_total

        # Pooled proportion under H0
        p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)
        se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treatment_total))

        z_stat = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
        p_value = 2 * (1 - float(stats.norm.cdf(abs(z_stat))))

        # CI uses unpooled SE (more accurate for the CI)
        se_unpooled = math.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        diff = p_t - p_c
        ci = (diff - z_crit * se_unpooled, diff + z_crit * se_unpooled)

        relative_effect = diff / p_c if p_c > 0 else 0.0

        return FrequentistResult(
            test_type=TestType.Z_TEST_PROPORTIONS,
            test_statistic=z_stat,
            p_value=p_value,
            confidence_level=1 - alpha,
            is_significant=p_value < alpha,
            confidence_interval=ci,
            effect_size=diff,
            relative_effect=relative_effect,
            control_mean=p_c,
            treatment_mean=p_t,
            control_n=control_total,
            treatment_n=treatment_total,
        )

    # ----------------------------------------------------------------
    # Bayesian Tests
    # ----------------------------------------------------------------

    def bayesian_beta_binomial(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        num_simulations: int | None = None,
        risk_threshold: float = 0.01,
    ) -> BayesianResult:
        """Bayesian analysis using Beta-Binomial model for conversion rates.

        Posterior: Beta(alpha + successes, beta + failures)
        Uses Monte Carlo simulation to compute P(B > A) and expected loss.

        Args:
            control_successes: Conversions in control.
            control_total: Total in control.
            treatment_successes: Conversions in treatment.
            treatment_total: Total in treatment.
            prior_alpha: Beta prior alpha (default: 1 = uniform).
            prior_beta: Beta prior beta.
            num_simulations: Number of MC samples.
            risk_threshold: Maximum acceptable expected loss.

        Returns:
            BayesianResult with posterior summaries and risk assessment.
        """
        num_simulations = num_simulations or self.default_num_simulations

        # Posterior parameters
        alpha_a = prior_alpha + control_successes
        beta_a = prior_beta + (control_total - control_successes)
        alpha_b = prior_alpha + treatment_successes
        beta_b = prior_beta + (treatment_total - treatment_successes)

        # Monte Carlo samples from posteriors
        samples_a = np.random.beta(alpha_a, beta_a, size=num_simulations)
        samples_b = np.random.beta(alpha_b, beta_b, size=num_simulations)

        # P(B > A)
        prob_b_better = float(np.mean(samples_b > samples_a))

        # Expected loss: E[max(A - B, 0)] for choosing B, and vice versa
        diff = samples_b - samples_a
        expected_loss_b = float(np.mean(np.maximum(-diff, 0)))  # loss if we pick B but A is better
        expected_loss_a = float(np.mean(np.maximum(diff, 0)))  # loss if we pick A but B is better

        # 95% HDI of the difference
        diff_sorted = np.sort(diff)
        n = len(diff_sorted)
        ci_size = int(np.ceil(0.95 * n))
        best_start = 0
        best_width = diff_sorted[ci_size - 1] - diff_sorted[0]
        for start in range(n - ci_size):
            width = diff_sorted[start + ci_size - 1] - diff_sorted[start]
            if width < best_width:
                best_width = width
                best_start = start
        hdi_95 = (float(diff_sorted[best_start]), float(diff_sorted[best_start + ci_size - 1]))

        # 95% equal-tailed credible interval
        ci_lower = float(np.percentile(diff, 2.5))
        ci_upper = float(np.percentile(diff, 97.5))

        return BayesianResult(
            test_type=TestType.BAYESIAN_BETA_BINOMIAL,
            probability_b_better=prob_b_better,
            expected_loss_b=expected_loss_b,
            expected_loss_a=expected_loss_a,
            credible_interval=(ci_lower, ci_upper),
            posterior_a={"alpha": alpha_a, "beta": beta_a, "mean": alpha_a / (alpha_a + beta_a)},
            posterior_b={"alpha": alpha_b, "beta": beta_b, "mean": alpha_b / (alpha_b + beta_b)},
            risk_threshold_met=expected_loss_b < risk_threshold,
            hdi_95=hdi_95,
        )

    def bayesian_normal(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        prior_mean: float = 0.0,
        prior_variance: float = 1e6,
        num_simulations: int | None = None,
        risk_threshold: float = 0.01,
    ) -> BayesianResult:
        """Bayesian Normal-Normal conjugate model for continuous metrics.

        Assumes known variance (estimated from data) with Normal prior on mean.
        Posterior: N(posterior_mean, posterior_var) where:
            posterior_var = 1 / (1/prior_var + n/data_var)
            posterior_mean = posterior_var * (prior_mean/prior_var + n*data_mean/data_var)

        Args:
            control: Continuous observations from control.
            treatment: Continuous observations from treatment.
            prior_mean: Prior mean for the group means.
            prior_variance: Prior variance (default: very diffuse).
            num_simulations: Number of MC samples.
            risk_threshold: Maximum acceptable expected loss.

        Returns:
            BayesianResult with posterior summaries.
        """
        num_simulations = num_simulations or self.default_num_simulations
        control = np.asarray(control, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=np.float64)

        def _posterior_params(
            data: np.ndarray,
        ) -> tuple[float, float]:
            n = len(data)
            data_mean = float(np.mean(data))
            data_var = float(np.var(data, ddof=1))
            if data_var == 0:
                data_var = 1e-10

            posterior_var = 1.0 / (1.0 / prior_variance + n / data_var)
            posterior_mean = posterior_var * (
                prior_mean / prior_variance + n * data_mean / data_var
            )
            return posterior_mean, posterior_var

        mean_a, var_a = _posterior_params(control)
        mean_b, var_b = _posterior_params(treatment)

        samples_a = np.random.normal(mean_a, math.sqrt(var_a), size=num_simulations)
        samples_b = np.random.normal(mean_b, math.sqrt(var_b), size=num_simulations)

        prob_b_better = float(np.mean(samples_b > samples_a))

        diff = samples_b - samples_a
        expected_loss_b = float(np.mean(np.maximum(-diff, 0)))
        expected_loss_a = float(np.mean(np.maximum(diff, 0)))

        ci_lower = float(np.percentile(diff, 2.5))
        ci_upper = float(np.percentile(diff, 97.5))

        # HDI
        diff_sorted = np.sort(diff)
        n = len(diff_sorted)
        ci_size = int(np.ceil(0.95 * n))
        best_start = 0
        best_width = diff_sorted[ci_size - 1] - diff_sorted[0]
        for start in range(n - ci_size):
            width = diff_sorted[start + ci_size - 1] - diff_sorted[start]
            if width < best_width:
                best_width = width
                best_start = start
        hdi_95 = (float(diff_sorted[best_start]), float(diff_sorted[best_start + ci_size - 1]))

        return BayesianResult(
            test_type=TestType.BAYESIAN_NORMAL,
            probability_b_better=prob_b_better,
            expected_loss_b=expected_loss_b,
            expected_loss_a=expected_loss_a,
            credible_interval=(ci_lower, ci_upper),
            posterior_a={"mean": mean_a, "variance": var_a},
            posterior_b={"mean": mean_b, "variance": var_b},
            risk_threshold_met=expected_loss_b < risk_threshold,
            hdi_95=hdi_95,
        )

    # ----------------------------------------------------------------
    # Sequential Testing
    # ----------------------------------------------------------------

    def sequential_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        max_sample_size: int,
        num_analyses: int = 5,
        alpha: float | None = None,
        spending_function: str = "obrien_fleming",
    ) -> SequentialTestResult:
        """Group sequential test using alpha spending functions.

        Addresses the peeking problem by controlling the overall Type I error
        rate across multiple interim analyses using O'Brien-Fleming or Pocock
        spending functions.

        Args:
            control: Current control observations.
            treatment: Current treatment observations.
            max_sample_size: Planned maximum total sample size.
            num_analyses: Total number of planned analyses (including final).
            alpha: Overall significance level.
            spending_function: "obrien_fleming" or "pocock".

        Returns:
            SequentialTestResult with boundaries and decision.
        """
        alpha = alpha or self.default_alpha
        control = np.asarray(control, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=np.float64)

        current_n = len(control) + len(treatment)
        info_fraction = min(current_n / max_sample_size, 1.0)

        # Compute alpha spent at this information fraction
        spent_alpha = self._alpha_spending(info_fraction, alpha, spending_function)

        # Two-sample z-statistic
        n_c, n_t = len(control), len(treatment)
        mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
        var_c = float(np.var(control, ddof=1)) if n_c > 1 else 1.0
        var_t = float(np.var(treatment, ddof=1)) if n_t > 1 else 1.0
        se = math.sqrt(var_c / n_c + var_t / n_t) if (n_c > 0 and n_t > 0) else 1.0

        z_stat = (mean_t - mean_c) / se if se > 0 else 0.0
        p_value = 2 * (1 - float(stats.norm.cdf(abs(z_stat))))

        # Boundaries
        z_boundary = float(stats.norm.ppf(1 - spent_alpha / 2)) if spent_alpha > 0 else float("inf")

        if abs(z_stat) >= z_boundary:
            decision = "reject_null"
        elif info_fraction >= 1.0:
            decision = "accept_null" if abs(z_stat) < z_boundary else "reject_null"
        else:
            decision = "continue"

        return SequentialTestResult(
            test_statistic=z_stat,
            upper_boundary=z_boundary,
            lower_boundary=-z_boundary,
            decision=decision,
            information_fraction=info_fraction,
            adjusted_alpha=spent_alpha,
            p_value=p_value,
        )

    def _alpha_spending(
        self,
        info_fraction: float,
        alpha: float,
        function: str,
    ) -> float:
        """Compute cumulative alpha spent at a given information fraction.

        O'Brien-Fleming: alpha_spent(t) = 2 - 2*Phi(z_{alpha/2} / sqrt(t))
        Pocock: alpha_spent(t) = alpha * ln(1 + (e-1)*t)
        """
        t = max(info_fraction, 1e-10)

        if function == "obrien_fleming":
            z_alpha = float(stats.norm.ppf(1 - alpha / 2))
            spent = 2.0 * (1.0 - float(stats.norm.cdf(z_alpha / math.sqrt(t))))
        elif function == "pocock":
            spent = alpha * math.log(1 + (math.e - 1) * t)
        else:
            msg = f"Unknown spending function: {function}"
            raise ValueError(msg)

        return min(spent, alpha)

    # ----------------------------------------------------------------
    # Multiple Comparison Corrections
    # ----------------------------------------------------------------

    def bonferroni_correction(
        self, p_values: list[float], alpha: float | None = None
    ) -> list[dict[str, Any]]:
        """Bonferroni correction for multiple comparisons.

        Adjusts alpha by dividing by the number of tests. Most conservative.

        Args:
            p_values: List of raw p-values from multiple tests.
            alpha: Family-wise error rate to control.

        Returns:
            List of dicts with original p-value, adjusted p-value, and significance.
        """
        alpha = alpha or self.default_alpha
        m = len(p_values)

        results = []
        for p in p_values:
            adjusted = min(p * m, 1.0)
            results.append(
                {
                    "original_p_value": p,
                    "adjusted_p_value": adjusted,
                    "is_significant": adjusted < alpha,
                    "method": "bonferroni",
                }
            )
        return results

    def fdr_correction(
        self,
        p_values: list[float],
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Benjamini-Hochberg False Discovery Rate correction.

        Controls the expected proportion of false discoveries. Less conservative
        than Bonferroni.

        Args:
            p_values: List of raw p-values.
            alpha: FDR level to control.

        Returns:
            List of dicts with original p-value, adjusted p-value, and significance.
        """
        alpha = alpha or self.default_alpha
        m = len(p_values)

        # Sort p-values and track original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        # Compute BH adjusted p-values
        adjusted = [0.0] * m
        prev_adj = 0.0
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            bh_val = p * m / rank
            # Enforce monotonicity: adjusted p-values should be non-decreasing
            adjusted[orig_idx] = bh_val

        # Step-up procedure: working backwards to enforce monotonicity
        sorted_indices = [idx for idx, _ in indexed]
        for i in range(m - 2, -1, -1):
            idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])

        # Cap at 1.0
        adjusted = [min(a, 1.0) for a in adjusted]

        results = []
        for i, p in enumerate(p_values):
            results.append(
                {
                    "original_p_value": p,
                    "adjusted_p_value": adjusted[i],
                    "is_significant": adjusted[i] < alpha,
                    "method": "benjamini_hochberg",
                }
            )
        return results

    # ----------------------------------------------------------------
    # Power Analysis
    # ----------------------------------------------------------------

    def power_analysis_proportions(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float | None = None,
        power: float = 0.8,
    ) -> PowerAnalysis:
        """Sample size calculation for a two-proportion z-test.

        Uses the formula:
            n = (z_{alpha/2} * sqrt(2*p_bar*q_bar) + z_{beta} * sqrt(p1*q1 + p2*q2))^2 / delta^2

        where p_bar is the average proportion under H0.

        Args:
            baseline_rate: Expected conversion rate of control (p1).
            minimum_detectable_effect: Absolute difference to detect (delta = p2 - p1).
            alpha: Significance level.
            power: Desired statistical power (1 - beta).

        Returns:
            PowerAnalysis with required sample size per group.
        """
        alpha = alpha or self.default_alpha

        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        p_bar = (p1 + p2) / 2.0
        q_bar = 1.0 - p_bar

        z_alpha = float(stats.norm.ppf(1 - alpha / 2))
        z_beta = float(stats.norm.ppf(power))

        numerator = (
            z_alpha * math.sqrt(2 * p_bar * q_bar)
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = minimum_detectable_effect**2

        n = math.ceil(numerator / denominator)

        return PowerAnalysis(
            required_sample_size_per_group=n,
            power=power,
            alpha=alpha,
            minimum_detectable_effect=minimum_detectable_effect,
            baseline_rate=baseline_rate,
        )

    def power_analysis_continuous(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        alpha: float | None = None,
        power: float = 0.8,
    ) -> PowerAnalysis:
        """Sample size calculation for a two-sample t-test.

        Uses the formula:
            n = 2 * (z_{alpha/2} + z_{beta})^2 * sigma^2 / delta^2

        Args:
            baseline_mean: Expected mean of control group.
            baseline_std: Expected standard deviation (assumed equal for both groups).
            minimum_detectable_effect: Absolute difference in means to detect.
            alpha: Significance level.
            power: Desired statistical power.

        Returns:
            PowerAnalysis with required sample size per group.
        """
        alpha = alpha or self.default_alpha

        z_alpha = float(stats.norm.ppf(1 - alpha / 2))
        z_beta = float(stats.norm.ppf(power))

        n = math.ceil(2 * ((z_alpha + z_beta) * baseline_std / minimum_detectable_effect) ** 2)

        return PowerAnalysis(
            required_sample_size_per_group=n,
            power=power,
            alpha=alpha,
            minimum_detectable_effect=minimum_detectable_effect,
            baseline_rate=None,
        )

    # ----------------------------------------------------------------
    # Confidence Intervals
    # ----------------------------------------------------------------

    def confidence_interval_proportion(
        self,
        successes: int,
        total: int,
        alpha: float | None = None,
        method: str = "wilson",
    ) -> tuple[float, float]:
        """Compute confidence interval for a proportion.

        Supports Wilson score interval (default) and Clopper-Pearson exact interval.

        Args:
            successes: Number of successes.
            total: Total trials.
            alpha: Significance level.
            method: "wilson" or "clopper_pearson".

        Returns:
            Tuple of (lower, upper) bounds.
        """
        alpha = alpha or self.default_alpha
        p_hat = successes / total if total > 0 else 0.0
        z = float(stats.norm.ppf(1 - alpha / 2))

        if method == "wilson":
            denominator = 1 + z**2 / total
            centre = (p_hat + z**2 / (2 * total)) / denominator
            margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
            return (max(0, centre - margin), min(1, centre + margin))
        elif method == "clopper_pearson":
            lower = (
                float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
                if successes > 0
                else 0.0
            )
            upper = (
                float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
                if successes < total
                else 1.0
            )
            return (lower, upper)
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

    def confidence_interval_mean(
        self,
        data: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float]:
        """Compute confidence interval for a mean using the t-distribution.

        Args:
            data: Observations.
            alpha: Significance level.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        alpha = alpha or self.default_alpha
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        mean = float(np.mean(data))
        se = float(stats.sem(data))
        t_crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
        return (mean - t_crit * se, mean + t_crit * se)
