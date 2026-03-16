"""Tests for the statistical analysis engine.

Validates correctness of frequentist tests, Bayesian analysis,
sequential testing, multiple comparison corrections, and power analysis
against known analytical results and simulated distributions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from model_lab.analysis.statistical_engine import StatisticalEngine, TestType


class TestTwoSampleTTest:
    """Tests for Welch's two-sample t-test."""

    def test_identical_distributions_not_significant(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Two samples from the same distribution should not be significant."""
        control = rng.normal(100, 10, size=1000)
        treatment = rng.normal(100, 10, size=1000)

        result = statistical_engine.two_sample_ttest(control, treatment)

        assert result.test_type == TestType.TTEST
        assert not result.is_significant
        assert result.p_value > 0.05
        # CI for difference should contain 0
        assert result.confidence_interval[0] <= 0 <= result.confidence_interval[1]

    def test_different_means_significant(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Clearly different means should be detected."""
        control = rng.normal(100, 10, size=500)
        treatment = rng.normal(105, 10, size=500)  # 5% higher

        result = statistical_engine.two_sample_ttest(control, treatment)

        assert result.is_significant
        assert result.p_value < 0.05
        assert result.effect_size > 0  # treatment > control
        assert result.confidence_interval[0] > 0  # entire CI above 0

    def test_agrees_with_scipy(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Results should match scipy.stats.ttest_ind."""
        control = rng.normal(50, 15, size=200)
        treatment = rng.normal(55, 12, size=250)

        result = statistical_engine.two_sample_ttest(control, treatment)
        scipy_stat, scipy_p = stats.ttest_ind(control, treatment, equal_var=False)

        assert abs(result.test_statistic - scipy_stat) < 1e-10
        assert abs(result.p_value - scipy_p) < 1e-10

    def test_confidence_interval_coverage(self, statistical_engine: StatisticalEngine) -> None:
        """95% CI should cover the true difference ~95% of the time."""
        true_diff = 3.0
        covered = 0
        n_simulations = 500
        rng = np.random.default_rng(123)

        for _ in range(n_simulations):
            c = rng.normal(100, 10, size=100)
            t = rng.normal(100 + true_diff, 10, size=100)
            result = statistical_engine.two_sample_ttest(c, t)
            if result.confidence_interval[0] <= true_diff <= result.confidence_interval[1]:
                covered += 1

        coverage = covered / n_simulations
        # Should be approximately 0.95, allow some tolerance
        assert 0.90 <= coverage <= 0.99, f"Coverage was {coverage}, expected ~0.95"


class TestChiSquaredTest:
    """Tests for chi-squared test of independence."""

    def test_equal_proportions_not_significant(self, statistical_engine: StatisticalEngine) -> None:
        """Equal conversion rates should not be significant."""
        result = statistical_engine.chi_squared_test(
            control_successes=100,
            control_total=1000,
            treatment_successes=105,
            treatment_total=1000,
        )

        assert result.test_type == TestType.CHI_SQUARED
        assert not result.is_significant

    def test_large_difference_significant(self, statistical_engine: StatisticalEngine) -> None:
        """Large difference in proportions should be significant."""
        result = statistical_engine.chi_squared_test(
            control_successes=100,
            control_total=1000,
            treatment_successes=150,
            treatment_total=1000,
        )

        assert result.is_significant
        assert result.p_value < 0.05
        assert result.effect_size == pytest.approx(0.05, abs=0.001)
        assert result.relative_effect == pytest.approx(0.50, abs=0.01)


class TestZTestProportions:
    """Tests for two-proportion z-test."""

    def test_known_result(self, statistical_engine: StatisticalEngine) -> None:
        """Test against a known analytical result."""
        # 10% vs 12% with large samples
        result = statistical_engine.z_test_proportions(
            control_successes=1000,
            control_total=10000,
            treatment_successes=1200,
            treatment_total=10000,
        )

        assert result.test_type == TestType.Z_TEST_PROPORTIONS
        # This should be highly significant
        assert result.is_significant
        assert result.effect_size == pytest.approx(0.02, abs=0.001)
        assert result.control_mean == pytest.approx(0.10, abs=0.001)
        assert result.treatment_mean == pytest.approx(0.12, abs=0.001)

    def test_small_difference_large_sample(self, statistical_engine: StatisticalEngine) -> None:
        """Small but real difference should be detectable with large sample."""
        result = statistical_engine.z_test_proportions(
            control_successes=5000,
            control_total=50000,
            treatment_successes=5250,
            treatment_total=50000,
        )

        # 10% vs 10.5% with 50k samples should be significant
        assert result.is_significant
        assert result.confidence_interval[0] > 0

    def test_no_difference(self, statistical_engine: StatisticalEngine) -> None:
        """Identical proportions should not be significant."""
        result = statistical_engine.z_test_proportions(
            control_successes=100,
            control_total=1000,
            treatment_successes=100,
            treatment_total=1000,
        )

        assert not result.is_significant
        assert result.effect_size == pytest.approx(0.0)


class TestBayesianBetaBinomial:
    """Tests for Bayesian Beta-Binomial analysis."""

    def test_clear_winner(self, statistical_engine: StatisticalEngine) -> None:
        """Treatment clearly better should have high P(B>A)."""
        result = statistical_engine.bayesian_beta_binomial(
            control_successes=100,
            control_total=1000,
            treatment_successes=150,
            treatment_total=1000,
            num_simulations=50000,
        )

        assert result.test_type == TestType.BAYESIAN_BETA_BINOMIAL
        assert result.probability_b_better > 0.95
        assert result.expected_loss_b < result.expected_loss_a
        assert result.credible_interval[0] > 0  # Entire CI positive

    def test_equal_arms(self, statistical_engine: StatisticalEngine) -> None:
        """Equal arms should have ~50% probability."""
        result = statistical_engine.bayesian_beta_binomial(
            control_successes=100,
            control_total=1000,
            treatment_successes=100,
            treatment_total=1000,
            num_simulations=100000,
        )

        assert 0.40 <= result.probability_b_better <= 0.60
        # Credible interval should contain 0
        assert result.credible_interval[0] < 0 < result.credible_interval[1]

    def test_posterior_parameters(self, statistical_engine: StatisticalEngine) -> None:
        """Posterior parameters should follow Beta(alpha+s, beta+f)."""
        result = statistical_engine.bayesian_beta_binomial(
            control_successes=50,
            control_total=200,
            treatment_successes=60,
            treatment_total=200,
            prior_alpha=1.0,
            prior_beta=1.0,
        )

        # Control: Beta(1+50, 1+150) = Beta(51, 151)
        assert result.posterior_a["alpha"] == pytest.approx(51.0)
        assert result.posterior_a["beta"] == pytest.approx(151.0)
        assert result.posterior_a["mean"] == pytest.approx(51.0 / 202.0, abs=0.001)

        # Treatment: Beta(1+60, 1+140) = Beta(61, 141)
        assert result.posterior_b["alpha"] == pytest.approx(61.0)
        assert result.posterior_b["beta"] == pytest.approx(141.0)

    def test_risk_threshold(self, statistical_engine: StatisticalEngine) -> None:
        """Risk threshold should be met when expected loss is small."""
        result = statistical_engine.bayesian_beta_binomial(
            control_successes=100,
            control_total=10000,
            treatment_successes=200,
            treatment_total=10000,
            risk_threshold=0.01,
            num_simulations=50000,
        )

        # Clear winner, so risk of choosing treatment should be small
        assert result.risk_threshold_met


class TestBayesianNormal:
    """Tests for Bayesian Normal-Normal analysis."""

    def test_different_means(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Clearly different means should be detected."""
        control = rng.normal(100, 10, size=500)
        treatment = rng.normal(110, 10, size=500)

        result = statistical_engine.bayesian_normal(control, treatment, num_simulations=50000)

        assert result.test_type == TestType.BAYESIAN_NORMAL
        assert result.probability_b_better > 0.99
        assert result.credible_interval[0] > 0

    def test_same_distribution(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Same distribution should give ~50% probability."""
        control = rng.normal(100, 10, size=500)
        treatment = rng.normal(100, 10, size=500)

        result = statistical_engine.bayesian_normal(control, treatment, num_simulations=50000)

        assert 0.30 <= result.probability_b_better <= 0.70


class TestSequentialTesting:
    """Tests for group sequential testing."""

    def test_continue_with_insufficient_data(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Should recommend continuing with small sample."""
        control = rng.normal(100, 10, size=50)
        treatment = rng.normal(100, 10, size=50)

        result = statistical_engine.sequential_test(control, treatment, max_sample_size=10000)

        assert result.decision == "continue"
        assert result.information_fraction < 0.05

    def test_reject_with_clear_signal(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """Should reject null with very strong signal at sufficient data."""
        control = rng.normal(100, 5, size=2500)
        treatment = rng.normal(110, 5, size=2500)

        result = statistical_engine.sequential_test(control, treatment, max_sample_size=5000)

        assert result.decision == "reject_null"
        assert abs(result.test_statistic) >= result.upper_boundary

    def test_obrien_fleming_more_conservative_early(
        self, statistical_engine: StatisticalEngine, rng: np.random.Generator
    ) -> None:
        """O'Brien-Fleming should have stricter boundaries early on."""
        control = rng.normal(100, 10, size=100)
        treatment = rng.normal(102, 10, size=100)

        obf_result = statistical_engine.sequential_test(
            control,
            treatment,
            max_sample_size=10000,
            spending_function="obrien_fleming",
        )
        pocock_result = statistical_engine.sequential_test(
            control,
            treatment,
            max_sample_size=10000,
            spending_function="pocock",
        )

        # O'Brien-Fleming boundary should be higher (more conservative) early
        assert obf_result.upper_boundary >= pocock_result.upper_boundary


class TestMultipleComparisonCorrections:
    """Tests for Bonferroni and FDR corrections."""

    def test_bonferroni_inflates_p_values(self, statistical_engine: StatisticalEngine) -> None:
        """Bonferroni should multiply p-values by number of tests."""
        p_values = [0.01, 0.03, 0.05]
        results = statistical_engine.bonferroni_correction(p_values)

        assert results[0]["adjusted_p_value"] == pytest.approx(0.03)
        assert results[1]["adjusted_p_value"] == pytest.approx(0.09)
        assert results[2]["adjusted_p_value"] == pytest.approx(0.15)

    def test_bonferroni_caps_at_one(self, statistical_engine: StatisticalEngine) -> None:
        """Adjusted p-values should never exceed 1.0."""
        p_values = [0.5, 0.8]
        results = statistical_engine.bonferroni_correction(p_values)

        assert results[0]["adjusted_p_value"] == 1.0
        assert results[1]["adjusted_p_value"] == 1.0

    def test_fdr_less_conservative(self, statistical_engine: StatisticalEngine) -> None:
        """FDR correction should be less conservative than Bonferroni."""
        p_values = [0.001, 0.01, 0.02, 0.04, 0.05]

        bonf = statistical_engine.bonferroni_correction(p_values)
        fdr = statistical_engine.fdr_correction(p_values)

        bonf_significant = sum(1 for r in bonf if r["is_significant"])
        fdr_significant = sum(1 for r in fdr if r["is_significant"])

        # FDR should find at least as many significant results
        assert fdr_significant >= bonf_significant

    def test_fdr_monotonicity(self, statistical_engine: StatisticalEngine) -> None:
        """FDR-adjusted p-values should maintain the original ordering."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05]
        results = statistical_engine.fdr_correction(p_values)

        adjusted = [r["adjusted_p_value"] for r in results]
        # Adjusted values should maintain monotonicity with original order
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1] + 1e-10


class TestPowerAnalysis:
    """Tests for sample size calculations."""

    def test_proportions_known_result(self, statistical_engine: StatisticalEngine) -> None:
        """Validate sample size against known formula results."""
        result = statistical_engine.power_analysis_proportions(
            baseline_rate=0.10,
            minimum_detectable_effect=0.02,
            alpha=0.05,
            power=0.80,
        )

        # For 10% -> 12% conversion, ~3,600-3,800 per group
        assert 3000 <= result.required_sample_size_per_group <= 4500

    def test_smaller_effect_needs_more_samples(self, statistical_engine: StatisticalEngine) -> None:
        """Smaller MDE should require larger sample size."""
        large_effect = statistical_engine.power_analysis_proportions(
            baseline_rate=0.10, minimum_detectable_effect=0.05
        )
        small_effect = statistical_engine.power_analysis_proportions(
            baseline_rate=0.10, minimum_detectable_effect=0.01
        )

        assert (
            small_effect.required_sample_size_per_group
            > large_effect.required_sample_size_per_group
        )

    def test_higher_power_needs_more_samples(self, statistical_engine: StatisticalEngine) -> None:
        """Higher power should require larger sample size."""
        low_power = statistical_engine.power_analysis_proportions(
            baseline_rate=0.10, minimum_detectable_effect=0.02, power=0.7
        )
        high_power = statistical_engine.power_analysis_proportions(
            baseline_rate=0.10, minimum_detectable_effect=0.02, power=0.95
        )

        assert high_power.required_sample_size_per_group > low_power.required_sample_size_per_group

    def test_continuous_power_analysis(self, statistical_engine: StatisticalEngine) -> None:
        """Test continuous metric power analysis."""
        result = statistical_engine.power_analysis_continuous(
            baseline_mean=100.0,
            baseline_std=20.0,
            minimum_detectable_effect=5.0,
            alpha=0.05,
            power=0.80,
        )

        # For delta=5, sigma=20: n ~= 2*(1.96+0.84)^2*400/25 = 2*7.84*16 ~= 252
        assert 200 <= result.required_sample_size_per_group <= 350


class TestConfidenceIntervals:
    """Tests for confidence interval computations."""

    def test_wilson_interval_covers_true_value(self, statistical_engine: StatisticalEngine) -> None:
        """Wilson interval should cover the true proportion ~95% of the time."""
        true_p = 0.15
        n_trials = 500
        n_samples = 200
        covered = 0
        rng = np.random.default_rng(42)

        for _ in range(n_trials):
            successes = int(rng.binomial(n_samples, true_p))
            ci = statistical_engine.confidence_interval_proportion(
                successes, n_samples, method="wilson"
            )
            if ci[0] <= true_p <= ci[1]:
                covered += 1

        coverage = covered / n_trials
        assert 0.90 <= coverage <= 0.99

    def test_clopper_pearson_conservative(self, statistical_engine: StatisticalEngine) -> None:
        """Clopper-Pearson should be at least as wide as Wilson."""
        ci_wilson = statistical_engine.confidence_interval_proportion(50, 500, method="wilson")
        ci_cp = statistical_engine.confidence_interval_proportion(50, 500, method="clopper_pearson")

        wilson_width = ci_wilson[1] - ci_wilson[0]
        cp_width = ci_cp[1] - ci_cp[0]
        assert cp_width >= wilson_width - 0.001  # CP is generally wider

    def test_mean_ci(self, statistical_engine: StatisticalEngine, rng: np.random.Generator) -> None:
        """Mean CI should contain the true mean ~95% of the time."""
        true_mean = 50.0
        covered = 0
        n_sims = 500

        for _ in range(n_sims):
            data = rng.normal(true_mean, 10, size=50)
            ci = statistical_engine.confidence_interval_mean(data)
            if ci[0] <= true_mean <= ci[1]:
                covered += 1

        coverage = covered / n_sims
        assert 0.90 <= coverage <= 0.99
