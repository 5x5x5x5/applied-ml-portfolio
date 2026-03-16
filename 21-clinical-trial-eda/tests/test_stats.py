"""Tests for statistical analysis utilities."""

import pandas as pd
import pytest

from clinical_eda.data_generator import generate_trial_dataset
from clinical_eda.stats import (
    baseline_balance_table,
    chi_squared_test,
    mann_whitney_test,
    response_rate_comparison,
    subgroup_analysis,
)


@pytest.fixture
def trial_df() -> pd.DataFrame:
    return generate_trial_dataset(n_patients=400, seed=42)


class TestChiSquaredTest:
    def test_returns_dict(self, trial_df: pd.DataFrame) -> None:
        result = chi_squared_test(trial_df, "sex")
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "chi2" in result

    def test_sex_balance(self, trial_df: pd.DataFrame) -> None:
        result = chi_squared_test(trial_df, "sex")
        # Randomized trial should have balanced sex
        assert result["p_value"] > 0.01


class TestMannWhitneyTest:
    def test_returns_dict(self, trial_df: pd.DataFrame) -> None:
        result = mann_whitney_test(trial_df, "age")
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "group_medians" in result


class TestResponseRateComparison:
    def test_returns_rates(self, trial_df: pd.DataFrame) -> None:
        result = response_rate_comparison(trial_df)
        assert "response_rates" in result
        assert "odds_ratio" in result
        assert len(result["response_rates"]) == 2


class TestSubgroupAnalysis:
    def test_subgroup_results(self, trial_df: pd.DataFrame) -> None:
        result = subgroup_analysis(trial_df, "sex")
        assert isinstance(result, pd.DataFrame)
        assert "subgroup" in result.columns
        assert len(result) >= 4  # 2 sexes x 2 arms


class TestBaselineBalanceTable:
    def test_table_shape(self, trial_df: pd.DataFrame) -> None:
        result = baseline_balance_table(
            trial_df,
            continuous_cols=["age", "bmi"],
            categorical_cols=["sex"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 2 continuous + 1 categorical
