"""Statistical analysis utilities for clinical trial data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger()


def chi_squared_test(
    df: pd.DataFrame, col: str, group_col: str = "treatment_arm"
) -> dict[str, Any]:
    """Run chi-squared test for categorical variable across treatment arms."""
    contingency = pd.crosstab(df[col], df[group_col])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "test": "chi-squared",
        "variable": col,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "significant": p_value < 0.05,
    }


def mann_whitney_test(
    df: pd.DataFrame, col: str, group_col: str = "treatment_arm"
) -> dict[str, Any]:
    """Run Mann-Whitney U test for continuous variable across arms."""
    groups = df[group_col].unique()
    g1 = df.loc[df[group_col] == groups[0], col].dropna()
    g2 = df.loc[df[group_col] == groups[1], col].dropna()
    u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "test": "Mann-Whitney U",
        "variable": col,
        "U_statistic": round(u_stat, 2),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "group_medians": {
            str(groups[0]): round(g1.median(), 3),
            str(groups[1]): round(g2.median(), 3),
        },
    }


def response_rate_comparison(
    df: pd.DataFrame, group_col: str = "treatment_arm"
) -> dict[str, Any]:
    """Compare responder rates between treatment arms."""
    rates = df.groupby(group_col)["responder"].mean()
    contingency = pd.crosstab(df[group_col], df["responder"])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    groups = rates.index.tolist()
    rate_diff = rates.iloc[0] - rates.iloc[1]

    # Odds ratio
    a = contingency.iloc[0, 1]  # treatment responders
    b = contingency.iloc[0, 0]  # treatment non-responders
    c = contingency.iloc[1, 1]  # placebo responders
    d = contingency.iloc[1, 0]  # placebo non-responders
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.inf

    return {
        "response_rates": {str(g): round(r, 4) for g, r in rates.items()},
        "rate_difference": round(rate_diff, 4),
        "odds_ratio": round(odds_ratio, 3),
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def subgroup_analysis(
    df: pd.DataFrame,
    subgroup_col: str,
    outcome_col: str = "responder",
) -> pd.DataFrame:
    """Compute treatment effect within subgroups."""
    results = []
    for name, group in df.groupby(subgroup_col):
        for arm, arm_group in group.groupby("treatment_arm"):
            results.append(
                {
                    "subgroup": name,
                    "treatment_arm": arm,
                    "n": len(arm_group),
                    "response_rate": round(arm_group[outcome_col].mean(), 4),
                    "mean_score": round(arm_group["response_score"].mean(), 4),
                }
            )
    return pd.DataFrame(results)


def baseline_balance_table(
    df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
    group_col: str = "treatment_arm",
) -> pd.DataFrame:
    """Generate Table 1 — baseline characteristics by treatment arm."""
    rows = []
    groups = sorted(df[group_col].unique())

    for col in continuous_cols:
        row: dict[str, Any] = {"variable": col, "type": "continuous"}
        for g in groups:
            subset = df.loc[df[group_col] == g, col].dropna()
            row[f"{g}_mean_sd"] = f"{subset.mean():.2f} ({subset.std():.2f})"
        result = mann_whitney_test(df, col, group_col)
        row["p_value"] = result["p_value"]
        rows.append(row)

    for col in categorical_cols:
        row = {"variable": col, "type": "categorical"}
        for g in groups:
            subset = df.loc[df[group_col] == g, col]
            mode_val = subset.mode().iloc[0] if len(subset) > 0 else "N/A"
            row[f"{g}_mean_sd"] = f"{mode_val} (n={len(subset)})"
        result = chi_squared_test(df, col, group_col)
        row["p_value"] = result["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)
