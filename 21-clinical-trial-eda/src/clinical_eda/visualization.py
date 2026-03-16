"""Visualization utilities for clinical trial EDA.

All functions return matplotlib Figure objects for notebook rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd

# Pharma-friendly palette
COLORS = {
    "RX-7281": "#2196F3",
    "Placebo": "#9E9E9E",
    "highlight": "#E91E63",
    "grid": "#E0E0E0",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_demographics_grid(df: pd.DataFrame) -> plt.Figure:
    """Create a 2x2 grid of demographic distributions by arm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Patient Demographics by Treatment Arm", fontsize=16, y=1.02)

    # Age distribution
    for arm, color in COLORS.items():
        if arm in ("highlight", "grid"):
            continue
        subset = df[df["treatment_arm"] == arm]
        axes[0, 0].hist(subset["age"], bins=25, alpha=0.6, label=arm, color=color)
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].set_title("Age Distribution")

    # BMI distribution
    for arm, color in COLORS.items():
        if arm in ("highlight", "grid"):
            continue
        subset = df[df["treatment_arm"] == arm]
        axes[0, 1].hist(subset["bmi"], bins=25, alpha=0.6, label=arm, color=color)
    axes[0, 1].set_xlabel("BMI (kg/m2)")
    axes[0, 1].set_title("BMI Distribution")

    # Sex balance
    sex_counts = df.groupby(["treatment_arm", "sex"]).size().unstack(fill_value=0)
    sex_counts.plot(kind="bar", ax=axes[1, 0], color=["#E91E63", "#2196F3"])
    axes[1, 0].set_title("Sex by Treatment Arm")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

    # Race distribution
    race_counts = df.groupby(["treatment_arm", "race"]).size().unstack(fill_value=0)
    race_counts.plot(kind="bar", ax=axes[1, 1], colormap="Set2")
    axes[1, 1].set_title("Race by Treatment Arm")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_biomarker_distributions(
    df: pd.DataFrame, biomarker_cols: list[str]
) -> plt.Figure:
    """Violin plots of biomarker distributions by treatment arm."""
    n_markers = len(biomarker_cols)
    n_cols = 4
    n_rows = (n_markers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten()

    palette = {k: v for k, v in COLORS.items() if k not in ("highlight", "grid")}

    for i, col in enumerate(biomarker_cols):
        sns.violinplot(
            data=df,
            x="treatment_arm",
            y=col,
            ax=axes_flat[i],
            palette=palette,
            inner="box",
            cut=0,
        )
        axes_flat[i].set_title(col.replace("_", " ").title(), fontsize=11)
        axes_flat[i].set_xlabel("")

    # Hide unused axes
    for j in range(n_markers, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Baseline Biomarker Distributions by Treatment Arm", fontsize=14)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    """Correlation heatmap for selected numeric columns."""
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Biomarker Correlation Matrix", fontsize=14)
    fig.tight_layout()
    return fig


def plot_response_rates(df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing response rates between arms."""
    rates = df.groupby("treatment_arm")["responder"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        rates.index,
        rates.values,
        color=[COLORS.get(arm, "#666") for arm in rates.index],
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_ylabel("Response Rate (%)")
    ax.set_title("Overall Response Rate (ACR20-like Endpoint)", fontsize=14)
    ax.set_ylim(0, max(rates.values) + 15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_missing_data(df: pd.DataFrame) -> plt.Figure:
    """Visualize missing data pattern across the dataset."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            "No missing data detected",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        return fig

    fig, ax = plt.subplots(figsize=(10, max(3, len(missing_pct) * 0.5)))
    bars = ax.barh(missing_pct.index, missing_pct.values, color=COLORS["highlight"])
    for bar, val in zip(bars, missing_pct.values):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Data by Variable", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_subgroup_forest(subgroup_df: pd.DataFrame) -> plt.Figure:
    """Forest plot of treatment effect across subgroups."""
    pivoted = subgroup_df.pivot_table(
        index="subgroup",
        columns="treatment_arm",
        values="response_rate",
        aggfunc="first",
    )
    pivoted["effect"] = pivoted.get("RX-7281", 0) - pivoted.get("Placebo", 0)
    pivoted = pivoted.sort_values("effect")

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivoted) * 0.6)))
    y_pos = range(len(pivoted))
    ax.barh(
        y_pos,
        pivoted["effect"] * 100,
        color=[
            COLORS["RX-7281"] if e > 0 else COLORS["Placebo"] for e in pivoted["effect"]
        ],
        height=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivoted.index)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Treatment Effect (Response Rate Difference, pp)")
    ax.set_title("Subgroup Analysis — Treatment Effect by Subgroup", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
