"""Shared test fixtures for PharmaForecast tests.

Provides synthetic pharmaceutical time series data with realistic patterns:
seasonal flu demand, trending growth, stationary generic drug demand, and
adverse event reporting data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_dates() -> pd.DatetimeIndex:
    """Two years of daily dates."""
    return pd.date_range(start="2023-01-01", periods=730, freq="D")


@pytest.fixture
def monthly_dates() -> pd.DatetimeIndex:
    """Five years of monthly dates."""
    return pd.date_range(start="2019-01-01", periods=60, freq="MS")


@pytest.fixture
def seasonal_demand_series(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Simulated daily drug demand with strong flu-season seasonality.

    Pattern: base demand ~5000 units/day with:
    - Annual seasonality (peak Jan-Feb, trough Jul-Aug)
    - Weekly seasonality (lower weekends)
    - Positive trend (~2 units/day growth)
    - Random noise
    """
    np.random.seed(42)
    n = len(daily_dates)
    t = np.arange(n)

    base = 5000.0
    trend = 2.0 * t
    yearly = 1500.0 * np.sin(2 * np.pi * t / 365.25 + np.pi / 2)  # Peak in winter
    weekly = 200.0 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 300, n)

    values = base + trend + yearly + weekly + noise
    values = np.maximum(values, 0)  # Demand cannot be negative

    return pd.Series(values, index=daily_dates, name="acetaminophen_demand")


@pytest.fixture
def stationary_series(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Stationary time series with no trend or seasonality.

    Represents stable generic drug demand.
    """
    np.random.seed(123)
    n = len(daily_dates)
    values = 1000.0 + np.random.normal(0, 50, n)
    return pd.Series(values, index=daily_dates, name="generic_demand")


@pytest.fixture
def trending_series(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Strong upward trend with changepoint.

    Simulates new drug launch with rapid adoption then plateau.
    """
    np.random.seed(7)
    n = len(daily_dates)
    t = np.arange(n)

    # Sigmoid-like growth with changepoint at day 365
    growth = 2000.0 / (1 + np.exp(-0.02 * (t - 365)))
    noise = np.random.normal(0, 100, n)

    values = 500.0 + growth + noise
    return pd.Series(values, index=daily_dates, name="new_drug_demand")


@pytest.fixture
def monthly_series(monthly_dates: pd.DatetimeIndex) -> pd.Series:
    """Monthly aggregated drug demand with quarterly patterns."""
    np.random.seed(99)
    n = len(monthly_dates)
    t = np.arange(n)

    base = 150000.0
    trend = 500.0 * t
    quarterly = 10000.0 * np.sin(2 * np.pi * t / 4)  # Quarterly pattern
    yearly = 20000.0 * np.sin(2 * np.pi * t / 12)  # Annual pattern
    noise = np.random.normal(0, 5000, n)

    values = base + trend + quarterly + yearly + noise
    return pd.Series(values, index=monthly_dates, name="monthly_demand")


@pytest.fixture
def adverse_event_series(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Simulated adverse event reporting counts.

    Poisson-distributed counts with seasonal reporting patterns
    (quarterly surges) and a gradual upward trend.
    """
    np.random.seed(55)
    n = len(daily_dates)
    t = np.arange(n)

    # Base rate with quarterly reporting surges
    base_rate = 10.0 + 0.005 * t
    quarterly_effect = 3.0 * np.sin(2 * np.pi * t / 91.25)

    rate = np.maximum(base_rate + quarterly_effect, 1.0)
    values = np.random.poisson(rate)

    return pd.Series(values.astype(float), index=daily_dates, name="ae_reports")


@pytest.fixture
def short_series() -> pd.Series:
    """Very short series (50 points) for edge case testing."""
    np.random.seed(11)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    values = 100.0 + np.random.normal(0, 10, 50)
    return pd.Series(values, index=dates, name="short_series")


@pytest.fixture
def series_with_outliers(daily_dates: pd.DatetimeIndex) -> pd.Series:
    """Series with injected outliers for robustness testing."""
    np.random.seed(42)
    n = len(daily_dates)
    values = 1000.0 + np.random.normal(0, 50, n)

    # Inject outliers
    outlier_indices = [100, 200, 365, 500, 600]
    for idx in outlier_indices:
        values[idx] = values[idx] * 5  # 5x spike

    return pd.Series(values, index=daily_dates, name="outlier_series")
