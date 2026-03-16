"""Time series feature engineering for pharmaceutical forecasting.

Generates lag features, rolling statistics, calendar features, Fourier terms
for seasonality, and polynomial trend features from raw time series data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Standard lag windows for pharmaceutical demand patterns
DEFAULT_LAGS: list[int] = [1, 7, 14, 30, 90, 365]

# Rolling windows for demand statistics
DEFAULT_ROLLING_WINDOWS: list[int] = [7, 14, 30, 60, 90]

# US federal holidays relevant to pharma (month, day) - simplified set
US_PHARMA_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (11, 28),  # Thanksgiving (approx)
    (12, 25),  # Christmas
]


def create_lag_features(
    series: pd.Series,
    lags: list[int] | None = None,
    prefix: str = "lag",
) -> pd.DataFrame:
    """Generate lag features from a time series.

    Lag features capture autoregressive behavior: demand yesterday,
    last week, last month, last quarter, and last year.

    Args:
        series: Input time series.
        lags: List of lag periods. Defaults to [1, 7, 14, 30, 90, 365].
        prefix: Column name prefix.

    Returns:
        DataFrame with lag columns (NaN where insufficient history).
    """
    lags = lags or DEFAULT_LAGS
    features: dict[str, pd.Series] = {}

    for lag in lags:
        col_name = f"{prefix}_{lag}"
        features[col_name] = series.shift(lag)

    # Lag differences (momentum features)
    if 1 in lags and 7 in lags:
        features[f"{prefix}_diff_1_7"] = series.shift(1) - series.shift(7)
    if 7 in lags and 30 in lags:
        features[f"{prefix}_diff_7_30"] = series.shift(7) - series.shift(30)

    result = pd.DataFrame(features, index=series.index)
    logger.debug("lag_features_created", n_features=len(features), lags=lags)
    return result


def create_rolling_features(
    series: pd.Series,
    windows: list[int] | None = None,
    prefix: str = "rolling",
) -> pd.DataFrame:
    """Generate rolling window statistics: mean, std, min, max, median.

    Captures recent demand levels and volatility over multiple time horizons.

    Args:
        series: Input time series.
        windows: List of window sizes. Defaults to [7, 14, 30, 60, 90].
        prefix: Column name prefix.

    Returns:
        DataFrame with rolling statistic columns.
    """
    windows = windows or DEFAULT_ROLLING_WINDOWS
    features: dict[str, pd.Series] = {}

    for w in windows:
        rolling = series.rolling(window=w, min_periods=1)
        features[f"{prefix}_mean_{w}"] = rolling.mean()
        features[f"{prefix}_std_{w}"] = rolling.std()
        features[f"{prefix}_min_{w}"] = rolling.min()
        features[f"{prefix}_max_{w}"] = rolling.max()
        features[f"{prefix}_median_{w}"] = rolling.median()

        # Range as fraction of mean (volatility indicator)
        range_vals = features[f"{prefix}_max_{w}"] - features[f"{prefix}_min_{w}"]
        mean_vals = features[f"{prefix}_mean_{w}"]
        features[f"{prefix}_range_ratio_{w}"] = range_vals / (mean_vals + 1e-8)

    # Exponentially weighted moving average for recent emphasis
    for span in [7, 30]:
        features[f"ewm_mean_{span}"] = series.ewm(span=span, min_periods=1).mean()
        features[f"ewm_std_{span}"] = series.ewm(span=span, min_periods=1).std()

    result = pd.DataFrame(features, index=series.index)
    logger.debug("rolling_features_created", n_features=len(features), windows=windows)
    return result


def create_calendar_features(
    index: pd.DatetimeIndex,
    include_holiday_flag: bool = True,
) -> pd.DataFrame:
    """Generate calendar-based features from a datetime index.

    Includes day of week, month, quarter, year, day of year, week of year,
    weekend flag, and a simplified holiday flag.

    Args:
        index: DatetimeIndex to extract calendar features from.
        include_holiday_flag: Whether to include a binary holiday indicator.

    Returns:
        DataFrame with calendar features aligned to the input index.
    """
    features: dict[str, Any] = {
        "day_of_week": index.dayofweek,
        "day_of_month": index.day,
        "month": index.month,
        "quarter": index.quarter,
        "year": index.year,
        "day_of_year": index.dayofyear,
        "week_of_year": index.isocalendar().week.values,
        "is_weekend": (index.dayofweek >= 5).astype(int),
        "is_month_start": index.is_month_start.astype(int),
        "is_month_end": index.is_month_end.astype(int),
        "is_quarter_start": index.is_quarter_start.astype(int),
        "is_quarter_end": index.is_quarter_end.astype(int),
    }

    if include_holiday_flag:
        month_day = list(zip(index.month, index.day))
        features["is_holiday"] = [1 if (m, d) in US_PHARMA_HOLIDAYS else 0 for m, d in month_day]

    # Season indicator (pharma-relevant: flu season Oct-Mar, allergy Apr-Jun)
    features["is_flu_season"] = ((index.month >= 10) | (index.month <= 3)).astype(int)
    features["is_allergy_season"] = ((index.month >= 4) & (index.month <= 6)).astype(int)

    result = pd.DataFrame(features, index=index)
    logger.debug("calendar_features_created", n_features=len(features))
    return result


def create_fourier_features(
    index: pd.DatetimeIndex,
    periods: list[float] | None = None,
    n_terms: int = 4,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """Generate Fourier terms for capturing periodic seasonality.

    Fourier features allow linear models to capture arbitrary seasonal patterns
    by decomposing them into sine/cosine components at different frequencies.

    Args:
        index: DatetimeIndex for the time series.
        periods: Seasonal periods in days. Defaults to [7, 30.44, 91.31, 365.25].
        n_terms: Number of Fourier terms per period (sin/cos pairs).
        prefix: Column name prefix.

    Returns:
        DataFrame with sine and cosine features.
    """
    if periods is None:
        periods = [
            7.0,  # Weekly
            30.44,  # Monthly
            91.31,  # Quarterly
            365.25,  # Yearly
        ]

    # Convert datetime to fractional day-of-year for periodicity
    t = (index - index[0]).total_seconds() / 86400.0  # days since start
    t_array = t.values.astype(float)

    features: dict[str, np.ndarray] = {}

    for period in periods:
        period_name = f"{period:.0f}d"
        for k in range(1, n_terms + 1):
            angle = 2.0 * np.pi * k * t_array / period
            features[f"{prefix}_sin_{period_name}_{k}"] = np.sin(angle)
            features[f"{prefix}_cos_{period_name}_{k}"] = np.cos(angle)

    result = pd.DataFrame(features, index=index)
    logger.debug(
        "fourier_features_created",
        n_features=len(features),
        periods=periods,
        n_terms=n_terms,
    )
    return result


def create_trend_features(
    index: pd.DatetimeIndex,
    degree: int = 2,
    prefix: str = "trend",
) -> pd.DataFrame:
    """Generate polynomial trend features from a time index.

    Creates linear, quadratic, and higher-order trend terms for capturing
    long-term growth or decline patterns in pharmaceutical demand.

    Args:
        index: DatetimeIndex for the time series.
        degree: Maximum polynomial degree.
        prefix: Column name prefix.

    Returns:
        DataFrame with trend features.
    """
    # Normalized time index (0 to 1 range for numerical stability)
    t = (index - index[0]).total_seconds()
    t_normalized = t / (t.max() + 1e-8)  # avoid division by zero
    t_values = t_normalized.values.astype(float)

    features: dict[str, np.ndarray] = {
        f"{prefix}_linear": t_values,
    }

    for d in range(2, degree + 1):
        features[f"{prefix}_poly_{d}"] = t_values**d

    # Log trend (common in growth scenarios)
    features[f"{prefix}_log"] = np.log1p(t_values * 1000)

    # Piecewise linear (allows for trend breaks)
    mid = 0.5
    features[f"{prefix}_piecewise"] = np.maximum(t_values - mid, 0)

    result = pd.DataFrame(features, index=index)
    logger.debug("trend_features_created", degree=degree, n_features=len(features))
    return result


def create_all_features(
    series: pd.Series,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    fourier_periods: list[float] | None = None,
    fourier_terms: int = 4,
    trend_degree: int = 2,
    include_holidays: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Generate a comprehensive feature matrix from a time series.

    Combines lag features, rolling statistics, calendar features, Fourier
    terms, and trend features into a single DataFrame.

    Args:
        series: Input time series with DatetimeIndex.
        lags: Lag periods for autoregressive features.
        rolling_windows: Windows for rolling statistics.
        fourier_periods: Seasonal periods for Fourier terms.
        fourier_terms: Number of Fourier terms per period.
        trend_degree: Polynomial degree for trend features.
        include_holidays: Include holiday and season flags.
        drop_na: Whether to drop rows with NaN values.

    Returns:
        DataFrame with all features combined.

    Raises:
        ValueError: If series does not have a DatetimeIndex.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex for full feature generation.")

    index = series.index

    parts = [
        create_lag_features(series, lags=lags),
        create_rolling_features(series, windows=rolling_windows),
        create_calendar_features(index, include_holiday_flag=include_holidays),
        create_fourier_features(index, periods=fourier_periods, n_terms=fourier_terms),
        create_trend_features(index, degree=trend_degree),
    ]

    result = pd.concat(parts, axis=1)

    if drop_na:
        n_before = len(result)
        result = result.dropna()
        n_dropped = n_before - len(result)
        if n_dropped > 0:
            logger.info("features_na_rows_dropped", n_dropped=n_dropped, n_remaining=len(result))

    logger.info(
        "all_features_created",
        n_features=result.shape[1],
        n_samples=result.shape[0],
        feature_groups=["lags", "rolling", "calendar", "fourier", "trend"],
    )

    return result
