"""Biomarker trend analysis with time-series decomposition.

Detects improving/worsening/stable trends, calculates rate of change,
and generates predictive alerts when values are projected to exit
normal ranges.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from biomarker_dash.schemas import (
    NORMAL_RANGES,
    BiomarkerReading,
    BiomarkerType,
    TrendDirection,
    TrendResult,
)

logger = logging.getLogger(__name__)

# Minimum data points to compute a trend
MIN_POINTS_FOR_TREND = 5
# Hours of prediction horizon
PREDICTION_HORIZON_HOURS = 24


class TrendAnalyzer:
    """Analyzes biomarker trends over time using statistical methods.

    Capabilities:
    - Time series decomposition (trend + seasonal + residual)
    - Linear trend detection with significance testing
    - Rate of change calculation
    - Forward prediction to detect imminent range exits
    """

    def __init__(self, significance_threshold: float = 0.6) -> None:
        self._significance_threshold = significance_threshold

    def analyze(
        self,
        readings: list[BiomarkerReading],
        biomarker_type: BiomarkerType,
        window_hours: int = 24,
    ) -> TrendResult:
        """Analyze a series of biomarker readings and determine trend.

        Args:
            readings: Time-ordered list of readings for one biomarker.
            biomarker_type: The biomarker type being analyzed.
            window_hours: Analysis window in hours.

        Returns:
            TrendResult with direction, rate, prediction, and confidence.
        """
        patient_id = readings[0].patient_id if readings else "unknown"

        if len(readings) < MIN_POINTS_FOR_TREND:
            return TrendResult(
                patient_id=patient_id,
                biomarker_type=biomarker_type,
                direction=TrendDirection.UNKNOWN,
                rate_of_change=0.0,
                confidence=0.0,
                window_hours=window_hours,
                data_points_used=len(readings),
            )

        # Filter to the analysis window
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        windowed = [r for r in readings if r.timestamp >= cutoff]
        if len(windowed) < MIN_POINTS_FOR_TREND:
            windowed = readings[-MIN_POINTS_FOR_TREND:]

        values = np.array([r.value for r in windowed], dtype=np.float64)
        timestamps = np.array([r.timestamp.timestamp() for r in windowed], dtype=np.float64)
        # Normalize timestamps to hours from first reading
        t_hours = (timestamps - timestamps[0]) / 3600.0

        # Decompose: trend via linear regression
        slope, intercept, r_squared = self._linear_fit(t_hours, values)

        # Rate of change (units per hour)
        rate_of_change = float(slope)

        # Determine direction based on slope significance and R-squared
        direction = self._classify_direction(slope, r_squared, values, biomarker_type)

        # Predict 24h ahead
        t_predict = t_hours[-1] + PREDICTION_HORIZON_HOURS
        predicted_value = float(intercept + slope * t_predict)

        # Check if predicted value exits normal range
        low, high = NORMAL_RANGES.get(biomarker_type, (0.0, 1000.0))
        predicted_exit = predicted_value < low or predicted_value > high

        # Confidence based on R-squared and sample size
        n = len(windowed)
        size_factor = min(1.0, n / 30.0)
        confidence = float(r_squared * size_factor)

        result = TrendResult(
            patient_id=patient_id,
            biomarker_type=biomarker_type,
            direction=direction,
            rate_of_change=round(rate_of_change, 6),
            predicted_value_24h=round(predicted_value, 4),
            predicted_exit_normal=predicted_exit,
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            window_hours=window_hours,
            data_points_used=n,
        )

        if predicted_exit and confidence > 0.5:
            logger.warning(
                "Predictive alert: patient %s %s trending toward range exit "
                "(predicted %.2f, range [%.2f, %.2f], confidence %.2f)",
                patient_id,
                biomarker_type.value,
                predicted_value,
                low,
                high,
                confidence,
            )

        return result

    def decompose(self, readings: list[BiomarkerReading]) -> dict[str, NDArray[np.float64]]:
        """Simple additive decomposition: trend + seasonal + residual.

        Uses a moving average for trend extraction and a simple
        periodic component estimate. For clinical biomarkers the
        'seasonal' component often captures circadian rhythms.
        """
        if len(readings) < 6:
            values = np.array([r.value for r in readings], dtype=np.float64)
            return {
                "original": values,
                "trend": values.copy(),
                "seasonal": np.zeros_like(values),
                "residual": np.zeros_like(values),
            }

        values = np.array([r.value for r in readings], dtype=np.float64)

        # Trend: centered moving average (window=5 or smaller)
        window = min(5, len(values) // 2)
        if window < 2:
            window = 2
        trend = self._moving_average(values, window)

        # Detrended signal
        detrended = values - trend

        # Seasonal: estimate periodic component (assume period = window)
        period = window
        seasonal = np.zeros_like(values)
        for i in range(period):
            indices = list(range(i, len(detrended), period))
            season_val = float(np.mean(detrended[indices]))
            for idx in indices:
                seasonal[idx] = season_val

        # Residual
        residual = values - trend - seasonal

        return {
            "original": values,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }

    def calculate_volatility(self, readings: list[BiomarkerReading], window: int = 10) -> float:
        """Calculate rolling volatility (std of returns) for a biomarker series."""
        if len(readings) < 3:
            return 0.0
        values = np.array([r.value for r in readings], dtype=np.float64)
        # Percentage changes
        returns = np.diff(values) / np.maximum(np.abs(values[:-1]), 1e-6)
        if len(returns) < window:
            return float(np.std(returns))
        # Rolling std of last `window` returns
        return float(np.std(returns[-window:]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float, float]:
        """Ordinary least squares linear fit.

        Returns:
            (slope, intercept, r_squared)
        """
        n = len(x)
        if n < 2:
            return 0.0, float(y[0]) if n else 0.0, 0.0

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        ss_xx = float(np.sum((x - x_mean) ** 2))
        ss_yy = float(np.sum((y - y_mean) ** 2))
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))

        if ss_xx < 1e-12:
            return 0.0, y_mean, 0.0

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_yy > 1e-12 else 0.0

        return slope, intercept, min(1.0, max(0.0, r_squared))

    def _classify_direction(
        self,
        slope: float,
        r_squared: float,
        values: NDArray[np.float64],
        biomarker_type: BiomarkerType,
    ) -> TrendDirection:
        """Classify trend direction using slope, significance, and clinical context."""
        # Need minimum R-squared for confidence
        if r_squared < self._significance_threshold * 0.3:
            return TrendDirection.STABLE

        value_range = float(np.ptp(values))
        mean_val = float(np.mean(values))
        if mean_val == 0:
            mean_val = 1.0

        # Relative slope: slope / mean as fraction of change per hour
        relative_slope = abs(slope) / abs(mean_val)

        # If slope is tiny relative to the values, call it stable
        if relative_slope < 0.001:
            return TrendDirection.STABLE

        # Determine if increasing is good or bad based on biomarker context
        low, high = NORMAL_RANGES.get(biomarker_type, (0.0, 1000.0))
        midpoint = (low + high) / 2.0

        # If current mean is above range and increasing -> worsening
        # If current mean is below range and decreasing -> worsening
        # Otherwise context-dependent
        if slope > 0:
            if mean_val > high:
                return TrendDirection.WORSENING
            elif mean_val < low:
                return TrendDirection.IMPROVING
            else:
                # Within range but going up - depends on proximity to boundary
                if mean_val > midpoint:
                    return TrendDirection.WORSENING
                return TrendDirection.IMPROVING
        else:
            if mean_val < low:
                return TrendDirection.WORSENING
            elif mean_val > high:
                return TrendDirection.IMPROVING
            else:
                if mean_val < midpoint:
                    return TrendDirection.WORSENING
                return TrendDirection.IMPROVING

    @staticmethod
    def _moving_average(data: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        """Centered moving average with edge handling."""
        result = np.empty_like(data)
        half = window // 2
        for i in range(len(data)):
            start = max(0, i - half)
            end = min(len(data), i + half + 1)
            result[i] = np.mean(data[start:end])
        return result
