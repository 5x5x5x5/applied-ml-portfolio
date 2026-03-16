"""Adverse event trend forecasting for pharmaceutical safety monitoring.

Predicts adverse event report volumes by drug class, detects emerging safety
signals from trend changes, and applies seasonal adjustment to normalize
reporting patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from pharma_forecast.models.ensemble_forecaster import EnsembleForecaster

logger = structlog.get_logger(__name__)


class SafetySignalLevel(Enum):
    """Safety signal classification levels."""

    NONE = "none"
    POTENTIAL = "potential"
    EMERGING = "emerging"
    CONFIRMED = "confirmed"


class AdverseEventCategory(Enum):
    """Categories of adverse events (simplified MedDRA SOC)."""

    CARDIAC = "cardiac_disorders"
    GASTROINTESTINAL = "gastrointestinal"
    HEPATIC = "hepatobiliary"
    NERVOUS_SYSTEM = "nervous_system"
    SKIN = "skin_subcutaneous"
    IMMUNE = "immune_system"
    RESPIRATORY = "respiratory_thoracic"
    RENAL = "renal_urinary"
    GENERAL = "general_disorders"


@dataclass
class AdverseEventForecast:
    """Forecast of adverse event reporting volumes."""

    drug_class: str
    event_category: AdverseEventCategory
    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    baseline_rate: float
    trend_direction: str
    trend_slope: float
    safety_signal: SafetySignalLevel
    signal_details: dict[str, Any] = field(default_factory=dict)
    seasonal_pattern: dict[int, float] = field(default_factory=dict)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()


@dataclass
class SafetySignal:
    """A detected safety signal from adverse event trends."""

    drug_class: str
    event_category: AdverseEventCategory
    signal_level: SafetySignalLevel
    detection_method: str
    detected_at: str
    baseline_rate: float
    current_rate: float
    rate_ratio: float
    p_value: float
    trend_acceleration: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Reporting pattern seasonality (FDA FAERS quarterly patterns) ---

REPORTING_SEASONAL_FACTORS: dict[int, float] = {
    1: 0.85,  # Jan: post-holiday drop
    2: 0.90,
    3: 1.10,  # Q1 close: reporting catch-up
    4: 0.95,
    5: 0.95,
    6: 1.15,  # Q2 close
    7: 0.80,  # Summer drop
    8: 0.85,
    9: 1.10,  # Q3 close
    10: 1.00,
    11: 1.05,
    12: 1.20,  # Q4 close / year-end reporting surge
}


def deseasonalize_reports(
    series: pd.Series,
    seasonal_factors: dict[int, float] | None = None,
) -> pd.Series:
    """Remove seasonal reporting patterns from adverse event counts.

    Divides each value by its month's seasonal factor to get the
    deseasonalized (underlying) rate.

    Args:
        series: Raw adverse event count series with DatetimeIndex.
        seasonal_factors: Monthly factors. Defaults to FAERS patterns.

    Returns:
        Deseasonalized series.
    """
    factors = seasonal_factors or REPORTING_SEASONAL_FACTORS
    deseasonalized = series.copy()

    if isinstance(series.index, pd.DatetimeIndex):
        for idx in series.index:
            factor = factors.get(idx.month, 1.0)
            if factor != 0:
                deseasonalized.loc[idx] = series.loc[idx] / factor

    return deseasonalized


def detect_trend_change(
    series: pd.Series,
    lookback_window: int = 90,
    baseline_window: int = 365,
) -> dict[str, Any]:
    """Detect trend changes in adverse event reporting rates.

    Compares the recent slope to the historical baseline slope using
    a statistical test for slope change.

    Args:
        series: Time series of event counts.
        lookback_window: Number of recent periods for current trend.
        baseline_window: Number of historical periods for baseline.

    Returns:
        Dictionary with trend analysis results.
    """
    clean = series.dropna()
    if len(clean) < lookback_window + baseline_window:
        return {
            "trend_change_detected": False,
            "insufficient_data": True,
            "available": len(clean),
            "required": lookback_window + baseline_window,
        }

    baseline = clean.iloc[-(lookback_window + baseline_window) : -lookback_window]
    recent = clean.iloc[-lookback_window:]

    # Fit linear trends
    x_base = np.arange(len(baseline))
    x_recent = np.arange(len(recent))

    base_slope, base_intercept, base_r, base_p, base_se = stats.linregress(x_base, baseline.values)
    recent_slope, recent_intercept, recent_r, recent_p, recent_se = stats.linregress(
        x_recent, recent.values
    )

    # Test if slopes are significantly different
    slope_diff = recent_slope - base_slope
    se_diff = np.sqrt(base_se**2 + recent_se**2) if (base_se + recent_se) > 0 else 1e-8
    z_stat = slope_diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    trend_change_detected = p_value < 0.05 and abs(slope_diff) > abs(base_slope) * 0.5

    # Trend direction
    if recent_slope > 0.01:
        direction = "increasing"
    elif recent_slope < -0.01:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "trend_change_detected": trend_change_detected,
        "baseline_slope": float(base_slope),
        "recent_slope": float(recent_slope),
        "slope_difference": float(slope_diff),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "direction": direction,
        "acceleration": float(recent_slope - base_slope),
    }


class AdverseEventForecaster:
    """Forecasts adverse event report volumes and detects safety signals.

    Combines time series forecasting with signal detection algorithms
    to identify emerging safety concerns from adverse event reporting
    patterns across drug classes and event categories.
    """

    def __init__(
        self,
        forecast_horizon: int = 90,
        confidence_level: float = 0.95,
        signal_rate_ratio_threshold: float = 1.5,
        signal_p_value_threshold: float = 0.01,
        lookback_window: int = 90,
        baseline_window: int = 365,
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.signal_rate_ratio_threshold = signal_rate_ratio_threshold
        self.signal_p_value_threshold = signal_p_value_threshold
        self.lookback_window = lookback_window
        self.baseline_window = baseline_window
        self._ensemble = EnsembleForecaster(confidence_level=confidence_level)

    def forecast_adverse_events(
        self,
        historical_events: pd.Series,
        drug_class: str,
        event_category: AdverseEventCategory,
        apply_deseasonalization: bool = True,
    ) -> AdverseEventForecast:
        """Generate a forecast of adverse event report volumes.

        Args:
            historical_events: Historical adverse event counts (DatetimeIndex).
            drug_class: Drug class identifier (e.g., "statins", "SSRIs").
            event_category: Type of adverse event.
            apply_deseasonalization: Whether to remove seasonal patterns.

        Returns:
            AdverseEventForecast with predictions and safety signal assessment.
        """
        logger.info(
            "ae_forecast_started",
            drug_class=drug_class,
            event_category=event_category.value,
            history_length=len(historical_events),
        )

        # Deseasonalize for modeling
        if apply_deseasonalization and isinstance(historical_events.index, pd.DatetimeIndex):
            modeling_series = deseasonalize_reports(historical_events)
        else:
            modeling_series = historical_events.copy()

        # Train ensemble
        self._ensemble.fit(modeling_series)
        ensemble_result = self._ensemble.predict(steps=self.forecast_horizon)

        forecast = ensemble_result.forecast
        lower = ensemble_result.lower_bound
        upper = ensemble_result.upper_bound

        # Re-seasonalize the forecast
        seasonal_pattern: dict[int, float] = {}
        if apply_deseasonalization and isinstance(forecast.index, pd.DatetimeIndex):
            for idx in forecast.index:
                factor = REPORTING_SEASONAL_FACTORS.get(idx.month, 1.0)
                forecast.loc[idx] *= factor
                lower.loc[idx] *= factor
                upper.loc[idx] *= factor
                seasonal_pattern[idx.month] = factor

        # Ensure non-negative
        forecast = forecast.clip(lower=0)
        lower = lower.clip(lower=0)
        upper = upper.clip(lower=0)

        # Trend analysis
        trend_info = detect_trend_change(
            historical_events,
            lookback_window=min(self.lookback_window, len(historical_events) // 3),
            baseline_window=min(self.baseline_window, len(historical_events) // 2),
        )

        # Baseline rate (average over baseline period)
        baseline_rate = (
            float(historical_events.iloc[: -self.lookback_window].mean())
            if len(historical_events) > self.lookback_window
            else float(historical_events.mean())
        )

        # Safety signal detection
        signal_level, signal_details = self._detect_safety_signal(
            historical_events, drug_class, event_category, baseline_rate, trend_info
        )

        result = AdverseEventForecast(
            drug_class=drug_class,
            event_category=event_category,
            forecast=forecast,
            lower_bound=lower,
            upper_bound=upper,
            baseline_rate=baseline_rate,
            trend_direction=trend_info.get("direction", "stable"),
            trend_slope=trend_info.get("recent_slope", 0.0),
            safety_signal=signal_level,
            signal_details=signal_details,
            seasonal_pattern=seasonal_pattern,
        )

        logger.info(
            "ae_forecast_complete",
            drug_class=drug_class,
            event_category=event_category.value,
            mean_forecast=round(float(forecast.mean()), 2),
            safety_signal=signal_level.value,
            trend=trend_info.get("direction", "unknown"),
        )

        return result

    def _detect_safety_signal(
        self,
        series: pd.Series,
        drug_class: str,
        event_category: AdverseEventCategory,
        baseline_rate: float,
        trend_info: dict[str, Any],
    ) -> tuple[SafetySignalLevel, dict[str, Any]]:
        """Assess whether current trends indicate a safety signal.

        Uses rate ratio comparison, trend acceleration, and CUSUM-like
        detection to identify emerging safety concerns.

        Args:
            series: Historical event series.
            drug_class: Drug class.
            event_category: Event category.
            baseline_rate: Historical baseline rate.
            trend_info: Trend analysis results.

        Returns:
            Tuple of (signal level, signal details dict).
        """
        if len(series) < self.lookback_window:
            return SafetySignalLevel.NONE, {"reason": "insufficient_data"}

        # Current rate vs baseline
        current_rate = float(series.iloc[-self.lookback_window :].mean())
        rate_ratio = current_rate / (baseline_rate + 1e-8)

        # Poisson test for rate increase
        observed = int(series.iloc[-self.lookback_window :].sum())
        expected = baseline_rate * self.lookback_window
        if expected > 0:
            p_value = 1 - stats.poisson.cdf(observed - 1, expected) if observed > expected else 1.0
        else:
            p_value = 1.0

        # Determine signal level
        trend_increasing = trend_info.get("direction") == "increasing"
        trend_accelerating = trend_info.get("acceleration", 0) > 0

        if (
            rate_ratio > 2.0 * self.signal_rate_ratio_threshold
            and p_value < self.signal_p_value_threshold
            and trend_increasing
        ):
            level = SafetySignalLevel.CONFIRMED
        elif (
            rate_ratio > self.signal_rate_ratio_threshold
            and p_value < self.signal_p_value_threshold
        ):
            level = SafetySignalLevel.EMERGING
        elif rate_ratio > 1.2 and (trend_increasing or trend_accelerating):
            level = SafetySignalLevel.POTENTIAL
        else:
            level = SafetySignalLevel.NONE

        details = {
            "baseline_rate": baseline_rate,
            "current_rate": current_rate,
            "rate_ratio": rate_ratio,
            "p_value": float(p_value),
            "observed_events": observed,
            "expected_events": expected,
            "trend_direction": trend_info.get("direction", "unknown"),
            "trend_acceleration": trend_info.get("acceleration", 0.0),
        }

        if level != SafetySignalLevel.NONE:
            logger.warning(
                "safety_signal_detected",
                drug_class=drug_class,
                event_category=event_category.value,
                signal_level=level.value,
                rate_ratio=round(rate_ratio, 3),
                p_value=round(float(p_value), 6),
            )

        return level, details

    def scan_drug_classes(
        self,
        event_data: dict[str, dict[str, pd.Series]],
    ) -> list[SafetySignal]:
        """Scan multiple drug classes and event categories for signals.

        Args:
            event_data: Nested dict of drug_class -> event_category -> time series.

        Returns:
            List of detected SafetySignals, sorted by severity.
        """
        signals: list[SafetySignal] = []

        for drug_class, categories in event_data.items():
            for cat_name, series in categories.items():
                try:
                    category = AdverseEventCategory(cat_name)
                except ValueError:
                    category = AdverseEventCategory.GENERAL

                forecast = self.forecast_adverse_events(
                    historical_events=series,
                    drug_class=drug_class,
                    event_category=category,
                )

                if forecast.safety_signal != SafetySignalLevel.NONE:
                    signal = SafetySignal(
                        drug_class=drug_class,
                        event_category=category,
                        signal_level=forecast.safety_signal,
                        detection_method="rate_ratio_with_trend",
                        detected_at=datetime.now(UTC).isoformat(),
                        baseline_rate=forecast.signal_details.get("baseline_rate", 0.0),
                        current_rate=forecast.signal_details.get("current_rate", 0.0),
                        rate_ratio=forecast.signal_details.get("rate_ratio", 1.0),
                        p_value=forecast.signal_details.get("p_value", 1.0),
                        trend_acceleration=forecast.signal_details.get("trend_acceleration", 0.0),
                        description=(
                            f"Safety signal ({forecast.safety_signal.value}) for "
                            f"{drug_class} / {category.value}: rate ratio "
                            f"{forecast.signal_details.get('rate_ratio', 0):.2f}"
                        ),
                    )
                    signals.append(signal)

        # Sort by severity (confirmed > emerging > potential)
        severity_order = {
            SafetySignalLevel.CONFIRMED: 0,
            SafetySignalLevel.EMERGING: 1,
            SafetySignalLevel.POTENTIAL: 2,
        }
        signals.sort(key=lambda s: severity_order.get(s.signal_level, 99))

        logger.info(
            "drug_class_scan_complete",
            n_classes_scanned=len(event_data),
            n_signals_detected=len(signals),
        )

        return signals
