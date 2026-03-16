"""Drug demand forecasting by region and pharmacy.

Predicts pharmaceutical demand accounting for seasonal patterns (flu, allergy),
new drug launches, patent expirations, and provides shortage early warnings
when predicted demand exceeds available supply capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

from pharma_forecast.models.ensemble_forecaster import EnsembleForecaster

logger = structlog.get_logger(__name__)


class DrugCategory(Enum):
    """Major drug categories with distinct demand patterns."""

    ANTIBIOTIC = "antibiotic"
    ANTIVIRAL = "antiviral"
    CARDIOVASCULAR = "cardiovascular"
    ONCOLOGY = "oncology"
    RESPIRATORY = "respiratory"
    IMMUNOLOGY = "immunology"
    CNS = "central_nervous_system"
    GENERIC = "generic"


class ShortageRisk(Enum):
    """Risk levels for drug shortage prediction."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DrugDemandForecast:
    """Complete demand forecast for a drug/region combination."""

    drug_id: str
    drug_name: str
    region: str
    category: DrugCategory
    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    seasonal_factors: dict[str, float]
    shortage_risk: ShortageRisk
    shortage_probability: float
    supply_gap: pd.Series | None = None
    event_impacts: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()


@dataclass
class MarketEvent:
    """An event that impacts drug demand (launch, patent expiry, recall)."""

    event_type: str
    drug_id: str
    effective_date: datetime
    impact_magnitude: float
    impact_duration_days: int
    description: str


# --- Seasonal demand multipliers by drug category ---

SEASONAL_PROFILES: dict[DrugCategory, dict[int, float]] = {
    DrugCategory.ANTIVIRAL: {
        1: 1.8,
        2: 1.6,
        3: 1.3,
        4: 0.8,
        5: 0.6,
        6: 0.5,
        7: 0.5,
        8: 0.6,
        9: 0.7,
        10: 1.0,
        11: 1.4,
        12: 1.7,
    },
    DrugCategory.RESPIRATORY: {
        1: 1.5,
        2: 1.4,
        3: 1.2,
        4: 1.3,
        5: 1.4,
        6: 1.1,
        7: 0.7,
        8: 0.7,
        9: 0.9,
        10: 1.0,
        11: 1.2,
        12: 1.4,
    },
    DrugCategory.CARDIOVASCULAR: {
        1: 1.05,
        2: 1.02,
        3: 1.0,
        4: 0.98,
        5: 0.97,
        6: 0.96,
        7: 0.95,
        8: 0.96,
        9: 0.98,
        10: 1.0,
        11: 1.03,
        12: 1.05,
    },
    DrugCategory.ANTIBIOTIC: {
        1: 1.4,
        2: 1.3,
        3: 1.1,
        4: 0.9,
        5: 0.8,
        6: 0.7,
        7: 0.7,
        8: 0.8,
        9: 0.9,
        10: 1.0,
        11: 1.2,
        12: 1.3,
    },
}


def get_seasonal_multiplier(category: DrugCategory, month: int) -> float:
    """Get the seasonal demand multiplier for a drug category and month.

    Args:
        category: Drug category.
        month: Month number (1-12).

    Returns:
        Multiplicative seasonal factor (1.0 = baseline).
    """
    profile = SEASONAL_PROFILES.get(category)
    if profile is None:
        return 1.0
    return profile.get(month, 1.0)


def compute_event_impact(
    forecast: pd.Series,
    events: list[MarketEvent],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Apply market event impacts to a demand forecast.

    Handles drug launches (demand increase), patent expirations (demand decrease
    for brand, increase for generics), and recalls (immediate demand shift).

    Args:
        forecast: Base demand forecast.
        events: List of market events to apply.

    Returns:
        Tuple of (adjusted forecast, list of impact details).
    """
    adjusted = forecast.copy()
    impact_details: list[dict[str, Any]] = []

    for event in events:
        event_date = pd.Timestamp(event.effective_date)

        # Only process events within forecast range
        if event_date > adjusted.index.max() or event_date < adjusted.index.min():
            continue

        # Calculate impact window
        mask = (adjusted.index >= event_date) & (
            adjusted.index <= event_date + pd.Timedelta(days=event.impact_duration_days)
        )

        if not mask.any():
            continue

        # Apply impact: gradual ramp for launches, step for recalls
        if event.event_type in ("launch", "patent_expiration"):
            # Gradual ramp-up/down over the impact duration
            n_impact_days = mask.sum()
            ramp = np.linspace(0, event.impact_magnitude, n_impact_days)
            adjusted.loc[mask] = adjusted.loc[mask] * (1 + ramp)
        elif event.event_type == "recall":
            # Immediate step change
            adjusted.loc[mask] = adjusted.loc[mask] * (1 + event.impact_magnitude)
        else:
            # Default: constant impact
            adjusted.loc[mask] = adjusted.loc[mask] * (1 + event.impact_magnitude)

        impact_details.append(
            {
                "event_type": event.event_type,
                "drug_id": event.drug_id,
                "effective_date": str(event_date),
                "magnitude": event.impact_magnitude,
                "affected_periods": int(mask.sum()),
                "description": event.description,
            }
        )

    return adjusted, impact_details


class DrugDemandForecaster:
    """Forecasts pharmaceutical demand by drug, region, and pharmacy.

    Combines statistical time series models with domain-specific adjustments
    for seasonal patterns, market events (launches, patent expirations),
    and supply chain constraints. Provides shortage early warnings.
    """

    def __init__(
        self,
        forecast_horizon: int = 90,
        confidence_level: float = 0.95,
        shortage_threshold: float = 0.85,
        critical_shortage_threshold: float = 0.95,
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.shortage_threshold = shortage_threshold
        self.critical_shortage_threshold = critical_shortage_threshold
        self._ensemble = EnsembleForecaster(confidence_level=confidence_level)
        self._regional_adjustments: dict[str, float] = {}

    def set_regional_adjustment(self, region: str, factor: float) -> None:
        """Set a demand adjustment factor for a region.

        Args:
            region: Region identifier.
            factor: Multiplicative adjustment (e.g., 1.1 = 10% higher demand).
        """
        self._regional_adjustments[region] = factor
        logger.info("regional_adjustment_set", region=region, factor=factor)

    def forecast_demand(
        self,
        historical_demand: pd.Series,
        drug_id: str,
        drug_name: str,
        region: str,
        category: DrugCategory,
        supply_capacity: pd.Series | None = None,
        market_events: list[MarketEvent] | None = None,
    ) -> DrugDemandForecast:
        """Generate a demand forecast for a specific drug in a region.

        Args:
            historical_demand: Historical demand time series.
            drug_id: Unique drug identifier (e.g., NDC code).
            drug_name: Human-readable drug name.
            region: Geographic region.
            category: Drug category for seasonal profiling.
            supply_capacity: Expected supply capacity over forecast horizon.
            market_events: Upcoming market events affecting demand.

        Returns:
            DrugDemandForecast with predictions and shortage assessment.
        """
        logger.info(
            "demand_forecast_started",
            drug_id=drug_id,
            region=region,
            category=category.value,
            history_length=len(historical_demand),
        )

        # Train ensemble on historical data
        self._ensemble.fit(historical_demand)

        # Generate base forecast
        ensemble_result = self._ensemble.predict(steps=self.forecast_horizon)
        base_forecast = ensemble_result.forecast
        lower = ensemble_result.lower_bound
        upper = ensemble_result.upper_bound

        # Apply seasonal adjustments
        seasonal_factors: dict[str, float] = {}
        if isinstance(base_forecast.index, pd.DatetimeIndex):
            for idx in base_forecast.index:
                month = idx.month
                multiplier = get_seasonal_multiplier(category, month)
                base_forecast.loc[idx] *= multiplier
                lower.loc[idx] *= multiplier
                upper.loc[idx] *= multiplier
                seasonal_factors[f"month_{month}"] = multiplier

        # Apply regional adjustments
        regional_factor = self._regional_adjustments.get(region, 1.0)
        base_forecast *= regional_factor
        lower *= regional_factor
        upper *= regional_factor

        # Apply market event impacts
        event_impacts: list[dict[str, Any]] = []
        if market_events:
            base_forecast, event_impacts = compute_event_impact(base_forecast, market_events)

        # Ensure forecasts are non-negative (demand cannot be negative)
        base_forecast = base_forecast.clip(lower=0)
        lower = lower.clip(lower=0)
        upper = upper.clip(lower=0)

        # Assess shortage risk
        shortage_risk, shortage_prob, supply_gap = self._assess_shortage_risk(
            base_forecast, upper, supply_capacity
        )

        result = DrugDemandForecast(
            drug_id=drug_id,
            drug_name=drug_name,
            region=region,
            category=category,
            forecast=base_forecast,
            lower_bound=lower,
            upper_bound=upper,
            seasonal_factors=seasonal_factors,
            shortage_risk=shortage_risk,
            shortage_probability=shortage_prob,
            supply_gap=supply_gap,
            event_impacts=event_impacts,
        )

        logger.info(
            "demand_forecast_complete",
            drug_id=drug_id,
            region=region,
            mean_forecast=round(float(base_forecast.mean()), 2),
            shortage_risk=shortage_risk.value,
            shortage_probability=round(shortage_prob, 4),
        )

        return result

    def _assess_shortage_risk(
        self,
        forecast: pd.Series,
        upper_bound: pd.Series,
        supply_capacity: pd.Series | None,
    ) -> tuple[ShortageRisk, float, pd.Series | None]:
        """Assess drug shortage risk by comparing forecast to supply.

        Args:
            forecast: Point forecast of demand.
            upper_bound: Upper confidence bound of demand.
            supply_capacity: Available supply (if known).

        Returns:
            Tuple of (risk level, probability, supply gap series).
        """
        if supply_capacity is None:
            return ShortageRisk.LOW, 0.0, None

        # Align supply with forecast
        common_idx = forecast.index.intersection(supply_capacity.index)
        if len(common_idx) == 0:
            return ShortageRisk.LOW, 0.0, None

        demand = forecast.loc[common_idx]
        supply = supply_capacity.loc[common_idx]
        demand_upper = upper_bound.loc[common_idx]

        # Supply gap: positive means demand exceeds supply
        gap = demand - supply

        # Fraction of periods where demand exceeds supply
        shortage_fraction = float((gap > 0).mean())

        # Fraction where upper bound exceeds supply (worst case)
        worst_case_fraction = float((demand_upper - supply > 0).mean())

        # Determine risk level
        if worst_case_fraction > self.critical_shortage_threshold:
            risk = ShortageRisk.CRITICAL
        elif shortage_fraction > self.shortage_threshold:
            risk = ShortageRisk.HIGH
        elif shortage_fraction > 0.5:
            risk = ShortageRisk.MODERATE
        else:
            risk = ShortageRisk.LOW

        logger.info(
            "shortage_risk_assessed",
            risk=risk.value,
            shortage_fraction=round(shortage_fraction, 4),
            worst_case_fraction=round(worst_case_fraction, 4),
        )

        return risk, shortage_fraction, gap

    def generate_shortage_report(
        self,
        forecasts: list[DrugDemandForecast],
    ) -> pd.DataFrame:
        """Generate a summary report of shortage risks across multiple drugs.

        Args:
            forecasts: List of demand forecasts to summarize.

        Returns:
            DataFrame with one row per drug/region showing risk metrics.
        """
        records = []
        for f in forecasts:
            records.append(
                {
                    "drug_id": f.drug_id,
                    "drug_name": f.drug_name,
                    "region": f.region,
                    "category": f.category.value,
                    "mean_forecast": float(f.forecast.mean()),
                    "peak_forecast": float(f.forecast.max()),
                    "shortage_risk": f.shortage_risk.value,
                    "shortage_probability": f.shortage_probability,
                    "mean_supply_gap": float(f.supply_gap.mean())
                    if f.supply_gap is not None
                    else 0.0,
                    "generated_at": f.generated_at,
                }
            )

        report = pd.DataFrame(records)
        report = report.sort_values("shortage_probability", ascending=False)

        logger.info(
            "shortage_report_generated",
            n_drugs=len(report),
            n_high_risk=int((report["shortage_risk"].isin(["high", "critical"])).sum()),
        )

        return report
