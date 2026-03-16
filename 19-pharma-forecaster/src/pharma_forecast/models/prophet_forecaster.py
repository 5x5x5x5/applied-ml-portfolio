"""Facebook Prophet wrapper for pharmaceutical time series forecasting.

Supports custom seasonality (flu season, quarterly reporting), holiday effects
(FDA deadlines), changepoint detection, and external regressors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = structlog.get_logger(__name__)


# --- Pharma-specific holiday and event calendars ---

FDA_SUBMISSION_DEADLINES: list[dict[str, Any]] = [
    {"holiday": "FDA_PDUFA_Q1", "month": 3, "day": 31},
    {"holiday": "FDA_PDUFA_Q2", "month": 6, "day": 30},
    {"holiday": "FDA_PDUFA_Q3", "month": 9, "day": 30},
    {"holiday": "FDA_PDUFA_Q4", "month": 12, "day": 31},
]

FLU_SEASON_PEAKS: list[dict[str, Any]] = [
    {"holiday": "flu_season_peak", "month": 1, "day": 15},
    {"holiday": "flu_season_peak", "month": 2, "day": 15},
    {"holiday": "flu_season_start", "month": 10, "day": 1},
    {"holiday": "flu_season_end", "month": 4, "day": 30},
]


def build_pharma_holidays(
    years: list[int],
    include_fda: bool = True,
    include_flu: bool = True,
    custom_events: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a holiday DataFrame for pharmaceutical-relevant events.

    Args:
        years: List of years to generate holidays for.
        include_fda: Include FDA submission deadlines.
        include_flu: Include flu season markers.
        custom_events: Additional custom events with 'holiday', 'month', 'day' keys.

    Returns:
        DataFrame with 'holiday', 'ds', 'lower_window', 'upper_window' columns.
    """
    events: list[dict[str, Any]] = []

    if include_fda:
        for year in years:
            for deadline in FDA_SUBMISSION_DEADLINES:
                events.append(
                    {
                        "holiday": deadline["holiday"],
                        "ds": pd.Timestamp(year=year, month=deadline["month"], day=deadline["day"]),
                        "lower_window": -7,
                        "upper_window": 7,
                    }
                )

    if include_flu:
        for year in years:
            for marker in FLU_SEASON_PEAKS:
                events.append(
                    {
                        "holiday": marker["holiday"],
                        "ds": pd.Timestamp(year=year, month=marker["month"], day=marker["day"]),
                        "lower_window": -14,
                        "upper_window": 14,
                    }
                )

    if custom_events:
        for year in years:
            for event in custom_events:
                events.append(
                    {
                        "holiday": event["holiday"],
                        "ds": pd.Timestamp(year=year, month=event["month"], day=event["day"]),
                        "lower_window": event.get("lower_window", -3),
                        "upper_window": event.get("upper_window", 3),
                    }
                )

    return pd.DataFrame(events)


@dataclass
class ProphetForecastResult:
    """Container for Prophet forecast output."""

    forecast: pd.DataFrame
    components: pd.DataFrame
    changepoints: pd.DataFrame
    cross_val_metrics: dict[str, float] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)


class ProphetForecaster:
    """Facebook Prophet wrapper with pharma-specific configuration.

    Provides custom seasonality for pharmaceutical patterns, holiday effects
    for FDA deadlines, changepoint detection for trend shifts, and support
    for external regressors like weather or population data.
    """

    def __init__(
        self,
        yearly_seasonality: bool | int = True,
        weekly_seasonality: bool | int = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_range: float = 0.8,
        n_changepoints: int = 25,
        growth: str = "linear",
        interval_width: float = 0.95,
        include_pharma_holidays: bool = True,
        include_flu_seasonality: bool = True,
        include_quarterly_seasonality: bool = True,
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_range = changepoint_range
        self.n_changepoints = n_changepoints
        self.growth = growth
        self.interval_width = interval_width
        self.include_pharma_holidays = include_pharma_holidays
        self.include_flu_seasonality = include_flu_seasonality
        self.include_quarterly_seasonality = include_quarterly_seasonality

        self._model: Prophet | None = None
        self._training_df: pd.DataFrame | None = None
        self._regressors: list[str] = []

    def _build_model(self, years: list[int]) -> Prophet:
        """Construct and configure a Prophet model instance."""
        holidays = None
        if self.include_pharma_holidays:
            holidays = build_pharma_holidays(
                years=years,
                include_fda=True,
                include_flu=True,
            )

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_range=self.changepoint_range,
            n_changepoints=self.n_changepoints,
            growth=self.growth,
            interval_width=self.interval_width,
            holidays=holidays,
        )

        # Add custom seasonalities for pharmaceutical patterns
        if self.include_flu_seasonality:
            model.add_seasonality(
                name="flu_season",
                period=365.25,
                fourier_order=5,
                prior_scale=8.0,
                mode="additive",
            )

        if self.include_quarterly_seasonality:
            model.add_seasonality(
                name="quarterly_reporting",
                period=365.25 / 4,
                fourier_order=3,
                prior_scale=5.0,
                mode="additive",
            )

        return model

    def add_regressor(
        self,
        name: str,
        prior_scale: float = 10.0,
        standardize: str = "auto",
        mode: str = "additive",
    ) -> ProphetForecaster:
        """Register an external regressor (e.g., weather, population).

        Must be called before fit(). The regressor column must be present
        in both training and forecast DataFrames.

        Args:
            name: Column name of the regressor.
            prior_scale: Regularization strength.
            standardize: Whether to standardize the regressor.
            mode: 'additive' or 'multiplicative'.

        Returns:
            self for method chaining.
        """
        self._regressors.append(name)
        logger.info("regressor_registered", name=name, mode=mode)
        return self

    def fit(
        self,
        df: pd.DataFrame,
        ds_col: str = "ds",
        y_col: str = "y",
    ) -> ProphetForecaster:
        """Fit the Prophet model to training data.

        Args:
            df: Training data with date and target columns.
            ds_col: Name of the date column.
            y_col: Name of the target column.

        Returns:
            self for method chaining.
        """
        train_df = df.rename(columns={ds_col: "ds", y_col: "y"})
        train_df["ds"] = pd.to_datetime(train_df["ds"])
        self._training_df = train_df.copy()

        years = sorted(train_df["ds"].dt.year.unique().tolist())
        # Extend for potential forecast horizon
        years.extend([max(years) + i for i in range(1, 4)])

        self._model = self._build_model(years)

        for reg_name in self._regressors:
            self._model.add_regressor(
                reg_name,
                prior_scale=10.0,
                standardize="auto",
                mode="additive",
            )

        logger.info(
            "prophet_fitting_started",
            n_rows=len(train_df),
            date_range=(str(train_df["ds"].min()), str(train_df["ds"].max())),
            regressors=self._regressors,
        )

        self._model.fit(train_df)

        changepoints = self._model.changepoints
        logger.info(
            "prophet_model_fitted",
            n_changepoints_detected=len(changepoints),
            changepoint_dates=[str(cp) for cp in changepoints[:5]],
        )

        return self

    def predict(
        self,
        periods: int,
        freq: str = "D",
        future_regressors: pd.DataFrame | None = None,
    ) -> ProphetForecastResult:
        """Generate forecast for future periods.

        Args:
            periods: Number of periods to forecast.
            freq: Frequency string ('D' for daily, 'W' for weekly, 'M' for monthly).
            future_regressors: DataFrame with regressor values for future dates.

        Returns:
            ProphetForecastResult with forecast, components, and changepoints.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before predicting. Call fit() first.")

        future_df = self._model.make_future_dataframe(periods=periods, freq=freq)

        # Merge future regressors if provided
        if future_regressors is not None and self._regressors:
            future_regressors["ds"] = pd.to_datetime(future_regressors["ds"])
            future_df = future_df.merge(future_regressors, on="ds", how="left")
            # Fill missing regressor values with training means
            if self._training_df is not None:
                for reg in self._regressors:
                    if reg in future_df.columns:
                        fill_val = (
                            self._training_df[reg].mean() if reg in self._training_df.columns else 0
                        )
                        future_df[reg] = future_df[reg].fillna(fill_val)
        elif self._regressors and self._training_df is not None:
            for reg in self._regressors:
                if reg in self._training_df.columns:
                    future_df[reg] = self._training_df[reg].mean()

        forecast = self._model.predict(future_df)

        # Extract components
        components_to_plot = ["trend"]
        if self.yearly_seasonality:
            components_to_plot.append("yearly")
        if self.weekly_seasonality:
            components_to_plot.append("weekly")
        if self.include_pharma_holidays:
            components_to_plot.append("holidays")

        available_components = [c for c in components_to_plot if c in forecast.columns]
        components = forecast[["ds"] + available_components].copy()

        # Changepoint info
        changepoint_df = pd.DataFrame(
            {
                "ds": self._model.changepoints,
                "delta": np.abs(
                    self._model.params["delta"].flatten()[: len(self._model.changepoints)]
                ),
            }
        )
        changepoint_df = changepoint_df.sort_values("delta", ascending=False)

        logger.info(
            "prophet_forecast_generated",
            periods=periods,
            freq=freq,
            forecast_end=str(forecast["ds"].max()),
        )

        return ProphetForecastResult(
            forecast=forecast,
            components=components,
            changepoints=changepoint_df,
            model_params={
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "growth": self.growth,
                "n_changepoints": self.n_changepoints,
            },
        )

    def cross_validate(
        self,
        initial: str = "365 days",
        period: str = "90 days",
        horizon: str = "90 days",
    ) -> dict[str, float]:
        """Run Prophet cross-validation and return performance metrics.

        Args:
            initial: Training period for first fold.
            period: Spacing between successive folds.
            horizon: Forecast horizon for each fold.

        Returns:
            Dictionary of metric name to average value.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before cross-validation.")

        logger.info(
            "prophet_cross_validation_started",
            initial=initial,
            period=period,
            horizon=horizon,
        )

        cv_results = cross_validation(
            self._model,
            initial=initial,
            period=period,
            horizon=horizon,
        )

        metrics_df = performance_metrics(cv_results)

        metrics = {
            "mae": float(metrics_df["mae"].mean()),
            "mape": float(metrics_df["mape"].mean()),
            "rmse": float(metrics_df["rmse"].mean()),
            "coverage": float(metrics_df["coverage"].mean()),
        }

        logger.info("prophet_cross_validation_complete", metrics=metrics)
        return metrics

    def detect_changepoints(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Identify the most significant trend changepoints.

        Args:
            top_n: Number of top changepoints to return.

        Returns:
            List of dicts with 'date' and 'magnitude' keys.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted first.")

        deltas = np.abs(self._model.params["delta"].flatten())
        changepoint_dates = self._model.changepoints

        n_available = min(len(deltas), len(changepoint_dates))
        deltas = deltas[:n_available]
        dates = changepoint_dates[:n_available]

        top_indices = np.argsort(deltas)[-top_n:][::-1]

        results = [
            {
                "date": str(dates.iloc[i]),
                "magnitude": float(deltas[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

        logger.info("changepoints_detected", top_n=top_n, results=results)
        return results
