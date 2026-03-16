"""ARIMA/SARIMA forecasting models with automatic parameter selection.

Provides stationarity testing, seasonal decomposition, and multi-step
ahead forecasting with confidence intervals for pharmaceutical time series.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

logger = structlog.get_logger(__name__)


@dataclass
class StationarityResult:
    """Result of stationarity tests on a time series."""

    is_stationary: bool
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    differencing_order: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Container for forecast output with confidence intervals."""

    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    model_order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    aic: float
    bic: float
    residual_diagnostics: dict[str, Any] = field(default_factory=dict)


class ARIMAForecaster:
    """ARIMA/SARIMA model with automatic parameter selection.

    Implements auto-ARIMA grid search, stationarity testing via ADF and KPSS,
    seasonal decomposition, and multi-step ahead forecasting with configurable
    confidence intervals.
    """

    def __init__(
        self,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        seasonal: bool = True,
        seasonal_period: int = 12,
        max_seasonal_p: int = 1,
        max_seasonal_d: int = 1,
        max_seasonal_q: int = 1,
        information_criterion: str = "aic",
        confidence_level: float = 0.95,
    ) -> None:
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.max_seasonal_p = max_seasonal_p
        self.max_seasonal_d = max_seasonal_d
        self.max_seasonal_q = max_seasonal_q
        self.information_criterion = information_criterion
        self.confidence_level = confidence_level

        self._model: SARIMAX | None = None
        self._fit_result: Any | None = None
        self._best_order: tuple[int, int, int] | None = None
        self._best_seasonal_order: tuple[int, int, int, int] | None = None
        self._training_data: pd.Series | None = None

    def test_stationarity(self, series: pd.Series) -> StationarityResult:
        """Run ADF and KPSS tests to determine stationarity and differencing order.

        The ADF test has H0: unit root (non-stationary).
        The KPSS test has H0: stationary.
        We use both to get a robust assessment.

        Args:
            series: Time series to test.

        Returns:
            StationarityResult with test statistics and recommended differencing.
        """
        clean_series = series.dropna()
        if len(clean_series) < 20:
            logger.warning("series_too_short_for_stationarity_test", length=len(clean_series))
            return StationarityResult(
                is_stationary=False,
                adf_statistic=0.0,
                adf_pvalue=1.0,
                kpss_statistic=0.0,
                kpss_pvalue=0.0,
                differencing_order=1,
            )

        adf_result = adfuller(clean_series, autolag="AIC")
        adf_stat, adf_p = adf_result[0], adf_result[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(clean_series, regression="c", nlags="auto")
        kpss_stat, kpss_p = kpss_result[0], kpss_result[1]

        # Determine stationarity: ADF rejects H0 (p < 0.05) AND KPSS fails to reject H0 (p > 0.05)
        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p > 0.05
        is_stationary = adf_stationary and kpss_stationary

        # Determine differencing order
        d = 0
        test_series = clean_series.copy()
        while d < self.max_d and not is_stationary:
            d += 1
            test_series = test_series.diff().dropna()
            if len(test_series) < 20:
                break
            adf_check = adfuller(test_series, autolag="AIC")
            if adf_check[1] < 0.05:
                is_stationary = True

        logger.info(
            "stationarity_test_complete",
            adf_statistic=round(adf_stat, 4),
            adf_pvalue=round(adf_p, 4),
            kpss_statistic=round(kpss_stat, 4),
            kpss_pvalue=round(kpss_p, 4),
            differencing_order=d,
            is_stationary=is_stationary,
        )

        return StationarityResult(
            is_stationary=is_stationary,
            adf_statistic=float(adf_stat),
            adf_pvalue=float(adf_p),
            kpss_statistic=float(kpss_stat),
            kpss_pvalue=float(kpss_p),
            differencing_order=d,
            details={
                "adf_critical_values": adf_result[4],
                "adf_num_lags": adf_result[2],
                "adf_num_obs": adf_result[3],
            },
        )

    def decompose(
        self,
        series: pd.Series,
        model: str = "additive",
        period: int | None = None,
    ) -> dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components.

        Args:
            series: Time series to decompose.
            model: 'additive' or 'multiplicative'.
            period: Seasonal period. Defaults to self.seasonal_period.

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual' keys.
        """
        period = period or self.seasonal_period
        clean_series = series.dropna()

        if len(clean_series) < 2 * period:
            logger.warning(
                "series_too_short_for_decomposition",
                length=len(clean_series),
                required=2 * period,
            )
            return {
                "trend": clean_series,
                "seasonal": pd.Series(0, index=clean_series.index),
                "residual": pd.Series(0, index=clean_series.index),
            }

        result = seasonal_decompose(clean_series, model=model, period=period)

        logger.info(
            "decomposition_complete",
            model=model,
            period=period,
            trend_strength=round(1 - result.resid.var() / (result.trend + result.resid).var(), 4)
            if result.resid.var() > 0
            else 0.0,
        )

        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }

    def _generate_parameter_grid(
        self, d: int
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int] | None]]:
        """Generate grid of (p,d,q) and seasonal (P,D,Q,s) combinations to search."""
        p_range = range(0, self.max_p + 1)
        q_range = range(0, self.max_q + 1)

        orders = list(itertools.product(p_range, [d], q_range))

        if not self.seasonal:
            return [(order, None) for order in orders]

        sp_range = range(0, self.max_seasonal_p + 1)
        sd_range = range(0, self.max_seasonal_d + 1)
        sq_range = range(0, self.max_seasonal_q + 1)

        seasonal_orders = [
            (sp, sd, sq, self.seasonal_period)
            for sp, sd, sq in itertools.product(sp_range, sd_range, sq_range)
        ]

        grid: list[tuple[tuple[int, int, int], tuple[int, int, int, int] | None]] = []
        for order in orders:
            for sorder in seasonal_orders:
                grid.append((order, sorder))

        return grid

    def auto_select(
        self, series: pd.Series
    ) -> tuple[tuple[int, int, int], tuple[int, int, int, int] | None]:
        """Perform auto-ARIMA parameter selection via grid search.

        Searches over (p,d,q) and (P,D,Q,s) combinations, selects best by
        AIC or BIC. The differencing order d is determined by stationarity tests.

        Args:
            series: Training time series.

        Returns:
            Tuple of (best_order, best_seasonal_order).
        """
        stationarity = self.test_stationarity(series)
        d = stationarity.differencing_order

        grid = self._generate_parameter_grid(d)
        logger.info("auto_arima_grid_search_started", num_candidates=len(grid))

        best_score = np.inf
        best_order: tuple[int, int, int] = (0, d, 0)
        best_seasonal: tuple[int, int, int, int] | None = None
        evaluated = 0

        for order, seasonal_order in grid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order or (0, 0, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False, maxiter=100)

                score = fit.aic if self.information_criterion == "aic" else fit.bic
                evaluated += 1

                if score < best_score:
                    best_score = score
                    best_order = order
                    best_seasonal = seasonal_order

            except Exception:
                continue

        logger.info(
            "auto_arima_grid_search_complete",
            evaluated=evaluated,
            best_order=best_order,
            best_seasonal_order=best_seasonal,
            best_score=round(best_score, 2),
        )

        self._best_order = best_order
        self._best_seasonal_order = best_seasonal
        return best_order, best_seasonal

    def fit(
        self,
        series: pd.Series,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        exog: pd.DataFrame | None = None,
    ) -> ARIMAForecaster:
        """Fit the ARIMA/SARIMA model to the training data.

        If order is not specified, runs auto parameter selection.

        Args:
            series: Training time series.
            order: (p, d, q) order. If None, auto-selects.
            seasonal_order: (P, D, Q, s) seasonal order.
            exog: Exogenous variables (optional).

        Returns:
            self for method chaining.
        """
        self._training_data = series.copy()

        if order is None:
            order, seasonal_order = self.auto_select(series)
        else:
            self._best_order = order
            self._best_seasonal_order = seasonal_order

        self._model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order or (0, 0, 0, 0),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit_result = self._model.fit(disp=False, maxiter=200)

        logger.info(
            "arima_model_fitted",
            order=order,
            seasonal_order=seasonal_order,
            aic=round(self._fit_result.aic, 2),
            bic=round(self._fit_result.bic, 2),
        )

        return self

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame | None = None,
        confidence_level: float | None = None,
    ) -> ForecastResult:
        """Generate multi-step ahead forecast with confidence intervals.

        Args:
            steps: Number of periods to forecast.
            exog: Future exogenous variables (must match training exog shape).
            confidence_level: Override default confidence level (0-1).

        Returns:
            ForecastResult with forecast, bounds, and diagnostics.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._fit_result is None or self._best_order is None:
            raise RuntimeError("Model must be fitted before predicting. Call fit() first.")

        alpha = 1.0 - (confidence_level or self.confidence_level)

        forecast_obj = self._fit_result.get_forecast(steps=steps, exog=exog, alpha=alpha)
        forecast_mean = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=alpha)

        # Residual diagnostics
        residuals = self._fit_result.resid
        ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)

        diagnostics = {
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
            "ljung_box_statistic": float(ljung_box["lb_stat"].iloc[0]),
            "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
            "residuals_white_noise": bool(ljung_box["lb_pvalue"].iloc[0] > 0.05),
        }

        logger.info(
            "forecast_generated",
            steps=steps,
            forecast_mean=round(float(forecast_mean.mean()), 2),
            confidence_level=confidence_level or self.confidence_level,
        )

        return ForecastResult(
            forecast=forecast_mean,
            lower_bound=conf_int.iloc[:, 0],
            upper_bound=conf_int.iloc[:, 1],
            confidence_level=confidence_level or self.confidence_level,
            model_order=self._best_order,
            seasonal_order=self._best_seasonal_order,
            aic=float(self._fit_result.aic),
            bic=float(self._fit_result.bic),
            residual_diagnostics=diagnostics,
        )

    def get_model_summary(self) -> str:
        """Return a text summary of the fitted model."""
        if self._fit_result is None:
            return "Model not fitted."
        return str(self._fit_result.summary())
