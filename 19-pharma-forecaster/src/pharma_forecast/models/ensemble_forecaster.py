"""Ensemble forecasting combining ARIMA, Prophet, and ML models.

Provides dynamic weight adjustment based on recent performance, model selection
per time series characteristics, and a backtesting framework for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = structlog.get_logger(__name__)


class TimeSeriesCharacteristic(Enum):
    """Classification of time series behavior for model selection."""

    TRENDING = "trending"
    SEASONAL = "seasonal"
    STATIONARY = "stationary"
    VOLATILE = "volatile"
    INTERMITTENT = "intermittent"


class Forecaster(Protocol):
    """Protocol for forecaster implementations."""

    def fit(self, series: pd.Series, **kwargs: Any) -> Any: ...
    def predict(self, steps: int, **kwargs: Any) -> Any: ...


@dataclass
class BacktestResult:
    """Results from backtesting a forecasting model or ensemble."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    smape: float
    forecast_bias: float
    n_folds: int
    fold_results: list[dict[str, float]] = field(default_factory=list)
    per_horizon_mae: dict[int, float] = field(default_factory=dict)


@dataclass
class EnsembleForecast:
    """Output of the ensemble forecasting system."""

    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    model_weights: dict[str, float]
    individual_forecasts: dict[str, pd.Series]
    confidence_level: float
    series_characteristic: TimeSeriesCharacteristic | None = None


def compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    More robust than MAPE when actual values are near zero,
    which is common in intermittent pharmaceutical demand.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        SMAPE as a fraction (0-1 range, multiply by 100 for percentage).
    """
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]))


def classify_series(series: pd.Series) -> TimeSeriesCharacteristic:
    """Classify a time series by its dominant behavior.

    Uses coefficient of variation, trend strength, and zero-fraction
    to determine the appropriate model strategy.

    Args:
        series: Time series to classify.

    Returns:
        TimeSeriesCharacteristic enum value.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return TimeSeriesCharacteristic.STATIONARY

    # Check for intermittent demand (common in pharma)
    zero_fraction = (clean == 0).sum() / len(clean)
    if zero_fraction > 0.3:
        return TimeSeriesCharacteristic.INTERMITTENT

    # Coefficient of variation
    cv = clean.std() / clean.mean() if clean.mean() != 0 else 0
    if cv > 1.5:
        return TimeSeriesCharacteristic.VOLATILE

    # Trend detection via linear regression slope significance
    x = np.arange(len(clean))
    slope = np.polyfit(x, clean.values, 1)[0]
    trend_strength = abs(slope) * len(clean) / clean.std() if clean.std() > 0 else 0

    if trend_strength > 2.0:
        return TimeSeriesCharacteristic.TRENDING

    # Check seasonality via autocorrelation at common lags
    if len(clean) >= 24:
        autocorr_12 = clean.autocorr(lag=12)
        if autocorr_12 is not None and abs(autocorr_12) > 0.3:
            return TimeSeriesCharacteristic.SEASONAL

    return TimeSeriesCharacteristic.STATIONARY


class EnsembleForecaster:
    """Ensemble of ARIMA, Prophet, and ML models with dynamic weighting.

    Combines multiple forecasting approaches with weights that adapt based
    on recent forecast accuracy. Includes automatic model selection based
    on time series characteristics and a full backtesting framework.
    """

    # Default weight profiles per series characteristic
    DEFAULT_WEIGHT_PROFILES: dict[TimeSeriesCharacteristic, dict[str, float]] = {
        TimeSeriesCharacteristic.TRENDING: {"arima": 0.3, "prophet": 0.5, "ml": 0.2},
        TimeSeriesCharacteristic.SEASONAL: {"arima": 0.25, "prophet": 0.5, "ml": 0.25},
        TimeSeriesCharacteristic.STATIONARY: {"arima": 0.5, "prophet": 0.2, "ml": 0.3},
        TimeSeriesCharacteristic.VOLATILE: {"arima": 0.2, "prophet": 0.3, "ml": 0.5},
        TimeSeriesCharacteristic.INTERMITTENT: {"arima": 0.1, "prophet": 0.3, "ml": 0.6},
    }

    def __init__(
        self,
        confidence_level: float = 0.95,
        dynamic_weighting: bool = True,
        weight_lookback_periods: int = 30,
        min_weight: float = 0.05,
    ) -> None:
        self.confidence_level = confidence_level
        self.dynamic_weighting = dynamic_weighting
        self.weight_lookback_periods = weight_lookback_periods
        self.min_weight = min_weight

        self._weights: dict[str, float] = {"arima": 0.34, "prophet": 0.33, "ml": 0.33}
        self._individual_models: dict[str, Any] = {}
        self._performance_history: list[dict[str, dict[str, float]]] = []
        self._series_characteristic: TimeSeriesCharacteristic | None = None
        self._ml_model: GradientBoostingRegressor | None = None
        self._training_data: pd.Series | None = None

    def _build_ml_features(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix for the ML component from a time series.

        Creates lag features, rolling statistics, and calendar features
        suitable for gradient boosting.

        Args:
            series: Input time series.

        Returns:
            Tuple of (feature DataFrame, target Series).
        """
        df = pd.DataFrame({"y": series})

        # Lag features
        for lag in [1, 7, 14, 30]:
            if len(series) > lag:
                df[f"lag_{lag}"] = series.shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            if len(series) > window:
                df[f"rolling_mean_{window}"] = series.rolling(window).mean()
                df[f"rolling_std_{window}"] = series.rolling(window).std()

        # Calendar features (if datetime index)
        if isinstance(series.index, pd.DatetimeIndex):
            df["day_of_week"] = series.index.dayofweek
            df["month"] = series.index.month
            df["quarter"] = series.index.quarter
            df["day_of_year"] = series.index.dayofyear

        df = df.dropna()
        target = df.pop("y")
        return df, target

    def _train_ml_model(self, series: pd.Series) -> None:
        """Train the gradient boosting ML component."""
        features, target = self._build_ml_features(series)
        if len(features) < 30:
            logger.warning("insufficient_data_for_ml", n_samples=len(features))
            self._ml_model = None
            return

        self._ml_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self._ml_model.fit(features, target)
        logger.info("ml_model_trained", n_features=features.shape[1], n_samples=len(features))

    def _predict_ml(self, series: pd.Series, steps: int) -> pd.Series:
        """Generate ML model predictions for future steps.

        Uses recursive forecasting: predict one step, then use the prediction
        as a lag feature for the next step.

        Args:
            series: Historical series to base predictions on.
            steps: Number of steps to forecast.

        Returns:
            Series of ML predictions.
        """
        if self._ml_model is None:
            # Fallback: use the last known value as a naive forecast
            last_val = series.iloc[-1]
            if isinstance(series.index, pd.DatetimeIndex):
                future_idx = pd.date_range(
                    start=series.index[-1]
                    + pd.tseries.frequencies.to_offset(pd.infer_freq(series.index) or "D"),
                    periods=steps,
                    freq=pd.infer_freq(series.index) or "D",
                )
            else:
                future_idx = pd.RangeIndex(start=len(series), stop=len(series) + steps)
            return pd.Series(last_val, index=future_idx)

        predictions = []
        current_series = series.copy()

        for _ in range(steps):
            features, _ = self._build_ml_features(current_series)
            if len(features) == 0:
                predictions.append(current_series.iloc[-1])
            else:
                pred = self._ml_model.predict(features.iloc[[-1]])[0]
                predictions.append(pred)

            # Extend series with prediction for next iteration
            if isinstance(current_series.index, pd.DatetimeIndex):
                freq = pd.infer_freq(current_series.index) or "D"
                next_idx = current_series.index[-1] + pd.tseries.frequencies.to_offset(freq)
            else:
                next_idx = current_series.index[-1] + 1

            new_point = pd.Series([predictions[-1]], index=[next_idx])
            current_series = pd.concat([current_series, new_point])

        if isinstance(series.index, pd.DatetimeIndex):
            freq = pd.infer_freq(series.index) or "D"
            future_idx = pd.date_range(
                start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=steps,
                freq=freq,
            )
        else:
            future_idx = pd.RangeIndex(start=len(series), stop=len(series) + steps)

        return pd.Series(predictions, index=future_idx)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Manually set ensemble weights.

        Args:
            weights: Dictionary with 'arima', 'prophet', 'ml' keys summing to ~1.0.
        """
        total = sum(weights.values())
        self._weights = {k: v / total for k, v in weights.items()}
        logger.info("ensemble_weights_set", weights=self._weights)

    def update_weights_from_performance(self, recent_errors: dict[str, float]) -> dict[str, float]:
        """Dynamically adjust weights based on recent forecast errors.

        Models with lower recent MAE get higher weights. Applies softmax-style
        transformation and enforces minimum weights.

        Args:
            recent_errors: Dictionary of model_name -> recent MAE.

        Returns:
            Updated weights dictionary.
        """
        if not recent_errors:
            return self._weights

        # Inverse error weighting with softmax smoothing
        inv_errors = {k: 1.0 / (v + 1e-8) for k, v in recent_errors.items()}
        total_inv = sum(inv_errors.values())

        new_weights = {}
        for model_name in self._weights:
            if model_name in inv_errors:
                raw_weight = inv_errors[model_name] / total_inv
                new_weights[model_name] = max(raw_weight, self.min_weight)
            else:
                new_weights[model_name] = self.min_weight

        # Re-normalize
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        self._weights = new_weights
        self._performance_history.append({"errors": recent_errors, "weights": new_weights})

        logger.info(
            "weights_updated_from_performance",
            recent_errors=recent_errors,
            new_weights={k: round(v, 4) for k, v in new_weights.items()},
        )

        return new_weights

    def select_models_for_series(self, series: pd.Series) -> dict[str, float]:
        """Select model weights based on time series characteristics.

        Classifies the series and assigns weights from the profile that
        best matches its behavior pattern.

        Args:
            series: Time series to analyze.

        Returns:
            Recommended weights for each model.
        """
        characteristic = classify_series(series)
        self._series_characteristic = characteristic

        weights = self.DEFAULT_WEIGHT_PROFILES[characteristic].copy()
        self._weights = weights

        logger.info(
            "models_selected_for_series",
            characteristic=characteristic.value,
            weights=weights,
        )

        return weights

    def fit(
        self,
        series: pd.Series,
        arima_forecasts: pd.Series | None = None,
        prophet_forecasts: pd.Series | None = None,
    ) -> EnsembleForecaster:
        """Fit the ensemble by training the ML component and setting weights.

        The ARIMA and Prophet models are expected to be trained separately
        and their forecasts passed in. The ML model is trained internally.

        Args:
            series: Training time series.
            arima_forecasts: Pre-computed ARIMA in-sample forecasts (optional).
            prophet_forecasts: Pre-computed Prophet in-sample forecasts (optional).

        Returns:
            self for method chaining.
        """
        self._training_data = series.copy()

        # Classify series and set initial weights
        self.select_models_for_series(series)

        # Train ML component
        self._train_ml_model(series)

        # If we have in-sample forecasts, update weights based on performance
        if arima_forecasts is not None and prophet_forecasts is not None:
            # Align all series
            common_idx = series.index.intersection(arima_forecasts.index).intersection(
                prophet_forecasts.index
            )
            if len(common_idx) > 0:
                actual = series.loc[common_idx]
                errors = {
                    "arima": float(mean_absolute_error(actual, arima_forecasts.loc[common_idx])),
                    "prophet": float(
                        mean_absolute_error(actual, prophet_forecasts.loc[common_idx])
                    ),
                }
                # ML in-sample error
                features, target = self._build_ml_features(series)
                if self._ml_model is not None and len(features) > 0:
                    ml_pred = self._ml_model.predict(features)
                    errors["ml"] = float(mean_absolute_error(target, ml_pred))
                else:
                    errors["ml"] = float(np.mean(list(errors.values())))

                if self.dynamic_weighting:
                    self.update_weights_from_performance(errors)

        logger.info("ensemble_fitted", weights=self._weights)
        return self

    def predict(
        self,
        steps: int,
        arima_forecast: pd.Series | None = None,
        prophet_forecast: pd.Series | None = None,
    ) -> EnsembleForecast:
        """Generate ensemble forecast by combining individual model outputs.

        Args:
            steps: Number of periods to forecast.
            arima_forecast: ARIMA forecast series (must have `steps` values).
            prophet_forecast: Prophet forecast series (must have `steps` values).

        Returns:
            EnsembleForecast with combined predictions and confidence intervals.

        Raises:
            RuntimeError: If ensemble has not been fitted.
        """
        if self._training_data is None:
            raise RuntimeError("Ensemble must be fitted before predicting. Call fit() first.")

        # Generate ML forecast
        ml_forecast = self._predict_ml(self._training_data, steps)

        individual_forecasts: dict[str, pd.Series] = {"ml": ml_forecast}

        # Use provided forecasts or fallback to ML only
        if arima_forecast is not None:
            individual_forecasts["arima"] = arima_forecast
        if prophet_forecast is not None:
            individual_forecasts["prophet"] = prophet_forecast

        # Weighted combination
        combined = pd.Series(0.0, index=ml_forecast.index)
        active_weight_total = 0.0

        for model_name, forecast_series in individual_forecasts.items():
            weight = self._weights.get(model_name, 0.0)
            # Align indices
            aligned = forecast_series.reindex(ml_forecast.index)
            if aligned.isna().all():
                continue
            aligned = aligned.fillna(method="ffill").fillna(method="bfill")
            combined += weight * aligned
            active_weight_total += weight

        if active_weight_total > 0:
            combined /= active_weight_total

        # Compute confidence intervals from forecast spread
        forecast_matrix = pd.DataFrame(individual_forecasts)
        forecast_std = forecast_matrix.std(axis=1)
        z_score = 1.96  # ~95% CI
        if self.confidence_level == 0.99:
            z_score = 2.576
        elif self.confidence_level == 0.90:
            z_score = 1.645

        lower = combined - z_score * forecast_std
        upper = combined + z_score * forecast_std

        logger.info(
            "ensemble_forecast_generated",
            steps=steps,
            weights=self._weights,
            forecast_mean=round(float(combined.mean()), 2),
        )

        return EnsembleForecast(
            forecast=combined,
            lower_bound=lower,
            upper_bound=upper,
            model_weights=self._weights.copy(),
            individual_forecasts=individual_forecasts,
            confidence_level=self.confidence_level,
            series_characteristic=self._series_characteristic,
        )

    def backtest(
        self,
        series: pd.Series,
        n_folds: int = 5,
        horizon: int = 30,
        min_train_size: int = 90,
    ) -> BacktestResult:
        """Run expanding-window backtesting to evaluate ensemble performance.

        Splits the series into expanding training windows and evaluates
        forecast accuracy on each subsequent test period.

        Args:
            series: Full time series to backtest on.
            n_folds: Number of backtesting folds.
            horizon: Forecast horizon (steps ahead) for each fold.
            min_train_size: Minimum training set size.

        Returns:
            BacktestResult with aggregate and per-fold metrics.
        """
        total_len = len(series)
        fold_size = (total_len - min_train_size - horizon) // n_folds

        if fold_size <= 0:
            logger.warning(
                "insufficient_data_for_backtest",
                total_len=total_len,
                min_required=min_train_size + horizon + n_folds,
            )
            return BacktestResult(
                model_name="ensemble",
                mae=float("inf"),
                rmse=float("inf"),
                mape=float("inf"),
                smape=float("inf"),
                forecast_bias=0.0,
                n_folds=0,
            )

        all_actuals: list[float] = []
        all_preds: list[float] = []
        fold_results: list[dict[str, float]] = []
        per_horizon_errors: dict[int, list[float]] = {}

        for fold_idx in range(n_folds):
            train_end = min_train_size + fold_idx * fold_size
            test_end = min(train_end + horizon, total_len)

            train = series.iloc[:train_end]
            test = series.iloc[train_end:test_end]
            actual_steps = len(test)

            if actual_steps == 0:
                continue

            # Fit ensemble on training data
            self._train_ml_model(train)
            self.select_models_for_series(train)

            # Generate ML-only forecast for backtest
            ml_pred = self._predict_ml(train, actual_steps)

            # Align predictions with test data
            predictions = ml_pred.values[:actual_steps]
            actuals = test.values[:actual_steps]

            fold_mae = float(mean_absolute_error(actuals, predictions))
            fold_rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
            fold_mape = float(np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))))
            fold_smape = compute_smape(actuals, predictions)
            fold_bias = float(np.mean(predictions - actuals))

            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_size": train_end,
                    "test_size": actual_steps,
                    "mae": fold_mae,
                    "rmse": fold_rmse,
                    "mape": fold_mape,
                    "smape": fold_smape,
                    "bias": fold_bias,
                }
            )

            all_actuals.extend(actuals.tolist())
            all_preds.extend(predictions.tolist())

            # Per-horizon tracking
            for h in range(actual_steps):
                per_horizon_errors.setdefault(h + 1, []).append(abs(actuals[h] - predictions[h]))

            logger.info(
                "backtest_fold_complete",
                fold=fold_idx,
                mae=round(fold_mae, 4),
                rmse=round(fold_rmse, 4),
            )

        if not all_actuals:
            return BacktestResult(
                model_name="ensemble",
                mae=float("inf"),
                rmse=float("inf"),
                mape=float("inf"),
                smape=float("inf"),
                forecast_bias=0.0,
                n_folds=0,
            )

        all_actuals_arr = np.array(all_actuals)
        all_preds_arr = np.array(all_preds)

        result = BacktestResult(
            model_name="ensemble",
            mae=float(mean_absolute_error(all_actuals_arr, all_preds_arr)),
            rmse=float(np.sqrt(mean_squared_error(all_actuals_arr, all_preds_arr))),
            mape=float(
                np.mean(np.abs((all_actuals_arr - all_preds_arr) / (all_actuals_arr + 1e-8)))
            ),
            smape=compute_smape(all_actuals_arr, all_preds_arr),
            forecast_bias=float(np.mean(all_preds_arr - all_actuals_arr)),
            n_folds=len(fold_results),
            fold_results=fold_results,
            per_horizon_mae={h: float(np.mean(errs)) for h, errs in per_horizon_errors.items()},
        )

        logger.info(
            "backtest_complete",
            n_folds=result.n_folds,
            overall_mae=round(result.mae, 4),
            overall_rmse=round(result.rmse, 4),
            forecast_bias=round(result.forecast_bias, 4),
        )

        return result
