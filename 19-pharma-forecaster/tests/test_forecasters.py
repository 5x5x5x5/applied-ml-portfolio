"""Tests for ARIMA, Prophet, and Ensemble forecasting models.

Validates model fitting, prediction, stationarity testing, backtesting,
and edge case handling with synthetic pharmaceutical time series data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharma_forecast.features.time_features import (
    create_all_features,
    create_calendar_features,
    create_fourier_features,
    create_lag_features,
    create_rolling_features,
    create_trend_features,
)
from pharma_forecast.models.arima_forecaster import ARIMAForecaster, StationarityResult
from pharma_forecast.models.ensemble_forecaster import (
    EnsembleForecaster,
    TimeSeriesCharacteristic,
    classify_series,
    compute_smape,
)
from pharma_forecast.monitoring.forecast_monitor import (
    AlertType,
    ForecastMonitor,
    MonitorConfig,
)

# --- ARIMA Forecaster Tests ---


class TestARIMAForecaster:
    """Tests for the ARIMA/SARIMA forecasting model."""

    def test_stationarity_on_stationary_series(self, stationary_series: pd.Series) -> None:
        """Stationary series should be detected as stationary."""
        forecaster = ARIMAForecaster()
        result = forecaster.test_stationarity(stationary_series)

        assert isinstance(result, StationarityResult)
        assert result.adf_pvalue < 0.05  # ADF rejects unit root
        assert result.differencing_order == 0

    def test_stationarity_on_trending_series(self, trending_series: pd.Series) -> None:
        """Trending series should require differencing."""
        forecaster = ARIMAForecaster()
        result = forecaster.test_stationarity(trending_series)

        assert isinstance(result, StationarityResult)
        assert result.differencing_order >= 1

    def test_fit_and_predict(self, stationary_series: pd.Series) -> None:
        """Model should fit and produce forecasts of correct length."""
        # Use a shorter subseries for speed
        series = stationary_series.iloc[:200]
        forecaster = ARIMAForecaster(max_p=1, max_q=1, seasonal=False, confidence_level=0.95)
        forecaster.fit(series, order=(1, 0, 1))
        result = forecaster.predict(steps=30)

        assert len(result.forecast) == 30
        assert len(result.lower_bound) == 30
        assert len(result.upper_bound) == 30
        assert result.confidence_level == 0.95
        assert result.model_order == (1, 0, 1)

    def test_confidence_intervals_ordering(self, stationary_series: pd.Series) -> None:
        """Lower bound should be below forecast, upper above."""
        series = stationary_series.iloc[:200]
        forecaster = ARIMAForecaster(seasonal=False)
        forecaster.fit(series, order=(1, 0, 0))
        result = forecaster.predict(steps=10)

        assert (result.lower_bound <= result.forecast).all()
        assert (result.upper_bound >= result.forecast).all()

    def test_predict_without_fit_raises(self) -> None:
        """Calling predict before fit should raise RuntimeError."""
        forecaster = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            forecaster.predict(steps=10)

    def test_seasonal_decomposition(self, seasonal_demand_series: pd.Series) -> None:
        """Decomposition should return trend, seasonal, and residual."""
        series = seasonal_demand_series.iloc[:400]
        forecaster = ARIMAForecaster(seasonal_period=30)
        result = forecaster.decompose(series, period=30)

        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert len(result["seasonal"]) == len(series)

    def test_model_summary(self, stationary_series: pd.Series) -> None:
        """Fitted model should produce a non-empty summary."""
        series = stationary_series.iloc[:200]
        forecaster = ARIMAForecaster(seasonal=False)
        forecaster.fit(series, order=(1, 0, 0))
        summary = forecaster.get_model_summary()

        assert len(summary) > 100
        assert "coef" in summary.lower() or "ar" in summary.lower()

    def test_auto_select_returns_valid_order(self, stationary_series: pd.Series) -> None:
        """Auto-selection should return a valid (p,d,q) tuple."""
        series = stationary_series.iloc[:200]
        forecaster = ARIMAForecaster(max_p=1, max_q=1, seasonal=False)
        order, seasonal = forecaster.auto_select(series)

        assert len(order) == 3
        assert all(isinstance(x, int) for x in order)
        assert 0 <= order[0] <= 1
        assert 0 <= order[2] <= 1

    def test_residual_diagnostics(self, stationary_series: pd.Series) -> None:
        """Residual diagnostics should include Ljung-Box test results."""
        series = stationary_series.iloc[:200]
        forecaster = ARIMAForecaster(seasonal=False)
        forecaster.fit(series, order=(1, 0, 1))
        result = forecaster.predict(steps=10)

        diag = result.residual_diagnostics
        assert "residual_mean" in diag
        assert "ljung_box_pvalue" in diag
        assert isinstance(diag["residuals_white_noise"], bool)


# --- Ensemble Forecaster Tests ---


class TestEnsembleForecaster:
    """Tests for the ensemble forecasting model."""

    def test_classify_stationary(self, stationary_series: pd.Series) -> None:
        """Stationary series should be classified correctly."""
        result = classify_series(stationary_series)
        assert result in (
            TimeSeriesCharacteristic.STATIONARY,
            TimeSeriesCharacteristic.SEASONAL,
        )

    def test_classify_trending(self, trending_series: pd.Series) -> None:
        """Trending series should be classified as trending."""
        result = classify_series(trending_series)
        assert result in (
            TimeSeriesCharacteristic.TRENDING,
            TimeSeriesCharacteristic.VOLATILE,
        )

    def test_smape_computation(self) -> None:
        """SMAPE should handle basic cases correctly."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 190.0, 310.0])
        result = compute_smape(actual, predicted)
        assert 0.0 < result < 1.0

    def test_smape_perfect_forecast(self) -> None:
        """SMAPE of a perfect forecast should be 0."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([100.0, 200.0])
        assert compute_smape(actual, predicted) == 0.0

    def test_fit_and_predict(self, seasonal_demand_series: pd.Series) -> None:
        """Ensemble should fit and produce forecasts."""
        series = seasonal_demand_series.iloc[:300]
        ensemble = EnsembleForecaster()
        ensemble.fit(series)
        result = ensemble.predict(steps=30)

        assert len(result.forecast) == 30
        assert len(result.lower_bound) == 30
        assert len(result.upper_bound) == 30
        assert "ml" in result.model_weights

    def test_weight_update(self) -> None:
        """Weight updates should favor models with lower errors."""
        ensemble = EnsembleForecaster()
        errors = {"arima": 100.0, "prophet": 50.0, "ml": 75.0}
        new_weights = ensemble.update_weights_from_performance(errors)

        # Prophet had lowest error, should have highest weight
        assert new_weights["prophet"] > new_weights["arima"]
        assert new_weights["prophet"] > new_weights["ml"]

        # Weights should sum to ~1.0
        assert abs(sum(new_weights.values()) - 1.0) < 1e-6

    def test_manual_weight_setting(self) -> None:
        """Manual weights should be normalized."""
        ensemble = EnsembleForecaster()
        ensemble.set_weights({"arima": 2, "prophet": 6, "ml": 2})

        assert abs(ensemble._weights["arima"] - 0.2) < 1e-6
        assert abs(ensemble._weights["prophet"] - 0.6) < 1e-6
        assert abs(ensemble._weights["ml"] - 0.2) < 1e-6

    def test_backtest_returns_metrics(self, seasonal_demand_series: pd.Series) -> None:
        """Backtesting should return valid accuracy metrics."""
        series = seasonal_demand_series.iloc[:400]
        ensemble = EnsembleForecaster()
        result = ensemble.backtest(series, n_folds=3, horizon=14, min_train_size=200)

        assert result.n_folds > 0
        assert result.mae >= 0
        assert result.rmse >= 0
        assert result.mape >= 0
        assert len(result.fold_results) > 0

    def test_predict_without_fit_raises(self) -> None:
        """Should raise error if not fitted."""
        ensemble = EnsembleForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            ensemble.predict(steps=10)

    def test_model_selection_per_characteristic(self, seasonal_demand_series: pd.Series) -> None:
        """Model selection should set weights based on series type."""
        ensemble = EnsembleForecaster()
        weights = ensemble.select_models_for_series(seasonal_demand_series.iloc[:300])

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w > 0 for w in weights.values())


# --- Feature Engineering Tests ---


class TestFeatureEngineering:
    """Tests for time series feature generation."""

    def test_lag_features_shape(self, seasonal_demand_series: pd.Series) -> None:
        """Lag features should have expected columns."""
        lags = create_lag_features(seasonal_demand_series, lags=[1, 7, 30])
        assert "lag_1" in lags.columns
        assert "lag_7" in lags.columns
        assert "lag_30" in lags.columns
        assert len(lags) == len(seasonal_demand_series)

    def test_rolling_features_values(self, stationary_series: pd.Series) -> None:
        """Rolling mean should be close to the series mean for stationary data."""
        rolling = create_rolling_features(stationary_series, windows=[30])
        # The 30-day rolling mean of a stationary series should be close to global mean
        last_100 = rolling["rolling_mean_30"].iloc[-100:]
        assert abs(last_100.mean() - stationary_series.mean()) < 100

    def test_calendar_features(self, daily_dates: pd.DatetimeIndex) -> None:
        """Calendar features should have correct ranges."""
        cal = create_calendar_features(daily_dates)
        assert cal["day_of_week"].min() == 0
        assert cal["day_of_week"].max() == 6
        assert cal["month"].min() == 1
        assert cal["month"].max() == 12
        assert set(cal["is_weekend"].unique()) == {0, 1}
        assert "is_flu_season" in cal.columns

    def test_fourier_features_shape(self, daily_dates: pd.DatetimeIndex) -> None:
        """Fourier features should generate sin/cos pairs."""
        fourier = create_fourier_features(daily_dates, periods=[365.25], n_terms=3)
        assert fourier.shape[1] == 6  # 3 sin + 3 cos terms
        # Values should be in [-1, 1]
        assert fourier.min().min() >= -1.0 - 1e-10
        assert fourier.max().max() <= 1.0 + 1e-10

    def test_trend_features(self, daily_dates: pd.DatetimeIndex) -> None:
        """Trend features should be monotonically increasing."""
        trend = create_trend_features(daily_dates, degree=2)
        assert "trend_linear" in trend.columns
        assert "trend_poly_2" in trend.columns
        linear = trend["trend_linear"].values
        assert np.all(linear[1:] >= linear[:-1])

    def test_all_features_combined(self, seasonal_demand_series: pd.Series) -> None:
        """Combined feature set should have many columns and no NaN after drop."""
        features = create_all_features(
            seasonal_demand_series.iloc[:400],
            lags=[1, 7, 30],
            rolling_windows=[7, 14],
            fourier_terms=2,
            drop_na=True,
        )
        assert features.shape[1] > 20
        assert not features.isna().any().any()

    def test_all_features_requires_datetime_index(self) -> None:
        """Should raise ValueError if series has no DatetimeIndex."""
        series = pd.Series([1, 2, 3, 4, 5], name="test")
        with pytest.raises(ValueError, match="DatetimeIndex"):
            create_all_features(series)


# --- Forecast Monitor Tests ---


class TestForecastMonitor:
    """Tests for the forecast monitoring system."""

    def test_accuracy_computation(self) -> None:
        """Accuracy metrics should be computed correctly."""
        monitor = ForecastMonitor()
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        actual = pd.Series(np.random.normal(100, 10, 100), index=index)
        predicted = actual + np.random.normal(0, 5, 100)

        metrics = monitor.compute_accuracy(actual, predicted, series_id="test")

        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.rmse >= metrics.mae  # RMSE >= MAE always
        assert metrics.n_observations == 100

    def test_bias_detection(self) -> None:
        """Monitor should detect systematic over-forecasting."""
        monitor = ForecastMonitor(config=MonitorConfig(bias_threshold=0.05))
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        actual = pd.Series(100.0, index=index)
        predicted = pd.Series(120.0, index=index)  # Consistent 20% over-forecast

        alert = monitor.detect_bias(actual, predicted, "biased_series")

        assert alert is not None
        assert alert.alert_type == AlertType.FORECAST_BIAS
        assert "over" in alert.message

    def test_no_bias_on_unbiased_forecast(self) -> None:
        """No bias alert for unbiased forecasts."""
        monitor = ForecastMonitor()
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        actual = pd.Series(100.0 + np.random.normal(0, 1, 100), index=index)
        predicted = pd.Series(100.0 + np.random.normal(0, 1, 100), index=index)

        alert = monitor.detect_bias(actual, predicted, "unbiased_series")
        assert alert is None

    def test_accuracy_threshold_alert(self) -> None:
        """Should trigger alert when MAE exceeds threshold."""
        config = MonitorConfig(mae_threshold=10.0)
        monitor = ForecastMonitor(config=config)
        index = pd.date_range("2024-01-01", periods=50, freq="D")

        actual = pd.Series(100.0, index=index)
        predicted = pd.Series(120.0, index=index)  # MAE = 20

        metrics = monitor.compute_accuracy(actual, predicted)
        alerts = monitor.check_accuracy_threshold(metrics, "test_series")

        assert len(alerts) > 0
        assert any(a.metric_name == "mae" for a in alerts)

    def test_drift_detection_no_drift(self, stationary_series: pd.Series) -> None:
        """Stationary series should show no drift."""
        monitor = ForecastMonitor()
        result = monitor.detect_drift(stationary_series, "stationary_test", window_size=60)

        assert not result.is_drifted

    def test_drift_detection_with_drift(self) -> None:
        """Shifted series should show drift."""
        monitor = ForecastMonitor(config=MonitorConfig(drift_p_value_threshold=0.05))
        np.random.seed(42)
        # Create series with mean shift at halfway point
        part1 = np.random.normal(100, 5, 100)
        part2 = np.random.normal(150, 5, 100)  # Mean shifted by 50
        values = np.concatenate([part1, part2])
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        series = pd.Series(values, index=dates)

        result = monitor.detect_drift(series, "drift_test", window_size=50)

        assert result.is_drifted
        assert result.p_value < 0.05

    def test_retrain_trigger(self) -> None:
        """Should trigger retraining after critical alerts."""
        monitor = ForecastMonitor()
        config = MonitorConfig(mae_threshold=5.0)
        monitor.config = config

        index = pd.date_range("2024-01-01", periods=50, freq="D")
        actual = pd.Series(100.0, index=index)
        predicted = pd.Series(130.0, index=index)  # Large error

        metrics = monitor.compute_accuracy(actual, predicted)
        monitor.check_accuracy_threshold(metrics, "test_series")

        should = monitor.should_retrain("test_series")
        assert should is True

    def test_alert_resolution(self) -> None:
        """Resolved alerts should not appear in active list."""
        monitor = ForecastMonitor(config=MonitorConfig(mae_threshold=5.0))
        index = pd.date_range("2024-01-01", periods=50, freq="D")
        actual = pd.Series(100.0, index=index)
        predicted = pd.Series(120.0, index=index)

        metrics = monitor.compute_accuracy(actual, predicted)
        alerts = monitor.check_accuracy_threshold(metrics, "resolve_test")
        assert len(alerts) > 0

        alert_id = alerts[0].alert_id
        monitor.resolve_alert(alert_id)
        active = monitor.get_active_alerts(series_id="resolve_test")
        resolved_ids = [a.alert_id for a in active]
        assert alert_id not in resolved_ids
