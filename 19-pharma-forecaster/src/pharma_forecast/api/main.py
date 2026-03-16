"""FastAPI application for the PharmaForecast service.

Endpoints:
    POST /forecast         - Generate forecast for a time series
    GET  /forecast/{id}/accuracy - Forecast accuracy metrics
    POST /backtest         - Run backtesting on historical data
    GET  /alerts           - Active forecast alerts
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pharma_forecast.models.arima_forecaster import ARIMAForecaster
from pharma_forecast.models.ensemble_forecaster import EnsembleForecaster
from pharma_forecast.monitoring.forecast_monitor import (
    AlertSeverity,
    ForecastMonitor,
    MonitorConfig,
)

logger = structlog.get_logger(__name__)

# --- Pydantic request/response models ---


class TimeSeriesPoint(BaseModel):
    """A single observation in a time series."""

    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    value: float = Field(..., description="Observed value")


class ForecastRequest(BaseModel):
    """Request body for generating a forecast."""

    series_id: str = Field(..., description="Unique identifier for this time series")
    data: list[TimeSeriesPoint] = Field(..., min_length=30, description="Historical observations")
    horizon: int = Field(default=90, ge=1, le=365, description="Forecast steps ahead")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="CI level")
    seasonal_period: int = Field(default=12, ge=1, description="Seasonal period")
    use_ensemble: bool = Field(default=True, description="Use ensemble of models")


class ForecastPointResponse(BaseModel):
    """A single point in the forecast output."""

    date: str
    forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    """Response body for a forecast request."""

    series_id: str
    model: str
    horizon: int
    confidence_level: float
    forecast: list[ForecastPointResponse]
    metrics: dict[str, float]
    generated_at: str


class BacktestRequest(BaseModel):
    """Request body for a backtesting run."""

    series_id: str
    data: list[TimeSeriesPoint] = Field(..., min_length=60)
    n_folds: int = Field(default=5, ge=2, le=20)
    horizon: int = Field(default=30, ge=1, le=180)
    min_train_size: int = Field(default=90, ge=30)


class BacktestFoldResult(BaseModel):
    """Results for a single backtest fold."""

    fold: int
    train_size: int
    test_size: int
    mae: float
    rmse: float
    mape: float
    smape: float


class BacktestResponse(BaseModel):
    """Response body for a backtesting run."""

    series_id: str
    model: str
    mae: float
    rmse: float
    mape: float
    smape: float
    forecast_bias: float
    n_folds: int
    fold_results: list[BacktestFoldResult]
    per_horizon_mae: dict[str, float]


class AccuracyResponse(BaseModel):
    """Response body for forecast accuracy metrics."""

    series_id: str
    history: list[dict[str, Any]]
    trend: dict[str, Any]


class AlertResponse(BaseModel):
    """A single forecast alert."""

    alert_id: str
    alert_type: str
    severity: str
    series_id: str
    message: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    timestamp: str
    resolved: bool


# --- Application setup ---


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PharmaForecast API",
        description=(
            "Time Series Forecasting for pharmaceutical demand, "
            "drug shortages, and adverse event trends."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    monitor = ForecastMonitor(config=MonitorConfig())
    forecast_store: dict[str, dict[str, Any]] = {}

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "PharmaForecast",
            "version": "1.0.0",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

    @app.post("/forecast", response_model=ForecastResponse)
    async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
        """Generate a forecast for the provided time series data."""
        logger.info(
            "forecast_request",
            series_id=request.series_id,
            n_points=len(request.data),
            horizon=request.horizon,
        )

        try:
            # Build pandas Series from request
            dates = pd.to_datetime([p.date for p in request.data])
            values = [p.value for p in request.data]
            series = pd.Series(values, index=dates, name=request.series_id)
            series = series.sort_index()

            model_name: str
            forecast_points: list[ForecastPointResponse] = []
            metrics: dict[str, float] = {}

            if request.use_ensemble:
                ensemble = EnsembleForecaster(confidence_level=request.confidence_level)
                ensemble.fit(series)
                result = ensemble.predict(steps=request.horizon)

                model_name = "ensemble"

                for i in range(len(result.forecast)):
                    idx = result.forecast.index[i]
                    date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
                    forecast_points.append(
                        ForecastPointResponse(
                            date=date_str,
                            forecast=round(float(result.forecast.iloc[i]), 4),
                            lower_bound=round(float(result.lower_bound.iloc[i]), 4),
                            upper_bound=round(float(result.upper_bound.iloc[i]), 4),
                        )
                    )

                metrics = {f"weight_{k}": round(v, 4) for k, v in result.model_weights.items()}
                if result.series_characteristic:
                    metrics["series_type"] = 0.0  # placeholder for JSON compatibility

            else:
                arima = ARIMAForecaster(
                    seasonal=True,
                    seasonal_period=request.seasonal_period,
                    confidence_level=request.confidence_level,
                )
                arima.fit(series)
                result_arima = arima.predict(steps=request.horizon)

                model_name = f"SARIMA{result_arima.model_order}"

                for i in range(len(result_arima.forecast)):
                    idx = result_arima.forecast.index[i]
                    date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
                    forecast_points.append(
                        ForecastPointResponse(
                            date=date_str,
                            forecast=round(float(result_arima.forecast.iloc[i]), 4),
                            lower_bound=round(float(result_arima.lower_bound.iloc[i]), 4),
                            upper_bound=round(float(result_arima.upper_bound.iloc[i]), 4),
                        )
                    )

                metrics = {
                    "aic": round(result_arima.aic, 2),
                    "bic": round(result_arima.bic, 2),
                }
                metrics.update(
                    {
                        k: round(v, 4)
                        for k, v in result_arima.residual_diagnostics.items()
                        if isinstance(v, (int, float))
                    }
                )

            # Store for later accuracy tracking
            forecast_store[request.series_id] = {
                "forecast": forecast_points,
                "model": model_name,
                "series_data": series,
                "generated_at": datetime.now(UTC).isoformat(),
            }

            response = ForecastResponse(
                series_id=request.series_id,
                model=model_name,
                horizon=request.horizon,
                confidence_level=request.confidence_level,
                forecast=forecast_points,
                metrics=metrics,
                generated_at=datetime.now(UTC).isoformat(),
            )

            logger.info("forecast_generated", series_id=request.series_id, model=model_name)
            return response

        except Exception as exc:
            logger.error("forecast_failed", series_id=request.series_id, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}")

    @app.get("/forecast/{series_id}/accuracy", response_model=AccuracyResponse)
    async def get_accuracy(series_id: str) -> AccuracyResponse:
        """Get forecast accuracy metrics and trend for a series."""
        history = monitor.get_accuracy_history(series_id)

        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No accuracy history found for series '{series_id}'",
            )

        history_dicts = [
            {
                "mae": m.mae,
                "rmse": m.rmse,
                "mape": m.mape,
                "smape": m.smape,
                "n_observations": m.n_observations,
            }
            for m in history
        ]

        trend = monitor.get_accuracy_trend(series_id)

        return AccuracyResponse(
            series_id=series_id,
            history=history_dicts,
            trend=trend,
        )

    @app.post("/backtest", response_model=BacktestResponse)
    async def run_backtest(request: BacktestRequest) -> BacktestResponse:
        """Run expanding-window backtesting on historical data."""
        logger.info(
            "backtest_request",
            series_id=request.series_id,
            n_folds=request.n_folds,
            horizon=request.horizon,
        )

        try:
            dates = pd.to_datetime([p.date for p in request.data])
            values = [p.value for p in request.data]
            series = pd.Series(values, index=dates, name=request.series_id)
            series = series.sort_index()

            ensemble = EnsembleForecaster()
            result = ensemble.backtest(
                series,
                n_folds=request.n_folds,
                horizon=request.horizon,
                min_train_size=request.min_train_size,
            )

            fold_results = [
                BacktestFoldResult(
                    fold=int(fr["fold"]),
                    train_size=int(fr["train_size"]),
                    test_size=int(fr["test_size"]),
                    mae=round(fr["mae"], 4),
                    rmse=round(fr["rmse"], 4),
                    mape=round(fr["mape"], 4),
                    smape=round(fr["smape"], 4),
                )
                for fr in result.fold_results
            ]

            return BacktestResponse(
                series_id=request.series_id,
                model="ensemble",
                mae=round(result.mae, 4),
                rmse=round(result.rmse, 4),
                mape=round(result.mape, 4),
                smape=round(result.smape, 4),
                forecast_bias=round(result.forecast_bias, 4),
                n_folds=result.n_folds,
                fold_results=fold_results,
                per_horizon_mae={str(k): round(v, 4) for k, v in result.per_horizon_mae.items()},
            )

        except Exception as exc:
            logger.error("backtest_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    @app.get("/alerts", response_model=list[AlertResponse])
    async def get_alerts(
        series_id: str | None = Query(default=None),
        severity: str | None = Query(default=None),
    ) -> list[AlertResponse]:
        """Get active forecast alerts, optionally filtered by series and severity."""
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity. Must be one of: {[s.value for s in AlertSeverity]}",
                )

        alerts = monitor.get_active_alerts(
            series_id=series_id,
            severity=severity_enum,
        )

        return [
            AlertResponse(
                alert_id=a.alert_id,
                alert_type=a.alert_type.value,
                severity=a.severity.value,
                series_id=a.series_id,
                message=a.message,
                metric_name=a.metric_name,
                metric_value=a.metric_value,
                threshold=a.threshold,
                timestamp=a.timestamp,
                resolved=a.resolved,
            )
            for a in alerts
        ]

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the uvicorn server."""
    uvicorn.run(
        "pharma_forecast.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
