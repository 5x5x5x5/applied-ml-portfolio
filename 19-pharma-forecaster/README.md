# PharmaForecast - Time Series Forecasting for Pharma

Ensemble forecasting system for pharmaceutical demand prediction, drug shortage early warning, and adverse event trend analysis. Combines classical statistics (ARIMA), Prophet, and ML feature engineering, orchestrated with Airflow and monitored for accuracy drift.

## Use Cases

- **Drug demand forecasting** (`use_cases/demand_forecast.py`): predicts demand by region and pharmacy, accounting for seasonal patterns (flu, allergy), new drug launches, and patent expirations. Emits shortage early warnings when predicted demand exceeds supply capacity.
- **Adverse event trend forecasting** (`use_cases/adverse_event_forecast.py`): predicts adverse event report volumes by drug class, detects emerging safety signals from trend changes, and applies seasonal adjustment to normalize reporting patterns.

## Architecture

```
19-pharma-forecaster/
├── src/pharma_forecast/
│   ├── models/
│   │   ├── arima_forecaster.py      # ARIMA with stationarity testing (ADF)
│   │   ├── prophet_forecaster.py    # Prophet with pharma-specific holidays
│   │   └── ensemble_forecaster.py   # Weighted ensemble + series classification
│   ├── features/time_features.py    # Lag, rolling, calendar, Fourier, trend features
│   ├── use_cases/                   # Demand + adverse event forecasting
│   ├── monitoring/forecast_monitor.py  # Accuracy tracking, drift detection, alerts
│   ├── pipeline/forecast_pipeline.py   # S3-backed batch pipeline with retry
│   └── api/main.py                  # FastAPI service
├── dags/daily_forecast_dag.py       # Airflow DAG (daily at 6 AM UTC)
├── frontend/                        # Plotly dashboard
├── infrastructure/cloudformation.yaml
├── tests/
└── Dockerfile / docker-compose.yml
```

The ensemble classifies each series (trend, seasonality, intermittency) and weights ARIMA, Prophet, and ML models accordingly. Backtesting uses rolling-origin evaluation with sMAPE as the headline metric.

## Setup

```bash
uv sync            # install dependencies
uv sync --extra dev
```

## Running

```bash
# API server
uv run uvicorn pharma_forecast.api.main:app --reload
```

Endpoints:
- `POST /forecast` - Generate forecast for a time series
- `GET /forecast/{id}/accuracy` - Forecast accuracy metrics
- `POST /backtest` - Run backtesting on historical data
- `GET /alerts` - Active forecast alerts

## Airflow

`dags/daily_forecast_dag.py` runs daily at 6 AM UTC: data quality check → per-series demand forecasts → adverse event forecast → accuracy monitoring → retraining triggers for drifting models.

## Testing

```bash
uv run pytest tests/
```

## Key Dependencies

- statsmodels (ARIMA, stationarity tests)
- prophet (trend/seasonality decomposition)
- scikit-learn (ML ensemble members)
- FastAPI + uvicorn (serving)
- Plotly (dashboard)
- boto3 (S3 pipeline storage)
