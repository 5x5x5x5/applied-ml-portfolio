"""End-to-end pharmaceutical forecasting pipeline.

Orchestrates data ingestion from S3, feature engineering, model training with
expanding-window cross-validation, forecast generation, result storage, and
alerting. Includes retry logic and pipeline state management.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from io import StringIO
from typing import Any

import boto3
import pandas as pd
import structlog
from botocore.exceptions import ClientError

from pharma_forecast.features.time_features import create_all_features
from pharma_forecast.models.arima_forecaster import ARIMAForecaster
from pharma_forecast.models.ensemble_forecaster import EnsembleForecaster
from pharma_forecast.monitoring.forecast_monitor import ForecastMonitor

logger = structlog.get_logger(__name__)


class PipelineStage(Enum):
    """Stages of the forecasting pipeline."""

    INGEST = "ingest"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    FORECASTING = "forecasting"
    STORAGE = "storage"
    ALERTING = "alerting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the forecasting pipeline."""

    # S3 configuration
    s3_bucket: str = "pharma-forecast-data"
    s3_input_prefix: str = "raw/demand/"
    s3_output_prefix: str = "forecasts/"
    s3_model_prefix: str = "models/"
    aws_region: str = "us-east-1"

    # Forecasting parameters
    forecast_horizon: int = 90
    confidence_level: float = 0.95
    min_training_samples: int = 180

    # Cross-validation
    cv_n_folds: int = 5
    cv_min_train_size: int = 365

    # Pipeline behavior
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_factor: float = 2.0
    alert_on_failure: bool = True

    # Alert thresholds
    mape_alert_threshold: float = 0.15
    bias_alert_threshold: float = 0.10


@dataclass
class PipelineResult:
    """Output of a pipeline execution."""

    pipeline_id: str
    series_id: str
    stage: PipelineStage
    started_at: str
    completed_at: str | None = None
    forecast: pd.DataFrame | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _retry_with_backoff(
    func: Any,
    max_retries: int = 3,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (ClientError, ConnectionError),
) -> Any:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: Callable to execute.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay on each retry.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        Result of the function call.

    Raises:
        Exception: The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    "retry_scheduled",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(exc),
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    "all_retries_exhausted",
                    attempts=max_retries + 1,
                    error=str(exc),
                )

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception captured")


class S3DataStore:
    """Handles data I/O with AWS S3 for the forecasting pipeline."""

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region
        self._client = boto3.client("s3", region_name=region)

    def read_timeseries(self, key: str) -> pd.Series:
        """Read a time series CSV from S3.

        Expects CSV with 'date' and 'value' columns.

        Args:
            key: S3 object key.

        Returns:
            Time series with DatetimeIndex.
        """

        def _download() -> pd.Series:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(body), parse_dates=["date"])
            df = df.set_index("date").sort_index()
            return df["value"]

        return _retry_with_backoff(_download)

    def write_forecast(self, key: str, forecast_df: pd.DataFrame) -> None:
        """Write forecast results to S3 as CSV.

        Args:
            key: S3 object key.
            forecast_df: DataFrame with forecast data.
        """

        def _upload() -> None:
            csv_buffer = forecast_df.to_csv()
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=csv_buffer.encode("utf-8"),
                ContentType="text/csv",
            )

        _retry_with_backoff(_upload)
        logger.info("forecast_written_to_s3", bucket=self.bucket, key=key)

    def write_metrics(self, key: str, metrics: dict[str, Any]) -> None:
        """Write metrics JSON to S3.

        Args:
            key: S3 object key.
            metrics: Dictionary of metrics.
        """

        def _upload() -> None:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(metrics, indent=2, default=str).encode("utf-8"),
                ContentType="application/json",
            )

        _retry_with_backoff(_upload)

    def list_series(self, prefix: str) -> list[str]:
        """List available time series files under an S3 prefix.

        Args:
            prefix: S3 key prefix to search.

        Returns:
            List of S3 object keys.
        """

        def _list() -> list[str]:
            response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = response.get("Contents", [])
            return [obj["Key"] for obj in contents if obj["Key"].endswith(".csv")]

        return _retry_with_backoff(_list)


class ForecastPipeline:
    """End-to-end pharmaceutical forecasting pipeline.

    Orchestrates the full forecasting workflow: data ingestion, feature
    engineering, model training with cross-validation, forecast generation,
    result persistence, and accuracy monitoring with alerts.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._store = S3DataStore(
            bucket=self.config.s3_bucket,
            region=self.config.aws_region,
        )
        self._monitor = ForecastMonitor()
        self._current_stage = PipelineStage.INGEST

    def _update_stage(self, stage: PipelineStage, pipeline_id: str) -> None:
        """Update and log the current pipeline stage."""
        self._current_stage = stage
        logger.info("pipeline_stage_changed", pipeline_id=pipeline_id, stage=stage.value)

    def ingest_data(self, series_id: str) -> pd.Series:
        """Ingest time series data from S3.

        Args:
            series_id: Identifier for the time series (maps to S3 key).

        Returns:
            Raw time series data.

        Raises:
            ValueError: If insufficient data for forecasting.
        """
        key = f"{self.config.s3_input_prefix}{series_id}.csv"
        series = self._store.read_timeseries(key)

        if len(series) < self.config.min_training_samples:
            raise ValueError(
                f"Insufficient data for {series_id}: "
                f"{len(series)} samples < {self.config.min_training_samples} minimum"
            )

        # Basic data quality checks
        n_missing = series.isna().sum()
        if n_missing > 0:
            logger.warning("missing_values_detected", series_id=series_id, n_missing=int(n_missing))
            series = series.interpolate(method="time").fillna(method="bfill").fillna(method="ffill")

        # Detect and handle outliers using IQR method
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 3.0 * iqr
        upper_fence = q3 + 3.0 * iqr
        n_outliers = ((series < lower_fence) | (series > upper_fence)).sum()
        if n_outliers > 0:
            logger.warning("outliers_detected", series_id=series_id, n_outliers=int(n_outliers))
            series = series.clip(lower=lower_fence, upper=upper_fence)

        logger.info(
            "data_ingested",
            series_id=series_id,
            n_samples=len(series),
            date_range=(str(series.index.min()), str(series.index.max())),
        )

        return series

    def engineer_features(self, series: pd.Series) -> pd.DataFrame:
        """Generate feature matrix from raw time series.

        Args:
            series: Raw time series data.

        Returns:
            Feature DataFrame for ML training.
        """
        features = create_all_features(
            series,
            lags=[1, 7, 14, 30, 90],
            rolling_windows=[7, 14, 30, 60],
            fourier_terms=3,
            trend_degree=2,
            drop_na=True,
        )
        logger.info(
            "features_engineered",
            n_features=features.shape[1],
            n_samples=features.shape[0],
        )
        return features

    def train_model(
        self,
        series: pd.Series,
        features: pd.DataFrame,
    ) -> tuple[ARIMAForecaster, EnsembleForecaster, dict[str, float]]:
        """Train ARIMA and ensemble models with cross-validation.

        Uses expanding-window CV to evaluate forecast quality and select
        the best model configuration.

        Args:
            series: Training time series.
            features: Feature matrix for ML component.

        Returns:
            Tuple of (fitted ARIMA model, fitted ensemble, CV metrics).
        """
        # Train ARIMA
        arima = ARIMAForecaster(
            seasonal=True,
            seasonal_period=12,
            max_p=2,
            max_q=2,
            confidence_level=self.config.confidence_level,
        )
        arima.fit(series)

        # Train ensemble (with ARIMA in-sample forecasts)
        ensemble = EnsembleForecaster(
            confidence_level=self.config.confidence_level,
            dynamic_weighting=True,
        )
        ensemble.fit(series)

        # Run backtesting
        backtest_result = ensemble.backtest(
            series,
            n_folds=self.config.cv_n_folds,
            horizon=min(self.config.forecast_horizon, 30),
            min_train_size=min(self.config.cv_min_train_size, len(series) // 2),
        )

        cv_metrics = {
            "mae": backtest_result.mae,
            "rmse": backtest_result.rmse,
            "mape": backtest_result.mape,
            "smape": backtest_result.smape,
            "bias": backtest_result.forecast_bias,
            "n_folds": backtest_result.n_folds,
        }

        logger.info("model_training_complete", cv_metrics=cv_metrics)
        return arima, ensemble, cv_metrics

    def generate_forecast(
        self,
        arima: ARIMAForecaster,
        ensemble: EnsembleForecaster,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Generate forecasts from the trained models.

        Args:
            arima: Fitted ARIMA forecaster.
            ensemble: Fitted ensemble forecaster.
            series: Training series (for ML predictions).

        Returns:
            DataFrame with forecast, lower/upper bounds, and model source.
        """
        horizon = self.config.forecast_horizon

        # ARIMA forecast
        arima_result = arima.predict(steps=horizon)

        # Ensemble forecast (using ARIMA output)
        ensemble_result = ensemble.predict(
            steps=horizon,
            arima_forecast=arima_result.forecast,
        )

        forecast_df = pd.DataFrame(
            {
                "forecast": ensemble_result.forecast,
                "lower_bound": ensemble_result.lower_bound,
                "upper_bound": ensemble_result.upper_bound,
                "arima_forecast": arima_result.forecast.reindex(ensemble_result.forecast.index),
            }
        )

        # Add model weights metadata
        for model_name, weight in ensemble_result.model_weights.items():
            forecast_df[f"weight_{model_name}"] = weight

        logger.info(
            "forecast_generated",
            horizon=horizon,
            forecast_mean=round(float(ensemble_result.forecast.mean()), 2),
        )

        return forecast_df

    def store_results(
        self,
        series_id: str,
        pipeline_id: str,
        forecast_df: pd.DataFrame,
        metrics: dict[str, float],
    ) -> None:
        """Persist forecast and metrics to S3.

        Args:
            series_id: Time series identifier.
            pipeline_id: Pipeline run identifier.
            forecast_df: Forecast DataFrame.
            metrics: Training/validation metrics.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        forecast_key = (
            f"{self.config.s3_output_prefix}{series_id}/{pipeline_id}/forecast_{timestamp}.csv"
        )
        self._store.write_forecast(forecast_key, forecast_df)

        metrics_key = (
            f"{self.config.s3_output_prefix}{series_id}/{pipeline_id}/metrics_{timestamp}.json"
        )
        self._store.write_metrics(metrics_key, metrics)

    def check_alerts(
        self,
        series_id: str,
        metrics: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Check forecast metrics against alert thresholds.

        Args:
            series_id: Time series identifier.
            metrics: Forecast accuracy metrics.

        Returns:
            List of alert dictionaries.
        """
        alerts: list[dict[str, Any]] = []

        mape = metrics.get("mape", 0.0)
        if mape > self.config.mape_alert_threshold:
            alerts.append(
                {
                    "type": "accuracy_degradation",
                    "series_id": series_id,
                    "metric": "mape",
                    "value": mape,
                    "threshold": self.config.mape_alert_threshold,
                    "severity": "warning"
                    if mape < 2 * self.config.mape_alert_threshold
                    else "critical",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        bias = abs(metrics.get("bias", 0.0))
        if bias > self.config.bias_alert_threshold:
            alerts.append(
                {
                    "type": "forecast_bias",
                    "series_id": series_id,
                    "metric": "absolute_bias",
                    "value": bias,
                    "threshold": self.config.bias_alert_threshold,
                    "severity": "warning",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        if alerts:
            logger.warning("forecast_alerts_triggered", series_id=series_id, n_alerts=len(alerts))

        return alerts

    def run(self, series_id: str, pipeline_id: str | None = None) -> PipelineResult:
        """Execute the full forecasting pipeline.

        Runs all stages: ingest -> features -> train -> forecast -> store -> alert.

        Args:
            series_id: Identifier for the target time series.
            pipeline_id: Optional run ID. Auto-generated if not provided.

        Returns:
            PipelineResult with forecast, metrics, and any alerts.
        """
        if pipeline_id is None:
            pipeline_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{series_id}"

        started_at = datetime.now(UTC).isoformat()

        logger.info("pipeline_started", pipeline_id=pipeline_id, series_id=series_id)

        try:
            # Stage 1: Ingest
            self._update_stage(PipelineStage.INGEST, pipeline_id)
            series = self.ingest_data(series_id)

            # Stage 2: Feature Engineering
            self._update_stage(PipelineStage.FEATURE_ENGINEERING, pipeline_id)
            features = self.engineer_features(series)

            # Stage 3: Training
            self._update_stage(PipelineStage.TRAINING, pipeline_id)
            arima, ensemble, cv_metrics = self.train_model(series, features)

            # Stage 4: Forecasting
            self._update_stage(PipelineStage.FORECASTING, pipeline_id)
            forecast_df = self.generate_forecast(arima, ensemble, series)

            # Stage 5: Storage
            self._update_stage(PipelineStage.STORAGE, pipeline_id)
            self.store_results(series_id, pipeline_id, forecast_df, cv_metrics)

            # Stage 6: Alerting
            self._update_stage(PipelineStage.ALERTING, pipeline_id)
            alerts = self.check_alerts(series_id, cv_metrics)

            self._update_stage(PipelineStage.COMPLETE, pipeline_id)

            result = PipelineResult(
                pipeline_id=pipeline_id,
                series_id=series_id,
                stage=PipelineStage.COMPLETE,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                forecast=forecast_df,
                metrics=cv_metrics,
                alerts=alerts,
                metadata={
                    "n_training_samples": len(series),
                    "n_features": features.shape[1],
                    "forecast_horizon": self.config.forecast_horizon,
                },
            )

            logger.info(
                "pipeline_complete",
                pipeline_id=pipeline_id,
                duration_seconds=round(
                    (
                        datetime.fromisoformat(result.completed_at)
                        - datetime.fromisoformat(started_at)
                    ).total_seconds(),
                    2,
                ),
            )

            return result

        except Exception as exc:
            self._update_stage(PipelineStage.FAILED, pipeline_id)
            logger.error(
                "pipeline_failed",
                pipeline_id=pipeline_id,
                stage=self._current_stage.value,
                error=str(exc),
            )

            return PipelineResult(
                pipeline_id=pipeline_id,
                series_id=series_id,
                stage=PipelineStage.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                error=str(exc),
            )

    def run_batch(self, series_ids: list[str]) -> list[PipelineResult]:
        """Run the pipeline for multiple time series.

        Args:
            series_ids: List of series identifiers to process.

        Returns:
            List of PipelineResults, one per series.
        """
        results = []
        for series_id in series_ids:
            result = self.run(series_id)
            results.append(result)

        n_success = sum(1 for r in results if r.stage == PipelineStage.COMPLETE)
        n_failed = sum(1 for r in results if r.stage == PipelineStage.FAILED)

        logger.info(
            "batch_pipeline_complete",
            total=len(series_ids),
            success=n_success,
            failed=n_failed,
        )

        return results
