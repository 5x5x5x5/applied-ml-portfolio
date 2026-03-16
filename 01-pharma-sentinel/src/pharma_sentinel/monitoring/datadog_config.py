"""DataDog integration for PharmaSentinel monitoring.

Provides custom metrics collection, APM setup, structured logging
configuration, and dashboard-as-code definitions.
"""

from __future__ import annotations

import logging
from typing import Any

from pharma_sentinel.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Metrics Collector
# ─────────────────────────────────────────────────────────────────────


class MetricsCollector:
    """Collects and ships custom metrics to DataDog via DogStatsD.

    Tracks prediction latency, model accuracy, pipeline throughput,
    error rates, and business metrics for the PharmaSentinel service.

    Attributes:
        settings: Application configuration.
        statsd: DogStatsD client for metric submission.
        enabled: Whether DataDog metrics are enabled.
    """

    METRIC_PREFIX = "pharma_sentinel"

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            settings: Application settings. Uses default if None.
        """
        self.settings = settings or get_settings()
        self.statsd: Any = None
        self.enabled = False
        self._default_tags: list[str] = [
            f"env:{self.settings.datadog.env}",
            f"service:{self.settings.datadog.service}",
            f"version:{self.settings.datadog.version}",
        ]

    def initialize(self) -> None:
        """Initialize the DogStatsD client and APM tracing.

        Fails gracefully if DataDog agent is not available.
        """
        if not self.settings.datadog.api_key:
            logger.info("DataDog API key not configured; metrics collection disabled")
            return

        try:
            from datadog import DogStatsd, initialize

            initialize(
                api_key=self.settings.datadog.api_key,
                app_key=self.settings.datadog.app_key,
            )

            self.statsd = DogStatsd(
                host=self.settings.datadog.statsd_host,
                port=self.settings.datadog.statsd_port,
                constant_tags=self._default_tags,
                namespace=self.METRIC_PREFIX,
            )

            self.enabled = True
            logger.info(
                "DataDog metrics initialized: host=%s, port=%d",
                self.settings.datadog.statsd_host,
                self.settings.datadog.statsd_port,
            )

        except ImportError:
            logger.warning("datadog package not installed; metrics disabled")
        except Exception:
            logger.exception("Failed to initialize DataDog metrics")

        # Initialize APM tracing
        if self.settings.datadog.trace_enabled:
            self._setup_apm_tracing()

    def _setup_apm_tracing(self) -> None:
        """Configure DataDog APM tracing with ddtrace."""
        try:
            from ddtrace import config as dd_config
            from ddtrace import patch_all, tracer

            tracer.configure(
                hostname=self.settings.datadog.agent_host,
                port=self.settings.datadog.agent_port,
            )

            dd_config.env = self.settings.datadog.env
            dd_config.service = self.settings.datadog.service
            dd_config.version = self.settings.datadog.version

            # Auto-instrument common libraries
            patch_all(
                fastapi=True,
                requests=True,
                botocore=True,
                redis=True,
                logging=self.settings.datadog.logs_injection,
            )

            logger.info("DataDog APM tracing configured")

        except ImportError:
            logger.warning("ddtrace package not installed; APM tracing disabled")
        except Exception:
            logger.exception("Failed to setup APM tracing")

    def record_prediction(
        self,
        severity: str,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Record a single prediction event with metrics.

        Args:
            severity: Predicted severity level.
            confidence: Prediction confidence score.
            latency_ms: Time taken for prediction in milliseconds.
        """
        tags = [f"severity:{severity}"]

        if self.enabled and self.statsd:
            self.statsd.increment("predictions.count", tags=tags)
            self.statsd.histogram("predictions.latency_ms", latency_ms, tags=tags)
            self.statsd.histogram("predictions.confidence", confidence, tags=tags)
            self.statsd.gauge(
                "predictions.confidence_avg",
                confidence,
                tags=tags,
            )

            # Track low-confidence predictions separately
            if confidence < 0.5:
                self.statsd.increment("predictions.low_confidence", tags=tags)

        logger.debug(
            "Prediction metric: severity=%s confidence=%.3f latency=%.1fms",
            severity,
            confidence,
            latency_ms,
        )

    def record_batch_prediction(
        self,
        count: int,
        latency_ms: float,
    ) -> None:
        """Record batch prediction metrics.

        Args:
            count: Number of predictions in batch.
            latency_ms: Total batch processing time in milliseconds.
        """
        if self.enabled and self.statsd:
            self.statsd.increment("batch_predictions.count")
            self.statsd.histogram("batch_predictions.size", count)
            self.statsd.histogram("batch_predictions.latency_ms", latency_ms)
            self.statsd.histogram(
                "batch_predictions.per_item_latency_ms",
                latency_ms / max(count, 1),
            )

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record HTTP request metrics.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path.
            status_code: HTTP response status code.
            latency_ms: Request processing time in milliseconds.
        """
        tags = [
            f"method:{method}",
            f"path:{path}",
            f"status:{status_code}",
            f"status_class:{status_code // 100}xx",
        ]

        if self.enabled and self.statsd:
            self.statsd.increment("http.requests.count", tags=tags)
            self.statsd.histogram("http.requests.latency_ms", latency_ms, tags=tags)

            if status_code >= 500:
                self.statsd.increment("http.requests.errors.5xx", tags=tags)
            elif status_code >= 400:
                self.statsd.increment("http.requests.errors.4xx", tags=tags)

    def record_pipeline_event(
        self,
        event_type: str,
        records_processed: int,
        duration_seconds: float,
        errors: int = 0,
    ) -> None:
        """Record data pipeline processing metrics.

        Args:
            event_type: Type of pipeline event (ingest, classify, etc.).
            records_processed: Number of records processed.
            duration_seconds: Total processing duration.
            errors: Number of errors during processing.
        """
        tags = [f"event_type:{event_type}"]

        if self.enabled and self.statsd:
            self.statsd.increment("pipeline.runs", tags=tags)
            self.statsd.gauge("pipeline.records_processed", records_processed, tags=tags)
            self.statsd.histogram("pipeline.duration_seconds", duration_seconds, tags=tags)
            self.statsd.gauge(
                "pipeline.throughput_rps",
                records_processed / max(duration_seconds, 0.001),
                tags=tags,
            )

            if errors > 0:
                self.statsd.increment("pipeline.errors", errors, tags=tags)

    def record_model_metric(
        self,
        metric_name: str,
        value: float,
        tags: list[str] | None = None,
    ) -> None:
        """Record a custom model performance metric.

        Args:
            metric_name: Name of the metric (e.g., 'accuracy', 'f1_score').
            value: Metric value.
            tags: Additional tags for the metric.
        """
        all_tags = tags or []

        if self.enabled and self.statsd:
            self.statsd.gauge(f"model.{metric_name}", value, tags=all_tags)

    def flush(self) -> None:
        """Flush any buffered metrics to DataDog."""
        if self.enabled and self.statsd:
            try:
                self.statsd.flush()
                logger.debug("Metrics flushed to DataDog")
            except Exception:
                logger.exception("Failed to flush metrics")

    def health_check(self) -> dict[str, Any]:
        """Check DataDog connectivity health.

        Returns:
            Dictionary with health status information.
        """
        return {
            "enabled": self.enabled,
            "agent_host": self.settings.datadog.agent_host,
            "agent_port": self.settings.datadog.agent_port,
            "trace_enabled": self.settings.datadog.trace_enabled,
            "default_tags": self._default_tags,
        }


# ─────────────────────────────────────────────────────────────────────
# Structured logging configuration
# ─────────────────────────────────────────────────────────────────────


def configure_logging(settings: AppSettings | None = None) -> None:
    """Configure structured logging with DataDog integration.

    Sets up Python logging with structured format compatible with
    DataDog log ingestion. Injects trace IDs when APM is enabled.

    Args:
        settings: Application settings. Uses default if None.
    """
    settings = settings or get_settings()

    log_format = (
        '{"timestamp":"%(asctime)s",'
        '"level":"%(levelname)s",'
        '"logger":"%(name)s",'
        '"message":"%(message)s",'
        f'"service":"{settings.datadog.service}",'
        f'"env":"{settings.datadog.env}",'
        f'"version":"{settings.datadog.version}"'
        "}"
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    # Reduce noise from third-party loggers
    for noisy_logger in ("botocore", "urllib3", "s3transfer", "boto3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.info("Logging configured: level=%s, env=%s", settings.log_level, settings.datadog.env)


# ─────────────────────────────────────────────────────────────────────
# Dashboard as code (JSON definition)
# ─────────────────────────────────────────────────────────────────────


def get_dashboard_definition() -> dict[str, Any]:
    """Generate a DataDog dashboard definition as code.

    Returns:
        Dictionary representing the full DataDog dashboard configuration
        for PharmaSentinel monitoring.
    """
    return {
        "title": "PharmaSentinel - Adverse Event Detection Pipeline",
        "description": "Monitoring dashboard for the PharmaSentinel drug adverse event detection pipeline",
        "layout_type": "ordered",
        "is_read_only": False,
        "template_variables": [
            {"name": "env", "prefix": "env", "default": "production"},
            {"name": "service", "prefix": "service", "default": "pharma-sentinel"},
        ],
        "widgets": [
            # Row 1: Overview
            {
                "definition": {
                    "title": "Service Overview",
                    "type": "group",
                    "layout_type": "ordered",
                    "widgets": [
                        {
                            "definition": {
                                "title": "Predictions / Second",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "sum:pharma_sentinel.predictions.count{$env,$service}.as_rate()",
                                        "display_type": "line",
                                        "style": {"palette": "dog_classic"},
                                    }
                                ],
                            },
                        },
                        {
                            "definition": {
                                "title": "Prediction Latency (p50/p95/p99)",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.predictions.latency_ms.median{$env,$service}",
                                        "display_type": "line",
                                        "style": {"palette": "cool"},
                                    },
                                    {
                                        "q": "avg:pharma_sentinel.predictions.latency_ms.95percentile{$env,$service}",
                                        "display_type": "line",
                                        "style": {"palette": "warm"},
                                    },
                                    {
                                        "q": "avg:pharma_sentinel.predictions.latency_ms.99percentile{$env,$service}",
                                        "display_type": "line",
                                        "style": {"palette": "orange"},
                                    },
                                ],
                                "yaxis": {"label": "ms"},
                            },
                        },
                    ],
                },
            },
            # Row 2: Predictions
            {
                "definition": {
                    "title": "Prediction Distribution",
                    "type": "group",
                    "layout_type": "ordered",
                    "widgets": [
                        {
                            "definition": {
                                "title": "Severity Distribution",
                                "type": "toplist",
                                "requests": [
                                    {
                                        "q": "sum:pharma_sentinel.predictions.count{$env,$service} by {severity}",
                                    }
                                ],
                            },
                        },
                        {
                            "definition": {
                                "title": "Confidence Score Distribution",
                                "type": "distribution",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.predictions.confidence{$env,$service}",
                                    }
                                ],
                            },
                        },
                        {
                            "definition": {
                                "title": "Low Confidence Predictions",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "sum:pharma_sentinel.predictions.low_confidence{$env,$service}.as_count()",
                                        "display_type": "bars",
                                        "style": {"palette": "red"},
                                    }
                                ],
                            },
                        },
                    ],
                },
            },
            # Row 3: Errors and pipeline
            {
                "definition": {
                    "title": "Error Rates & Pipeline Health",
                    "type": "group",
                    "layout_type": "ordered",
                    "widgets": [
                        {
                            "definition": {
                                "title": "HTTP Error Rate",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "sum:pharma_sentinel.http.requests.errors.5xx{$env,$service}.as_rate()",
                                        "display_type": "bars",
                                        "style": {"palette": "red"},
                                    },
                                    {
                                        "q": "sum:pharma_sentinel.http.requests.errors.4xx{$env,$service}.as_rate()",
                                        "display_type": "bars",
                                        "style": {"palette": "yellow"},
                                    },
                                ],
                            },
                        },
                        {
                            "definition": {
                                "title": "Pipeline Throughput (records/sec)",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.pipeline.throughput_rps{$env,$service}",
                                        "display_type": "line",
                                        "style": {"palette": "green"},
                                    }
                                ],
                            },
                        },
                        {
                            "definition": {
                                "title": "Pipeline Errors",
                                "type": "query_value",
                                "requests": [
                                    {
                                        "q": "sum:pharma_sentinel.pipeline.errors{$env,$service}",
                                    }
                                ],
                                "precision": 0,
                            },
                        },
                    ],
                },
            },
            # Row 4: Model performance
            {
                "definition": {
                    "title": "Model Performance",
                    "type": "group",
                    "layout_type": "ordered",
                    "widgets": [
                        {
                            "definition": {
                                "title": "Model Accuracy",
                                "type": "query_value",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.model.accuracy{$env,$service}",
                                    }
                                ],
                                "precision": 4,
                            },
                        },
                        {
                            "definition": {
                                "title": "Model F1 Score",
                                "type": "query_value",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.model.f1_score{$env,$service}",
                                    }
                                ],
                                "precision": 4,
                            },
                        },
                        {
                            "definition": {
                                "title": "Average Confidence by Severity",
                                "type": "timeseries",
                                "requests": [
                                    {
                                        "q": "avg:pharma_sentinel.predictions.confidence_avg{$env,$service} by {severity}",
                                        "display_type": "line",
                                    }
                                ],
                            },
                        },
                    ],
                },
            },
        ],
    }
