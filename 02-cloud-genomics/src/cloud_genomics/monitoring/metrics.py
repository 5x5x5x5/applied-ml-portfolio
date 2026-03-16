"""DataDog custom metrics for the CloudGenomics service.

Tracks classification latency, variant distribution, model confidence,
throughput, error rates, and operational health metrics. Provides both
real-time monitoring and historical trend analysis.

Metrics are emitted via the DataDog DogStatsD protocol and the
datadog Python library.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import DataDog libraries; fall back gracefully
try:
    from datadog import DogStatsd
    from datadog import initialize as dd_initialize

    _DD_AVAILABLE = True
except ImportError:
    _DD_AVAILABLE = False
    logger.info("DataDog library not available; metrics will be logged locally only")

try:
    from ddtrace import tracer as dd_tracer

    _DDTRACE_AVAILABLE = True
except ImportError:
    _DDTRACE_AVAILABLE = False


@dataclass
class MetricPoint:
    """A single metric data point for local buffering."""

    name: str
    value: float
    tags: list[str]
    timestamp: float
    metric_type: str  # gauge, counter, histogram, distribution


@dataclass
class MetricsConfig:
    """Configuration for the DataDog metrics collector."""

    api_key: str = ""
    app_key: str = ""
    statsd_host: str = "localhost"
    statsd_port: int = 8125
    service_name: str = "cloud-genomics"
    environment: str = "development"
    version: str = "1.0.0"
    flush_interval_seconds: int = 10
    enable_runtime_metrics: bool = True
    custom_tags: list[str] = field(default_factory=list)

    @property
    def default_tags(self) -> list[str]:
        """Generate default tags applied to all metrics."""
        tags = [
            f"service:{self.service_name}",
            f"env:{self.environment}",
            f"version:{self.version}",
        ]
        tags.extend(self.custom_tags)
        return tags


class MetricsCollector:
    """DataDog custom metrics collector for CloudGenomics.

    Emits metrics via DogStatsD for real-time dashboards and alerting.
    Includes fallback to local buffering when DataDog is unavailable.
    """

    # Metric name constants
    PREFIX = "cloudgenomics"

    # Classification metrics
    CLASSIFICATION_LATENCY = f"{PREFIX}.classification.latency"
    CLASSIFICATION_CONFIDENCE = f"{PREFIX}.classification.confidence"
    CLASSIFICATION_COUNT = f"{PREFIX}.classification.count"
    CLASSIFICATION_DISTRIBUTION = f"{PREFIX}.classification.distribution"

    # Throughput metrics
    VARIANTS_PROCESSED = f"{PREFIX}.variants.processed"
    VARIANTS_FILTERED = f"{PREFIX}.variants.filtered"
    VCF_PROCESSING_LATENCY = f"{PREFIX}.vcf.processing_latency"
    VCF_FILE_SIZE = f"{PREFIX}.vcf.file_size_bytes"

    # Model metrics
    MODEL_ACCURACY = f"{PREFIX}.model.accuracy"
    MODEL_PREDICTION_COUNT = f"{PREFIX}.model.prediction_count"
    MODEL_FEATURE_IMPORTANCE = f"{PREFIX}.model.feature_importance"

    # Error metrics
    ERROR_COUNT = f"{PREFIX}.errors.count"
    ERROR_RATE = f"{PREFIX}.errors.rate"

    # API metrics
    REQUEST_LATENCY = f"{PREFIX}.api.request_latency"
    REQUEST_COUNT = f"{PREFIX}.api.request_count"
    ACTIVE_REQUESTS = f"{PREFIX}.api.active_requests"

    # Infrastructure metrics
    MEMORY_USAGE = f"{PREFIX}.infra.memory_usage_mb"
    CPU_USAGE = f"{PREFIX}.infra.cpu_usage_percent"

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self._config = config or MetricsConfig()
        self._statsd: Any = None
        self._local_buffer: list[MetricPoint] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the DataDog client."""
        if _DD_AVAILABLE and self._config.api_key:
            try:
                dd_initialize(
                    api_key=self._config.api_key,
                    app_key=self._config.app_key,
                    statsd_host=self._config.statsd_host,
                    statsd_port=self._config.statsd_port,
                )
                self._statsd = DogStatsd(
                    host=self._config.statsd_host,
                    port=self._config.statsd_port,
                    constant_tags=self._config.default_tags,
                )
                self._initialized = True
                logger.info(
                    "DataDog metrics initialized: host=%s port=%d",
                    self._config.statsd_host,
                    self._config.statsd_port,
                )
            except Exception:
                logger.exception("Failed to initialize DataDog metrics client")
                self._initialized = False
        else:
            logger.info("DataDog not configured; using local metrics buffer")
            self._initialized = False

    def _emit(
        self,
        metric_type: str,
        name: str,
        value: float,
        tags: list[str] | None = None,
    ) -> None:
        """Emit a metric point to DataDog or local buffer.

        Args:
            metric_type: One of 'gauge', 'counter', 'histogram', 'distribution'.
            name: Metric name.
            value: Metric value.
            tags: Additional tags for this metric point.
        """
        all_tags = list(self._config.default_tags)
        if tags:
            all_tags.extend(tags)

        if self._initialized and self._statsd:
            try:
                if metric_type == "gauge":
                    self._statsd.gauge(name, value, tags=all_tags)
                elif metric_type == "counter":
                    self._statsd.increment(name, value, tags=all_tags)
                elif metric_type == "histogram":
                    self._statsd.histogram(name, value, tags=all_tags)
                elif metric_type == "distribution":
                    self._statsd.distribution(name, value, tags=all_tags)
            except Exception:
                logger.exception("Failed to emit metric %s", name)

        # Always buffer locally for debugging/testing
        point = MetricPoint(
            name=name,
            value=value,
            tags=all_tags,
            timestamp=time.time(),
            metric_type=metric_type,
        )
        self._local_buffer.append(point)

        # Keep buffer bounded
        if len(self._local_buffer) > 10000:
            self._local_buffer = self._local_buffer[-5000:]

    # -----------------------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------------------
    def record_classification(
        self,
        variant_class: str,
        confidence: float,
        latency_seconds: float,
        variant_type: str = "SNV",
    ) -> None:
        """Record a variant classification event.

        Args:
            variant_class: Predicted ACMG class (benign, pathogenic, etc.).
            confidence: Model confidence score (0-1).
            latency_seconds: Time taken for classification.
            variant_type: Type of variant (SNV, insertion, deletion).
        """
        tags = [
            f"variant_class:{variant_class}",
            f"variant_type:{variant_type}",
        ]

        self._emit("histogram", self.CLASSIFICATION_LATENCY, latency_seconds, tags)
        self._emit("histogram", self.CLASSIFICATION_CONFIDENCE, confidence, tags)
        self._emit("counter", self.CLASSIFICATION_COUNT, 1, tags)
        self._emit("counter", self.CLASSIFICATION_DISTRIBUTION, 1, [f"class:{variant_class}"])

        self._counters["total_classifications"] += 1
        self._counters[f"class_{variant_class}"] += 1

        logger.debug(
            "metric classification class=%s confidence=%.3f latency=%.3fs",
            variant_class,
            confidence,
            latency_seconds,
        )

    # -----------------------------------------------------------------
    # VCF / throughput metrics
    # -----------------------------------------------------------------
    def record_vcf_processing(
        self,
        total_variants: int,
        passed_variants: int,
        latency_seconds: float,
        file_size_bytes: int = 0,
    ) -> None:
        """Record VCF file processing metrics.

        Args:
            total_variants: Total variants in the file.
            passed_variants: Variants that passed quality filters.
            latency_seconds: Total processing time.
            file_size_bytes: Size of the VCF file.
        """
        filtered = total_variants - passed_variants

        self._emit("counter", self.VARIANTS_PROCESSED, total_variants)
        self._emit("counter", self.VARIANTS_FILTERED, filtered)
        self._emit("histogram", self.VCF_PROCESSING_LATENCY, latency_seconds)

        if file_size_bytes > 0:
            self._emit("histogram", self.VCF_FILE_SIZE, file_size_bytes)

        # Throughput: variants per second
        if latency_seconds > 0:
            throughput = total_variants / latency_seconds
            self._emit("gauge", f"{self.PREFIX}.vcf.throughput_vps", throughput)

        self._counters["total_vcf_files"] += 1
        self._counters["total_variants_processed"] += total_variants

        logger.info(
            "metric vcf_processing total=%d passed=%d filtered=%d latency=%.3fs",
            total_variants,
            passed_variants,
            filtered,
            latency_seconds,
        )

    # -----------------------------------------------------------------
    # Model performance metrics
    # -----------------------------------------------------------------
    def record_model_metrics(
        self,
        accuracy: float,
        cross_val_mean: float,
        cross_val_std: float,
        feature_importances: dict[str, float] | None = None,
    ) -> None:
        """Record model training/evaluation metrics.

        Args:
            accuracy: Model training accuracy.
            cross_val_mean: Cross-validation mean accuracy.
            cross_val_std: Cross-validation standard deviation.
            feature_importances: Top feature importance scores.
        """
        self._emit("gauge", self.MODEL_ACCURACY, accuracy)
        self._emit("gauge", f"{self.PREFIX}.model.cv_mean", cross_val_mean)
        self._emit("gauge", f"{self.PREFIX}.model.cv_std", cross_val_std)

        if feature_importances:
            for feature_name, importance in list(feature_importances.items())[:10]:
                self._emit(
                    "gauge",
                    self.MODEL_FEATURE_IMPORTANCE,
                    importance,
                    [f"feature:{feature_name}"],
                )

    # -----------------------------------------------------------------
    # Error metrics
    # -----------------------------------------------------------------
    def increment_error_count(
        self,
        error_type: str,
        count: int = 1,
    ) -> None:
        """Increment the error counter.

        Args:
            error_type: Category of error (e.g., classification_error, vcf_parse_error).
            count: Number of errors to record.
        """
        tags = [f"error_type:{error_type}"]
        self._emit("counter", self.ERROR_COUNT, count, tags)
        self._counters[f"error_{error_type}"] += count

        logger.warning("metric error type=%s count=%d", error_type, count)

    # -----------------------------------------------------------------
    # API request metrics
    # -----------------------------------------------------------------
    def record_request_latency(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record API request latency and count.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            status_code: Response status code.
            duration_seconds: Request duration.
        """
        tags = [
            f"endpoint:{endpoint}",
            f"method:{method}",
            f"status_code:{status_code}",
        ]

        self._emit("histogram", self.REQUEST_LATENCY, duration_seconds, tags)
        self._emit("counter", self.REQUEST_COUNT, 1, tags)

        self._counters["total_requests"] += 1
        if status_code >= 400:
            self._counters["error_requests"] += 1

    # -----------------------------------------------------------------
    # Infrastructure metrics
    # -----------------------------------------------------------------
    def record_infrastructure_metrics(
        self,
        memory_usage_mb: float,
        cpu_usage_percent: float,
    ) -> None:
        """Record infrastructure resource utilization.

        Args:
            memory_usage_mb: Current memory usage in megabytes.
            cpu_usage_percent: Current CPU utilization percentage.
        """
        self._emit("gauge", self.MEMORY_USAGE, memory_usage_mb)
        self._emit("gauge", self.CPU_USAGE, cpu_usage_percent)

    # -----------------------------------------------------------------
    # Custom dashboard helpers
    # -----------------------------------------------------------------
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected metrics for debugging.

        Returns:
            Dictionary with counter values and buffer statistics.
        """
        return {
            "counters": dict(self._counters),
            "buffer_size": len(self._local_buffer),
            "datadog_connected": self._initialized,
            "recent_metrics": [
                {
                    "name": p.name,
                    "value": p.value,
                    "type": p.metric_type,
                    "tags": p.tags,
                }
                for p in self._local_buffer[-10:]
            ],
        }

    def get_classification_distribution(self) -> dict[str, int]:
        """Get the distribution of classifications seen so far.

        Returns:
            Dictionary mapping class names to counts.
        """
        return {
            key.replace("class_", ""): value
            for key, value in self._counters.items()
            if key.startswith("class_")
        }

    def flush(self) -> None:
        """Flush buffered metrics to DataDog.

        Called automatically by the DogStatsD client, but can be
        triggered manually for testing.
        """
        if self._initialized and self._statsd:
            try:
                self._statsd.flush()
                logger.debug("Flushed metrics buffer to DataDog")
            except Exception:
                logger.exception("Failed to flush metrics")


def create_metrics_collector(
    config: MetricsConfig | None = None,
) -> MetricsCollector:
    """Factory function to create a configured MetricsCollector.

    Args:
        config: Optional metrics configuration. If None, uses defaults
                suitable for local development.

    Returns:
        Configured MetricsCollector instance.
    """
    if config is None:
        config = MetricsConfig()

    collector = MetricsCollector(config=config)

    if _DDTRACE_AVAILABLE:
        logger.info("DataDog tracing (ddtrace) is available and active")

    return collector


def build_datadog_dashboard() -> dict[str, Any]:
    """Generate a DataDog dashboard definition for CloudGenomics.

    Returns a dashboard JSON structure that can be imported into DataDog
    for monitoring the service.

    Returns:
        Dashboard definition dictionary.
    """
    prefix = MetricsCollector.PREFIX

    return {
        "title": "CloudGenomics - Variant Classification Service",
        "description": "Real-time monitoring of the genomic variant classification pipeline",
        "layout_type": "ordered",
        "widgets": [
            {
                "definition": {
                    "title": "Classification Throughput",
                    "type": "timeseries",
                    "requests": [
                        {
                            "q": f"sum:{prefix}.classification.count{{*}}.as_rate()",
                            "display_type": "bars",
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "Classification Latency (p95)",
                    "type": "timeseries",
                    "requests": [
                        {
                            "q": f"p95:{prefix}.classification.latency{{*}}",
                            "display_type": "line",
                        }
                    ],
                    "yaxis": {"label": "seconds"},
                },
            },
            {
                "definition": {
                    "title": "Model Confidence Distribution",
                    "type": "distribution",
                    "requests": [
                        {
                            "q": f"{prefix}.classification.confidence{{*}}",
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "Variant Classification Distribution",
                    "type": "toplist",
                    "requests": [
                        {
                            "q": f"sum:{prefix}.classification.distribution{{*}} by {{class}}",
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "Error Rate",
                    "type": "timeseries",
                    "requests": [
                        {
                            "q": f"sum:{prefix}.errors.count{{*}}.as_rate()",
                            "display_type": "bars",
                            "style": {"palette": "warm"},
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "API Request Latency by Endpoint",
                    "type": "timeseries",
                    "requests": [
                        {
                            "q": f"avg:{prefix}.api.request_latency{{*}} by {{endpoint}}",
                            "display_type": "line",
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "VCF Processing Throughput",
                    "type": "query_value",
                    "requests": [
                        {
                            "q": f"avg:{prefix}.vcf.throughput_vps{{*}}",
                        }
                    ],
                    "precision": 1,
                },
            },
            {
                "definition": {
                    "title": "Model Accuracy",
                    "type": "query_value",
                    "requests": [
                        {
                            "q": f"avg:{prefix}.model.accuracy{{*}}",
                        }
                    ],
                    "precision": 4,
                },
            },
            {
                "definition": {
                    "title": "Top Feature Importances",
                    "type": "toplist",
                    "requests": [
                        {
                            "q": (f"avg:{prefix}.model.feature_importance{{*}} by {{feature}}"),
                        }
                    ],
                },
            },
            {
                "definition": {
                    "title": "Infrastructure - Memory & CPU",
                    "type": "timeseries",
                    "requests": [
                        {
                            "q": f"avg:{prefix}.infra.memory_usage_mb{{*}}",
                            "display_type": "line",
                        },
                        {
                            "q": f"avg:{prefix}.infra.cpu_usage_percent{{*}}",
                            "display_type": "line",
                        },
                    ],
                },
            },
        ],
        "notify_list": [],
        "template_variables": [
            {"name": "env", "default": "production", "prefix": "env"},
            {"name": "service", "default": "cloud-genomics", "prefix": "service"},
        ],
    }
