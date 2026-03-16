"""Performance monitoring with Prometheus metrics and latency tracking.

Tracks p50/p95/p99 latency, throughput, model performance,
and SLA violation alerting for the prediction service.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = structlog.get_logger(__name__)

# --- Prometheus Metrics ---

REQUEST_LATENCY = Histogram(
    "rxpredict_request_latency_seconds",
    "Request latency in seconds",
    labelnames=["endpoint", "method"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

INFERENCE_LATENCY = Histogram(
    "rxpredict_inference_latency_seconds",
    "Model inference latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
)

REQUEST_COUNT = Counter(
    "rxpredict_requests_total",
    "Total request count",
    labelnames=["endpoint", "method", "status"],
)

PREDICTION_COUNT = Counter(
    "rxpredict_predictions_total",
    "Total predictions made",
    labelnames=["predicted_class", "risk_level"],
)

CACHE_OPERATIONS = Counter(
    "rxpredict_cache_operations_total",
    "Cache operations",
    labelnames=["operation", "result"],
)

ERROR_COUNT = Counter(
    "rxpredict_errors_total",
    "Total errors",
    labelnames=["error_type"],
)

MODEL_LOAD_TIME = Gauge(
    "rxpredict_model_load_time_seconds",
    "Time taken to load the model",
)

ACTIVE_REQUESTS = Gauge(
    "rxpredict_active_requests",
    "Number of currently active requests",
)

SLA_VIOLATIONS = Counter(
    "rxpredict_sla_violations_total",
    "Number of requests that violated the SLA",
    labelnames=["sla_type"],
)

INFERENCE_LATENCY_SUMMARY = Summary(
    "rxpredict_inference_latency_summary",
    "Inference latency summary with quantiles",
)

THROUGHPUT_GAUGE = Gauge(
    "rxpredict_throughput_rps",
    "Current requests per second",
)


# --- SLA Configuration ---


@dataclass
class SLAConfig:
    """SLA thresholds for the prediction service."""

    p50_latency_ms: float = 20.0
    p95_latency_ms: float = 50.0
    p99_latency_ms: float = 100.0
    max_error_rate: float = 0.01  # 1%
    min_cache_hit_rate: float = 0.3  # 30%
    max_model_staleness_hours: float = 168.0  # 1 week


@dataclass
class LatencyWindow:
    """Sliding window of latency measurements."""

    window_size: int = 1000
    _measurements: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _lock: Lock = field(default_factory=Lock)

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._measurements.append(latency_ms)

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self._measurements:
                return 0.0
            sorted_vals = sorted(self._measurements)
            idx = int(len(sorted_vals) * p / 100.0)
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._measurements)

    @property
    def mean(self) -> float:
        with self._lock:
            if not self._measurements:
                return 0.0
            return sum(self._measurements) / len(self._measurements)


class PerformanceMonitor:
    """Central performance monitoring for the prediction service.

    Tracks latency percentiles, throughput, SLA violations,
    and exposes Prometheus-compatible metrics.
    """

    def __init__(self, sla_config: SLAConfig | None = None) -> None:
        self.sla = sla_config or SLAConfig()
        self.request_latency = LatencyWindow(window_size=5000)
        self.inference_latency = LatencyWindow(window_size=5000)
        self._request_timestamps: deque[float] = deque(maxlen=10000)
        self._error_count = 0
        self._total_count = 0
        self._lock = Lock()
        self._model_load_time: float | None = None
        self._last_model_update: float | None = None

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record a request with its latency and status."""
        self.request_latency.record(latency_ms)

        # Prometheus metrics
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(latency_ms / 1000.0)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(status_code)).inc()

        with self._lock:
            self._total_count += 1
            self._request_timestamps.append(time.monotonic())
            if status_code >= 400:
                self._error_count += 1
                ERROR_COUNT.labels(error_type=f"http_{status_code}").inc()

        # Check SLA violations
        if latency_ms > self.sla.p99_latency_ms:
            SLA_VIOLATIONS.labels(sla_type="p99_latency").inc()
            logger.warning(
                "sla_violation_latency",
                endpoint=endpoint,
                latency_ms=round(latency_ms, 3),
                threshold_ms=self.sla.p99_latency_ms,
            )

    def record_inference(self, latency_ms: float) -> None:
        """Record model inference latency."""
        self.inference_latency.record(latency_ms)
        INFERENCE_LATENCY.observe(latency_ms / 1000.0)
        INFERENCE_LATENCY_SUMMARY.observe(latency_ms / 1000.0)

    def record_prediction(self, predicted_class: str, risk_level: str) -> None:
        """Record a prediction outcome for distribution tracking."""
        PREDICTION_COUNT.labels(predicted_class=predicted_class, risk_level=risk_level).inc()

    def record_cache_operation(self, operation: str, hit: bool) -> None:
        """Record a cache operation."""
        result = "hit" if hit else "miss"
        CACHE_OPERATIONS.labels(operation=operation, result=result).inc()

    def record_model_load(self, load_time_seconds: float) -> None:
        """Record model loading time."""
        self._model_load_time = load_time_seconds
        self._last_model_update = time.time()
        MODEL_LOAD_TIME.set(load_time_seconds)
        logger.info("model_load_recorded", load_time_s=round(load_time_seconds, 3))

    def get_throughput(self) -> float:
        """Calculate current requests per second (over last 60 seconds)."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            recent = sum(1 for ts in self._request_timestamps if ts > cutoff)
        rps = recent / 60.0
        THROUGHPUT_GAUGE.set(rps)
        return rps

    def get_error_rate(self) -> float:
        """Current error rate."""
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._error_count / self._total_count

    def check_sla_compliance(self) -> dict[str, Any]:
        """Check all SLA metrics and return compliance status."""
        p50 = self.request_latency.p50
        p95 = self.request_latency.p95
        p99 = self.request_latency.p99
        error_rate = self.get_error_rate()

        violations: list[str] = []
        if p50 > self.sla.p50_latency_ms:
            violations.append(f"p50 latency {p50:.1f}ms > {self.sla.p50_latency_ms}ms")
        if p95 > self.sla.p95_latency_ms:
            violations.append(f"p95 latency {p95:.1f}ms > {self.sla.p95_latency_ms}ms")
        if p99 > self.sla.p99_latency_ms:
            violations.append(f"p99 latency {p99:.1f}ms > {self.sla.p99_latency_ms}ms")
        if error_rate > self.sla.max_error_rate:
            violations.append(f"error rate {error_rate:.4f} > {self.sla.max_error_rate}")

        # Model staleness check
        model_stale = False
        if self._last_model_update is not None:
            hours_since_update = (time.time() - self._last_model_update) / 3600.0
            if hours_since_update > self.sla.max_model_staleness_hours:
                model_stale = True
                violations.append(
                    f"model stale: {hours_since_update:.1f}h > {self.sla.max_model_staleness_hours}h"
                )

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "metrics": {
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "error_rate": round(error_rate, 6),
                "throughput_rps": round(self.get_throughput(), 2),
                "model_stale": model_stale,
                "total_requests": self.request_latency.count,
            },
        }

    def get_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        return {
            "request_latency": {
                "p50_ms": round(self.request_latency.p50, 3),
                "p95_ms": round(self.request_latency.p95, 3),
                "p99_ms": round(self.request_latency.p99, 3),
                "mean_ms": round(self.request_latency.mean, 3),
                "count": self.request_latency.count,
            },
            "inference_latency": {
                "p50_ms": round(self.inference_latency.p50, 3),
                "p95_ms": round(self.inference_latency.p95, 3),
                "p99_ms": round(self.inference_latency.p99, 3),
                "mean_ms": round(self.inference_latency.mean, 3),
                "count": self.inference_latency.count,
            },
            "throughput_rps": round(self.get_throughput(), 2),
            "error_rate": round(self.get_error_rate(), 6),
            "sla_compliance": self.check_sla_compliance(),
            "model_load_time_s": (
                round(self._model_load_time, 3) if self._model_load_time else None
            ),
        }
