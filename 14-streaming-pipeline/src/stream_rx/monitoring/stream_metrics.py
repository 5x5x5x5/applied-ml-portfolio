"""
Stream monitoring with consumer lag tracking, throughput metrics, latency
measurement, error rate monitoring, and DataDog integration.

Provides both a pull-based metrics collector (for periodic scraping) and
a push-based integration with DataDog's StatsD/API for production alerting.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from stream_rx.config import MonitoringConfig, get_config
from stream_rx.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric data structures
# ---------------------------------------------------------------------------


@dataclass
class ThroughputSample:
    """A single throughput measurement."""

    timestamp: float
    count: int
    topic: str = ""
    consumer_group: str = ""


@dataclass
class LatencySample:
    """A single processing latency measurement."""

    timestamp: float
    latency_ms: float
    topic: str = ""


@dataclass
class ConsumerLagSnapshot:
    """Consumer lag for a single partition at a point in time."""

    timestamp: float
    topic: str
    partition: int
    consumer_group: str
    current_offset: int
    log_end_offset: int

    @property
    def lag(self) -> int:
        return max(0, self.log_end_offset - self.current_offset)


@dataclass
class ErrorSample:
    """Record of a processing error."""

    timestamp: float
    error_type: str
    topic: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class StreamMetricsCollector:
    """
    Collects and aggregates streaming pipeline metrics.

    Tracks:
    - Consumer lag per partition
    - Throughput (events/sec)
    - Processing latency (p50, p95, p99)
    - Error rates
    - DataDog metric submission

    Thread-safe for use across multiple consumer threads.
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        window_seconds: int = 300,
    ) -> None:
        self._config = config or get_config().monitoring
        self._window_seconds = window_seconds
        self._lock = threading.Lock()

        # Ring buffers for time-windowed metrics
        self._throughput_samples: deque[ThroughputSample] = deque(maxlen=10_000)
        self._latency_samples: deque[LatencySample] = deque(maxlen=50_000)
        self._error_samples: deque[ErrorSample] = deque(maxlen=5_000)

        # Latest lag snapshots per (topic, partition, group)
        self._lag_snapshots: dict[str, ConsumerLagSnapshot] = {}

        # Cumulative counters
        self._total_processed = 0
        self._total_errors = 0
        self._start_time = time.monotonic()

        # DataDog client (lazy init)
        self._dd_client: Any = None
        self._dd_initialized = False

        logger.info(
            "metrics_collector_initialized",
            window_seconds=window_seconds,
            datadog_enabled=bool(self._config.datadog_api_key),
        )

    # ------------------------------------------------------------------
    # Recording metrics
    # ------------------------------------------------------------------

    def record_throughput(self, count: int, topic: str = "", consumer_group: str = "") -> None:
        """Record a throughput measurement."""
        sample = ThroughputSample(
            timestamp=time.time(),
            count=count,
            topic=topic,
            consumer_group=consumer_group,
        )
        with self._lock:
            self._throughput_samples.append(sample)
            self._total_processed += count

    def record_latency(self, latency_ms: float, topic: str = "") -> None:
        """Record a processing latency measurement in milliseconds."""
        sample = LatencySample(
            timestamp=time.time(),
            latency_ms=latency_ms,
            topic=topic,
        )
        with self._lock:
            self._latency_samples.append(sample)

    def record_consumer_lag(
        self,
        topic: str,
        partition: int,
        consumer_group: str,
        current_offset: int,
        log_end_offset: int,
    ) -> None:
        """Record the consumer lag for a specific partition."""
        snapshot = ConsumerLagSnapshot(
            timestamp=time.time(),
            topic=topic,
            partition=partition,
            consumer_group=consumer_group,
            current_offset=current_offset,
            log_end_offset=log_end_offset,
        )
        key = f"{topic}:{partition}:{consumer_group}"
        with self._lock:
            self._lag_snapshots[key] = snapshot

        # Check thresholds
        lag = snapshot.lag
        if lag >= self._config.lag_critical_threshold:
            logger.error(
                "consumer_lag_critical",
                topic=topic,
                partition=partition,
                lag=lag,
                threshold=self._config.lag_critical_threshold,
            )
        elif lag >= self._config.lag_warning_threshold:
            logger.warning(
                "consumer_lag_warning",
                topic=topic,
                partition=partition,
                lag=lag,
                threshold=self._config.lag_warning_threshold,
            )

    def record_error(self, error_type: str, topic: str = "", message: str = "") -> None:
        """Record a processing error."""
        sample = ErrorSample(
            timestamp=time.time(),
            error_type=error_type,
            topic=topic,
            message=message,
        )
        with self._lock:
            self._error_samples.append(sample)
            self._total_errors += 1

    # ------------------------------------------------------------------
    # Querying metrics
    # ------------------------------------------------------------------

    def get_throughput(self, window_seconds: int | None = None) -> dict[str, Any]:
        """
        Get throughput metrics over a time window.

        Returns:
            Dict with events/sec, total count, and per-topic breakdown.
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        with self._lock:
            recent = [s for s in self._throughput_samples if s.timestamp >= cutoff]

        if not recent:
            return {
                "events_per_sec": 0.0,
                "total_in_window": 0,
                "window_seconds": window,
                "topics": {},
            }

        total = sum(s.count for s in recent)
        elapsed = max(recent[-1].timestamp - recent[0].timestamp, 1.0)
        eps = total / elapsed

        # Per-topic breakdown
        topics: dict[str, int] = {}
        for s in recent:
            topic_key = s.topic or "unknown"
            topics[topic_key] = topics.get(topic_key, 0) + s.count

        return {
            "events_per_sec": round(eps, 2),
            "total_in_window": total,
            "window_seconds": window,
            "topics": topics,
        }

    def get_latency_percentiles(self, window_seconds: int | None = None) -> dict[str, float]:
        """
        Get latency percentiles (p50, p95, p99) over a time window.

        Returns:
            Dict with percentile values in milliseconds.
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        with self._lock:
            recent = [s.latency_ms for s in self._latency_samples if s.timestamp >= cutoff]

        if not recent:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "count": 0}

        recent.sort()
        n = len(recent)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return recent[min(idx, n - 1)]

        return {
            "p50": round(percentile(50), 2),
            "p95": round(percentile(95), 2),
            "p99": round(percentile(99), 2),
            "avg": round(sum(recent) / n, 2),
            "count": n,
        }

    def get_consumer_lag(self) -> dict[str, Any]:
        """
        Get current consumer lag for all tracked partitions.

        Returns:
            Dict with per-partition lag and aggregate totals.
        """
        with self._lock:
            snapshots = dict(self._lag_snapshots)

        partitions: list[dict[str, Any]] = []
        total_lag = 0
        for key, snap in snapshots.items():
            lag = snap.lag
            total_lag += lag
            partitions.append(
                {
                    "topic": snap.topic,
                    "partition": snap.partition,
                    "consumer_group": snap.consumer_group,
                    "current_offset": snap.current_offset,
                    "log_end_offset": snap.log_end_offset,
                    "lag": lag,
                    "timestamp": datetime.fromtimestamp(snap.timestamp).isoformat(),
                }
            )

        return {
            "total_lag": total_lag,
            "partitions": partitions,
            "partition_count": len(partitions),
        }

    def get_error_rate(self, window_seconds: int | None = None) -> dict[str, Any]:
        """
        Get error rate over a time window.

        Returns:
            Dict with error count, rate, and breakdown by type.
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        with self._lock:
            recent_errors = [s for s in self._error_samples if s.timestamp >= cutoff]
            recent_throughput = [s for s in self._throughput_samples if s.timestamp >= cutoff]

        total_processed = sum(s.count for s in recent_throughput)
        error_count = len(recent_errors)
        error_rate = error_count / max(total_processed, 1)

        # Breakdown by type
        by_type: dict[str, int] = {}
        for e in recent_errors:
            by_type[e.error_type] = by_type.get(e.error_type, 0) + 1

        is_above_threshold = error_rate > self._config.error_rate_threshold
        if is_above_threshold:
            logger.error(
                "error_rate_exceeded",
                rate=round(error_rate, 4),
                threshold=self._config.error_rate_threshold,
                error_count=error_count,
            )

        return {
            "error_count": error_count,
            "total_processed": total_processed,
            "error_rate": round(error_rate, 6),
            "threshold": self._config.error_rate_threshold,
            "above_threshold": is_above_threshold,
            "by_type": by_type,
            "window_seconds": window,
        }

    # ------------------------------------------------------------------
    # DataDog integration
    # ------------------------------------------------------------------

    def _init_datadog(self) -> bool:
        """Initialize the DataDog client lazily."""
        if self._dd_initialized:
            return self._dd_client is not None

        if not self._config.datadog_api_key:
            self._dd_initialized = True
            return False

        try:
            from datadog import initialize, statsd

            initialize(
                api_key=self._config.datadog_api_key,
                app_key=self._config.datadog_app_key,
            )
            self._dd_client = statsd
            self._dd_initialized = True
            logger.info("datadog_initialized")
            return True
        except ImportError:
            logger.warning("datadog_package_not_installed")
            self._dd_initialized = True
            return False
        except Exception as exc:
            logger.error("datadog_init_failed", error=str(exc))
            self._dd_initialized = True
            return False

    def push_to_datadog(self) -> None:
        """Push current metrics snapshot to DataDog."""
        if not self._init_datadog() or self._dd_client is None:
            return

        prefix = self._config.metrics_prefix
        dd = self._dd_client

        try:
            # Throughput
            throughput = self.get_throughput()
            dd.gauge(f"{prefix}.throughput.events_per_sec", throughput["events_per_sec"])

            # Latency
            latency = self.get_latency_percentiles()
            dd.gauge(f"{prefix}.latency.p50", latency["p50"])
            dd.gauge(f"{prefix}.latency.p95", latency["p95"])
            dd.gauge(f"{prefix}.latency.p99", latency["p99"])

            # Consumer lag
            lag_data = self.get_consumer_lag()
            dd.gauge(f"{prefix}.consumer_lag.total", lag_data["total_lag"])
            for partition in lag_data["partitions"]:
                tags = [
                    f"topic:{partition['topic']}",
                    f"partition:{partition['partition']}",
                    f"consumer_group:{partition['consumer_group']}",
                ]
                dd.gauge(f"{prefix}.consumer_lag.partition", partition["lag"], tags=tags)

            # Errors
            errors = self.get_error_rate()
            dd.gauge(f"{prefix}.error_rate", errors["error_rate"])
            dd.increment(f"{prefix}.errors.total", errors["error_count"])

            logger.debug("datadog_metrics_pushed")
        except Exception as exc:
            logger.error("datadog_push_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Get a complete metrics summary."""
        uptime = time.monotonic() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "throughput": self.get_throughput(),
            "latency": self.get_latency_percentiles(),
            "consumer_lag": self.get_consumer_lag(),
            "error_rate": self.get_error_rate(),
        }


# ---------------------------------------------------------------------------
# Periodic metrics flusher
# ---------------------------------------------------------------------------


class MetricsFlusher:
    """
    Background thread that periodically flushes metrics to DataDog and logs.
    """

    def __init__(
        self,
        collector: StreamMetricsCollector,
        interval_sec: int | None = None,
    ) -> None:
        config = get_config().monitoring
        self._collector = collector
        self._interval = interval_sec or config.metrics_flush_interval_sec
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background metrics flusher."""
        self._running = True
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="metrics-flusher",
            daemon=True,
        )
        self._thread.start()
        logger.info("metrics_flusher_started", interval_sec=self._interval)

    def _flush_loop(self) -> None:
        """Main flush loop."""
        while self._running:
            try:
                self._collector.push_to_datadog()
                summary = self._collector.get_summary()
                logger.info(
                    "metrics_snapshot",
                    throughput_eps=summary["throughput"]["events_per_sec"],
                    latency_p99=summary["latency"]["p99"],
                    total_lag=summary["consumer_lag"]["total_lag"],
                    error_rate=summary["error_rate"]["error_rate"],
                )
            except Exception as exc:
                logger.error("metrics_flush_error", error=str(exc))

            time.sleep(self._interval)

    def stop(self) -> None:
        """Stop the background flusher."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("metrics_flusher_stopped")
