"""
Load testing script for the StreamRx pipeline.

Generates high-throughput prescription and adverse event traffic to stress-test
the pipeline. Supports configurable concurrency, duration, and throughput
targets. Reports real-time statistics including achieved throughput, latency,
and error rates.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from stream_rx.logging_setup import configure_logging, get_logger
from stream_rx.producers.adverse_event_producer import AdverseEventProducer
from stream_rx.producers.prescription_producer import PrescriptionProducer

logger = get_logger(__name__)


@dataclass
class LoadTestStats:
    """Thread-safe statistics collector for load testing."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_events: int = 0
    total_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    _interval_events: int = 0
    _interval_start: float = field(default_factory=time.monotonic)

    def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self.total_events += 1
            self._interval_events += 1
            self.latencies_ms.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self.total_errors += 1

    def get_interval_throughput(self) -> float:
        """Get throughput since last interval reset."""
        with self._lock:
            elapsed = max(time.monotonic() - self._interval_start, 0.001)
            eps = self._interval_events / elapsed
            self._interval_events = 0
            self._interval_start = time.monotonic()
            return eps

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(time.monotonic() - self.start_time, 0.001)
            latencies = self.latencies_ms[-10000:]  # last 10k for percentiles

            summary: dict[str, Any] = {
                "total_events": self.total_events,
                "total_errors": self.total_errors,
                "elapsed_seconds": round(elapsed, 1),
                "overall_eps": round(self.total_events / elapsed, 1),
                "error_rate": round(
                    self.total_errors / max(self.total_events + self.total_errors, 1), 4
                ),
            }

            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                summary["latency_p50_ms"] = round(sorted_lat[int(n * 0.50)], 2)
                summary["latency_p95_ms"] = round(sorted_lat[int(n * 0.95)], 2)
                summary["latency_p99_ms"] = round(sorted_lat[min(int(n * 0.99), n - 1)], 2)
                summary["latency_avg_ms"] = round(statistics.mean(latencies), 2)

            return summary


class LoadGenerator:
    """
    Multi-threaded load generator for the StreamRx pipeline.

    Creates multiple producer instances running in parallel to achieve
    high aggregate throughput. Monitors and reports performance metrics
    in real-time.
    """

    def __init__(
        self,
        rx_eps: float = 500.0,
        ae_eps: float = 50.0,
        rx_threads: int = 4,
        ae_threads: int = 2,
        duration_seconds: int = 300,
        patient_pool_size: int = 50_000,
    ) -> None:
        self._rx_eps = rx_eps
        self._ae_eps = ae_eps
        self._rx_threads = rx_threads
        self._ae_threads = ae_threads
        self._duration = duration_seconds
        self._patient_pool_size = patient_pool_size
        self._stats = LoadTestStats()
        self._running = False

        logger.info(
            "load_generator_configured",
            rx_eps=rx_eps,
            ae_eps=ae_eps,
            rx_threads=rx_threads,
            ae_threads=ae_threads,
            duration_sec=duration_seconds,
        )

    def _run_rx_producer(self, thread_id: int, eps_per_thread: float) -> None:
        """Run a single prescription producer thread."""
        producer = PrescriptionProducer(
            events_per_second=eps_per_thread,
            patient_pool_size=self._patient_pool_size,
        )
        interval = 1.0 / max(eps_per_thread, 0.1)

        logger.info("rx_producer_thread_started", thread_id=thread_id, eps=eps_per_thread)

        while self._running:
            start = time.monotonic()
            event = producer.produce_one()
            elapsed_ms = (time.monotonic() - start) * 1000

            if event is not None:
                self._stats.record_success(elapsed_ms)
            else:
                self._stats.record_error()

            sleep_time = interval - (elapsed_ms / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)

        producer.shutdown()
        logger.info("rx_producer_thread_stopped", thread_id=thread_id)

    def _run_ae_producer(self, thread_id: int, eps_per_thread: float) -> None:
        """Run a single adverse event producer thread."""
        producer = AdverseEventProducer(
            events_per_second=eps_per_thread,
            burst_probability=0.08,
        )
        interval = 1.0 / max(eps_per_thread, 0.1)

        logger.info("ae_producer_thread_started", thread_id=thread_id, eps=eps_per_thread)

        while self._running:
            start = time.monotonic()
            event = producer.produce_one()
            elapsed_ms = (time.monotonic() - start) * 1000

            if event is not None:
                self._stats.record_success(elapsed_ms)
            else:
                self._stats.record_error()

            sleep_time = interval - (elapsed_ms / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)

        producer.shutdown()
        logger.info("ae_producer_thread_stopped", thread_id=thread_id)

    def _stats_reporter(self) -> None:
        """Periodically report load test statistics."""
        report_interval = 10  # seconds
        while self._running:
            time.sleep(report_interval)
            if not self._running:
                break
            interval_eps = self._stats.get_interval_throughput()
            summary = self._stats.get_summary()
            logger.info(
                "load_test_progress",
                interval_eps=round(interval_eps, 1),
                **summary,
            )

    def run(self) -> dict[str, Any]:
        """
        Execute the load test.

        Returns:
            Final statistics summary.
        """
        self._running = True
        self._stats = LoadTestStats()

        rx_eps_per_thread = self._rx_eps / max(self._rx_threads, 1)
        ae_eps_per_thread = self._ae_eps / max(self._ae_threads, 1)

        logger.info(
            "load_test_starting",
            total_threads=self._rx_threads + self._ae_threads,
            target_total_eps=self._rx_eps + self._ae_eps,
            duration_sec=self._duration,
        )

        threads: list[threading.Thread] = []

        # Stats reporter
        reporter = threading.Thread(target=self._stats_reporter, daemon=True)
        reporter.start()

        # Prescription producer threads
        for i in range(self._rx_threads):
            t = threading.Thread(
                target=self._run_rx_producer,
                args=(i, rx_eps_per_thread),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Adverse event producer threads
        for i in range(self._ae_threads):
            t = threading.Thread(
                target=self._run_ae_producer,
                args=(i + self._rx_threads, ae_eps_per_thread),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait for duration
        try:
            time.sleep(self._duration)
        except KeyboardInterrupt:
            logger.info("load_test_interrupted")

        self._running = False

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=10)

        summary = self._stats.get_summary()
        logger.info("load_test_completed", **summary)

        return summary


def main() -> None:
    """CLI entry point for load generation."""
    parser = argparse.ArgumentParser(
        description="StreamRx Load Generator - Stress test the streaming pipeline"
    )
    parser.add_argument(
        "--rx-eps",
        type=float,
        default=500.0,
        help="Target prescription events/sec (default: 500)",
    )
    parser.add_argument(
        "--ae-eps",
        type=float,
        default=50.0,
        help="Target adverse event events/sec (default: 50)",
    )
    parser.add_argument(
        "--rx-threads",
        type=int,
        default=4,
        help="Number of prescription producer threads (default: 4)",
    )
    parser.add_argument(
        "--ae-threads",
        type=int,
        default=2,
        help="Number of adverse event producer threads (default: 2)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=50_000,
        help="Patient pool size (default: 50000)",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    configure_logging(level=args.log_level)

    generator = LoadGenerator(
        rx_eps=args.rx_eps,
        ae_eps=args.ae_eps,
        rx_threads=args.rx_threads,
        ae_threads=args.ae_threads,
        duration_seconds=args.duration,
        patient_pool_size=args.patients,
    )

    summary = generator.run()

    # Print final report
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:.<30} {value}")
    print("=" * 60)

    # Exit with error if error rate too high
    if summary.get("error_rate", 0) > 0.05:
        logger.error("load_test_high_error_rate", error_rate=summary["error_rate"])
        sys.exit(1)


if __name__ == "__main__":
    main()
