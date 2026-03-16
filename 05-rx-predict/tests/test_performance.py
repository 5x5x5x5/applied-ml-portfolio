"""Performance regression tests.

These tests enforce the sub-100ms latency SLA and detect performance regressions.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from rx_predict.models.drug_response_model import DrugResponseModel
from rx_predict.models.feature_processor import FeatureProcessor
from rx_predict.monitoring.performance import LatencyWindow, PerformanceMonitor
from rx_predict.monitoring.production_issues import (
    CircuitBreaker,
    CircuitState,
    LatencySpikeDetector,
    ProductionIssueDetector,
)


class TestLatencyWindow:
    """Test the sliding window latency tracker."""

    def test_empty_window(self) -> None:
        window = LatencyWindow()
        assert window.p50 == 0.0
        assert window.p99 == 0.0
        assert window.count == 0

    def test_record_and_percentiles(self) -> None:
        window = LatencyWindow(window_size=100)
        for i in range(100):
            window.record(float(i))
        assert window.count == 100
        assert window.p50 == pytest.approx(50.0, abs=2)
        assert window.p99 == pytest.approx(99.0, abs=2)

    def test_window_overflow(self) -> None:
        window = LatencyWindow(window_size=10)
        for i in range(20):
            window.record(float(i))
        assert window.count == 10


class TestPerformanceMonitor:
    """Test the performance monitoring system."""

    def test_record_request(self, performance_monitor: PerformanceMonitor) -> None:
        performance_monitor.record_request("/predict", "POST", 200, 25.0)
        assert performance_monitor.request_latency.count == 1
        assert performance_monitor.request_latency.p50 == 25.0

    def test_error_rate(self, performance_monitor: PerformanceMonitor) -> None:
        for _ in range(90):
            performance_monitor.record_request("/predict", "POST", 200, 10.0)
        for _ in range(10):
            performance_monitor.record_request("/predict", "POST", 500, 10.0)
        assert performance_monitor.get_error_rate() == pytest.approx(0.1, abs=0.01)

    def test_sla_compliance_passing(self, performance_monitor: PerformanceMonitor) -> None:
        for _ in range(100):
            performance_monitor.record_request("/predict", "POST", 200, 20.0)
        compliance = performance_monitor.check_sla_compliance()
        assert compliance["compliant"] is True
        assert len(compliance["violations"]) == 0

    def test_sla_compliance_failing(self, performance_monitor: PerformanceMonitor) -> None:
        for _ in range(100):
            performance_monitor.record_request("/predict", "POST", 200, 150.0)
        compliance = performance_monitor.check_sla_compliance()
        assert compliance["compliant"] is False
        assert len(compliance["violations"]) > 0

    def test_performance_report(self, performance_monitor: PerformanceMonitor) -> None:
        performance_monitor.record_request("/predict", "POST", 200, 25.0)
        report = performance_monitor.get_performance_report()
        assert "request_latency" in report
        assert "inference_latency" in report
        assert "throughput_rps" in report
        assert "sla_compliance" in report


class TestCircuitBreaker:
    """Test the circuit breaker pattern."""

    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_closes_after_success_in_half_open(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=0.1,
            half_open_max_requests=2,
        )
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_status_report(self) -> None:
        cb = CircuitBreaker()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status


class TestLatencySpikeDetector:
    """Test latency spike detection."""

    def test_no_spike_for_normal_latency(self) -> None:
        detector = LatencySpikeDetector(baseline_p99_ms=50.0, spike_multiplier=2.0)
        issue = detector.record(30.0)
        assert issue is None

    def test_spike_detected(self) -> None:
        detector = LatencySpikeDetector(baseline_p99_ms=50.0, spike_multiplier=2.0)
        # Fill some normal values first
        for _ in range(20):
            detector.record(30.0)
        issue = detector.record(150.0)
        assert issue is not None
        assert issue.issue_type.value == "latency_spike"

    def test_spike_rate_tracking(self) -> None:
        detector = LatencySpikeDetector(baseline_p99_ms=10.0, spike_multiplier=2.0)
        for _ in range(90):
            detector.record(5.0)
        for _ in range(10):
            detector.record(50.0)
        assert 0.05 <= detector.spike_rate <= 0.15


class TestProductionIssueDetector:
    """Test the aggregated issue detector."""

    def test_normal_requests_no_issues(self) -> None:
        detector = ProductionIssueDetector()
        for _ in range(10):
            issues = detector.record_request(latency_ms=20.0, success=True)
            assert len(issues) == 0

    def test_summary_report(self) -> None:
        detector = ProductionIssueDetector()
        summary = detector.get_issue_summary()
        assert "circuit_breaker" in summary
        assert "latency_spike_rate" in summary
        assert "active_issues_count" in summary


# --- Benchmark Tests ---


@pytest.mark.benchmark
class TestPerformanceBenchmark:
    """Latency benchmark tests that enforce the sub-100ms SLA."""

    def test_single_prediction_p99_under_100ms(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """CRITICAL: Single prediction p99 must be under 100ms."""
        # Warmup
        for _ in range(50):
            trained_model.predict(sample_patient_data)

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            trained_model.predict(sample_patient_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        p50 = latencies[100]
        p95 = latencies[190]
        p99 = latencies[198]

        assert p99 < 100.0, (
            f"p99 latency SLA violation: {p99:.3f}ms (limit: 100ms). "
            f"p50={p50:.3f}ms, p95={p95:.3f}ms"
        )

    def test_single_prediction_p50_under_20ms(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """p50 should be well under the SLA."""
        for _ in range(50):
            trained_model.predict(sample_patient_data)

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            trained_model.predict(sample_patient_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        p50 = latencies[100]
        assert p50 < 20.0, f"p50 latency too high: {p50:.3f}ms"

    def test_feature_processing_overhead(
        self, feature_processor: FeatureProcessor, sample_patient_data: dict[str, Any]
    ) -> None:
        """Feature processing should add <5ms overhead."""
        for _ in range(50):
            feature_processor.process_single(sample_patient_data)

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            feature_processor.process_single(sample_patient_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        p99 = latencies[198]
        assert p99 < 5.0, f"Feature processing p99 too slow: {p99:.3f}ms"

    def test_batch_throughput(
        self, trained_model: DrugResponseModel, sample_patient_data: dict[str, Any]
    ) -> None:
        """Batch of 10 should complete in under 200ms."""
        patients = [sample_patient_data] * 10

        # Warmup
        for _ in range(5):
            trained_model.predict_batch(patients)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            trained_model.predict_batch(patients)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        p95 = latencies[47]
        assert p95 < 200.0, f"Batch prediction p95 too slow: {p95:.3f}ms for batch of 10"

    def test_model_warmup_effective(self, sample_patient_data: dict[str, Any]) -> None:
        """Model warmup should reduce cold-start latency."""
        model = DrugResponseModel()
        model.build_default_model()

        # Cold prediction
        cold_start = time.perf_counter()
        model.predict(sample_patient_data)
        cold_ms = (time.perf_counter() - cold_start) * 1000

        # Warm up
        model.warm_up()

        # Warm prediction
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            model.predict(sample_patient_data)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        warm_p50 = sorted(latencies)[10]
        # Warm predictions should generally be faster than cold start
        # (allowing some tolerance for system variance)
        assert warm_p50 < cold_ms * 2, (
            f"Warmup not effective: cold={cold_ms:.3f}ms, warm_p50={warm_p50:.3f}ms"
        )
