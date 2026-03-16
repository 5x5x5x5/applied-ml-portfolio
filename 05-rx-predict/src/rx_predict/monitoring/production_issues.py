"""Production issue detection and automated resolution.

Detects common production issues in ML services:
- Memory leaks
- Latency spikes with auto-diagnosis
- Model staleness
- Error rate monitoring with circuit breaker
- Automated runbook suggestions
"""

from __future__ import annotations

import gc
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class IssueType(str, Enum):
    """Types of production issues."""

    MEMORY_LEAK = "memory_leak"
    LATENCY_SPIKE = "latency_spike"
    MODEL_STALE = "model_stale"
    HIGH_ERROR_RATE = "high_error_rate"
    CACHE_DEGRADATION = "cache_degradation"
    THROUGHPUT_DROP = "throughput_drop"
    CPU_SATURATION = "cpu_saturation"


class Severity(str, Enum):
    """Issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ProductionIssue:
    """Represents a detected production issue."""

    issue_type: IssueType
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    runbook: str = ""
    auto_remediation: str = ""


# --- Runbooks ---

RUNBOOKS: dict[IssueType, str] = {
    IssueType.MEMORY_LEAK: """
RUNBOOK: Memory Leak Detected
1. Check current memory usage: `docker stats` or check /proc/meminfo
2. Identify leak source:
   - Feature processor cache growing unbounded? Check LRU cache size.
   - Redis connection pool leak? Verify connections are being returned.
   - Large numpy arrays not being freed? Check for circular references.
3. Immediate mitigation:
   - Trigger garbage collection: POST /admin/gc
   - Reduce cache sizes in configuration
   - If critical: rolling restart of service pods
4. Root cause fix:
   - Add memory profiling with tracemalloc
   - Set hard limits on cache sizes
   - Review recent code changes for array allocations
""",
    IssueType.LATENCY_SPIKE: """
RUNBOOK: Latency Spike Detected
1. Check if spike is isolated or systemic:
   - Single endpoint? Check feature processing for that input type.
   - All endpoints? Check infrastructure (CPU, memory, network).
2. Common causes:
   - GC pause: Check gc.get_stats() for collection counts
   - Redis latency: Check Redis SLOWLOG
   - Model inference regression: Compare model version benchmarks
   - CPU throttling: Check container CPU limits
3. Immediate mitigation:
   - Enable response caching if not already active
   - Scale horizontally if load-related
   - Switch to simpler model variant if model-related
4. Prevention:
   - Set up p99 latency alerting at 80ms (below 100ms SLA)
   - Run continuous benchmarks in CI/CD
""",
    IssueType.MODEL_STALE: """
RUNBOOK: Model Staleness Detected
1. Check when the model was last updated: GET /health
2. Verify model training pipeline status
3. Check if new training data is available
4. Validate model performance metrics haven't degraded
5. Trigger model retraining if needed
6. After retraining:
   - Run A/B test against current model
   - Verify latency hasn't regressed
   - Gradual rollout (canary deploy)
""",
    IssueType.HIGH_ERROR_RATE: """
RUNBOOK: High Error Rate Detected
1. Check error types in logs: filter by error_type label
2. Common causes:
   - Invalid input data: Check request validation logs
   - Model failure: Check model health, possibly corrupted model file
   - Dependency failure: Check Redis, downstream services
3. Immediate mitigation:
   - Circuit breaker should be activated automatically
   - Return cached results for known inputs
   - Return graceful degradation responses
4. Recovery:
   - Fix root cause
   - Reset circuit breaker: POST /admin/circuit-breaker/reset
   - Monitor error rate for 15 minutes before closing
""",
    IssueType.CACHE_DEGRADATION: """
RUNBOOK: Cache Degradation Detected
1. Check Redis health: redis-cli INFO
2. Check cache hit rate: GET /metrics
3. Common causes:
   - Redis memory full: Check maxmemory setting
   - Network issues: Check Redis latency
   - TTL too short: Review cache TTL configuration
4. Mitigation:
   - Service continues without cache (graceful degradation)
   - Increase Redis memory or add eviction policy
   - Pre-warm cache with common requests
""",
    IssueType.THROUGHPUT_DROP: """
RUNBOOK: Throughput Drop Detected
1. Compare current RPS with baseline
2. Check if drop correlates with latency increase
3. Common causes:
   - Connection pool exhaustion
   - Thread/worker starvation
   - Upstream rate limiting
4. Scale workers if needed
""",
    IssueType.CPU_SATURATION: """
RUNBOOK: CPU Saturation Detected
1. Check process CPU usage
2. Profile hot code paths
3. Consider:
   - Reducing model complexity
   - Batch processing optimization
   - Horizontal scaling
""",
}


class MemoryMonitor:
    """Detects memory leaks by tracking RSS growth over time."""

    def __init__(
        self,
        check_interval_seconds: float = 60.0,
        growth_threshold_mb: float = 100.0,
        window_size: int = 60,
    ) -> None:
        self._check_interval = check_interval_seconds
        self._growth_threshold_mb = growth_threshold_mb
        self._memory_samples: deque[tuple[float, float]] = deque(maxlen=window_size)
        self._last_check = 0.0

    def _get_rss_mb(self) -> float:
        """Get current RSS memory in MB."""
        try:
            with open(f"/proc/{os.getpid()}/statm") as f:
                pages = int(f.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 * 1024)
        except (FileNotFoundError, ValueError, IndexError):
            # Fallback for non-Linux systems
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                return usage.ru_maxrss / 1024  # KB to MB on Linux
            except ImportError:
                return 0.0

    def check(self) -> ProductionIssue | None:
        """Check for memory leaks. Returns an issue if detected."""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return None
        self._last_check = now

        rss_mb = self._get_rss_mb()
        self._memory_samples.append((now, rss_mb))

        if len(self._memory_samples) < 10:
            return None

        # Calculate memory growth rate
        oldest_time, oldest_rss = self._memory_samples[0]
        newest_time, newest_rss = self._memory_samples[-1]
        elapsed_minutes = (newest_time - oldest_time) / 60.0

        if elapsed_minutes < 1.0:
            return None

        growth_mb = newest_rss - oldest_rss
        growth_rate_mb_per_min = growth_mb / elapsed_minutes

        if growth_mb > self._growth_threshold_mb:
            severity = (
                Severity.CRITICAL if growth_mb > self._growth_threshold_mb * 2 else Severity.WARNING
            )
            return ProductionIssue(
                issue_type=IssueType.MEMORY_LEAK,
                severity=severity,
                message=f"Memory growth detected: {growth_mb:.1f}MB over {elapsed_minutes:.1f}min",
                details={
                    "current_rss_mb": round(newest_rss, 1),
                    "growth_mb": round(growth_mb, 1),
                    "growth_rate_mb_per_min": round(growth_rate_mb_per_min, 2),
                },
                runbook=RUNBOOKS[IssueType.MEMORY_LEAK],
                auto_remediation="gc.collect() + cache.clear()",
            )

        return None

    def force_gc(self) -> dict[str, Any]:
        """Force garbage collection and return stats."""
        before_rss = self._get_rss_mb()
        gc_stats_before = gc.get_stats()

        collected = gc.collect()

        after_rss = self._get_rss_mb()
        freed_mb = before_rss - after_rss

        result = {
            "collected_objects": collected,
            "rss_before_mb": round(before_rss, 1),
            "rss_after_mb": round(after_rss, 1),
            "freed_mb": round(freed_mb, 1),
            "gc_generations": [
                {"collections": s.get("collections", 0), "collected": s.get("collected", 0)}
                for s in gc_stats_before
            ],
        }

        logger.info("forced_gc_complete", **result)
        return result


class LatencySpikeDetector:
    """Detects latency spikes and attempts auto-diagnosis."""

    def __init__(
        self,
        baseline_p99_ms: float = 50.0,
        spike_multiplier: float = 2.0,
        window_size: int = 100,
    ) -> None:
        self._baseline_p99 = baseline_p99_ms
        self._spike_multiplier = spike_multiplier
        self._recent_latencies: deque[float] = deque(maxlen=window_size)
        self._spike_count = 0
        self._total_count = 0

    def record(self, latency_ms: float) -> ProductionIssue | None:
        """Record a latency measurement and check for spikes."""
        self._recent_latencies.append(latency_ms)
        self._total_count += 1

        threshold = self._baseline_p99 * self._spike_multiplier

        if latency_ms > threshold:
            self._spike_count += 1
            diagnosis = self._diagnose(latency_ms)

            severity = Severity.CRITICAL if latency_ms > 100.0 else Severity.WARNING
            return ProductionIssue(
                issue_type=IssueType.LATENCY_SPIKE,
                severity=severity,
                message=f"Latency spike: {latency_ms:.1f}ms (threshold: {threshold:.1f}ms)",
                details={
                    "latency_ms": round(latency_ms, 3),
                    "threshold_ms": round(threshold, 3),
                    "spike_rate": round(self._spike_count / max(self._total_count, 1), 4),
                    "diagnosis": diagnosis,
                },
                runbook=RUNBOOKS[IssueType.LATENCY_SPIKE],
                auto_remediation=diagnosis.get("recommended_action", "investigate"),
            )

        return None

    def _diagnose(self, latency_ms: float) -> dict[str, Any]:
        """Auto-diagnose the cause of a latency spike."""
        diagnosis: dict[str, Any] = {"probable_causes": []}

        # Check GC pressure
        gc_stats = gc.get_stats()
        gen2_collections = gc_stats[2].get("collections", 0) if len(gc_stats) > 2 else 0
        if gen2_collections > 0:
            diagnosis["probable_causes"].append("gc_pressure_gen2")

        # Check if it's an extreme outlier vs sustained
        if len(self._recent_latencies) >= 10:
            recent = list(self._recent_latencies)[-10:]
            avg_recent = sum(recent) / len(recent)
            if latency_ms > avg_recent * 3:
                diagnosis["probable_causes"].append("isolated_outlier")
                diagnosis["recommended_action"] = "monitor"
            else:
                diagnosis["probable_causes"].append("sustained_degradation")
                diagnosis["recommended_action"] = "investigate_infrastructure"

        if latency_ms > 200:
            diagnosis["probable_causes"].append("possible_cold_start_or_cache_miss")
            diagnosis["recommended_action"] = "check_cache_and_warmup"

        return diagnosis

    @property
    def spike_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._spike_count / self._total_count


class CircuitBreaker:
    """Circuit breaker pattern for error rate protection.

    States:
    - CLOSED: Normal operation, tracking error rate
    - OPEN: Blocking requests, returning fallback
    - HALF_OPEN: Allowing limited requests to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_requests: int = 3,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._half_open_max = half_open_max_requests
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_requests = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info("circuit_breaker_half_open")
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_requests < self._half_open_max:
                    self._half_open_requests += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("circuit_breaker_closed", msg="recovered")
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("circuit_breaker_reopened")
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_opened",
                    failures=self._failure_count,
                    threshold=self._failure_threshold,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
        logger.info("circuit_breaker_reset")

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_s": self._recovery_timeout,
        }


class ModelStalenessDetector:
    """Detects when the model hasn't been updated within acceptable limits."""

    def __init__(self, max_staleness_hours: float = 168.0) -> None:
        self._max_staleness_hours = max_staleness_hours
        self._last_model_update: float | None = None
        self._model_version: str = "unknown"

    def record_model_update(self, version: str) -> None:
        self._last_model_update = time.time()
        self._model_version = version

    def check(self) -> ProductionIssue | None:
        if self._last_model_update is None:
            return None

        hours_since_update = (time.time() - self._last_model_update) / 3600.0
        if hours_since_update > self._max_staleness_hours:
            return ProductionIssue(
                issue_type=IssueType.MODEL_STALE,
                severity=Severity.WARNING,
                message=f"Model {self._model_version} is {hours_since_update:.1f}h old",
                details={
                    "model_version": self._model_version,
                    "hours_since_update": round(hours_since_update, 1),
                    "max_staleness_hours": self._max_staleness_hours,
                },
                runbook=RUNBOOKS[IssueType.MODEL_STALE],
            )
        return None


class ProductionIssueDetector:
    """Aggregates all issue detectors and provides unified monitoring."""

    def __init__(self) -> None:
        self.memory_monitor = MemoryMonitor()
        self.latency_detector = LatencySpikeDetector()
        self.circuit_breaker = CircuitBreaker()
        self.staleness_detector = ModelStalenessDetector()
        self._active_issues: list[ProductionIssue] = []
        self._issue_history: deque[ProductionIssue] = deque(maxlen=1000)
        self._lock = Lock()

    def record_request(self, latency_ms: float, success: bool) -> list[ProductionIssue]:
        """Record a request and return any detected issues."""
        new_issues: list[ProductionIssue] = []

        # Latency check
        latency_issue = self.latency_detector.record(latency_ms)
        if latency_issue:
            new_issues.append(latency_issue)

        # Circuit breaker
        if success:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()

        # Memory check (rate-limited internally)
        memory_issue = self.memory_monitor.check()
        if memory_issue:
            new_issues.append(memory_issue)

        # Model staleness check
        staleness_issue = self.staleness_detector.check()
        if staleness_issue:
            new_issues.append(staleness_issue)

        # Store issues
        if new_issues:
            with self._lock:
                for issue in new_issues:
                    self._active_issues.append(issue)
                    self._issue_history.append(issue)
                    logger.warning(
                        "production_issue_detected",
                        issue_type=issue.issue_type.value,
                        severity=issue.severity.value,
                        message=issue.message,
                    )

        return new_issues

    def get_active_issues(self) -> list[dict[str, Any]]:
        """Get currently active issues."""
        with self._lock:
            # Prune old issues (older than 5 minutes)
            cutoff = time.time() - 300
            self._active_issues = [i for i in self._active_issues if i.timestamp > cutoff]
            return [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity.value,
                    "message": i.message,
                    "details": i.details,
                    "timestamp": i.timestamp,
                    "runbook": i.runbook,
                    "auto_remediation": i.auto_remediation,
                }
                for i in self._active_issues
            ]

    def get_issue_summary(self) -> dict[str, Any]:
        """Summary of all issue detection systems."""
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "latency_spike_rate": round(self.latency_detector.spike_rate, 4),
            "active_issues_count": len(self._active_issues),
            "total_issues_detected": len(self._issue_history),
            "memory_rss_mb": round(self.memory_monitor._get_rss_mb(), 1),
        }
