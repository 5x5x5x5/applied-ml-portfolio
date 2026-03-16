"""
Real-time pharmacovigilance safety signal detection.

Consumes adverse event reports and computes disproportionality metrics
(Proportional Reporting Ratio, Reporting Odds Ratio) in sliding windows.
When metrics exceed thresholds, safety alerts are triggered for review.
Failed processing is routed to a dead-letter queue for later investigation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import orjson

from stream_rx.config import KafkaConfig, SignalDetectionConfig, get_config
from stream_rx.logging_setup import get_logger
from stream_rx.models import (
    AdverseEventReport,
    AlertPriority,
    SafetySignalAlert,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Contingency table and disproportionality analysis
# ---------------------------------------------------------------------------


@dataclass
class ContingencyCell:
    """
    2x2 contingency table for a drug-reaction pair.

    Layout:
                    | Reaction R | Not Reaction R |
    Drug D          |     a      |       b        |
    Not Drug D      |     c      |       d        |

    Used to compute PRR, ROR, and chi-square statistics.
    """

    a: int = 0  # Drug D + Reaction R
    b: int = 0  # Drug D + Not Reaction R
    c: int = 0  # Not Drug D + Reaction R
    d: int = 0  # Not Drug D + Not Reaction R

    @property
    def total(self) -> int:
        return self.a + self.b + self.c + self.d

    def prr(self) -> float | None:
        """
        Proportional Reporting Ratio.

        PRR = (a / (a+b)) / (c / (c+d))
        Returns None if denominators are zero.
        """
        if (self.a + self.b) == 0 or (self.c + self.d) == 0 or self.c == 0:
            return None
        numerator = self.a / (self.a + self.b)
        denominator = self.c / (self.c + self.d)
        if denominator == 0:
            return None
        return numerator / denominator

    def ror(self) -> float | None:
        """
        Reporting Odds Ratio.

        ROR = (a * d) / (b * c)
        Returns None if denominator is zero.
        """
        if self.b == 0 or self.c == 0:
            return None
        return (self.a * self.d) / (self.b * self.c)

    def chi_square(self) -> float | None:
        """
        Yates-corrected chi-square statistic for a 2x2 table.

        chi2 = N * (|ad - bc| - N/2)^2 / ((a+b)(c+d)(a+c)(b+d))
        """
        n = self.total
        if n == 0:
            return None
        row1 = self.a + self.b
        row2 = self.c + self.d
        col1 = self.a + self.c
        col2 = self.b + self.d
        denom = row1 * row2 * col1 * col2
        if denom == 0:
            return None
        numer = n * (max(abs(self.a * self.d - self.b * self.c) - n / 2, 0)) ** 2
        return numer / denom

    def prr_ci_lower(self, z: float = 1.96) -> float | None:
        """Lower bound of PRR 95% confidence interval."""
        prr_val = self.prr()
        if prr_val is None or prr_val <= 0:
            return None
        if self.a == 0:
            return None
        se = math.sqrt(1 / self.a - 1 / (self.a + self.b) + 1 / self.c - 1 / (self.c + self.d))
        return math.exp(math.log(prr_val) - z * se)


@dataclass
class WindowState:
    """
    Sliding window state for signal detection.

    Tracks drug-reaction pair counts and total event counts within the window.
    """

    window_start: datetime = field(default_factory=datetime.utcnow)
    window_end: datetime = field(default_factory=datetime.utcnow)

    # Total adverse events in window
    total_events: int = 0

    # Count of events per drug
    drug_event_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Count of events per reaction
    reaction_event_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Count of events for each (drug, reaction) pair
    pair_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def add_event(self, drug_names: list[str], reaction_terms: list[str]) -> None:
        """Register a single adverse event in the window state."""
        self.total_events += 1
        for drug in drug_names:
            self.drug_event_counts[drug] += 1
            for reaction in reaction_terms:
                self.pair_counts[(drug, reaction)] += 1
        for reaction in reaction_terms:
            self.reaction_event_counts[reaction] += 1

    def build_contingency(self, drug: str, reaction: str) -> ContingencyCell:
        """
        Build a 2x2 contingency table for a specific drug-reaction pair.

        Uses the marginal counts tracked in this window.
        """
        a = self.pair_counts.get((drug, reaction), 0)
        drug_total = self.drug_event_counts.get(drug, 0)
        reaction_total = self.reaction_event_counts.get(reaction, 0)
        b = drug_total - a
        c = reaction_total - a
        d = self.total_events - a - b - c
        return ContingencyCell(a=a, b=max(b, 0), c=max(c, 0), d=max(d, 0))


# ---------------------------------------------------------------------------
# Signal detector
# ---------------------------------------------------------------------------


class SignalDetector:
    """
    Real-time safety signal detector using disproportionality analysis.

    Maintains a sliding window of adverse event data and periodically
    evaluates all drug-reaction pairs for statistical signals. When a
    signal is detected (PRR or ROR exceeds threshold with sufficient
    case count), a SafetySignalAlert is emitted.
    """

    def __init__(
        self,
        signal_config: SignalDetectionConfig | None = None,
        kafka_config: KafkaConfig | None = None,
    ) -> None:
        cfg = get_config()
        self._signal_config = signal_config or cfg.signal_detection
        self._kafka_config = kafka_config or cfg.kafka

        self._current_window = WindowState(
            window_start=datetime.utcnow(),
            window_end=datetime.utcnow()
            + timedelta(hours=self._signal_config.sliding_window_hours),
        )

        # Historical baseline for comparison
        self._baseline: WindowState | None = None

        # Detected signals
        self._active_signals: list[SafetySignalAlert] = []
        self._signal_history: list[SafetySignalAlert] = []

        # Dead letter queue for failed processing
        self._dlq: list[dict[str, Any]] = []

        # Metrics
        self._events_processed = 0
        self._signals_detected = 0
        self._dlq_count = 0
        self._last_evaluation_time: datetime | None = None

        logger.info(
            "signal_detector_initialized",
            prr_threshold=self._signal_config.prr_threshold,
            ror_threshold=self._signal_config.ror_threshold,
            min_case_count=self._signal_config.min_case_count,
            window_hours=self._signal_config.sliding_window_hours,
        )

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def process_event(self, raw_event: bytes | dict[str, Any]) -> dict[str, Any]:
        """
        Process a single adverse event through the signal detection pipeline.

        Args:
            raw_event: Raw bytes or pre-parsed dict of an AdverseEventReport.

        Returns:
            Processing result including any triggered signals.
        """
        result: dict[str, Any] = {
            "status": "ok",
            "signals": [],
        }

        try:
            if isinstance(raw_event, bytes):
                data = orjson.loads(raw_event)
            else:
                data = raw_event

            event = AdverseEventReport(**data)
            self._events_processed += 1

            # Extract drug names and reaction terms
            drug_names = [d.drug_name for d in event.suspect_drugs]
            reaction_terms = [r.term for r in event.reactions]

            # Update window state
            self._current_window.add_event(drug_names, reaction_terms)

            # Check if window needs rotation
            if datetime.utcnow() >= self._current_window.window_end:
                signals = self._evaluate_and_rotate()
                result["signals"] = [s.model_dump() for s in signals]

        except orjson.JSONDecodeError as exc:
            self._send_to_dlq(raw_event, f"JSON decode error: {exc}")
            result["status"] = "dlq_routed"
        except Exception as exc:
            self._send_to_dlq(raw_event, f"Processing error: {exc}")
            result["status"] = "dlq_routed"

        return result

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _evaluate_and_rotate(self) -> list[SafetySignalAlert]:
        """
        Evaluate all drug-reaction pairs in the current window and rotate.

        Returns:
            List of newly detected safety signals.
        """
        signals = self.evaluate_signals()
        self._active_signals.extend(signals)
        self._signal_history.extend(signals)

        # Rotate: current window becomes baseline context
        self._baseline = self._current_window
        now = datetime.utcnow()
        self._current_window = WindowState(
            window_start=now,
            window_end=now + timedelta(hours=self._signal_config.sliding_window_hours),
        )
        self._last_evaluation_time = now

        logger.info(
            "signal_evaluation_completed",
            window_events=self._baseline.total_events if self._baseline else 0,
            signals_found=len(signals),
            total_signals=self._signals_detected,
        )
        return signals

    def evaluate_signals(self) -> list[SafetySignalAlert]:
        """
        Evaluate all drug-reaction pairs for disproportionality signals.

        Computes PRR and ROR for each pair and checks against thresholds.
        A signal is triggered when:
        - PRR >= threshold AND case count >= minimum
        - OR ROR >= threshold AND case count >= minimum
        - AND chi-square >= chi-square threshold (statistical significance)

        Returns:
            List of SafetySignalAlert objects for detected signals.
        """
        signals: list[SafetySignalAlert] = []
        window = self._current_window

        if window.total_events < self._signal_config.min_case_count:
            return signals

        for (drug, reaction), count in window.pair_counts.items():
            if count < self._signal_config.min_case_count:
                continue

            cell = window.build_contingency(drug, reaction)
            prr_val = cell.prr()
            ror_val = cell.ror()
            chi2_val = cell.chi_square()

            # Check statistical significance
            if chi2_val is not None and chi2_val < self._signal_config.chi_square_threshold:
                continue

            # Check PRR signal
            prr_signal = prr_val is not None and prr_val >= self._signal_config.prr_threshold
            ror_signal = ror_val is not None and ror_val >= self._signal_config.ror_threshold

            if prr_signal or ror_signal:
                # Determine priority based on metric strength
                metric_val = max(
                    prr_val if prr_val is not None else 0,
                    ror_val if ror_val is not None else 0,
                )
                if metric_val >= 10.0 or count >= 20:
                    priority = AlertPriority.CRITICAL
                elif metric_val >= 5.0 or count >= 10:
                    priority = AlertPriority.HIGH
                elif metric_val >= 3.0:
                    priority = AlertPriority.MEDIUM
                else:
                    priority = AlertPriority.LOW

                # Determine which metric triggered
                if prr_signal and ror_signal:
                    metric_name = "PRR+ROR"
                    metric_value = prr_val if prr_val is not None else 0.0
                elif prr_signal:
                    metric_name = "PRR"
                    metric_value = prr_val if prr_val is not None else 0.0
                else:
                    metric_name = "ROR"
                    metric_value = ror_val if ror_val is not None else 0.0

                signal = SafetySignalAlert(
                    drug_name=drug,
                    reaction_term=reaction,
                    metric_name=metric_name,
                    metric_value=round(metric_value, 3),
                    threshold=self._signal_config.prr_threshold,
                    case_count=count,
                    window_start=window.window_start,
                    window_end=window.window_end,
                    priority=priority,
                    description=(
                        f"Signal detected for {drug} - {reaction}: "
                        f"PRR={prr_val:.2f}, ROR={ror_val:.2f}, "
                        f"cases={count}, chi2={chi2_val:.2f}"
                        if prr_val and ror_val and chi2_val
                        else f"Signal detected for {drug} - {reaction}: cases={count}"
                    ),
                )
                signals.append(signal)
                self._signals_detected += 1

                logger.warning(
                    "safety_signal_detected",
                    drug=drug,
                    reaction=reaction,
                    prr=prr_val,
                    ror=ror_val,
                    chi2=chi2_val,
                    cases=count,
                    priority=priority.value,
                    alert_id=signal.alert_id,
                )

        return signals

    # ------------------------------------------------------------------
    # Dead letter queue
    # ------------------------------------------------------------------

    def _send_to_dlq(self, raw_event: Any, reason: str) -> None:
        """Route a failed event to the dead letter queue."""
        self._dlq_count += 1
        dlq_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "raw_event": raw_event if isinstance(raw_event, dict) else str(raw_event)[:10_000],
            "attempt": 1,
        }
        self._dlq.append(dlq_entry)
        logger.error(
            "event_routed_to_dlq",
            reason=reason,
            dlq_size=len(self._dlq),
        )

    def get_dlq_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve entries from the dead letter queue."""
        return self._dlq[:limit]

    def retry_dlq_entry(self, index: int) -> dict[str, Any]:
        """Retry processing a specific DLQ entry."""
        if index >= len(self._dlq):
            return {"status": "not_found"}
        entry = self._dlq[index]
        result = self.process_event(entry["raw_event"])
        if result["status"] == "ok":
            self._dlq.pop(index)
        return result

    # ------------------------------------------------------------------
    # Query active signals
    # ------------------------------------------------------------------

    def get_active_signals(self) -> list[SafetySignalAlert]:
        """Return currently active safety signals."""
        return list(self._active_signals)

    def acknowledge_signal(self, alert_id: str) -> bool:
        """Acknowledge and remove an active signal."""
        for i, sig in enumerate(self._active_signals):
            if sig.alert_id == alert_id:
                self._active_signals.pop(i)
                logger.info("signal_acknowledged", alert_id=alert_id)
                return True
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "events_processed": self._events_processed,
            "signals_detected": self._signals_detected,
            "active_signals": len(self._active_signals),
            "dlq_size": len(self._dlq),
            "current_window_events": self._current_window.total_events,
            "unique_drug_reaction_pairs": len(self._current_window.pair_counts),
            "last_evaluation": (
                self._last_evaluation_time.isoformat() if self._last_evaluation_time else None
            ),
        }
