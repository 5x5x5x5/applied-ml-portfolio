"""
Tests for the real-time safety signal detection module.

Covers contingency table math, PRR/ROR computation, signal evaluation,
dead letter queue behavior, and alert lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from stream_rx.config import SignalDetectionConfig
from stream_rx.consumers.signal_detector import (
    ContingencyCell,
    SignalDetector,
    WindowState,
)
from stream_rx.models import AdverseEventReport

# ---------------------------------------------------------------------------
# ContingencyCell tests
# ---------------------------------------------------------------------------


class TestContingencyCell:
    """Test 2x2 contingency table computations."""

    def test_prr_calculation(self) -> None:
        # a=10, b=90 (Drug D: 100 reports, 10 with reaction R)
        # c=5, d=795 (Not Drug D: 800 reports, 5 with reaction R)
        cell = ContingencyCell(a=10, b=90, c=5, d=795)
        prr = cell.prr()
        assert prr is not None
        # PRR = (10/100) / (5/800) = 0.1 / 0.00625 = 16.0
        assert abs(prr - 16.0) < 0.01

    def test_ror_calculation(self) -> None:
        cell = ContingencyCell(a=10, b=90, c=5, d=795)
        ror = cell.ror()
        assert ror is not None
        # ROR = (10*795) / (90*5) = 7950 / 450 = 17.67
        assert abs(ror - 17.667) < 0.01

    def test_chi_square_calculation(self) -> None:
        cell = ContingencyCell(a=10, b=90, c=5, d=795)
        chi2 = cell.chi_square()
        assert chi2 is not None
        assert chi2 > 3.84  # Significant at p<0.05

    def test_prr_zero_denominator(self) -> None:
        cell = ContingencyCell(a=5, b=0, c=0, d=0)
        assert cell.prr() is None

    def test_ror_zero_denominator(self) -> None:
        cell = ContingencyCell(a=5, b=0, c=5, d=10)
        assert cell.ror() is None

    def test_prr_ci_lower(self) -> None:
        cell = ContingencyCell(a=20, b=80, c=10, d=890)
        ci_lower = cell.prr_ci_lower()
        assert ci_lower is not None
        prr = cell.prr()
        assert prr is not None
        assert ci_lower < prr  # Lower bound must be less than point estimate

    def test_all_zeros(self) -> None:
        cell = ContingencyCell(a=0, b=0, c=0, d=0)
        assert cell.total == 0
        assert cell.prr() is None
        assert cell.ror() is None
        assert cell.chi_square() is None

    def test_perfect_association(self) -> None:
        # All Drug D patients have reaction, no non-Drug D patients do
        cell = ContingencyCell(a=50, b=0, c=0, d=950)
        prr = cell.prr()
        # c=0 leads to None (zero denominator in denominator fraction)
        assert prr is None

    def test_no_association(self) -> None:
        # Equal proportions in both groups
        cell = ContingencyCell(a=10, b=90, c=100, d=900)
        prr = cell.prr()
        assert prr is not None
        assert abs(prr - 1.0) < 0.01  # PRR ~1.0 means no signal


# ---------------------------------------------------------------------------
# WindowState tests
# ---------------------------------------------------------------------------


class TestWindowState:
    """Test sliding window state management."""

    def test_add_single_event(self) -> None:
        window = WindowState()
        window.add_event(["DrugA"], ["Nausea"])
        assert window.total_events == 1
        assert window.drug_event_counts["DrugA"] == 1
        assert window.reaction_event_counts["Nausea"] == 1
        assert window.pair_counts[("DrugA", "Nausea")] == 1

    def test_add_multi_drug_event(self) -> None:
        window = WindowState()
        window.add_event(["DrugA", "DrugB"], ["Nausea", "Headache"])
        assert window.total_events == 1
        assert window.drug_event_counts["DrugA"] == 1
        assert window.drug_event_counts["DrugB"] == 1
        assert window.pair_counts[("DrugA", "Nausea")] == 1
        assert window.pair_counts[("DrugB", "Headache")] == 1

    def test_accumulate_multiple_events(self) -> None:
        window = WindowState()
        for _ in range(5):
            window.add_event(["DrugA"], ["Nausea"])
        for _ in range(3):
            window.add_event(["DrugA"], ["Headache"])
        for _ in range(10):
            window.add_event(["DrugB"], ["Nausea"])

        assert window.total_events == 18
        assert window.drug_event_counts["DrugA"] == 8
        assert window.pair_counts[("DrugA", "Nausea")] == 5

    def test_build_contingency_table(self) -> None:
        window = WindowState()
        # 10 events: DrugA + Rhabdomyolysis
        for _ in range(10):
            window.add_event(["DrugA"], ["Rhabdomyolysis"])
        # 5 events: DrugA + Nausea
        for _ in range(5):
            window.add_event(["DrugA"], ["Nausea"])
        # 3 events: DrugB + Rhabdomyolysis
        for _ in range(3):
            window.add_event(["DrugB"], ["Rhabdomyolysis"])
        # 82 events: DrugB + Nausea (background)
        for _ in range(82):
            window.add_event(["DrugB"], ["Nausea"])

        cell = window.build_contingency("DrugA", "Rhabdomyolysis")
        assert cell.a == 10  # DrugA + Rhabdomyolysis
        assert cell.b == 5  # DrugA + NOT Rhabdomyolysis
        assert cell.c == 3  # NOT DrugA + Rhabdomyolysis
        assert cell.d == 82  # NOT DrugA + NOT Rhabdomyolysis
        assert cell.total == 100


# ---------------------------------------------------------------------------
# SignalDetector tests
# ---------------------------------------------------------------------------


class TestSignalDetector:
    """Test the signal detection engine."""

    def test_initialization(self) -> None:
        detector = SignalDetector()
        stats = detector.stats
        assert stats["events_processed"] == 0
        assert stats["signals_detected"] == 0
        assert stats["active_signals"] == 0
        assert stats["dlq_size"] == 0

    def test_process_valid_event(self, sample_adverse_event: AdverseEventReport) -> None:
        detector = SignalDetector()
        result = detector.process_event(sample_adverse_event.model_dump())
        assert result["status"] == "ok"
        assert detector.stats["events_processed"] == 1

    def test_process_event_from_bytes(self, sample_adverse_event: AdverseEventReport) -> None:
        detector = SignalDetector()
        result = detector.process_event(sample_adverse_event.to_bytes())
        assert result["status"] == "ok"

    def test_invalid_event_goes_to_dlq(self) -> None:
        detector = SignalDetector()
        result = detector.process_event(b"this is not valid json")
        assert result["status"] == "dlq_routed"
        assert detector.stats["dlq_size"] == 1

    def test_malformed_event_goes_to_dlq(self) -> None:
        detector = SignalDetector()
        result = detector.process_event({"not": "a valid adverse event"})
        assert result["status"] == "dlq_routed"
        assert detector.stats["dlq_size"] == 1

    def test_signal_detection_with_batch(
        self, adverse_event_batch_for_signal: list[dict[str, Any]]
    ) -> None:
        # Use a short window so we can evaluate
        config = SignalDetectionConfig()
        detector = SignalDetector(signal_config=config)

        for event in adverse_event_batch_for_signal:
            detector.process_event(event)

        assert detector.stats["events_processed"] == len(adverse_event_batch_for_signal)

        # Manually trigger evaluation
        signals = detector.evaluate_signals()

        # There should be a signal for Atorvastatin-Rhabdomyolysis
        ator_rhabdo = [
            s
            for s in signals
            if s.drug_name == "Atorvastatin" and s.reaction_term == "Rhabdomyolysis"
        ]
        assert len(ator_rhabdo) >= 1, (
            f"Expected Atorvastatin-Rhabdomyolysis signal, got: "
            f"{[(s.drug_name, s.reaction_term) for s in signals]}"
        )

        signal = ator_rhabdo[0]
        assert signal.metric_value > 0
        assert signal.case_count >= 3

    def test_dlq_retrieval(self) -> None:
        detector = SignalDetector()
        detector.process_event(b"bad1")
        detector.process_event(b"bad2")
        detector.process_event(b"bad3")

        entries = detector.get_dlq_entries(limit=10)
        assert len(entries) == 3
        assert entries[0]["reason"].startswith("JSON decode error")

    def test_dlq_limit(self) -> None:
        detector = SignalDetector()
        for i in range(10):
            detector.process_event(b"bad")

        entries = detector.get_dlq_entries(limit=5)
        assert len(entries) == 5

    def test_acknowledge_signal(self) -> None:
        config = SignalDetectionConfig()
        detector = SignalDetector(signal_config=config)

        # Create a signal by processing enough events
        for i in range(20):
            detector.process_event(
                {
                    "report_id": f"rpt-{i}",
                    "event_type": "adverse_event",
                    "timestamp": datetime.utcnow().isoformat(),
                    "patient_id": f"PAT-XX-{i:06d}",
                    "suspect_drugs": [
                        {
                            "drug_name": "TestDrug",
                            "drug_class": "test",
                            "role": "suspect",
                        }
                    ],
                    "reactions": [
                        {
                            "term": "TestReaction",
                            "meddra_pt_code": "00000000",
                            "seriousness": "serious",
                        }
                    ],
                    "severity": "severe",
                    "outcome": "recovered",
                    "hospitalized": False,
                    "reporter_type": "physician",
                    "reporter_country": "US",
                }
            )

        # Manually evaluate
        signals = detector.evaluate_signals()
        if signals:
            alert_id = signals[0].alert_id
            detector._active_signals.extend(signals)
            assert detector.acknowledge_signal(alert_id) is True
            assert detector.acknowledge_signal("nonexistent-id") is False

    def test_process_multi_drug_event(
        self, sample_adverse_event_multi_drug: AdverseEventReport
    ) -> None:
        detector = SignalDetector()
        result = detector.process_event(sample_adverse_event_multi_drug.model_dump())
        assert result["status"] == "ok"

        # Both drugs should be tracked
        window = detector._current_window
        assert window.drug_event_counts["Warfarin"] == 1
        assert window.drug_event_counts["Ibuprofen"] == 1
        assert window.reaction_event_counts["Gastrointestinal Haemorrhage"] == 1


# ---------------------------------------------------------------------------
# Metrics computation tests
# ---------------------------------------------------------------------------


class TestDisproportionalityMetrics:
    """Test mathematical correctness of PRR/ROR computations."""

    def test_known_prr_example(self) -> None:
        """Test with a known textbook example."""
        # From pharmacovigilance literature:
        # Drug X with Liver failure: a=15, b=85, c=100, d=9800
        cell = ContingencyCell(a=15, b=85, c=100, d=9800)
        prr = cell.prr()
        assert prr is not None
        # PRR = (15/100) / (100/9900) = 0.15 / 0.01010 = 14.85
        assert 14.0 < prr < 16.0

    def test_ror_vs_prr_relationship(self) -> None:
        """ROR should be close to PRR when events are rare."""
        cell = ContingencyCell(a=5, b=95, c=10, d=9890)
        prr = cell.prr()
        ror = cell.ror()
        assert prr is not None
        assert ror is not None
        # For rare events, PRR and ROR should be similar
        assert abs(prr - ror) / max(prr, ror) < 0.15

    def test_chi_square_not_significant(self) -> None:
        """Small differences should not be significant."""
        cell = ContingencyCell(a=2, b=98, c=3, d=897)
        chi2 = cell.chi_square()
        if chi2 is not None:
            assert chi2 < 3.84  # Not significant at p<0.05
