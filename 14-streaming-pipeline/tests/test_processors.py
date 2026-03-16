"""
Tests for the prescription processor including drug interaction checking,
windowed aggregation, and anomaly detection.
"""

from __future__ import annotations

from typing import Any

import pytest

from stream_rx.consumers.prescription_processor import PrescriptionProcessor
from stream_rx.models import PrescriptionEvent


class TestPrescriptionEventModel:
    """Test PrescriptionEvent model validation."""

    def test_valid_prescription_creation(self, sample_prescription: PrescriptionEvent) -> None:
        assert sample_prescription.patient_id == "PAT-NE-123456"
        assert sample_prescription.drug_ndc == "00002-4462-30"
        assert sample_prescription.drug_class == "statin"
        assert sample_prescription.quantity == 30.0

    def test_prescription_partition_key(self, sample_prescription: PrescriptionEvent) -> None:
        assert sample_prescription.partition_key() == "PAT-NE-123456"

    def test_prescription_serialization(self, sample_prescription: PrescriptionEvent) -> None:
        raw = sample_prescription.to_bytes()
        assert isinstance(raw, bytes)
        assert b"PAT-NE-123456" in raw
        assert b"Atorvastatin" in raw

    def test_invalid_ndc_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid NDC"):
            PrescriptionEvent(
                patient_id="PAT-NE-123456",
                drug_ndc="INVALID",
                drug_name="Test Drug",
                prescriber_npi="1234567890",
                pharmacy_ncpdp="0312456",
                quantity=30.0,
                days_supply=30,
                refill_number=0,
            )

    def test_invalid_npi_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid NPI"):
            PrescriptionEvent(
                patient_id="PAT-NE-123456",
                drug_ndc="00002-4462-30",
                drug_name="Test Drug",
                prescriber_npi="123",  # Too short
                pharmacy_ncpdp="0312456",
                quantity=30.0,
                days_supply=30,
                refill_number=0,
            )

    def test_zero_quantity_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrescriptionEvent(
                patient_id="PAT-NE-123456",
                drug_ndc="00002-4462-30",
                drug_name="Test Drug",
                prescriber_npi="1234567890",
                pharmacy_ncpdp="0312456",
                quantity=0,  # Must be > 0
                days_supply=30,
                refill_number=0,
            )


class TestDrugInteractionChecking:
    """Test drug interaction detection logic."""

    def test_no_interaction_single_drug(self) -> None:
        processor = PrescriptionProcessor()
        alerts = processor.check_interactions(
            patient_id="PAT-TEST-001",
            new_drug_class="statin",
            new_drug_name="Atorvastatin",
            prescriber_npi="1234567890",
        )
        assert len(alerts) == 0

    def test_anticoagulant_nsaid_interaction(self) -> None:
        processor = PrescriptionProcessor()

        # First, add an anticoagulant
        alerts1 = processor.check_interactions(
            patient_id="PAT-TEST-002",
            new_drug_class="anticoagulant",
            new_drug_name="Warfarin",
            prescriber_npi="1234567890",
        )
        assert len(alerts1) == 0

        # Then add an NSAID for the same patient
        alerts2 = processor.check_interactions(
            patient_id="PAT-TEST-002",
            new_drug_class="nsaid",
            new_drug_name="Ibuprofen",
            prescriber_npi="2345678901",
        )
        assert len(alerts2) == 1
        alert = alerts2[0]
        assert alert.drug_a == "Warfarin"
        assert alert.drug_b == "Ibuprofen"
        assert alert.risk_level == "high"
        assert "bleeding" in alert.interaction_description.lower()

    def test_opioid_ssri_interaction(self) -> None:
        processor = PrescriptionProcessor()

        processor.check_interactions(
            patient_id="PAT-TEST-003",
            new_drug_class="ssri",
            new_drug_name="Sertraline",
            prescriber_npi="1234567890",
        )

        alerts = processor.check_interactions(
            patient_id="PAT-TEST-003",
            new_drug_class="opioid",
            new_drug_name="Tramadol",
            prescriber_npi="1234567890",
        )
        assert len(alerts) == 1
        assert alerts[0].risk_level == "moderate"

    def test_no_interaction_different_patients(self) -> None:
        processor = PrescriptionProcessor()

        processor.check_interactions(
            patient_id="PAT-A-001",
            new_drug_class="anticoagulant",
            new_drug_name="Warfarin",
            prescriber_npi="1234567890",
        )

        # Different patient gets NSAID - no interaction
        alerts = processor.check_interactions(
            patient_id="PAT-B-002",
            new_drug_class="nsaid",
            new_drug_name="Ibuprofen",
            prescriber_npi="1234567890",
        )
        assert len(alerts) == 0

    def test_no_interaction_safe_combination(self) -> None:
        processor = PrescriptionProcessor()

        processor.check_interactions(
            patient_id="PAT-TEST-004",
            new_drug_class="statin",
            new_drug_name="Atorvastatin",
            prescriber_npi="1234567890",
        )

        alerts = processor.check_interactions(
            patient_id="PAT-TEST-004",
            new_drug_class="ppi",
            new_drug_name="Omeprazole",
            prescriber_npi="1234567890",
        )
        assert len(alerts) == 0


class TestPrescriptionProcessing:
    """Test the full event processing pipeline."""

    def test_process_valid_event(self, sample_prescription: PrescriptionEvent) -> None:
        processor = PrescriptionProcessor()
        result = processor.process_event(sample_prescription.model_dump())
        assert result["status"] == "ok"
        assert processor.stats["processed"] == 1
        assert processor.stats["errors"] == 0

    def test_process_event_from_bytes(self, sample_prescription: PrescriptionEvent) -> None:
        processor = PrescriptionProcessor()
        result = processor.process_event(sample_prescription.to_bytes())
        assert result["status"] == "ok"

    def test_process_invalid_json(self) -> None:
        processor = PrescriptionProcessor()
        result = processor.process_event(b"not valid json{{{")
        assert result["status"] == "deserialization_error"
        assert processor.stats["errors"] == 1

    def test_process_batch_updates_aggregation(
        self, prescription_batch: list[dict[str, Any]]
    ) -> None:
        processor = PrescriptionProcessor()
        for event in prescription_batch:
            result = processor.process_event(event)
            assert result["status"] == "ok"

        assert processor.stats["processed"] == 20
        assert processor.stats["current_window_size"] == 20

    def test_interaction_detected_during_processing(
        self,
        sample_prescription_nsaid: PrescriptionEvent,
        sample_prescription_anticoagulant: PrescriptionEvent,
    ) -> None:
        processor = PrescriptionProcessor()

        # Process NSAID first
        result1 = processor.process_event(sample_prescription_nsaid.model_dump())
        assert result1["status"] == "ok"
        assert len(result1["interaction_alerts"]) == 0

        # Process anticoagulant for same patient - should trigger interaction
        result2 = processor.process_event(sample_prescription_anticoagulant.model_dump())
        assert result2["status"] == "ok"
        assert len(result2["interaction_alerts"]) == 1
        assert result2["interaction_alerts"][0]["risk_level"] == "high"

    def test_pending_alerts_cleared(self) -> None:
        processor = PrescriptionProcessor()

        # Create an interaction
        processor.check_interactions("PAT-TEST-999", "anticoagulant", "Warfarin", "1234567890")
        processor.check_interactions("PAT-TEST-999", "nsaid", "Ibuprofen", "1234567890")

        alerts = processor.get_pending_alerts()
        assert len(alerts) == 1

        # Second call should return empty (cleared)
        alerts2 = processor.get_pending_alerts()
        assert len(alerts2) == 0


class TestAnomalyDetection:
    """Test prescribing anomaly detection."""

    def test_no_anomaly_normal_volume(self) -> None:
        processor = PrescriptionProcessor()
        window_counts = {"statin": 100, "biguanide": 80, "ppi": 50}
        anomalies = processor.detect_anomalies(window_counts)
        assert len(anomalies) == 0

    def test_anomaly_detected_high_volume(self) -> None:
        processor = PrescriptionProcessor()
        # opioid baseline is 50, so 50 * 3.0 = 150 threshold
        window_counts = {"opioid": 200}
        anomalies = processor.detect_anomalies(window_counts)
        assert len(anomalies) == 1
        assert anomalies[0]["drug_class"] == "opioid"
        assert anomalies[0]["count"] == 200
        assert anomalies[0]["ratio"] > 3.0

    def test_no_anomaly_unknown_drug_class(self) -> None:
        processor = PrescriptionProcessor()
        window_counts = {"unknown_class": 9999}
        anomalies = processor.detect_anomalies(window_counts)
        assert len(anomalies) == 0
