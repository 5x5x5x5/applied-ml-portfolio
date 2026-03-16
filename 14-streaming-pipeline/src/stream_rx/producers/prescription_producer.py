"""
Kafka producer that generates realistic prescription fill events.

Simulates a high-throughput stream of prescription dispense events as they
would arrive from pharmacy point-of-sale systems. Events are validated via
Pydantic before publishing, and partitioned by patient_id to guarantee
per-patient ordering.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta
from typing import Any

from confluent_kafka import KafkaError, KafkaException, Producer
from pydantic import ValidationError

from stream_rx.config import KafkaConfig, get_config
from stream_rx.logging_setup import get_logger
from stream_rx.models import PrescriptionEvent

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Realistic reference data for event generation
# ---------------------------------------------------------------------------

DRUG_CATALOG: list[dict[str, str]] = [
    {"ndc": "00002-4462-30", "name": "Atorvastatin 40mg", "drug_class": "statin"},
    {"ndc": "00093-7180-01", "name": "Metformin 500mg", "drug_class": "biguanide"},
    {"ndc": "00378-1800-01", "name": "Lisinopril 10mg", "drug_class": "ace_inhibitor"},
    {"ndc": "00069-0150-30", "name": "Amlodipine 5mg", "drug_class": "calcium_channel_blocker"},
    {"ndc": "00591-0405-01", "name": "Omeprazole 20mg", "drug_class": "ppi"},
    {"ndc": "00074-3799-13", "name": "Levothyroxine 50mcg", "drug_class": "thyroid"},
    {"ndc": "00093-0058-01", "name": "Losartan 50mg", "drug_class": "arb"},
    {"ndc": "00172-3637-60", "name": "Albuterol Inhaler", "drug_class": "bronchodilator"},
    {"ndc": "00781-1506-10", "name": "Gabapentin 300mg", "drug_class": "anticonvulsant"},
    {"ndc": "55111-0160-30", "name": "Sertraline 50mg", "drug_class": "ssri"},
    {"ndc": "00228-2658-11", "name": "Clopidogrel 75mg", "drug_class": "antiplatelet"},
    {"ndc": "00186-0377-28", "name": "Apixaban 5mg", "drug_class": "anticoagulant"},
    {"ndc": "00002-8215-01", "name": "Dulaglutide 1.5mg", "drug_class": "glp1_agonist"},
    {"ndc": "00169-4130-12", "name": "Insulin Glargine 100u/mL", "drug_class": "insulin"},
    {"ndc": "68462-0254-90", "name": "Rosuvastatin 20mg", "drug_class": "statin"},
    {"ndc": "00173-0882-20", "name": "Fluticasone/Salmeterol 250/50", "drug_class": "ics_laba"},
    {"ndc": "00310-0280-30", "name": "Warfarin 5mg", "drug_class": "anticoagulant"},
    {"ndc": "61958-1001-01", "name": "Oxycodone 5mg", "drug_class": "opioid"},
    {"ndc": "00591-5613-01", "name": "Tramadol 50mg", "drug_class": "opioid"},
    {"ndc": "00904-6214-61", "name": "Ibuprofen 800mg", "drug_class": "nsaid"},
]

PHARMACIES: list[dict[str, str]] = [
    {"ncpdp": "0312456", "name": "CVS Pharmacy #1823", "state": "NY"},
    {"ncpdp": "0487921", "name": "Walgreens #12450", "state": "CA"},
    {"ncpdp": "0623187", "name": "Rite Aid #4521", "state": "PA"},
    {"ncpdp": "0798432", "name": "Kroger Pharmacy #903", "state": "OH"},
    {"ncpdp": "0145698", "name": "Walmart Pharmacy #2871", "state": "TX"},
    {"ncpdp": "0956321", "name": "HEB Pharmacy #120", "state": "TX"},
    {"ncpdp": "0234789", "name": "Publix Pharmacy #687", "state": "FL"},
    {"ncpdp": "0567123", "name": "Costco Pharmacy #445", "state": "WA"},
]

PRESCRIBER_NPIS: list[str] = [
    "1234567890",
    "2345678901",
    "3456789012",
    "4567890123",
    "5678901234",
    "6789012345",
    "7890123456",
    "8901234567",
    "9012345678",
    "1357924680",
    "2468013579",
    "1122334455",
    "5544332211",
    "9988776655",
    "6655443322",
]

DIAGNOSIS_CODES: list[str] = [
    "E11.9",
    "I10",
    "J45.20",
    "E03.9",
    "E78.5",
    "F32.1",
    "M54.5",
    "K21.0",
    "G40.909",
    "I25.10",
]


def _generate_patient_id() -> str:
    """Generate a deterministic-looking patient ID from a fixed pool."""
    region = random.choice(["NE", "SE", "MW", "SW", "WE"])
    number = random.randint(100000, 999999)
    return f"PAT-{region}-{number}"


class PrescriptionProducer:
    """
    High-throughput Kafka producer for prescription fill events.

    Generates schema-validated events and publishes them to a Kafka topic,
    partitioned by patient_id to ensure per-patient ordering. Supports
    configurable throughput and graceful shutdown.
    """

    def __init__(
        self,
        kafka_config: KafkaConfig | None = None,
        events_per_second: float = 100.0,
        patient_pool_size: int = 10_000,
    ) -> None:
        self._config = kafka_config or get_config().kafka
        self._events_per_second = events_per_second
        self._patient_pool: list[str] = [_generate_patient_id() for _ in range(patient_pool_size)]
        self._running = False
        self._total_produced = 0
        self._total_errors = 0

        producer_conf: dict[str, Any] = {
            "bootstrap.servers": self._config.bootstrap_servers,
            "enable.idempotence": self._config.enable_idempotence,
            "acks": self._config.acks,
            "compression.type": self._config.compression_type,
            "batch.size": self._config.batch_size,
            "linger.ms": self._config.linger_ms,
            "retries": 5,
            "retry.backoff.ms": 200,
            "delivery.timeout.ms": 120_000,
            "client.id": "streamrx-prescription-producer",
        }
        self._producer = Producer(producer_conf)
        logger.info(
            "prescription_producer_initialized",
            bootstrap_servers=self._config.bootstrap_servers,
            topic=self._config.prescription_topic,
            target_eps=self._events_per_second,
        )

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _generate_event(self) -> PrescriptionEvent:
        """Build a single realistic prescription event."""
        drug = random.choice(DRUG_CATALOG)
        pharmacy = random.choice(PHARMACIES)
        prescriber_npi = random.choice(PRESCRIBER_NPIS)
        patient_id = random.choice(self._patient_pool)

        # Realistic quantity/supply distributions
        if drug["drug_class"] in ("insulin", "bronchodilator", "ics_laba"):
            quantity = float(random.choice([1, 2, 3]))
            days_supply = random.choice([30, 60, 90])
        elif drug["drug_class"] == "opioid":
            quantity = float(random.choice([10, 15, 20, 30, 60]))
            days_supply = random.choice([5, 7, 10, 14, 30])
        else:
            quantity = float(random.choice([30, 60, 90]))
            days_supply = random.choice([30, 60, 90])

        num_diag = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        diagnoses = random.sample(DIAGNOSIS_CODES, k=min(num_diag, len(DIAGNOSIS_CODES)))

        return PrescriptionEvent(
            patient_id=patient_id,
            drug_ndc=drug["ndc"],
            drug_name=drug["name"],
            drug_class=drug["drug_class"],
            prescriber_npi=prescriber_npi,
            pharmacy_ncpdp=pharmacy["ncpdp"],
            pharmacy_name=pharmacy["name"],
            pharmacy_state=pharmacy["state"],
            quantity=quantity,
            days_supply=days_supply,
            refill_number=random.randint(0, 11),
            diagnosis_codes=diagnoses,
            plan_id=f"PLAN-{random.randint(1000, 9999)}",
            timestamp=datetime.utcnow() - timedelta(seconds=random.uniform(0, 2)),
        )

    # ------------------------------------------------------------------
    # Delivery callback
    # ------------------------------------------------------------------

    def _delivery_callback(self, err: KafkaError | None, msg: Any) -> None:
        """Called once per message to indicate delivery result."""
        if err is not None:
            self._total_errors += 1
            logger.error(
                "delivery_failed",
                error=str(err),
                topic=msg.topic(),
                partition=msg.partition(),
            )
        else:
            self._total_produced += 1
            if self._total_produced % 10_000 == 0:
                logger.info(
                    "delivery_progress",
                    total_produced=self._total_produced,
                    total_errors=self._total_errors,
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                )

    # ------------------------------------------------------------------
    # Produce loop
    # ------------------------------------------------------------------

    def produce_one(self) -> PrescriptionEvent | None:
        """
        Generate, validate, and publish a single prescription event.

        Returns:
            The produced event, or None if validation/publish failed.
        """
        try:
            event = self._generate_event()
        except ValidationError as exc:
            logger.warning("event_validation_failed", errors=exc.errors())
            self._total_errors += 1
            return None

        try:
            self._producer.produce(
                topic=self._config.prescription_topic,
                key=event.partition_key().encode("utf-8"),
                value=event.to_bytes(),
                callback=self._delivery_callback,
            )
            self._producer.poll(0)  # trigger delivery callbacks
            return event
        except KafkaException as exc:
            logger.error("produce_failed", error=str(exc))
            self._total_errors += 1
            return None

    def run(self, max_events: int | None = None) -> None:
        """
        Run the continuous production loop.

        Args:
            max_events: Stop after this many events (None = run forever).
        """
        self._running = True
        interval = 1.0 / self._events_per_second
        count = 0
        logger.info("producer_started", target_eps=self._events_per_second)

        try:
            while self._running:
                if max_events is not None and count >= max_events:
                    break
                start = time.monotonic()
                self.produce_one()
                count += 1

                # Rate limiting
                elapsed = time.monotonic() - start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("producer_interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Flush remaining messages and clean up."""
        self._running = False
        remaining = self._producer.flush(timeout=30)
        logger.info(
            "producer_shutdown",
            total_produced=self._total_produced,
            total_errors=self._total_errors,
            unflushed=remaining,
        )

    @property
    def stats(self) -> dict[str, int]:
        """Return current production statistics."""
        return {
            "total_produced": self._total_produced,
            "total_errors": self._total_errors,
        }


def main() -> None:
    """Entry point for the prescription producer CLI."""
    import argparse

    from stream_rx.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="StreamRx Prescription Producer")
    parser.add_argument("--eps", type=float, default=100.0, help="Events per second")
    parser.add_argument("--max-events", type=int, default=None, help="Max events then exit")
    parser.add_argument("--patients", type=int, default=10_000, help="Patient pool size")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    producer = PrescriptionProducer(
        events_per_second=args.eps,
        patient_pool_size=args.patients,
    )
    producer.run(max_events=args.max_events)


if __name__ == "__main__":
    main()
