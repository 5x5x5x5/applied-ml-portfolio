"""
Faust stream processor for real-time prescription event analysis.

Consumes prescription fill events from Kafka and performs:
- Drug interaction checking against a curated knowledge base
- Tumbling-window aggregation of prescription counts by drug
- Anomaly detection for unusual prescribing patterns
- Alert emission for dangerous drug combinations
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import faust
import orjson
from faust import Stream

from stream_rx.config import KafkaConfig, RedisConfig, get_config
from stream_rx.logging_setup import get_logger
from stream_rx.models import (
    AlertPriority,
    DrugInteractionAlert,
    PrescriptionEvent,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known drug interaction pairs (severity-rated)
# ---------------------------------------------------------------------------

KNOWN_INTERACTIONS: list[dict[str, Any]] = [
    {
        "drug_a_class": "anticoagulant",
        "drug_b_class": "nsaid",
        "risk": "high",
        "description": "Increased bleeding risk: anticoagulant + NSAID combination",
    },
    {
        "drug_a_class": "anticoagulant",
        "drug_b_class": "antiplatelet",
        "risk": "high",
        "description": "Triple antithrombotic risk: anticoagulant + antiplatelet",
    },
    {
        "drug_a_class": "opioid",
        "drug_b_class": "ssri",
        "risk": "moderate",
        "description": "Serotonin syndrome risk: opioid (tramadol) + SSRI",
    },
    {
        "drug_a_class": "ace_inhibitor",
        "drug_b_class": "arb",
        "risk": "high",
        "description": "Dual RAAS blockade: ACE inhibitor + ARB combination",
    },
    {
        "drug_a_class": "statin",
        "drug_b_class": "anticoagulant",
        "risk": "moderate",
        "description": "Increased myopathy risk with certain statin-anticoagulant combos",
    },
    {
        "drug_a_class": "insulin",
        "drug_b_class": "biguanide",
        "risk": "low",
        "description": "Additive hypoglycemia risk: insulin + metformin",
    },
    {
        "drug_a_class": "opioid",
        "drug_b_class": "anticonvulsant",
        "risk": "moderate",
        "description": "CNS depression risk: opioid + gabapentinoid",
    },
    {
        "drug_a_class": "ssri",
        "drug_b_class": "nsaid",
        "risk": "moderate",
        "description": "Increased GI bleeding risk: SSRI + NSAID",
    },
]

# Build a lookup set for fast interaction checking
_INTERACTION_LOOKUP: dict[tuple[str, str], dict[str, str]] = {}
for _ix in KNOWN_INTERACTIONS:
    pair = (_ix["drug_a_class"], _ix["drug_b_class"])
    reverse = (_ix["drug_b_class"], _ix["drug_a_class"])
    _INTERACTION_LOOKUP[pair] = {"risk": _ix["risk"], "description": _ix["description"]}
    _INTERACTION_LOOKUP[reverse] = {"risk": _ix["risk"], "description": _ix["description"]}


# ---------------------------------------------------------------------------
# Faust record types
# ---------------------------------------------------------------------------


class RxRecord(faust.Record):
    """Faust-compatible record for prescription events."""

    event_id: str
    event_type: str
    timestamp: str
    patient_id: str
    drug_ndc: str
    drug_name: str
    drug_class: str
    prescriber_npi: str
    prescriber_name: str
    pharmacy_ncpdp: str
    pharmacy_name: str
    pharmacy_state: str
    quantity: float
    days_supply: int
    refill_number: int
    diagnosis_codes: list[str]
    plan_id: str


class DrugCount(faust.Record):
    """Aggregated prescription count for a drug in a time window."""

    drug_name: str
    drug_class: str
    count: int
    window_start: str
    window_end: str


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------


class PrescriptionProcessor:
    """
    Real-time prescription stream processor.

    Provides drug interaction checking, windowed aggregations, and anomaly
    detection. Designed to run as a Faust worker or be used standalone for
    testing.
    """

    def __init__(
        self,
        kafka_config: KafkaConfig | None = None,
        redis_config: RedisConfig | None = None,
    ) -> None:
        cfg = get_config()
        self._kafka_config = kafka_config or cfg.kafka
        self._redis_config = redis_config or cfg.redis

        # In-memory state for interaction checking
        self._patient_active_drugs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._drug_window_ttl_sec = 86_400 * 30  # 30-day window for active drugs

        # Aggregation state
        self._drug_counts: dict[str, int] = defaultdict(int)
        self._prescriber_counts: dict[str, int] = defaultdict(int)
        self._window_start: datetime = datetime.utcnow()
        self._window_duration = timedelta(minutes=5)

        # Anomaly detection baselines (populated from historical data)
        self._drug_baselines: dict[str, float] = {
            "opioid": 50.0,
            "statin": 200.0,
            "anticoagulant": 80.0,
            "ssri": 120.0,
            "ace_inhibitor": 150.0,
            "biguanide": 180.0,
            "nsaid": 100.0,
            "ppi": 90.0,
        }
        self._anomaly_stddev_factor = 3.0

        # Metrics
        self._processed_count = 0
        self._interaction_alerts_count = 0
        self._anomaly_alerts_count = 0
        self._errors_count = 0
        self._alerts_buffer: list[DrugInteractionAlert] = []

        logger.info("prescription_processor_initialized")

    # ------------------------------------------------------------------
    # Drug interaction checking
    # ------------------------------------------------------------------

    def check_interactions(
        self, patient_id: str, new_drug_class: str, new_drug_name: str, prescriber_npi: str
    ) -> list[DrugInteractionAlert]:
        """
        Check a newly prescribed drug against the patient's active medications.

        Maintains a per-patient sliding window of active drugs and checks each
        new prescription against known dangerous pairs.

        Returns:
            List of interaction alerts (empty if no interactions found).
        """
        alerts: list[DrugInteractionAlert] = []
        now = time.time()

        # Prune expired drugs for this patient
        active = self._patient_active_drugs[patient_id]
        active = [d for d in active if (now - d["timestamp"]) < self._drug_window_ttl_sec]
        self._patient_active_drugs[patient_id] = active

        # Check each active drug against the new one
        for existing in active:
            pair = (existing["drug_class"], new_drug_class)
            interaction = _INTERACTION_LOOKUP.get(pair)
            if interaction is not None:
                priority = {
                    "high": AlertPriority.CRITICAL,
                    "moderate": AlertPriority.HIGH,
                    "low": AlertPriority.MEDIUM,
                }.get(interaction["risk"], AlertPriority.LOW)

                alert = DrugInteractionAlert(
                    patient_id=patient_id,
                    drug_a=existing["drug_name"],
                    drug_b=new_drug_name,
                    interaction_description=interaction["description"],
                    risk_level=interaction["risk"],
                    priority=priority,
                    prescriber_npi=prescriber_npi,
                )
                alerts.append(alert)
                self._interaction_alerts_count += 1
                logger.warning(
                    "drug_interaction_detected",
                    patient_id=patient_id,
                    drug_a=existing["drug_name"],
                    drug_b=new_drug_name,
                    risk=interaction["risk"],
                    alert_id=alert.alert_id,
                )

        # Add the new drug to the patient's active list
        active.append(
            {
                "drug_name": new_drug_name,
                "drug_class": new_drug_class,
                "timestamp": now,
            }
        )

        return alerts

    # ------------------------------------------------------------------
    # Windowed aggregation
    # ------------------------------------------------------------------

    def _check_window_rotation(self) -> dict[str, int] | None:
        """
        Rotate the tumbling window if the current window has expired.

        Returns:
            The completed window's drug counts, or None if window is still active.
        """
        now = datetime.utcnow()
        if now - self._window_start >= self._window_duration:
            completed = dict(self._drug_counts)
            self._drug_counts = defaultdict(int)
            old_start = self._window_start
            self._window_start = now
            logger.info(
                "window_rotated",
                window_start=old_start.isoformat(),
                window_end=now.isoformat(),
                total_drugs=sum(completed.values()),
                unique_drugs=len(completed),
            )
            return completed
        return None

    def aggregate_prescription(self, drug_class: str) -> None:
        """Increment the count for a drug class in the current window."""
        self._drug_counts[drug_class] += 1

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(self, window_counts: dict[str, int]) -> list[dict[str, Any]]:
        """
        Detect anomalous prescription volumes compared to historical baselines.

        Uses a simple z-score approach: if the count for a drug class exceeds
        baseline * anomaly_stddev_factor, flag it.

        Returns:
            List of anomaly descriptors.
        """
        anomalies: list[dict[str, Any]] = []
        for drug_class, count in window_counts.items():
            baseline = self._drug_baselines.get(drug_class)
            if baseline is None:
                continue
            # Scale baseline to window duration (baselines are per 5-min window)
            if count > baseline * self._anomaly_stddev_factor:
                anomaly = {
                    "drug_class": drug_class,
                    "count": count,
                    "baseline": baseline,
                    "ratio": round(count / baseline, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                anomalies.append(anomaly)
                self._anomaly_alerts_count += 1
                logger.warning(
                    "prescribing_anomaly_detected",
                    drug_class=drug_class,
                    count=count,
                    baseline=baseline,
                    ratio=anomaly["ratio"],
                )
        return anomalies

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process_event(self, raw_event: bytes | dict[str, Any]) -> dict[str, Any]:
        """
        Process a single prescription event through the full pipeline.

        Steps:
            1. Deserialize and validate
            2. Check drug interactions
            3. Update windowed aggregation
            4. Check for window rotation and run anomaly detection
            5. Return processing result

        Args:
            raw_event: Raw bytes from Kafka or a pre-parsed dict.

        Returns:
            Dict with processing results including any alerts.
        """
        result: dict[str, Any] = {
            "status": "ok",
            "interaction_alerts": [],
            "anomalies": [],
            "window_completed": False,
        }

        try:
            if isinstance(raw_event, bytes):
                data = orjson.loads(raw_event)
            else:
                data = raw_event

            event = PrescriptionEvent(**data)
            self._processed_count += 1

            # Step 1: Drug interaction checking
            alerts = self.check_interactions(
                patient_id=event.patient_id,
                new_drug_class=event.drug_class,
                new_drug_name=event.drug_name,
                prescriber_npi=event.prescriber_npi,
            )
            result["interaction_alerts"] = [a.model_dump() for a in alerts]
            self._alerts_buffer.extend(alerts)

            # Step 2: Aggregate
            self.aggregate_prescription(event.drug_class)

            # Step 3: Window rotation + anomaly detection
            completed_window = self._check_window_rotation()
            if completed_window is not None:
                result["window_completed"] = True
                anomalies = self.detect_anomalies(completed_window)
                result["anomalies"] = anomalies

        except orjson.JSONDecodeError as exc:
            self._errors_count += 1
            logger.error("event_deserialization_failed", error=str(exc))
            result["status"] = "deserialization_error"
        except Exception as exc:
            self._errors_count += 1
            logger.error("event_processing_failed", error=str(exc), exc_info=True)
            result["status"] = "processing_error"

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "processed": self._processed_count,
            "interaction_alerts": self._interaction_alerts_count,
            "anomaly_alerts": self._anomaly_alerts_count,
            "errors": self._errors_count,
            "current_window_size": sum(self._drug_counts.values()),
            "active_patients": len(self._patient_active_drugs),
        }

    def get_pending_alerts(self) -> list[DrugInteractionAlert]:
        """Return and clear buffered alerts."""
        alerts = list(self._alerts_buffer)
        self._alerts_buffer.clear()
        return alerts


# ---------------------------------------------------------------------------
# Faust application factory
# ---------------------------------------------------------------------------


def create_faust_app(kafka_config: KafkaConfig | None = None) -> faust.App:
    """
    Create and configure a Faust application for prescription processing.

    Returns:
        Configured faust.App with agents and tables.
    """
    cfg = kafka_config or get_config().kafka

    app = faust.App(
        "streamrx-prescription-processor",
        broker=f"kafka://{cfg.bootstrap_servers}",
        topic_partitions=cfg.num_partitions,
        topic_replication_factor=cfg.replication_factor,
        consumer_auto_offset_reset="latest",
        processing_guarantee="exactly_once",
    )

    rx_topic = app.topic(cfg.prescription_topic, value_type=RxRecord)
    alerts_topic = app.topic(cfg.alerts_topic, value_serializer="json")

    # Tumbling window table for drug counts
    drug_counts_table = app.Table(
        "drug_counts",
        default=int,
    ).tumbling(timedelta(minutes=5), expires=timedelta(hours=1))

    processor = PrescriptionProcessor(kafka_config=cfg)

    @app.agent(rx_topic)
    async def process_prescriptions(stream: Stream[RxRecord]) -> None:
        """Faust agent that processes each prescription event."""
        async for event in stream:
            raw = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "patient_id": event.patient_id,
                "drug_ndc": event.drug_ndc,
                "drug_name": event.drug_name,
                "drug_class": event.drug_class,
                "prescriber_npi": event.prescriber_npi,
                "prescriber_name": event.prescriber_name,
                "pharmacy_ncpdp": event.pharmacy_ncpdp,
                "pharmacy_name": event.pharmacy_name,
                "pharmacy_state": event.pharmacy_state,
                "quantity": event.quantity,
                "days_supply": event.days_supply,
                "refill_number": event.refill_number,
                "diagnosis_codes": event.diagnosis_codes,
                "plan_id": event.plan_id,
            }
            result = processor.process_event(raw)

            # Update windowed table
            drug_counts_table[event.drug_class] += 1

            # Forward alerts to the alerts topic
            for alert_data in result.get("interaction_alerts", []):
                await alerts_topic.send(value=alert_data)

    @app.timer(interval=60.0)
    async def log_processor_stats() -> None:
        """Periodically log processing statistics."""
        logger.info("processor_stats", **processor.stats)

    return app
