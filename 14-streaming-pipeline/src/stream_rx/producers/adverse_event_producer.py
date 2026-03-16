"""
Kafka producer for adverse event (FAERS-like) reports.

Simulates spontaneous adverse event submissions with variable severity,
multi-drug involvement, and realistic MedDRA reaction terms. Events arrive
at a lower but burstier rate than prescriptions, reflecting real-world
reporting patterns.
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
from stream_rx.models import (
    AdverseEventReport,
    EventOutcome,
    ReactionTerm,
    SeverityLevel,
    SuspectDrug,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reference data for adverse event generation
# ---------------------------------------------------------------------------

SUSPECT_DRUG_POOL: list[dict[str, str]] = [
    {"name": "Atorvastatin", "ndc": "00002-4462-30", "drug_class": "statin", "route": "oral"},
    {"name": "Metformin", "ndc": "00093-7180-01", "drug_class": "biguanide", "route": "oral"},
    {"name": "Lisinopril", "ndc": "00378-1800-01", "drug_class": "ace_inhibitor", "route": "oral"},
    {"name": "Omeprazole", "ndc": "00591-0405-01", "drug_class": "ppi", "route": "oral"},
    {"name": "Sertraline", "ndc": "55111-0160-30", "drug_class": "ssri", "route": "oral"},
    {
        "name": "Gabapentin",
        "ndc": "00781-1506-10",
        "drug_class": "anticonvulsant",
        "route": "oral",
    },
    {"name": "Apixaban", "ndc": "00186-0377-28", "drug_class": "anticoagulant", "route": "oral"},
    {"name": "Warfarin", "ndc": "00310-0280-30", "drug_class": "anticoagulant", "route": "oral"},
    {"name": "Oxycodone", "ndc": "61958-1001-01", "drug_class": "opioid", "route": "oral"},
    {"name": "Tramadol", "ndc": "00591-5613-01", "drug_class": "opioid", "route": "oral"},
    {"name": "Ibuprofen", "ndc": "00904-6214-61", "drug_class": "nsaid", "route": "oral"},
    {"name": "Clopidogrel", "ndc": "00228-2658-11", "drug_class": "antiplatelet", "route": "oral"},
    {
        "name": "Insulin Glargine",
        "ndc": "00169-4130-12",
        "drug_class": "insulin",
        "route": "subcutaneous",
    },
    {
        "name": "Dulaglutide",
        "ndc": "00002-8215-01",
        "drug_class": "glp1_agonist",
        "route": "subcutaneous",
    },
    {
        "name": "Fluticasone/Salmeterol",
        "ndc": "00173-0882-20",
        "drug_class": "ics_laba",
        "route": "inhalation",
    },
]

# MedDRA Preferred Terms commonly seen in FAERS
REACTION_TERMS: list[dict[str, str]] = [
    {"term": "Nausea", "code": "10028813", "seriousness": "non_serious"},
    {"term": "Dizziness", "code": "10013573", "seriousness": "non_serious"},
    {"term": "Headache", "code": "10019211", "seriousness": "non_serious"},
    {"term": "Fatigue", "code": "10016256", "seriousness": "non_serious"},
    {"term": "Diarrhoea", "code": "10012735", "seriousness": "non_serious"},
    {"term": "Rash", "code": "10037844", "seriousness": "non_serious"},
    {"term": "Myalgia", "code": "10028411", "seriousness": "non_serious"},
    {"term": "Hepatotoxicity", "code": "10019851", "seriousness": "serious"},
    {"term": "Rhabdomyolysis", "code": "10039020", "seriousness": "serious"},
    {"term": "Anaphylaxis", "code": "10002198", "seriousness": "serious"},
    {"term": "Stevens-Johnson Syndrome", "code": "10042033", "seriousness": "serious"},
    {"term": "QT Prolongation", "code": "10037703", "seriousness": "serious"},
    {"term": "Gastrointestinal Haemorrhage", "code": "10017955", "seriousness": "serious"},
    {"term": "Acute Kidney Injury", "code": "10069339", "seriousness": "serious"},
    {"term": "Lactic Acidosis", "code": "10023676", "seriousness": "serious"},
    {"term": "Respiratory Depression", "code": "10038687", "seriousness": "serious"},
    {"term": "Serotonin Syndrome", "code": "10040108", "seriousness": "serious"},
    {"term": "Thrombocytopenia", "code": "10043554", "seriousness": "serious"},
    {"term": "Angioedema", "code": "10002424", "seriousness": "serious"},
    {"term": "Hypoglycaemia", "code": "10020993", "seriousness": "serious"},
]

# Drug-specific reaction affinities (some drugs more likely to cause certain reactions)
DRUG_REACTION_AFFINITY: dict[str, list[str]] = {
    "statin": ["Myalgia", "Rhabdomyolysis", "Hepatotoxicity"],
    "opioid": ["Respiratory Depression", "Nausea", "Dizziness"],
    "nsaid": ["Gastrointestinal Haemorrhage", "Acute Kidney Injury", "Nausea"],
    "anticoagulant": ["Gastrointestinal Haemorrhage", "Thrombocytopenia"],
    "ssri": ["Serotonin Syndrome", "Nausea", "Headache", "Dizziness"],
    "ace_inhibitor": ["Angioedema", "Dizziness", "Rash"],
    "biguanide": ["Lactic Acidosis", "Diarrhoea", "Nausea"],
    "insulin": ["Hypoglycaemia", "Rash"],
    "glp1_agonist": ["Nausea", "Diarrhoea", "Headache"],
    "anticonvulsant": ["Dizziness", "Fatigue", "Stevens-Johnson Syndrome"],
    "ppi": ["Headache", "Diarrhoea", "Nausea"],
    "antiplatelet": ["Gastrointestinal Haemorrhage", "Rash", "Thrombocytopenia"],
}

REPORTER_TYPES: list[str] = ["consumer", "physician", "pharmacist", "nurse", "other"]
SEVERITY_WEIGHTS: list[float] = [0.30, 0.35, 0.20, 0.10, 0.05]
OUTCOME_WEIGHTS: list[float] = [0.35, 0.25, 0.20, 0.05, 0.15]


class AdverseEventProducer:
    """
    Kafka producer for FAERS-like adverse event reports.

    Generates multi-drug adverse event submissions with realistic severity
    distributions and drug-reaction affinities. Supports bursty arrival
    patterns to simulate real-world reporting dynamics.
    """

    def __init__(
        self,
        kafka_config: KafkaConfig | None = None,
        events_per_second: float = 10.0,
        burst_probability: float = 0.05,
        burst_size_range: tuple[int, int] = (5, 20),
    ) -> None:
        self._config = kafka_config or get_config().kafka
        self._events_per_second = events_per_second
        self._burst_probability = burst_probability
        self._burst_size_range = burst_size_range
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
            "client.id": "streamrx-adverse-event-producer",
        }
        self._producer = Producer(producer_conf)
        logger.info(
            "adverse_event_producer_initialized",
            topic=self._config.adverse_event_topic,
            target_eps=self._events_per_second,
        )

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _pick_reactions(self, drug_class: str) -> list[ReactionTerm]:
        """Select reactions with affinity towards the given drug class."""
        affine_terms = DRUG_REACTION_AFFINITY.get(drug_class, [])
        all_terms = [r for r in REACTION_TERMS]

        # Weight affine reactions higher
        weights = []
        for rt in all_terms:
            w = 5.0 if rt["term"] in affine_terms else 1.0
            weights.append(w)

        num_reactions = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.15, 0.1])[0]
        chosen = random.choices(all_terms, weights=weights, k=num_reactions)

        # Deduplicate
        seen: set[str] = set()
        unique: list[ReactionTerm] = []
        for r in chosen:
            if r["term"] not in seen:
                seen.add(r["term"])
                unique.append(
                    ReactionTerm(
                        term=r["term"],
                        meddra_pt_code=r["code"],
                        seriousness=r["seriousness"],
                    )
                )
        return unique or [ReactionTerm(term="Nausea", meddra_pt_code="10028813")]

    def _build_suspect_drugs(self) -> list[SuspectDrug]:
        """Create a list of suspect drugs (1-3, occasionally more)."""
        num_suspects = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        drugs = random.sample(SUSPECT_DRUG_POOL, k=min(num_suspects, len(SUSPECT_DRUG_POOL)))

        result: list[SuspectDrug] = []
        for i, d in enumerate(drugs):
            start = datetime.utcnow() - timedelta(days=random.randint(1, 365))
            result.append(
                SuspectDrug(
                    drug_name=d["name"],
                    drug_ndc=d["ndc"],
                    drug_class=d["drug_class"],
                    role="suspect" if i == 0 else random.choice(["suspect", "interacting"]),
                    dose=f"{random.choice([5, 10, 20, 25, 50, 100, 200, 500])}mg",
                    route=d["route"],
                    start_date=start,
                    end_date=start + timedelta(days=random.randint(1, 90))
                    if random.random() > 0.3
                    else None,
                )
            )
        return result

    def _generate_event(self) -> AdverseEventReport:
        """Build a single FAERS-like adverse event report."""
        suspect_drugs = self._build_suspect_drugs()
        primary_class = suspect_drugs[0].drug_class
        reactions = self._pick_reactions(primary_class)

        # Determine severity from reaction seriousness
        has_serious = any(r.seriousness == "serious" for r in reactions)
        if has_serious:
            severity = random.choices(
                list(SeverityLevel),
                weights=[0.05, 0.20, 0.40, 0.25, 0.10],
            )[0]
        else:
            severity = random.choices(
                list(SeverityLevel),
                weights=SEVERITY_WEIGHTS,
            )[0]

        outcome = random.choices(list(EventOutcome), weights=OUTCOME_WEIGHTS)[0]
        if severity == SeverityLevel.FATAL:
            outcome = EventOutcome.FATAL

        patient_age = random.choices(
            [random.randint(18, 45), random.randint(45, 65), random.randint(65, 95)],
            weights=[0.2, 0.35, 0.45],
        )[0]

        concomitant = [
            random.choice(SUSPECT_DRUG_POOL)["name"] for _ in range(random.randint(0, 4))
        ]

        region = random.choice(["NE", "SE", "MW", "SW", "WE"])
        patient_id = f"PAT-{region}-{random.randint(100000, 999999)}"

        return AdverseEventReport(
            patient_id=patient_id,
            patient_age=patient_age,
            patient_sex=random.choice(["M", "F", "U"]),
            patient_weight_kg=round(random.gauss(80, 20), 1) if random.random() > 0.3 else None,
            suspect_drugs=suspect_drugs,
            concomitant_drugs=concomitant,
            reactions=reactions,
            severity=severity,
            outcome=outcome,
            hospitalized=severity
            in (SeverityLevel.SEVERE, SeverityLevel.LIFE_THREATENING, SeverityLevel.FATAL),
            reporter_type=random.choice(REPORTER_TYPES),
            reporter_country=random.choices(
                ["US", "GB", "DE", "FR", "JP", "CA", "AU"],
                weights=[0.50, 0.10, 0.08, 0.07, 0.10, 0.08, 0.07],
            )[0],
            narrative=self._generate_narrative(suspect_drugs, reactions, severity),
        )

    @staticmethod
    def _generate_narrative(
        drugs: list[SuspectDrug],
        reactions: list[ReactionTerm],
        severity: SeverityLevel,
    ) -> str:
        """Generate a brief narrative summary for the adverse event."""
        drug_names = ", ".join(d.drug_name for d in drugs)
        reaction_names = ", ".join(r.term for r in reactions)
        return (
            f"Patient experienced {reaction_names} while taking {drug_names}. "
            f"Severity assessed as {severity.value}. "
            f"Reporter noted onset approximately "
            f"{random.randint(1, 30)} days after starting therapy."
        )

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def _delivery_callback(self, err: KafkaError | None, msg: Any) -> None:
        if err is not None:
            self._total_errors += 1
            logger.error("ae_delivery_failed", error=str(err), topic=msg.topic())
        else:
            self._total_produced += 1
            if self._total_produced % 1000 == 0:
                logger.info(
                    "ae_delivery_progress",
                    total_produced=self._total_produced,
                    total_errors=self._total_errors,
                )

    def produce_one(self) -> AdverseEventReport | None:
        """Generate and publish a single adverse event report."""
        try:
            event = self._generate_event()
        except ValidationError as exc:
            logger.warning("ae_validation_failed", errors=exc.errors())
            self._total_errors += 1
            return None

        try:
            self._producer.produce(
                topic=self._config.adverse_event_topic,
                key=event.partition_key().encode("utf-8"),
                value=event.to_bytes(),
                callback=self._delivery_callback,
            )
            self._producer.poll(0)
            return event
        except KafkaException as exc:
            logger.error("ae_produce_failed", error=str(exc))
            self._total_errors += 1
            return None

    def produce_burst(self) -> int:
        """Produce a burst of events to simulate a batch submission."""
        count = random.randint(*self._burst_size_range)
        produced = 0
        for _ in range(count):
            if self.produce_one() is not None:
                produced += 1
        logger.info("ae_burst_produced", burst_size=count, produced=produced)
        return produced

    def run(self, max_events: int | None = None) -> None:
        """Run the continuous adverse event production loop."""
        self._running = True
        interval = 1.0 / self._events_per_second
        count = 0
        logger.info("ae_producer_started", target_eps=self._events_per_second)

        try:
            while self._running:
                if max_events is not None and count >= max_events:
                    break

                # Occasionally produce a burst
                if random.random() < self._burst_probability:
                    count += self.produce_burst()
                else:
                    start = time.monotonic()
                    self.produce_one()
                    count += 1
                    elapsed = time.monotonic() - start
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("ae_producer_interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Flush remaining messages and clean up."""
        self._running = False
        remaining = self._producer.flush(timeout=30)
        logger.info(
            "ae_producer_shutdown",
            total_produced=self._total_produced,
            total_errors=self._total_errors,
            unflushed=remaining,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_produced": self._total_produced,
            "total_errors": self._total_errors,
        }


def main() -> None:
    """Entry point for the adverse event producer CLI."""
    import argparse

    from stream_rx.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="StreamRx Adverse Event Producer")
    parser.add_argument("--eps", type=float, default=10.0, help="Events per second")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--burst-prob", type=float, default=0.05)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    producer = AdverseEventProducer(
        events_per_second=args.eps,
        burst_probability=args.burst_prob,
    )
    producer.run(max_events=args.max_events)


if __name__ == "__main__":
    main()
