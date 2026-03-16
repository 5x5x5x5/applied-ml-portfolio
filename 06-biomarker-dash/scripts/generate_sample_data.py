#!/usr/bin/env python3
"""Generate realistic synthetic biomarker data for demo and testing.

Creates patient profiles with realistic biomarker time series including
normal patterns, anomalies, trends, and clinical events.

Usage:
    python scripts/generate_sample_data.py [--patients N] [--hours H] [--output FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biomarker_dash.schemas import (
    NORMAL_RANGES,
    BiomarkerReading,
    BiomarkerType,
    PatientContext,
    Sex,
)

logger = logging.getLogger(__name__)

# Patient profile templates
PATIENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "patient_id": "P001",
        "age": 62,
        "sex": "male",
        "conditions": ["hypertension", "diabetes_type2"],
        "medications": ["metformin", "lisinopril"],
        "biomarker_adjustments": {
            "glucose": {"base_shift": 20, "volatility": 1.5},
            "blood_pressure_sys": {"base_shift": 15, "volatility": 1.2},
        },
    },
    {
        "patient_id": "P002",
        "age": 45,
        "sex": "female",
        "conditions": ["hypothyroidism"],
        "medications": ["levothyroxine"],
        "biomarker_adjustments": {
            "tsh": {"base_shift": 2.0, "volatility": 1.3},
        },
    },
    {
        "patient_id": "P003",
        "age": 78,
        "sex": "male",
        "conditions": ["coronary_artery_disease", "atrial_fibrillation"],
        "medications": ["aspirin", "warfarin", "metoprolol"],
        "biomarker_adjustments": {
            "heart_rate": {"base_shift": -5, "volatility": 1.8},
            "troponin": {"base_shift": 0.01, "volatility": 2.0},
        },
    },
    {
        "patient_id": "P004",
        "age": 33,
        "sex": "female",
        "conditions": [],
        "medications": [],
        "biomarker_adjustments": {},
    },
    {
        "patient_id": "P005",
        "age": 55,
        "sex": "male",
        "conditions": ["chronic_kidney_disease"],
        "medications": ["amlodipine"],
        "biomarker_adjustments": {
            "creatinine": {"base_shift": 0.5, "volatility": 1.4},
            "potassium": {"base_shift": 0.4, "volatility": 1.3},
        },
    },
]

# Biomarker generation parameters
BIOMARKER_PARAMS: dict[str, dict[str, float]] = {
    "glucose": {"noise_std": 8.0, "circadian_amplitude": 5.0, "circadian_phase": 6.0},
    "hemoglobin": {"noise_std": 0.3, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "wbc": {"noise_std": 0.8, "circadian_amplitude": 0.5, "circadian_phase": 14.0},
    "platelet": {"noise_std": 15.0, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "creatinine": {"noise_std": 0.05, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "alt": {"noise_std": 3.0, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "ast": {"noise_std": 2.5, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "heart_rate": {"noise_std": 5.0, "circadian_amplitude": 8.0, "circadian_phase": 14.0},
    "blood_pressure_sys": {"noise_std": 6.0, "circadian_amplitude": 8.0, "circadian_phase": 10.0},
    "blood_pressure_dia": {"noise_std": 4.0, "circadian_amplitude": 5.0, "circadian_phase": 10.0},
    "temperature": {"noise_std": 0.2, "circadian_amplitude": 0.4, "circadian_phase": 16.0},
    "oxygen_sat": {"noise_std": 0.5, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "potassium": {"noise_std": 0.15, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "sodium": {"noise_std": 1.0, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "tsh": {"noise_std": 0.3, "circadian_amplitude": 0.5, "circadian_phase": 2.0},
    "cholesterol_total": {"noise_std": 5.0, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "troponin": {"noise_std": 0.002, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
    "crp": {"noise_std": 0.3, "circadian_amplitude": 0.0, "circadian_phase": 0.0},
}

# Biomarker units
UNITS: dict[str, str] = {
    "glucose": "mg/dL",
    "hemoglobin": "g/dL",
    "wbc": "K/uL",
    "platelet": "K/uL",
    "creatinine": "mg/dL",
    "alt": "U/L",
    "ast": "U/L",
    "heart_rate": "bpm",
    "blood_pressure_sys": "mmHg",
    "blood_pressure_dia": "mmHg",
    "temperature": "F",
    "oxygen_sat": "%",
    "potassium": "mEq/L",
    "sodium": "mEq/L",
    "tsh": "mIU/L",
    "cholesterol_total": "mg/dL",
    "troponin": "ng/mL",
    "crp": "mg/L",
}


class BiomarkerDataGenerator:
    """Generates realistic synthetic biomarker time series data."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_patient_data(
        self,
        template: dict[str, Any],
        hours: int = 24,
        interval_minutes: int = 5,
    ) -> tuple[PatientContext, list[BiomarkerReading]]:
        """Generate a full set of biomarker data for one patient."""
        patient_ctx = PatientContext(
            patient_id=template["patient_id"],
            age=template["age"],
            sex=Sex(template["sex"]),
            conditions=template.get("conditions", []),
            medications=template.get("medications", []),
        )

        readings: list[BiomarkerReading] = []
        adjustments = template.get("biomarker_adjustments", {})

        # Select a subset of biomarkers for this patient (not all patients
        # have all lab tests at all times)
        # Vitals are always present
        vitals = [
            "heart_rate",
            "blood_pressure_sys",
            "blood_pressure_dia",
            "temperature",
            "oxygen_sat",
        ]
        # Labs based on conditions
        labs = ["glucose", "potassium", "sodium", "creatinine", "hemoglobin"]
        if "diabetes" in str(template.get("conditions", [])):
            labs.append("cholesterol_total")
        if "coronary" in str(template.get("conditions", [])):
            labs.extend(["troponin", "crp"])
        if "hypothyroidism" in str(template.get("conditions", [])):
            labs.append("tsh")

        active_biomarkers = list(set(vitals + labs))

        # Time points
        now = datetime.utcnow()
        start = now - timedelta(hours=hours)
        n_points = (hours * 60) // interval_minutes

        # Determine if there should be a clinical event
        has_event = self._rng.random() > 0.3  # 70% chance of an event
        event_start_idx = int(n_points * (0.5 + self._rng.random() * 0.3)) if has_event else -1
        event_duration = int(n_points * 0.1)
        event_biomarker = random.choice(active_biomarkers) if has_event else None

        for bm_name in active_biomarkers:
            bm_type = BiomarkerType(bm_name)
            low, high = NORMAL_RANGES.get(bm_type, (0.0, 100.0))
            midpoint = (low + high) / 2.0
            params = BIOMARKER_PARAMS.get(bm_name, {})
            adj = adjustments.get(bm_name, {})

            base = midpoint + adj.get("base_shift", 0.0)
            noise_std = params.get("noise_std", 1.0) * adj.get("volatility", 1.0)
            circadian_amp = params.get("circadian_amplitude", 0.0)
            circadian_phase = params.get("circadian_phase", 0.0)

            # Determine sample frequency (vitals more frequent than labs)
            if bm_name in vitals:
                sample_interval = 1  # Every time point
            else:
                sample_interval = max(1, 60 // interval_minutes)  # Roughly hourly for labs

            for i in range(0, n_points, sample_interval):
                t = start + timedelta(minutes=i * interval_minutes)
                hour_of_day = t.hour + t.minute / 60.0

                # Base value + circadian rhythm
                circadian = circadian_amp * np.sin(
                    2 * np.pi * (hour_of_day - circadian_phase) / 24.0
                )
                value = base + float(circadian)

                # Add noise
                value += float(self._rng.normal(0, noise_std))

                # Inject clinical event (spike or dip)
                if (
                    has_event
                    and bm_name == event_biomarker
                    and event_start_idx <= i < event_start_idx + event_duration
                ):
                    event_progress = (i - event_start_idx) / max(event_duration, 1)
                    # Bell curve for event intensity
                    intensity = np.exp(-4 * (event_progress - 0.5) ** 2)
                    event_magnitude = (high - low) * 0.4
                    # Direction: go above range or below range
                    direction = 1.0 if self._rng.random() > 0.5 else -1.0
                    value += float(direction * event_magnitude * intensity)

                # Slow trend for some biomarkers
                if bm_name in adj:
                    trend_rate = adj.get("base_shift", 0) * 0.001
                    value += trend_rate * i

                # Clamp to physically possible ranges
                value = max(0.0, value)
                if bm_name == "oxygen_sat":
                    value = min(100.0, value)

                reading = BiomarkerReading(
                    reading_id=str(uuid.uuid4()),
                    patient_id=template["patient_id"],
                    biomarker_type=bm_type,
                    value=round(value, 3),
                    unit=UNITS.get(bm_name, ""),
                    timestamp=t,
                    source="synthetic",
                )
                readings.append(reading)

        # Sort by timestamp
        readings.sort(key=lambda r: r.timestamp)
        return patient_ctx, readings

    def generate_all_patients(
        self,
        hours: int = 24,
        interval_minutes: int = 5,
    ) -> tuple[list[PatientContext], list[BiomarkerReading]]:
        """Generate data for all patient templates."""
        all_contexts: list[PatientContext] = []
        all_readings: list[BiomarkerReading] = []

        for template in PATIENT_TEMPLATES:
            ctx, readings = self.generate_patient_data(
                template, hours=hours, interval_minutes=interval_minutes
            )
            all_contexts.append(ctx)
            all_readings.extend(readings)
            logger.info(
                "Generated %d readings for patient %s (%d biomarkers)",
                len(readings),
                ctx.patient_id,
                len({r.biomarker_type for r in readings}),
            )

        logger.info(
            "Total: %d patients, %d readings",
            len(all_contexts),
            len(all_readings),
        )
        return all_contexts, all_readings


async def send_to_api(
    readings: list[BiomarkerReading],
    patients: list[PatientContext],
    base_url: str = "http://localhost:8000",
) -> None:
    """Send generated data to the running BiomarkerDash API."""
    import httpx

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Register patients
        for ctx in patients:
            resp = await client.post(
                "/api/patients",
                json=ctx.model_dump(mode="json"),
            )
            logger.info("Registered patient %s: %d", ctx.patient_id, resp.status_code)

        # Send readings in chronological order
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)
        batch_size = 50
        for i in range(0, len(sorted_readings), batch_size):
            batch = sorted_readings[i : i + batch_size]
            for reading in batch:
                try:
                    resp = await client.post(
                        "/api/biomarkers",
                        json=reading.model_dump(mode="json"),
                    )
                    if resp.status_code != 201:
                        logger.warning("Failed to submit reading: %d", resp.status_code)
                except httpx.ConnectError:
                    logger.error("Cannot connect to API at %s", base_url)
                    return

            logger.info(
                "Sent %d/%d readings",
                min(i + batch_size, len(sorted_readings)),
                len(sorted_readings),
            )
            await asyncio.sleep(0.05)  # Small delay to avoid overwhelming the API


def save_to_file(
    patients: list[PatientContext],
    readings: list[BiomarkerReading],
    output_path: str,
) -> None:
    """Save generated data to a JSON file."""
    data = {
        "generated_at": datetime.utcnow().isoformat(),
        "patients": [p.model_dump(mode="json") for p in patients],
        "readings": [r.model_dump(mode="json") for r in readings],
        "summary": {
            "total_patients": len(patients),
            "total_readings": len(readings),
            "biomarker_types": list({r.biomarker_type.value for r in readings}),
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved %d readings to %s", len(readings), output_path)


def main() -> None:
    """Main entry point for sample data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic biomarker data")
    parser.add_argument(
        "--patients",
        type=int,
        default=5,
        help="Number of patients (uses templates, max 5)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours of data to generate",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Data point interval in minutes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_data.json",
        help="Output file path",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send data to running API instead of saving to file",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (used with --send)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    generator = BiomarkerDataGenerator(seed=args.seed)
    patients, readings = generator.generate_all_patients(
        hours=args.hours,
        interval_minutes=args.interval,
    )

    # Limit patients if requested
    patient_ids = {p.patient_id for p in patients[: args.patients]}
    patients = [p for p in patients if p.patient_id in patient_ids]
    readings = [r for r in readings if r.patient_id in patient_ids]

    if args.send:
        asyncio.run(send_to_api(readings, patients, base_url=args.api_url))
    else:
        save_to_file(patients, readings, args.output)

    print(f"Generated {len(readings)} readings for {len(patients)} patients")


if __name__ == "__main__":
    main()
