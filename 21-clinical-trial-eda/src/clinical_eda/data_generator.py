"""Synthetic clinical trial data generator.

Generates realistic Phase III trial data for a novel anti-inflammatory
compound (RX-7281) vs placebo, including patient demographics, lab
biomarkers, treatment outcomes, and adverse events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

# Reproducible defaults
DEFAULT_SEED = 42
DEFAULT_N_PATIENTS = 1200

# Biomarker panels
BIOMARKER_COLS = [
    "crp_baseline",  # C-reactive protein (mg/L)
    "il6_baseline",  # Interleukin-6 (pg/mL)
    "tnf_alpha_baseline",  # TNF-alpha (pg/mL)
    "esr_baseline",  # Erythrocyte sedimentation rate (mm/hr)
    "wbc_baseline",  # White blood cell count (10^3/uL)
    "neutrophil_pct",  # Neutrophil percentage
    "albumin_baseline",  # Albumin (g/dL)
    "hemoglobin_baseline",  # Hemoglobin (g/dL)
]


def generate_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate patient demographic data."""
    ages = rng.normal(52, 14, n).clip(18, 85).astype(int)
    sexes = rng.choice(["M", "F"], n, p=[0.45, 0.55])
    races = rng.choice(
        ["White", "Black", "Asian", "Hispanic", "Other"],
        n,
        p=[0.55, 0.18, 0.12, 0.10, 0.05],
    )
    bmis = rng.normal(28.5, 5.5, n).clip(16, 55).round(1)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.45, 0.35, 0.20])
    disease_duration_yrs = rng.exponential(5.0, n).clip(0.5, 30).round(1)

    return pd.DataFrame(
        {
            "age": ages,
            "sex": sexes,
            "race": races,
            "bmi": bmis,
            "smoking_status": smoking,
            "disease_duration_years": disease_duration_yrs,
        }
    )


def generate_biomarkers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate baseline biomarker panel values."""
    crp = rng.lognormal(2.0, 0.8, n).clip(0.5, 200).round(2)
    il6 = rng.lognormal(1.5, 0.9, n).clip(0.1, 500).round(2)
    tnf = rng.lognormal(1.2, 0.6, n).clip(0.5, 100).round(2)
    esr = rng.normal(35, 18, n).clip(2, 120).astype(int)
    wbc = rng.normal(8.0, 2.5, n).clip(2.0, 20.0).round(1)
    neutrophil = rng.normal(62, 10, n).clip(20, 95).round(1)
    albumin = rng.normal(3.8, 0.5, n).clip(2.0, 5.5).round(1)
    hemoglobin = rng.normal(13.0, 1.8, n).clip(7.0, 18.0).round(1)

    return pd.DataFrame(
        {
            "crp_baseline": crp,
            "il6_baseline": il6,
            "tnf_baseline": tnf,
            "esr_baseline": esr,
            "wbc_baseline": wbc,
            "neutrophil_pct": neutrophil,
            "albumin_baseline": albumin,
            "hemoglobin_baseline": hemoglobin,
        }
    )


def generate_treatment_and_outcomes(
    demographics: pd.DataFrame,
    biomarkers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate treatment assignment, response, and survival data.

    Treatment effect is modulated by biomarker levels — patients with
    high CRP and IL-6 respond better to RX-7281, creating a discoverable
    biomarker signature.
    """
    n = len(demographics)

    # 1:1 randomization with stratification noise
    treatment = rng.choice(["RX-7281", "Placebo"], n)

    # Composite response score (lower = better)
    # Base score influenced by disease severity proxies
    base_score = (
        0.3 * (demographics["age"] - 50) / 15
        + 0.2 * (demographics["bmi"] - 25) / 5
        + 0.15 * np.log1p(biomarkers["crp_baseline"]) / 3
        + 0.1 * np.log1p(biomarkers["il6_baseline"]) / 3
        + rng.normal(0, 0.5, n)
    )

    # Treatment effect: stronger for high-inflammation patients
    inflammation_index = (
        np.log1p(biomarkers["crp_baseline"])
        + np.log1p(biomarkers["il6_baseline"])
        + np.log1p(biomarkers["tnf_baseline"])
    ) / 3

    treatment_effect = np.where(
        treatment == "RX-7281",
        -1.2 - 0.4 * (inflammation_index - inflammation_index.median()),
        0.0,
    )

    response_score = (base_score + treatment_effect).round(3)

    # Binary responder (ACR20-like threshold)
    responder = (response_score < -0.3).astype(int)

    # Time-to-event: progression-free survival (weeks)
    hazard = np.exp(0.5 * response_score + rng.normal(0, 0.2, n))
    pfs_weeks = (rng.exponential(1 / hazard) * 40).clip(1, 104).round(1)
    event_observed = rng.binomial(1, 0.7, n)  # 30% censored

    # Adverse events
    ae_prob = np.where(treatment == "RX-7281", 0.25, 0.18)
    had_ae = rng.binomial(1, ae_prob)
    ae_grade = np.where(
        had_ae,
        rng.choice([1, 2, 3, 4], n, p=[0.40, 0.35, 0.20, 0.05]),
        0,
    )

    # Inject ~3% missing data across some columns
    missing_mask = rng.random(n) < 0.03
    pfs_weeks_with_na = pfs_weeks.copy().astype(float)
    pfs_weeks_with_na[missing_mask] = np.nan

    return pd.DataFrame(
        {
            "treatment_arm": treatment,
            "response_score": response_score,
            "responder": responder,
            "pfs_weeks": pfs_weeks_with_na,
            "event_observed": event_observed,
            "adverse_event": had_ae,
            "ae_grade": ae_grade,
            "inflammation_index": inflammation_index.round(3),
        }
    )


def generate_trial_dataset(
    n_patients: int = DEFAULT_N_PATIENTS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a complete clinical trial dataset.

    Args:
        n_patients: Number of patients to simulate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with patient demographics, biomarkers, and outcomes.
    """
    rng = np.random.default_rng(seed)
    logger.info("generating_trial_data", n_patients=n_patients, seed=seed)

    demographics = generate_demographics(n_patients, rng)
    biomarkers = generate_biomarkers(n_patients, rng)
    outcomes = generate_treatment_and_outcomes(demographics, biomarkers, rng)

    df = pd.concat([demographics, biomarkers, outcomes], axis=1)
    df.index.name = "patient_id"
    df.index = [f"PT-{i:04d}" for i in range(1, n_patients + 1)]

    # Add site and enrollment info
    df["site_id"] = rng.choice([f"SITE-{s:02d}" for s in range(1, 21)], n_patients)
    df["enrollment_date"] = pd.date_range("2024-01-15", periods=n_patients, freq="4h")[
        :n_patients
    ]

    logger.info(
        "trial_data_generated",
        shape=df.shape,
        treatment_balance=df["treatment_arm"].value_counts().to_dict(),
    )
    return df
