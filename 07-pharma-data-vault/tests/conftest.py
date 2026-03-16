"""
Shared fixtures for PharmaDataVault tests.

Provides:
  - Temporary directories for staging/archive operations
  - Sample CSV data files for testing
  - VaultConfig with test-appropriate settings
  - Mock database engine for unit tests
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pharma_vault._config import VaultConfig


@pytest.fixture
def tmp_staging_dir(tmp_path: Path) -> Path:
    """Create a temporary staging directory."""
    staging = tmp_path / "staging"
    staging.mkdir()
    return staging


@pytest.fixture
def tmp_archive_dir(tmp_path: Path) -> Path:
    """Create a temporary archive directory."""
    archive = tmp_path / "archive"
    archive.mkdir()
    return archive


@pytest.fixture
def test_config(tmp_staging_dir: Path, tmp_archive_dir: Path, tmp_path: Path) -> VaultConfig:
    """Create a VaultConfig pointing to temporary directories."""
    return VaultConfig(
        db_host="localhost",
        db_port=5432,
        db_name="pharma_vault_test",
        db_user="test_user",
        db_password="test_password",
        staging_dir=tmp_staging_dir,
        archive_dir=tmp_archive_dir,
        log_dir=tmp_path / "logs",
        batch_size=100,
        max_retries=2,
        retry_delay_seconds=1,
    )


@pytest.fixture
def sample_drug_csv(tmp_staging_dir: Path) -> Path:
    """Create a sample drug feed CSV file."""
    filepath = tmp_staging_dir / "drug_feed_20260305.csv"
    rows = [
        [
            "ndc_code",
            "drug_name",
            "generic_name",
            "manufacturer",
            "dosage_form",
            "strength",
            "route",
            "dea_schedule",
            "approval_date",
            "therapeutic_class",
            "nda_number",
        ],
        [
            "12345-6789-01",
            "Pharmazol",
            "pharmazolium",
            "ACME PHARMA",
            "tablet",
            "500mg",
            "oral",
            "N/A",
            "2020-01-15",
            "Antihypertensive",
            "NDA-2020-001",
        ],
        [
            "12345-6789-02",
            "Pharmazol XR",
            "pharmazolium",
            "ACME PHARMA",
            "capsule",
            "750mg",
            "oral",
            "N/A",
            "2021-06-20",
            "Antihypertensive",
            "NDA-2021-002",
        ],
        [
            "98765-4321-01",
            "Oncovex",
            "oncovexium",
            "BIOTECH LABS",
            "injection",
            "10mg/mL",
            "IV",
            "N/A",
            "2022-03-10",
            "Antineoplastic",
            "NDA-2022-003",
        ],
        [
            "55555-1111-01",
            "Neurostab",
            "neurostabilin",
            "NEURO INC",
            "tablet",
            "200mg",
            "oral",
            "IV",
            "2019-11-01",
            "Anticonvulsant",
            "NDA-2019-004",
        ],
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return filepath


@pytest.fixture
def sample_drug_csv_with_errors(tmp_staging_dir: Path) -> Path:
    """Create a drug feed CSV with intentional data quality issues."""
    filepath = tmp_staging_dir / "drug_feed_20260305.csv"
    rows = [
        [
            "ndc_code",
            "drug_name",
            "generic_name",
            "manufacturer",
            "dosage_form",
            "strength",
            "route",
            "dea_schedule",
            "approval_date",
            "therapeutic_class",
            "nda_number",
        ],
        # Valid record
        [
            "12345-6789-01",
            "GoodDrug",
            "gooddrugium",
            "PHARMA CO",
            "tablet",
            "100mg",
            "oral",
            "",
            "2020-01-01",
            "Analgesic",
            "NDA-001",
        ],
        # Missing NDC
        [
            "",
            "BadDrug1",
            "baddrugium",
            "PHARMA CO",
            "tablet",
            "50mg",
            "oral",
            "",
            "2020-02-01",
            "Analgesic",
            "",
        ],
        # Missing drug name
        [
            "11111-2222-33",
            "",
            "noname",
            "PHARMA CO",
            "capsule",
            "25mg",
            "oral",
            "",
            "2020-03-01",
            "Other",
            "",
        ],
        # Invalid date
        [
            "22222-3333-44",
            "WeirdDate",
            "weirdium",
            "PHARMA CO",
            "tablet",
            "10mg",
            "oral",
            "",
            "not-a-date",
            "Analgesic",
            "",
        ],
        # Missing manufacturer
        [
            "33333-4444-55",
            "NoMfr",
            "nomfrium",
            "",
            "injection",
            "5mg/mL",
            "IV",
            "",
            "2021-01-01",
            "Antineoplastic",
            "",
        ],
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return filepath


@pytest.fixture
def sample_patient_csv(tmp_staging_dir: Path) -> Path:
    """Create a sample patient feed CSV file."""
    filepath = tmp_staging_dir / "patient_feed_20260305.csv"
    rows = [
        [
            "mrn",
            "date_of_birth",
            "age",
            "sex",
            "ethnicity",
            "race",
            "weight_kg",
            "height_cm",
            "country",
            "state",
            "smoking_status",
        ],
        [
            "MRN-001",
            "1985-03-15",
            "41",
            "M",
            "Not Hispanic",
            "White",
            "82.5",
            "178",
            "USA",
            "CA",
            "never",
        ],
        [
            "MRN-002",
            "1972-08-22",
            "53",
            "F",
            "Hispanic",
            "White",
            "65.0",
            "162",
            "USA",
            "TX",
            "former",
        ],
        [
            "MRN-003",
            "1990-11-01",
            "35",
            "M",
            "Not Hispanic",
            "Asian",
            "75.0",
            "170",
            "USA",
            "NY",
            "current",
        ],
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return filepath


@pytest.fixture
def sample_ae_csv(tmp_staging_dir: Path) -> Path:
    """Create a sample adverse event feed CSV file."""
    filepath = tmp_staging_dir / "ae_feed_20260305.csv"
    rows = [
        [
            "ae_report_id",
            "drug_ndc",
            "patient_mrn",
            "ae_term",
            "description",
            "severity",
            "seriousness",
            "causality",
            "outcome",
            "onset_date",
            "resolution_date",
            "report_date",
            "reporter_type",
            "meddra_code",
            "meddra_soc",
            "expectedness",
            "action_taken",
        ],
        [
            "AE-2026-001",
            "12345-6789-01",
            "MRN-001",
            "Headache",
            "Mild frontal headache",
            "mild",
            "N",
            "possible",
            "recovered",
            "2026-02-15",
            "2026-02-16",
            "2026-02-17",
            "physician",
            "10019211",
            "Nervous system disorders",
            "expected",
            "none",
        ],
        [
            "AE-2026-002",
            "98765-4321-01",
            "MRN-002",
            "Nausea",
            "Persistent nausea requiring antiemetic",
            "moderate",
            "N",
            "probable",
            "recovered",
            "2026-02-20",
            "2026-02-25",
            "2026-02-21",
            "physician",
            "10028813",
            "Gastrointestinal disorders",
            "expected",
            "dose_reduced",
        ],
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return filepath


@pytest.fixture
def sample_mfg_fixed_width(tmp_staging_dir: Path) -> Path:
    """Create a sample manufacturing feed fixed-width file."""
    filepath = tmp_staging_dir / "mfg_feed_20260305.dat"
    content = textwrap.dedent("""\
        HDR MANUFACTURING FEED 20260305
        12345-6789-01LOT-2026-A001   50000.00    kg                  2026-01-152026-07-15FAC-MFG-001         LINE-A                        passed              2026-01-20John Smith QC Analyst                                                                                    98.50N                             2026-01-252-8C refrigerated
        12345-6789-01LOT-2026-A002   50000.00    kg                  2026-02-012026-08-01FAC-MFG-001         LINE-B                        pending             2026-02-05                                                                                                      97.20N
        TRL 00002 RECORDS
    """)

    filepath.write_text(content, encoding="utf-8")
    return filepath


@pytest.fixture
def checksum_file(sample_drug_csv: Path) -> Path:
    """Create a companion MD5 checksum file for the drug CSV."""
    import hashlib

    hasher = hashlib.md5()
    with open(sample_drug_csv, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    md5_file = sample_drug_csv.with_suffix(".csv.md5")
    md5_file.write_text(f"{hasher.hexdigest()}  {sample_drug_csv.name}\n")
    return md5_file


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock SQLAlchemy engine for unit tests."""
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=connection)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine
