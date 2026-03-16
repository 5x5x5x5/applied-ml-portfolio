"""
Shared test fixtures for PharmaFlow test suite.

Provides sample DataFrames, temporary file fixtures, and
pre-configured mapping/session objects for unit and integration tests.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_drug_df() -> pd.DataFrame:
    """Sample drug master data with various quality scenarios."""
    return pd.DataFrame(
        {
            "drug_name": [
                "Lipitor",
                "  atorvastatin calcium  ",
                "METFORMIN HCL",
                "Humira",
                "Keytruda",
                "Bad Drug",
                "Eliquis",
                "lipitor",  # fuzzy dup of row 0
            ],
            "generic_name": [
                "Atorvastatin",
                "Atorvastatin",
                "Metformin",
                "Adalimumab",
                "Pembrolizumab",
                "InvalidDrug",
                "Apixaban",
                "Atorvastatin",
            ],
            "ndc_code": [
                "00071-0155-23",
                "00093-5057-01",
                "00378-0123-01",
                "00074-4339-02",
                "00006-3026-02",
                "INVALID",
                "00003-0894-21",
                "00071-0155-23",  # same NDC as row 0
            ],
            "strength": [
                "10mg",
                "10mg",
                "500mg",
                "40mg/0.8mL",
                "100mg/4mL",
                "",
                "5mg",
                "10mg",
            ],
            "dosage_form": [
                "Tablet",
                "tablet",
                "TABLET",
                "Injection",
                "Injection",
                "Unknown",
                "Tablet",
                "Tablet",
            ],
            "route": [
                "Oral",
                "oral",
                "ORAL",
                "Subcutaneous",
                "Intravenous",
                "",
                "Oral",
                "Oral",
            ],
            "manufacturer": [
                "Pfizer",
                "Teva",
                "Mylan",
                "AbbVie",
                "Merck",
                "Unknown",
                "BMS/Pfizer",
                "Pfizer",
            ],
            "dea_schedule": ["", "", "", "", "", "", "", ""],
            "therapeutic_class": [
                "Statin",
                "Statin",
                "Biguanide",
                "TNF Inhibitor",
                "PD-1 Inhibitor",
                "",
                "Anticoagulant",
                "Statin",
            ],
            "supplier_code": [
                "SUP01",
                "SUP02",
                "SUP01",
                "SUP01",
                "SUP01",
                "SUP03",
                "SUP02",
                "SUP02",
            ],
        }
    )


@pytest.fixture
def sample_drug_csv(tmp_path: Path, sample_drug_df: pd.DataFrame) -> Path:
    """Write sample drug data to a CSV file and return the path."""
    csv_path = tmp_path / "drug_supplier_01.csv"
    sample_drug_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_trial_xml(tmp_path: Path) -> Path:
    """Create a sample clinical trial XML file."""
    xml_content = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <clinical_studies>
      <clinical_study>
        <nct_id>NCT00000001</nct_id>
        <brief_title>Phase III Study of Drug X in Type 2 Diabetes</brief_title>
        <overall_status>Completed</overall_status>
        <phase>Phase 3</phase>
        <study_type>Interventional</study_type>
        <enrollment>1500</enrollment>
        <start_date>2020-01-15</start_date>
        <completion_date>2024-06-30</completion_date>
        <lead_sponsor>PharmaCo Inc</lead_sponsor>
        <indication_code>C0011849</indication_code>
        <investigator_id>INV001</investigator_id>
        <site_id>SITE01</site_id>
      </clinical_study>
      <clinical_study>
        <nct_id>NCT00000002</nct_id>
        <brief_title>Phase I/II Oncology Trial for Compound Y</brief_title>
        <overall_status>Recruiting</overall_status>
        <phase>Phase 1/Phase 2</phase>
        <study_type>Interventional</study_type>
        <enrollment>250</enrollment>
        <start_date>2023-03-01</start_date>
        <completion_date>2027-12-31</completion_date>
        <lead_sponsor>BioTech Labs</lead_sponsor>
        <indication_code>C0006826</indication_code>
        <investigator_id>INV003</investigator_id>
        <site_id>SITE03</site_id>
      </clinical_study>
      <clinical_study>
        <nct_id>NCT00000003</nct_id>
        <brief_title>Alzheimer Disease Prevention Study</brief_title>
        <overall_status>Active, not recruiting</overall_status>
        <phase>Phase 2</phase>
        <study_type>Interventional</study_type>
        <enrollment>800</enrollment>
        <start_date>2021-09-15</start_date>
        <completion_date>2026-03-31</completion_date>
        <lead_sponsor>NeuroPharm</lead_sponsor>
        <indication_code>C0002395</indication_code>
        <investigator_id>INV002</investigator_id>
        <site_id>SITE02</site_id>
      </clinical_study>
    </clinical_studies>
    """)
    xml_path = tmp_path / "clinical_trials.xml"
    xml_path.write_text(xml_content)
    return xml_path


@pytest.fixture
def sample_faers_files(tmp_path: Path) -> dict[str, Path]:
    """Create sample FAERS quarterly files ($ delimited)."""
    # DEMO file
    demo_data = [
        [
            "primaryid",
            "caseid",
            "age",
            "age_cod",
            "sex",
            "wt",
            "wt_cod",
            "reporter_country",
            "event_dt",
            "init_fda_dt",
        ],
        ["10001", "1001", "55", "YR", "F", "70", "KG", "US", "20250101", "20250115"],
        ["10002", "1002", "42", "YR", "M", "85", "KG", "US", "20250201", "20250215"],
        ["10003", "1003", "68", "YR", "F", "60", "KG", "JP", "20250301", "20250315"],
        ["10004", "1004", "33", "YR", "M", "90", "KG", "US", "20250101", "20250201"],
        ["10005", "1005", "71", "YR", "F", "55", "KG", "GB", "20250115", "20250301"],
    ]
    demo_path = tmp_path / "DEMO25Q4.txt"
    _write_dollar_delimited(demo_path, demo_data)

    # DRUG file
    drug_data = [
        ["primaryid", "caseid", "drug_seq", "role_cod", "drugname", "prod_ai", "route"],
        ["10001", "1001", "1", "PS", "LIPITOR", "ATORVASTATIN", "Oral"],
        ["10002", "1002", "1", "PS", "METFORMIN", "METFORMIN HCL", "Oral"],
        ["10003", "1003", "1", "PS", "LIPITOR", "ATORVASTATIN", "Oral"],
        ["10004", "1004", "1", "PS", "HUMIRA", "ADALIMUMAB", "Subcutaneous"],
        ["10005", "1005", "1", "PS", "LIPITOR", "ATORVASTATIN", "Oral"],
    ]
    drug_path = tmp_path / "DRUG25Q4.txt"
    _write_dollar_delimited(drug_path, drug_data)

    # REAC file
    reac_data = [
        ["primaryid", "caseid", "pt", "drug_rec_act"],
        ["10001", "1001", "Rhabdomyolysis", ""],
        ["10002", "1002", "Nausea", ""],
        ["10003", "1003", "Myalgia", ""],
        ["10004", "1004", "Injection site reaction", ""],
        ["10005", "1005", "Rhabdomyolysis", ""],
    ]
    reac_path = tmp_path / "REAC25Q4.txt"
    _write_dollar_delimited(reac_path, reac_data)

    # OUTC file
    outc_data = [
        ["primaryid", "caseid", "outc_cod"],
        ["10001", "1001", "HO"],
        ["10002", "1002", "OT"],
        ["10003", "1003", "OT"],
        ["10004", "1004", "OT"],
        ["10005", "1005", "DE"],
    ]
    outc_path = tmp_path / "OUTC25Q4.txt"
    _write_dollar_delimited(outc_path, outc_data)

    return {
        "demo": demo_path,
        "drug": drug_path,
        "reac": reac_path,
        "outc": outc_path,
    }


def _write_dollar_delimited(path: Path, rows: list[list[str]]) -> None:
    """Write rows to a $-delimited file."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="$")
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def existing_dim_drug_df() -> pd.DataFrame:
    """Existing drug dimension records for SCD comparison."""
    return pd.DataFrame(
        {
            "drug_key": [1, 2, 3],
            "ndc_code": ["00071-0155-23", "00378-0123-01", "00074-4339-02"],
            "drug_name": ["LIPITOR", "METFORMIN HCL", "HUMIRA"],
            "record_hash": [
                "abc123",  # Will differ from incoming
                "same_hash",
                "def456",
            ],
            "current_flag": ["Y", "Y", "Y"],
        }
    )
