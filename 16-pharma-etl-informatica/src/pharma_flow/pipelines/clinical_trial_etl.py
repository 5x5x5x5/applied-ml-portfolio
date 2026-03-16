"""
Clinical Trial Data ETL Pipeline.

Loads clinical trial data from XML files and relational reference tables,
transforms it into fact and dimension tables for a clinical trial data warehouse.

Pipeline flow:
  1. Source: XML files with trial data (ClinicalTrials.gov format)
  2. Source: Relational reference tables (investigators, sites, indications)
  3. SourceQualifier: Parse XML and project columns
  4. Expression: Calculate trial duration, classify phase, derive status flags
  5. Joiner: Join trial data with investigator reference
  6. Joiner: Join with site reference
  7. Aggregator: Aggregate enrollment counts by phase and indication
  8. Normalizer: Normalize multi-arm trial designs into individual rows
  9. Lookup: Map indication codes to standard MedDRA terms
  10. SequenceGenerator: Generate fact table surrogate keys
  11. Target: fact_clinical_trial and dim_trial tables
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from pharma_flow.framework.mapping import (
    LoadType,
    Mapping,
    MappingParameter,
    SessionConfig,
    SourceDefinition,
    SourceType,
    TargetDefinition,
    TargetType,
)
from pharma_flow.framework.transformations import (
    Aggregator,
    Expression,
    Filter,
    Joiner,
    JoinType,
    Lookup,
    LookupMode,
    Normalizer,
    SequenceGenerator,
    Sorter,
    SourceQualifier,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Phase classification helper
# ---------------------------------------------------------------------------

PHASE_MAP = {
    "phase 1": "Phase I",
    "phase 1/phase 2": "Phase I/II",
    "phase 2": "Phase II",
    "phase 2/phase 3": "Phase II/III",
    "phase 3": "Phase III",
    "phase 4": "Phase IV",
    "not applicable": "N/A",
    "early phase 1": "Phase 0",
}


def classify_phase(phase_raw: str) -> str:
    """Normalize clinical trial phase to standard format."""
    if pd.isna(phase_raw):
        return "Unknown"
    return PHASE_MAP.get(str(phase_raw).strip().lower(), "Unknown")


def calculate_trial_duration_days(start_date: str, end_date: str) -> int:
    """Calculate trial duration in days from date strings."""
    try:
        if pd.isna(start_date) or pd.isna(end_date):
            return 0
        fmt_patterns = ["%Y-%m-%d", "%B %d, %Y", "%B %Y", "%Y-%m"]
        start_dt = None
        end_dt = None
        for fmt in fmt_patterns:
            try:
                start_dt = datetime.strptime(str(start_date).strip(), fmt)
                break
            except ValueError:
                continue
        for fmt in fmt_patterns:
            try:
                end_dt = datetime.strptime(str(end_date).strip(), fmt)
                break
            except ValueError:
                continue
        if start_dt and end_dt:
            return max((end_dt - start_dt).days, 0)
    except Exception:
        pass
    return 0


def parse_enrollment(enrollment_str: str) -> int:
    """Parse enrollment count, handling various formats."""
    if pd.isna(enrollment_str):
        return 0
    cleaned = re.sub(r"[^\d]", "", str(enrollment_str))
    return int(cleaned) if cleaned else 0


# ---------------------------------------------------------------------------
# Reference data builders (for testing / standalone use)
# ---------------------------------------------------------------------------


def build_investigator_reference() -> pd.DataFrame:
    """Build sample investigator reference data."""
    return pd.DataFrame(
        {
            "investigator_id": ["INV001", "INV002", "INV003", "INV004", "INV005"],
            "investigator_name": [
                "Dr. Sarah Chen",
                "Dr. Michael Roberts",
                "Dr. Aisha Patel",
                "Dr. James Wilson",
                "Dr. Maria Garcia",
            ],
            "institution": [
                "Johns Hopkins University",
                "Mayo Clinic",
                "Cleveland Clinic",
                "MD Anderson Cancer Center",
                "Massachusetts General Hospital",
            ],
            "specialization": [
                "Oncology",
                "Cardiology",
                "Neurology",
                "Oncology",
                "Immunology",
            ],
        }
    )


def build_site_reference() -> pd.DataFrame:
    """Build sample clinical trial site reference data."""
    return pd.DataFrame(
        {
            "site_id": ["SITE01", "SITE02", "SITE03", "SITE04", "SITE05"],
            "site_name": [
                "Hopkins Baltimore",
                "Mayo Rochester",
                "Cleveland Main",
                "MDA Houston",
                "MGH Boston",
            ],
            "country": ["US", "US", "US", "US", "US"],
            "state": ["MD", "MN", "OH", "TX", "MA"],
        }
    )


def build_indication_lookup() -> pd.DataFrame:
    """Build indication code to MedDRA term lookup."""
    return pd.DataFrame(
        {
            "indication_code": [
                "C0006826",
                "C0011849",
                "C0002395",
                "C0020538",
                "C0036341",
                "C0007097",
            ],
            "meddra_preferred_term": [
                "Malignant neoplasm",
                "Diabetes mellitus",
                "Alzheimer disease",
                "Hypertension",
                "Schizophrenia",
                "Non-small cell lung carcinoma",
            ],
            "meddra_soc": [
                "Neoplasms",
                "Metabolism and nutrition",
                "Nervous system",
                "Vascular",
                "Psychiatric",
                "Neoplasms",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------


def build_clinical_trial_mapping(
    xml_source_path: str,
    investigator_ref: pd.DataFrame | None = None,
    site_ref: pd.DataFrame | None = None,
    indication_lkp: pd.DataFrame | None = None,
    target_connection: str = "",
) -> Mapping:
    """
    Build the clinical trial ETL mapping.

    Args:
        xml_source_path: Path to XML file with trial data.
        investigator_ref: Investigator reference DataFrame.
        site_ref: Site reference DataFrame.
        indication_lkp: Indication code lookup DataFrame.
        target_connection: SQLAlchemy connection string for target.

    Returns:
        Configured Mapping.
    """
    # Use defaults if not provided
    inv_ref = investigator_ref if investigator_ref is not None else build_investigator_reference()
    s_ref = site_ref if site_ref is not None else build_site_reference()
    ind_lkp = indication_lkp if indication_lkp is not None else build_indication_lookup()

    # -- Sources --
    xml_source = SourceDefinition(
        name="SRC_TRIAL_XML",
        source_type=SourceType.XML,
        file_path=xml_source_path,
        xml_root_element="clinical_study",
    )

    # -- Targets --
    fact_target = TargetDefinition(
        name="TGT_FACT_TRIAL",
        target_type=TargetType.FLAT_FILE if not target_connection else TargetType.RELATIONAL,
        connection_name=target_connection,
        table_name="fact_clinical_trial",
        file_path="/tmp/pharmaflow_fact_clinical_trial.csv",
        load_type=LoadType.INSERT,
    )

    dim_target = TargetDefinition(
        name="TGT_DIM_TRIAL",
        target_type=TargetType.FLAT_FILE if not target_connection else TargetType.RELATIONAL,
        connection_name=target_connection,
        table_name="dim_trial",
        file_path="/tmp/pharmaflow_dim_trial.csv",
        load_type=LoadType.UPSERT,
        key_columns=["nct_id"],
    )

    # -- Mapping --
    mapping = Mapping(
        name="m_clinical_trial_load",
        description="Clinical Trial XML to Data Warehouse ETL",
        sources=[xml_source],
        targets=[fact_target, dim_target],
        session_config=SessionConfig(
            commit_interval=1000,
            error_threshold=50,
        ),
    )

    mapping.add_parameter(MappingParameter("$$LOAD_DATE", "2026-03-05"))
    mapping.add_parameter(MappingParameter("$$DATA_SOURCE", "ClinicalTrials.gov"))

    # -- Transformations --

    # 1. Source Qualifier
    sq = SourceQualifier(
        name="SQ_TRIAL_XML",
        source_filter="nct_id.notna()",
    )

    # 2. Expression: Derive computed columns
    exp_derive = Expression(name="EXP_TRIAL_DERIVED")
    exp_derive.add_expression(
        "phase_std",
        lambda df: df["phase"].apply(classify_phase) if "phase" in df.columns else "Unknown",
    )
    exp_derive.add_expression(
        "trial_duration_days",
        lambda df: df.apply(
            lambda row: calculate_trial_duration_days(
                row.get("start_date", ""),
                row.get("completion_date", ""),
            ),
            axis=1,
        ),
    )
    exp_derive.add_expression(
        "enrollment_count",
        lambda df: df["enrollment"].apply(parse_enrollment) if "enrollment" in df.columns else 0,
    )
    exp_derive.add_expression(
        "is_active",
        lambda df: (
            df["overall_status"]
            .str.lower()
            .isin(["recruiting", "active, not recruiting", "enrolling by invitation"])
            .astype(int)
            if "overall_status" in df.columns
            else 0
        ),
    )
    exp_derive.add_expression(
        "study_type_std",
        lambda df: (
            df["study_type"].fillna("").str.strip().str.title()
            if "study_type" in df.columns
            else ""
        ),
    )

    # 3. Filter: Remove trials with no NCT ID
    fil = Filter(
        name="FIL_VALID_TRIAL",
        condition_func=lambda df: df["nct_id"].notna() & (df["nct_id"] != ""),
    )

    # 4. Joiner: Join with investigator reference
    jnr_inv = Joiner(
        name="JNR_INVESTIGATOR",
        detail_source=inv_ref,
        join_type=JoinType.LEFT_OUTER,
        master_keys=["lead_investigator_id"]
        if "lead_investigator_id" in []
        else ["investigator_id"],
        detail_keys=["investigator_id"],
    )

    # 5. Joiner: Join with site reference
    jnr_site = Joiner(
        name="JNR_SITE",
        detail_source=s_ref,
        join_type=JoinType.LEFT_OUTER,
        master_keys=["site_id"],
        detail_keys=["site_id"],
    )

    # 6. Lookup: Map indication codes to MedDRA terms
    lkp_indication = Lookup(
        name="LKP_INDICATION",
        lookup_source=ind_lkp,
        lookup_keys=["indication_code"],
        return_columns=["meddra_preferred_term", "meddra_soc"],
        mode=LookupMode.CONNECTED,
        default_values={
            "meddra_preferred_term": "Unmapped",
            "meddra_soc": "Unmapped",
        },
    )
    lkp_indication.build_cache()

    # 7. Normalizer: Normalize multi-arm trials
    nrm = Normalizer(
        name="NRM_TRIAL_ARMS",
        group_columns=["arm_1", "arm_2", "arm_3"],
        normalized_column="arm_description",
        index_column="arm_number",
        id_columns=["nct_id", "phase_std", "enrollment_count"],
    )

    # 8. Aggregator: Enrollment by phase and indication
    agg = Aggregator(
        name="AGG_ENROLLMENT",
        group_by=["phase_std", "indication_code"],
        aggregations={
            "enrollment_count": "sum",
            "nct_id": "count",
            "trial_duration_days": "mean",
        },
    )

    # 9. Sequence Generator for fact keys
    seq = SequenceGenerator(
        name="SEQ_TRIAL_FACT_KEY",
        output_column="trial_fact_key",
        start_value=1,
        increment=1,
    )

    # 10. Sorter: Sort by NCT ID
    srt = Sorter(
        name="SRT_NCT_ID",
        sort_keys=["nct_id"],
        ascending=[True],
    )

    # Wire transformations
    mapping.add_transformation(sq)
    mapping.add_transformation(exp_derive)
    mapping.add_transformation(fil)
    mapping.add_transformation(srt)
    mapping.add_transformation(lkp_indication)
    mapping.add_transformation(seq)

    return mapping


def run_clinical_trial_pipeline(
    xml_path: str,
    target_connection: str = "",
    runtime_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build and execute the clinical trial ETL pipeline."""
    from pharma_flow.framework.session import Session

    mapping = build_clinical_trial_mapping(xml_path, target_connection=target_connection)
    session = Session(
        name="s_clinical_trial_load",
        mapping=mapping,
        runtime_params=runtime_params or {},
    )

    stats = session.execute()
    return stats.summary()
