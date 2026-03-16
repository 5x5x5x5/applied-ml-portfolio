"""
Drug Master Data ETL Pipeline.

Complete ETL for loading and maintaining a dimensional drug master table
from multiple supplier CSV files. Demonstrates all major transformation
types in an Informatica-style mapping.

Pipeline flow:
  1. Read CSV files from multiple drug suppliers
  2. SourceQualifier: Filter and project relevant columns
  3. Expression: Standardize drug names (uppercase, trim), validate NDC format,
     compute hash for SCD comparison
  4. Lookup: Check existing drug dimension for SCD Type 2
  5. Router: Route to INSERT (new drugs), UPDATE (changed drugs), NOCHANGE
  6. SequenceGenerator: Assign surrogate keys to new records
  7. UpdateStrategy: Set row disposition (DD_INSERT / DD_UPDATE)
  8. Target: Write to drug dimension table (SCD Type 2)

Fuzzy matching for deduplication uses Levenshtein distance on drug names.
"""

from __future__ import annotations

import re
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
    Expression,
    Filter,
    Lookup,
    LookupMode,
    Rank,
    Router,
    RowDisposition,
    SequenceGenerator,
    Sorter,
    SourceQualifier,
    UpdateStrategy,
    compute_row_hash,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# NDC validation regex: 5-4-2 or 5-3-2 or 4-4-2 format
# ---------------------------------------------------------------------------
NDC_PATTERN = re.compile(r"^\d{4,5}-\d{3,4}-\d{2}$")


def validate_ndc(ndc: str) -> bool:
    """Validate National Drug Code format."""
    if pd.isna(ndc):
        return False
    return bool(NDC_PATTERN.match(str(ndc).strip()))


def standardize_drug_name(name: str) -> str:
    """Standardize drug name: uppercase, trim, remove extra whitespace."""
    if pd.isna(name):
        return ""
    result = str(name).strip().upper()
    result = re.sub(r"\s+", " ", result)
    return result


def fuzzy_deduplicate(
    df: pd.DataFrame,
    name_column: str,
    threshold: int = 85,
) -> pd.DataFrame:
    """
    Deduplicate drug records using fuzzy matching on names.

    Groups similar drug names and keeps the first occurrence (by supplier
    priority or alphabetical order).
    """
    try:
        from thefuzz import fuzz
    except ImportError:
        logger.warning("thefuzz_not_available, skipping_fuzzy_dedup")
        return df

    if df.empty:
        return df

    names = df[name_column].tolist()
    groups: dict[int, list[int]] = {}  # group_id -> list of row indices
    assigned: set[int] = set()
    group_counter = 0

    for i in range(len(names)):
        if i in assigned:
            continue

        group_counter += 1
        groups[group_counter] = [i]
        assigned.add(i)

        for j in range(i + 1, len(names)):
            if j in assigned:
                continue
            score = fuzz.token_sort_ratio(str(names[i]), str(names[j]))
            if score >= threshold:
                groups[group_counter].append(j)
                assigned.add(j)

    # Keep first record from each group
    keep_indices = [indices[0] for indices in groups.values()]
    result = df.iloc[keep_indices].copy().reset_index(drop=True)

    logger.info(
        "fuzzy_dedup_complete",
        input_rows=len(df),
        output_rows=len(result),
        duplicates_removed=len(df) - len(result),
    )
    return result


# ---------------------------------------------------------------------------
# SCD Type 2 hash columns -- used to detect changes
# ---------------------------------------------------------------------------
SCD_COMPARE_COLUMNS = [
    "drug_name_std",
    "generic_name",
    "strength",
    "dosage_form",
    "route",
    "manufacturer",
    "dea_schedule",
    "therapeutic_class",
]


def build_drug_master_mapping(
    source_file_paths: list[str],
    target_connection: str = "",
    existing_dim_data: pd.DataFrame | None = None,
) -> Mapping:
    """
    Build the complete drug master ETL mapping.

    Args:
        source_file_paths: List of CSV file paths from drug suppliers.
        target_connection: SQLAlchemy connection string for target DB.
        existing_dim_data: Current dim_drug records for SCD comparison.

    Returns:
        Configured Mapping ready for Session execution.
    """
    # -- Sources --
    sources = []
    for i, path in enumerate(source_file_paths):
        sources.append(
            SourceDefinition(
                name=f"SRC_DRUG_SUPPLIER_{i + 1}",
                source_type=SourceType.FLAT_FILE,
                file_path=path,
                file_delimiter=",",
                file_has_header=True,
            )
        )

    # -- Target --
    target = TargetDefinition(
        name="TGT_DIM_DRUG",
        target_type=TargetType.FLAT_FILE if not target_connection else TargetType.RELATIONAL,
        connection_name=target_connection,
        table_name="dim_drug",
        file_path="/tmp/pharmaflow_dim_drug_output.csv",
        load_type=LoadType.UPSERT,
        key_columns=["drug_key"],
        update_columns=[
            "drug_name_std",
            "generic_name",
            "strength",
            "dosage_form",
            "route",
            "manufacturer",
            "ndc_code",
            "dea_schedule",
            "therapeutic_class",
            "record_hash",
            "effective_end_date",
            "current_flag",
            "update_timestamp",
        ],
    )

    # -- Mapping --
    mapping = Mapping(
        name="m_drug_master_scd2",
        description="Drug Master Data SCD Type 2 ETL",
        sources=sources,
        targets=[target],
        session_config=SessionConfig(
            commit_interval=5000,
            error_threshold=100,
            treat_source_rows_as="data-driven",
        ),
    )

    # -- Parameters --
    mapping.add_parameter(MappingParameter("$$PROCESS_DATE", "2026-03-05"))
    mapping.add_parameter(MappingParameter("$$SUPPLIER_PRIORITY", "1"))

    # -- Transformations --

    # 1. Source Qualifier: basic filtering
    sq = SourceQualifier(
        name="SQ_DRUG_FILES",
        source_filter="drug_name.notna()",
        select_columns=[
            "drug_name",
            "generic_name",
            "ndc_code",
            "strength",
            "dosage_form",
            "route",
            "manufacturer",
            "dea_schedule",
            "therapeutic_class",
            "supplier_code",
        ],
    )

    # 2. Expression: Standardize and validate
    exp = Expression(name="EXP_STANDARDIZE")
    exp.add_expression(
        "drug_name_std",
        lambda df: df["drug_name"].apply(standardize_drug_name),
    )
    exp.add_expression(
        "generic_name",
        lambda df: df["generic_name"].fillna("").str.strip().str.upper(),
    )
    exp.add_expression(
        "ndc_valid",
        lambda df: df["ndc_code"].apply(lambda x: validate_ndc(x)),
    )
    exp.add_expression(
        "strength",
        lambda df: df["strength"].fillna("").str.strip(),
    )
    exp.add_expression(
        "dosage_form",
        lambda df: df["dosage_form"].fillna("").str.strip().str.upper(),
    )
    exp.add_expression(
        "route",
        lambda df: df["route"].fillna("").str.strip().str.upper(),
    )
    exp.add_expression(
        "manufacturer",
        lambda df: df["manufacturer"].fillna("").str.strip().str.upper(),
    )
    exp.add_expression(
        "record_hash",
        lambda df: compute_row_hash(df, SCD_COMPARE_COLUMNS),
    )

    # 3. Filter: Keep only valid NDC records
    fil = Filter(
        name="FIL_VALID_NDC",
        condition="ndc_valid == True",
    )

    # 4. Sorter: Sort by drug name for dedup
    srt = Sorter(
        name="SRT_DRUG_NAME",
        sort_keys=["drug_name_std", "supplier_code"],
        ascending=[True, True],
    )

    # 5. Rank: Keep best record per NDC (rank by supplier priority)
    rnk = Rank(
        name="RNK_BEST_PER_NDC",
        group_by=["ndc_code"],
        rank_column="supplier_code",
        top_n=1,
        ascending=True,
    )

    # 6. Lookup: Check existing dimension
    existing = existing_dim_data if existing_dim_data is not None else pd.DataFrame()
    lkp = Lookup(
        name="LKP_EXISTING_DIM",
        lookup_source=existing,
        lookup_keys=["ndc_code"],
        return_columns=["drug_key", "record_hash"],
        mode=LookupMode.CONNECTED,
        default_values={"drug_key": -1, "record_hash": ""},
    )
    if not existing.empty:
        lkp.build_cache()

    # 7. Router: Route based on SCD comparison
    rtr = Router(name="RTR_SCD_ROUTE")
    rtr.add_group(
        "GRP_NEW",
        condition_func=lambda df: df["drug_key"] == -1,
    )
    rtr.add_group(
        "GRP_CHANGED",
        condition_func=lambda df: (
            (df["drug_key"] != -1) & (df["record_hash"] != df.get("record_hash_lkp", ""))
        ),
    )
    # Remaining rows go to DEFAULT (no change)

    # 8. Sequence Generator: Assign keys to new records
    seq = SequenceGenerator(
        name="SEQ_DRUG_KEY",
        output_column="drug_key_new",
        start_value=1000,
        increment=1,
    )

    # 9. Update Strategy
    upd = UpdateStrategy(
        name="UPD_DRUG_STRATEGY",
        strategy_func=lambda df: df.apply(
            lambda row: (
                RowDisposition.DD_INSERT
                if row.get("drug_key", -1) == -1
                else RowDisposition.DD_UPDATE
            ),
            axis=1,
        ),
    )

    # 10. Expression: Add SCD metadata columns
    exp_scd = Expression(name="EXP_SCD_METADATA")
    exp_scd.add_expression(
        "effective_start_date",
        lambda df: pd.Timestamp.now().strftime("%Y-%m-%d"),
    )
    exp_scd.add_expression(
        "effective_end_date",
        lambda df: "9999-12-31",
    )
    exp_scd.add_expression(
        "current_flag",
        lambda df: "Y",
    )
    exp_scd.add_expression(
        "update_timestamp",
        lambda df: pd.Timestamp.now().isoformat(),
    )

    # Wire up transformation pipeline
    mapping.add_transformation(sq)
    mapping.add_transformation(exp)
    mapping.add_transformation(fil)
    mapping.add_transformation(srt)
    mapping.add_transformation(rnk)
    mapping.add_transformation(lkp)
    mapping.add_transformation(rtr)
    mapping.add_transformation(seq)
    mapping.add_transformation(upd)
    mapping.add_transformation(exp_scd)

    return mapping


def run_drug_master_pipeline(
    source_files: list[str],
    target_connection: str = "",
    existing_dim: pd.DataFrame | None = None,
    runtime_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to build and execute the drug master pipeline.

    Returns the session performance statistics.
    """
    from pharma_flow.framework.session import Session

    mapping = build_drug_master_mapping(source_files, target_connection, existing_dim)
    session = Session(
        name="s_drug_master_scd2",
        mapping=mapping,
        runtime_params=runtime_params or {},
    )

    stats = session.execute()
    return stats.summary()
