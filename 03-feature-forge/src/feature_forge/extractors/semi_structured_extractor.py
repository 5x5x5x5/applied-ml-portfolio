"""Feature extraction from semi-structured data in Snowflake VARIANT columns.

Provides SemiStructuredFeatureExtractor for parsing nested clinical notes,
extracting entities from JSON medical records, flattening hierarchical data,
and handling missing/null values using Snowflake's LATERAL FLATTEN and PARSE_JSON.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector import DictCursor, SnowflakeConnection

from feature_forge.extractors.structured_extractor import SnowflakeConfig

logger = logging.getLogger(__name__)


@dataclass
class SemiStructuredQuery:
    """A query targeting semi-structured VARIANT data with extraction metadata."""

    name: str
    description: str
    source_table: str
    variant_column: str
    json_paths: list[str]
    flatten_path: str | None = None
    entity_key: str = "patient_id"


class SemiStructuredFeatureExtractor:
    """Extract features from semi-structured JSON/XML data stored in Snowflake VARIANT columns.

    Handles nested clinical notes, JSON medical records, hierarchical lab panels,
    and other semi-structured healthcare data. Uses LATERAL FLATTEN, PARSE_JSON,
    and path-based extraction to build flat feature vectors.
    """

    def __init__(self, config: SnowflakeConfig) -> None:
        self._config = config
        self._conn: SnowflakeConnection | None = None
        logger.info(
            "SemiStructuredFeatureExtractor initialised for %s.%s",
            config.database,
            config.schema,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> SnowflakeConnection:
        """Establish or return existing Snowflake connection."""
        if self._conn is not None and not self._conn.is_closed():
            return self._conn

        logger.info("Connecting to Snowflake for semi-structured extraction")
        self._conn = snowflake.connector.connect(
            account=self._config.account,
            user=self._config.user,
            password=self._config.password,
            warehouse=self._config.warehouse,
            database=self._config.database,
            schema=self._config.schema,
            role=self._config.role,
            login_timeout=self._config.login_timeout,
        )
        return self._conn

    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
            logger.info("Semi-structured extractor connection closed")
        self._conn = None

    def _execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        conn = self.connect()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            if not rows:
                logger.warning("Semi-structured query returned zero rows")
                return pd.DataFrame()
            return pd.DataFrame(rows)
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Clinical notes extraction
    # ------------------------------------------------------------------

    def extract_clinical_notes_features(
        self,
        as_of_date: datetime | None = None,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Extract features from JSON clinical notes stored in VARIANT columns.

        Parses nested note structures to extract: note types, encounter counts,
        diagnosis codes mentioned, procedure mentions, sentiment indicators,
        note length statistics, and provider specialty information.
        """
        sql = """
        WITH notes_flattened AS (
            SELECT
                cn.patient_id,
                cn.encounter_id,
                cn.note_date,
                cn.clinical_note:noteType::STRING AS note_type,
                cn.clinical_note:provider.specialty::STRING AS provider_specialty,
                cn.clinical_note:provider.npi::STRING AS provider_npi,
                cn.clinical_note:content.text::STRING AS note_text,
                LENGTH(cn.clinical_note:content.text::STRING) AS note_length,
                cn.clinical_note:content.sections AS sections,
                cn.clinical_note:metadata.priority::STRING AS priority,
                cn.clinical_note:metadata.is_addendum::BOOLEAN AS is_addendum
            FROM clinical_notes cn
            WHERE cn.note_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
              AND cn.note_date <= %(as_of_date)s::DATE
        ),
        diagnosis_extracted AS (
            SELECT
                nf.patient_id,
                nf.encounter_id,
                dx.value:code::STRING AS dx_code,
                dx.value:description::STRING AS dx_description,
                dx.value:type::STRING AS dx_type
            FROM notes_flattened nf,
                LATERAL FLATTEN(
                    input => nf.sections,
                    path => 'diagnoses',
                    outer => TRUE
                ) dx
        ),
        procedure_extracted AS (
            SELECT
                nf.patient_id,
                nf.encounter_id,
                proc.value:cpt_code::STRING AS cpt_code,
                proc.value:description::STRING AS proc_description
            FROM notes_flattened nf,
                LATERAL FLATTEN(
                    input => nf.sections,
                    path => 'procedures',
                    outer => TRUE
                ) proc
        ),
        patient_note_agg AS (
            SELECT
                nf.patient_id,
                COUNT(DISTINCT nf.encounter_id) AS encounter_count,
                COUNT(*) AS total_notes,
                AVG(nf.note_length) AS avg_note_length,
                MAX(nf.note_length) AS max_note_length,
                COUNT(DISTINCT nf.note_type) AS distinct_note_types,
                COUNT(DISTINCT nf.provider_specialty) AS distinct_provider_specialties,
                SUM(CASE WHEN nf.priority = 'URGENT' THEN 1 ELSE 0 END) AS urgent_note_count,
                SUM(CASE WHEN nf.is_addendum THEN 1 ELSE 0 END) AS addendum_count,
                MAX(nf.note_date) AS last_note_date,
                DATEDIFF('day', MAX(nf.note_date), %(as_of_date)s::DATE) AS days_since_last_note
            FROM notes_flattened nf
            GROUP BY nf.patient_id
        ),
        patient_dx_agg AS (
            SELECT
                patient_id,
                COUNT(DISTINCT dx_code) AS unique_diagnosis_count,
                COUNT(DISTINCT SUBSTR(dx_code, 1, 3)) AS unique_dx_category_count,
                SUM(CASE WHEN dx_type = 'PRIMARY' THEN 1 ELSE 0 END) AS primary_dx_count
            FROM diagnosis_extracted
            WHERE dx_code IS NOT NULL
            GROUP BY patient_id
        ),
        patient_proc_agg AS (
            SELECT
                patient_id,
                COUNT(DISTINCT cpt_code) AS unique_procedure_count
            FROM procedure_extracted
            WHERE cpt_code IS NOT NULL
            GROUP BY patient_id
        )
        SELECT
            pna.patient_id,
            pna.encounter_count,
            pna.total_notes,
            pna.avg_note_length,
            pna.max_note_length,
            pna.distinct_note_types,
            pna.distinct_provider_specialties,
            pna.urgent_note_count,
            pna.addendum_count,
            pna.days_since_last_note,
            COALESCE(pda.unique_diagnosis_count, 0) AS unique_diagnosis_count,
            COALESCE(pda.unique_dx_category_count, 0) AS unique_dx_category_count,
            COALESCE(pda.primary_dx_count, 0) AS primary_dx_count,
            COALESCE(ppa.unique_procedure_count, 0) AS unique_procedure_count,
            CURRENT_TIMESTAMP() AS feature_ts
        FROM patient_note_agg pna
        LEFT JOIN patient_dx_agg pda ON pna.patient_id = pda.patient_id
        LEFT JOIN patient_proc_agg ppa ON pna.patient_id = ppa.patient_id
        ORDER BY pna.patient_id
        """
        params: dict[str, Any] = {
            "lookback_days": lookback_days,
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        df = self._execute(sql, params)
        return self._handle_missing_values(df)

    # ------------------------------------------------------------------
    # Medical record JSON extraction
    # ------------------------------------------------------------------

    def extract_medical_record_entities(
        self,
        as_of_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Extract entity-level features from JSON medical records.

        Flattens nested allergy lists, medication arrays, vital sign
        objects, and immunization records using LATERAL FLATTEN.
        """
        sql = """
        WITH record_base AS (
            SELECT
                mr.patient_id,
                mr.record_id,
                mr.record_date,
                mr.medical_record:demographics.ethnicity::STRING AS ethnicity,
                mr.medical_record:demographics.language::STRING AS primary_language,
                mr.medical_record:demographics.marital_status::STRING AS marital_status,
                mr.medical_record:social_history.smoking_status::STRING AS smoking_status,
                mr.medical_record:social_history.alcohol_use::STRING AS alcohol_use,
                mr.medical_record:social_history.exercise_frequency::STRING AS exercise_freq,
                mr.medical_record:vitals AS vitals_obj,
                mr.medical_record:allergies AS allergies_arr,
                mr.medical_record:immunizations AS immunizations_arr,
                mr.medical_record:family_history AS family_history_arr
            FROM medical_records mr
            WHERE mr.record_date <= COALESCE(%(as_of_date)s::DATE, CURRENT_DATE())
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY mr.patient_id ORDER BY mr.record_date DESC
            ) = 1
        ),
        allergy_counts AS (
            SELECT
                rb.patient_id,
                COUNT(a.value) AS allergy_count,
                SUM(CASE WHEN a.value:severity::STRING = 'SEVERE' THEN 1 ELSE 0 END)
                    AS severe_allergy_count,
                SUM(CASE WHEN a.value:category::STRING = 'DRUG' THEN 1 ELSE 0 END)
                    AS drug_allergy_count
            FROM record_base rb,
                LATERAL FLATTEN(input => rb.allergies_arr, outer => TRUE) a
            GROUP BY rb.patient_id
        ),
        immunization_counts AS (
            SELECT
                rb.patient_id,
                COUNT(DISTINCT imm.value:vaccine_code::STRING) AS distinct_vaccines,
                MAX(imm.value:administration_date::DATE) AS last_immunization_date
            FROM record_base rb,
                LATERAL FLATTEN(input => rb.immunizations_arr, outer => TRUE) imm
            GROUP BY rb.patient_id
        ),
        family_hx AS (
            SELECT
                rb.patient_id,
                COUNT(fh.value) AS family_condition_count,
                SUM(CASE
                    WHEN fh.value:condition::STRING ILIKE '%diabetes%' THEN 1
                    ELSE 0
                END) AS family_diabetes_flag,
                SUM(CASE
                    WHEN fh.value:condition::STRING ILIKE '%heart%'
                      OR fh.value:condition::STRING ILIKE '%cardiac%' THEN 1
                    ELSE 0
                END) AS family_cardiac_flag,
                SUM(CASE
                    WHEN fh.value:condition::STRING ILIKE '%cancer%' THEN 1
                    ELSE 0
                END) AS family_cancer_flag
            FROM record_base rb,
                LATERAL FLATTEN(input => rb.family_history_arr, outer => TRUE) fh
            GROUP BY rb.patient_id
        )
        SELECT
            rb.patient_id,
            rb.ethnicity,
            rb.primary_language,
            rb.marital_status,
            rb.smoking_status,
            rb.alcohol_use,
            rb.exercise_freq,
            -- Vitals from nested JSON object
            rb.vitals_obj:systolic_bp::FLOAT AS systolic_bp,
            rb.vitals_obj:diastolic_bp::FLOAT AS diastolic_bp,
            rb.vitals_obj:heart_rate::FLOAT AS heart_rate,
            rb.vitals_obj:respiratory_rate::FLOAT AS respiratory_rate,
            rb.vitals_obj:temperature::FLOAT AS temperature,
            rb.vitals_obj:bmi::FLOAT AS bmi,
            rb.vitals_obj:weight_kg::FLOAT AS weight_kg,
            rb.vitals_obj:height_cm::FLOAT AS height_cm,
            -- Allergy features
            COALESCE(ac.allergy_count, 0) AS allergy_count,
            COALESCE(ac.severe_allergy_count, 0) AS severe_allergy_count,
            COALESCE(ac.drug_allergy_count, 0) AS drug_allergy_count,
            -- Immunization features
            COALESCE(ic.distinct_vaccines, 0) AS distinct_vaccines,
            ic.last_immunization_date,
            -- Family history features
            COALESCE(fhx.family_condition_count, 0) AS family_condition_count,
            COALESCE(fhx.family_diabetes_flag, 0) AS family_diabetes_flag,
            COALESCE(fhx.family_cardiac_flag, 0) AS family_cardiac_flag,
            COALESCE(fhx.family_cancer_flag, 0) AS family_cancer_flag,
            CURRENT_TIMESTAMP() AS feature_ts
        FROM record_base rb
        LEFT JOIN allergy_counts ac ON rb.patient_id = ac.patient_id
        LEFT JOIN immunization_counts ic ON rb.patient_id = ic.patient_id
        LEFT JOIN family_hx fhx ON rb.patient_id = fhx.patient_id
        ORDER BY rb.patient_id
        """
        params: dict[str, Any] = {
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        df = self._execute(sql, params)
        return self._encode_categorical_variants(df)

    # ------------------------------------------------------------------
    # Hierarchical data flattening
    # ------------------------------------------------------------------

    def extract_nested_lab_panels(
        self,
        as_of_date: datetime | None = None,
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """Flatten hierarchical lab panel data stored as nested JSON.

        Lab panels contain nested arrays of component results. This method
        uses recursive LATERAL FLATTEN to extract individual component
        values, then pivots them into a wide feature format.
        """
        sql = """
        WITH panel_components AS (
            SELECT
                lp.patient_id,
                lp.order_id,
                lp.collection_date,
                lp.lab_panel_data:panel_name::STRING AS panel_name,
                lp.lab_panel_data:ordering_provider::STRING AS ordering_provider,
                comp.value:component_name::STRING AS component_name,
                comp.value:result_value::FLOAT AS result_value,
                comp.value:unit::STRING AS unit,
                comp.value:reference_range.low::FLOAT AS ref_low,
                comp.value:reference_range.high::FLOAT AS ref_high,
                comp.value:abnormal_flag::STRING AS abnormal_flag,
                CASE
                    WHEN comp.value:result_value::FLOAT < comp.value:reference_range.low::FLOAT
                        THEN 'LOW'
                    WHEN comp.value:result_value::FLOAT > comp.value:reference_range.high::FLOAT
                        THEN 'HIGH'
                    ELSE 'NORMAL'
                END AS range_status
            FROM lab_panels lp,
                LATERAL FLATTEN(input => lp.lab_panel_data:components) comp
            WHERE lp.collection_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
              AND lp.collection_date <= %(as_of_date)s::DATE
        ),
        component_stats AS (
            SELECT
                patient_id,
                component_name,
                AVG(result_value) AS avg_value,
                STDDEV(result_value) AS std_value,
                MIN(result_value) AS min_value,
                MAX(result_value) AS max_value,
                COUNT(*) AS measurement_count,
                SUM(CASE WHEN range_status != 'NORMAL' THEN 1 ELSE 0 END) AS abnormal_count,
                -- Trend: difference between most recent and earliest
                LAST_VALUE(result_value) OVER (
                    PARTITION BY patient_id, component_name
                    ORDER BY collection_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) - FIRST_VALUE(result_value) OVER (
                    PARTITION BY patient_id, component_name
                    ORDER BY collection_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS value_trend
            FROM panel_components
            GROUP BY patient_id, component_name, result_value, collection_date
        ),
        pivoted AS (
            SELECT
                patient_id,
                -- Common metabolic panel components
                MAX(CASE WHEN component_name = 'GLUCOSE' THEN avg_value END) AS glucose_avg,
                MAX(CASE WHEN component_name = 'GLUCOSE' THEN std_value END) AS glucose_std,
                MAX(CASE WHEN component_name = 'GLUCOSE' THEN abnormal_count END)
                    AS glucose_abnormal_ct,
                MAX(CASE WHEN component_name = 'CREATININE' THEN avg_value END) AS creatinine_avg,
                MAX(CASE WHEN component_name = 'CREATININE' THEN value_trend END)
                    AS creatinine_trend,
                MAX(CASE WHEN component_name = 'BUN' THEN avg_value END) AS bun_avg,
                MAX(CASE WHEN component_name = 'SODIUM' THEN avg_value END) AS sodium_avg,
                MAX(CASE WHEN component_name = 'POTASSIUM' THEN avg_value END) AS potassium_avg,
                MAX(CASE WHEN component_name = 'CALCIUM' THEN avg_value END) AS calcium_avg,
                -- CBC components
                MAX(CASE WHEN component_name = 'WBC' THEN avg_value END) AS wbc_avg,
                MAX(CASE WHEN component_name = 'HEMOGLOBIN' THEN avg_value END) AS hemoglobin_avg,
                MAX(CASE WHEN component_name = 'PLATELET' THEN avg_value END) AS platelet_avg,
                -- Total abnormal rate across all components
                SUM(abnormal_count)::FLOAT / NULLIF(SUM(measurement_count), 0)
                    AS overall_abnormal_rate
            FROM component_stats
            GROUP BY patient_id
        )
        SELECT
            p.*,
            CURRENT_TIMESTAMP() AS feature_ts
        FROM pivoted p
        ORDER BY p.patient_id
        """
        params: dict[str, Any] = {
            "lookback_days": lookback_days,
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        return self._execute(sql, params)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle nulls in semi-structured extractions.

        Numeric columns get median imputation; categorical columns
        get a sentinel value 'UNKNOWN'.
        """
        if df.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
                logger.debug(
                    "Imputed %d nulls in %s with median=%.4f", df[col].isna().sum(), col, median
                )

        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna("UNKNOWN")
                logger.debug("Filled nulls in categorical column %s with 'UNKNOWN'", col)

        return df

    @staticmethod
    def _encode_categorical_variants(df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns from VARIANT extractions.

        Uses frequency encoding for high-cardinality and one-hot for
        low-cardinality categorical features.
        """
        if df.empty:
            return df

        HIGH_CARDINALITY_THRESHOLD = 10
        categorical_cols = df.select_dtypes(include=["object"]).columns
        cols_to_drop: list[str] = []

        for col in categorical_cols:
            if col in ("patient_id", "feature_ts"):
                continue

            n_unique = df[col].nunique()

            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq_enc"] = df[col].map(freq_map).astype(np.float64)
                cols_to_drop.append(col)
                logger.debug("Frequency-encoded %s (cardinality=%d)", col, n_unique)
            else:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, dtype=np.float64)
                df = pd.concat([df, dummies], axis=1)
                cols_to_drop.append(col)
                logger.debug("One-hot encoded %s (cardinality=%d)", col, n_unique)

        df = df.drop(columns=cols_to_drop)
        return df

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SemiStructuredFeatureExtractor:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
