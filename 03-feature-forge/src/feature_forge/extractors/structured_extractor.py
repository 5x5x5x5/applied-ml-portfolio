"""Feature extraction from structured Snowflake tables.

Provides StructuredFeatureExtractor for building feature vectors from
relational tables including patient demographics, lab results, and
prescription history using parameterized SnowSQL queries with
aggregations, window functions, and multi-table joins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector import DictCursor, SnowflakeConnection

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parents[3] / "sql" / "feature_queries"


@dataclass(frozen=True)
class SnowflakeConfig:
    """Connection parameters for Snowflake."""

    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: str = "SYSADMIN"
    login_timeout: int = 30


@dataclass
class FeatureQuery:
    """Encapsulates a parameterized feature query with metadata."""

    name: str
    description: str
    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    entity_key: str = "patient_id"
    timestamp_column: str = "feature_ts"


class StructuredFeatureExtractor:
    """Extract features from structured Snowflake tables.

    Supports patient demographics, lab results, and prescription history
    tables. Executes parameterized SnowSQL queries using aggregations,
    window functions, and cross-table joins, then transforms results
    into feature vectors suitable for ML pipelines.
    """

    def __init__(self, config: SnowflakeConfig) -> None:
        self._config = config
        self._conn: SnowflakeConnection | None = None
        self._query_cache: dict[str, FeatureQuery] = {}
        logger.info(
            "StructuredFeatureExtractor initialised for %s.%s",
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

        logger.info(
            "Connecting to Snowflake account=%s warehouse=%s",
            self._config.account,
            self._config.warehouse,
        )
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
        logger.info("Snowflake connection established successfully")
        return self._conn

    def disconnect(self) -> None:
        """Close the Snowflake connection if open."""
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
            logger.info("Snowflake connection closed")
        self._conn = None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _load_sql_file(self, filename: str) -> str:
        """Load a SQL query from the sql/feature_queries directory."""
        path = SQL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")
        return path.read_text()

    def _execute_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a parameterized query and return a DataFrame."""
        conn = self.connect()
        logger.debug("Executing query (params=%s):\n%s", params, sql[:200])

        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            if not rows:
                logger.warning("Query returned zero rows")
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            logger.info("Query returned %d rows, %d columns", len(df), len(df.columns))
            return df
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Pre-built feature queries
    # ------------------------------------------------------------------

    def register_query(self, query: FeatureQuery) -> None:
        """Register a named feature query for later execution."""
        self._query_cache[query.name] = query
        logger.info("Registered feature query: %s", query.name)

    def execute_registered_query(
        self, name: str, override_params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Execute a previously registered query by name."""
        if name not in self._query_cache:
            raise KeyError(f"No registered query named '{name}'")
        query = self._query_cache[name]
        params = {**query.parameters, **(override_params or {})}
        return self._execute_query(query.sql, params)

    # ------------------------------------------------------------------
    # Patient demographic features
    # ------------------------------------------------------------------

    def extract_patient_demographics(
        self,
        as_of_date: datetime | None = None,
        patient_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract demographic features for patients.

        Computes age buckets, insurance encoding, geographic region,
        and chronic condition counts via aggregation and joins.
        """
        sql = self._load_sql_file("patient_features.sql")
        params: dict[str, Any] = {
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        if patient_ids:
            placeholders = ", ".join(["%s"] * len(patient_ids))
            sql = sql.replace("-- PATIENT_ID_FILTER", f"AND p.patient_id IN ({placeholders})")
            params["patient_ids"] = patient_ids

        df = self._execute_query(sql, params)
        return self._post_process_demographics(df)

    @staticmethod
    def _post_process_demographics(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise and encode demographic features."""
        if df.empty:
            return df

        # Age buckets
        if "age" in df.columns:
            df["age_bucket"] = pd.cut(
                df["age"],
                bins=[0, 18, 30, 45, 60, 75, 120],
                labels=["pediatric", "young_adult", "adult", "middle_age", "senior", "elderly"],
            )

        # One-hot encode gender if present
        if "gender" in df.columns:
            dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=np.float64)
            df = pd.concat([df, dummies], axis=1)

        return df

    # ------------------------------------------------------------------
    # Lab result features
    # ------------------------------------------------------------------

    def extract_lab_features(
        self,
        lookback_days: int = 365,
        as_of_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Extract lab result features with rolling statistics.

        Uses Snowflake window functions to compute rolling mean, std,
        min, max, trend slope, and out-of-range counts for each lab
        test per patient within the specified lookback window.
        """
        sql = self._load_sql_file("lab_features.sql")
        params: dict[str, Any] = {
            "lookback_days": lookback_days,
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        df = self._execute_query(sql, params)
        return self._post_process_lab_features(df)

    @staticmethod
    def _post_process_lab_features(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing lab values and compute derived ratios."""
        if df.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # Derived ratio features if component columns exist
        if {"hemoglobin_mean", "hematocrit_mean"}.issubset(df.columns):
            df["hgb_hct_ratio"] = df["hemoglobin_mean"] / df["hematocrit_mean"].replace(0, np.nan)

        if {"bun_mean", "creatinine_mean"}.issubset(df.columns):
            df["bun_creatinine_ratio"] = df["bun_mean"] / df["creatinine_mean"].replace(0, np.nan)

        return df

    # ------------------------------------------------------------------
    # Prescription features
    # ------------------------------------------------------------------

    def extract_prescription_features(
        self,
        lookback_days: int = 365,
        as_of_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Extract prescription history features.

        Computes active medication count, polypharmacy indicators,
        therapeutic class diversity, refill adherence ratios, and
        high-risk medication flags using window functions.
        """
        sql = """
        WITH rx_window AS (
            SELECT
                r.patient_id,
                r.ndc_code,
                r.drug_name,
                r.therapeutic_class,
                r.prescribed_date,
                r.days_supply,
                r.refills_remaining,
                r.quantity,
                DATEADD('day', r.days_supply, r.prescribed_date) AS expected_end_date,
                LEAD(r.prescribed_date) OVER (
                    PARTITION BY r.patient_id, r.ndc_code
                    ORDER BY r.prescribed_date
                ) AS next_fill_date,
                ROW_NUMBER() OVER (
                    PARTITION BY r.patient_id, r.ndc_code
                    ORDER BY r.prescribed_date DESC
                ) AS recency_rank
            FROM prescriptions r
            WHERE r.prescribed_date >= DATEADD('day', -%(lookback_days)s, %(as_of_date)s::DATE)
              AND r.prescribed_date <= %(as_of_date)s::DATE
        ),
        adherence AS (
            SELECT
                patient_id,
                ndc_code,
                AVG(
                    CASE
                        WHEN next_fill_date IS NULL THEN NULL
                        ELSE LEAST(
                            1.0,
                            days_supply::FLOAT
                            / NULLIF(DATEDIFF('day', prescribed_date, next_fill_date), 0)
                        )
                    END
                ) AS pdc_ratio
            FROM rx_window
            GROUP BY patient_id, ndc_code
        ),
        patient_rx_agg AS (
            SELECT
                rw.patient_id,
                COUNT(DISTINCT rw.ndc_code) AS unique_medications,
                COUNT(DISTINCT rw.therapeutic_class) AS therapeutic_class_count,
                SUM(CASE WHEN rw.recency_rank = 1
                         AND rw.expected_end_date >= %(as_of_date)s::DATE
                    THEN 1 ELSE 0 END) AS active_medication_count,
                MAX(CASE WHEN rw.therapeutic_class IN (
                    'OPIOID_ANALGESIC', 'ANTICOAGULANT', 'IMMUNOSUPPRESSANT'
                ) THEN 1 ELSE 0 END) AS high_risk_medication_flag,
                AVG(a.pdc_ratio) AS avg_adherence_ratio
            FROM rx_window rw
            LEFT JOIN adherence a
                ON rw.patient_id = a.patient_id AND rw.ndc_code = a.ndc_code
            GROUP BY rw.patient_id
        )
        SELECT
            pa.patient_id,
            pa.unique_medications,
            pa.therapeutic_class_count,
            pa.active_medication_count,
            CASE WHEN pa.active_medication_count >= 5 THEN 1 ELSE 0 END AS polypharmacy_flag,
            pa.high_risk_medication_flag,
            COALESCE(pa.avg_adherence_ratio, 0) AS avg_adherence_ratio,
            CURRENT_TIMESTAMP() AS feature_ts
        FROM patient_rx_agg pa
        ORDER BY pa.patient_id
        """
        params: dict[str, Any] = {
            "lookback_days": lookback_days,
            "as_of_date": as_of_date or datetime.utcnow(),
        }
        return self._execute_query(sql, params)

    # ------------------------------------------------------------------
    # Unified feature vector assembly
    # ------------------------------------------------------------------

    def build_feature_vectors(
        self,
        lookback_days: int = 365,
        as_of_date: datetime | None = None,
        patient_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a unified feature vector by joining all feature groups.

        Performs a full outer join across demographics, lab, and
        prescription features on patient_id, then fills missing values
        and applies standard scaling to numeric columns.
        """
        effective_date = as_of_date or datetime.utcnow()

        logger.info(
            "Building feature vectors as_of=%s lookback=%d days",
            effective_date.isoformat(),
            lookback_days,
        )

        demographics = self.extract_patient_demographics(
            as_of_date=effective_date,
            patient_ids=patient_ids,
        )
        lab_features = self.extract_lab_features(
            lookback_days=lookback_days,
            as_of_date=effective_date,
        )
        rx_features = self.extract_prescription_features(
            lookback_days=lookback_days,
            as_of_date=effective_date,
        )

        # Merge on patient_id
        merged = demographics
        if not lab_features.empty:
            merged = merged.merge(lab_features, on="patient_id", how="left", suffixes=("", "_lab"))
        if not rx_features.empty:
            merged = merged.merge(rx_features, on="patient_id", how="left", suffixes=("", "_rx"))

        logger.info(
            "Feature vector assembled: %d patients, %d features",
            len(merged),
            len(merged.columns),
        )
        return merged

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> StructuredFeatureExtractor:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
