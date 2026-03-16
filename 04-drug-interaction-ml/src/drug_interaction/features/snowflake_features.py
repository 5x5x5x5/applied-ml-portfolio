"""Patient-level feature extraction from Snowflake.

Queries Snowflake for co-prescription patterns, patient demographics,
historical adverse event rates, and temporal prescription data.
Uses complex window functions and semi-structured data parsing.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SnowflakeConfig(BaseModel):
    """Connection parameters for Snowflake."""

    account: str
    user: str
    password: str = Field(repr=False)
    warehouse: str
    database: str
    schema_name: str = Field(alias="schema", default="PUBLIC")
    role: str = "DRUG_INTERACTION_ROLE"

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# SQL Templates
# ---------------------------------------------------------------------------

CO_PRESCRIPTION_SQL = """
WITH prescription_pairs AS (
    SELECT
        p1.patient_id,
        p1.drug_ndc AS drug_a_ndc,
        p2.drug_ndc AS drug_b_ndc,
        p1.prescribe_date AS date_a,
        p2.prescribe_date AS date_b,
        ABS(DATEDIFF('day', p1.prescribe_date, p2.prescribe_date)) AS day_gap,
        p1.dosage_mg AS dosage_a,
        p2.dosage_mg AS dosage_b,
        p1.duration_days AS duration_a,
        p2.duration_days AS duration_b,
        -- Overlap window in days
        GREATEST(0,
            LEAST(DATEADD('day', p1.duration_days, p1.prescribe_date),
                  DATEADD('day', p2.duration_days, p2.prescribe_date))
            - GREATEST(p1.prescribe_date, p2.prescribe_date)
        ) AS overlap_days
    FROM {database}.{schema}.prescriptions p1
    JOIN {database}.{schema}.prescriptions p2
        ON p1.patient_id = p2.patient_id
        AND p1.drug_ndc < p2.drug_ndc  -- avoid self-pairs and duplicates
        AND ABS(DATEDIFF('day', p1.prescribe_date, p2.prescribe_date)) <= 30
    WHERE p1.prescribe_date >= %(start_date)s
      AND p1.prescribe_date < %(end_date)s
),
aggregated AS (
    SELECT
        drug_a_ndc,
        drug_b_ndc,
        COUNT(DISTINCT patient_id) AS co_prescribed_patients,
        COUNT(*) AS co_prescription_events,
        AVG(day_gap) AS avg_day_gap,
        AVG(overlap_days) AS avg_overlap_days,
        AVG(dosage_a) AS avg_dosage_a,
        AVG(dosage_b) AS avg_dosage_b,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY overlap_days) AS median_overlap,
        STDDEV(overlap_days) AS stddev_overlap
    FROM prescription_pairs
    GROUP BY drug_a_ndc, drug_b_ndc
)
SELECT * FROM aggregated
ORDER BY co_prescribed_patients DESC
"""

PATIENT_DEMOGRAPHICS_SQL = """
WITH drug_patients AS (
    SELECT DISTINCT
        rx.drug_ndc,
        pt.patient_id,
        pt.age,
        pt.sex,
        pt.weight_kg,
        pt.height_cm,
        pt.bmi,
        pt.ethnicity,
        pt.renal_function_egfr,
        pt.hepatic_function_class,
        -- Parse semi-structured comorbidity data (stored as VARIANT)
        ARRAY_SIZE(pt.comorbidities:conditions) AS num_comorbidities,
        pt.comorbidities:conditions[0]:code::STRING AS primary_comorbidity_icd,
        pt.comorbidities:polypharmacy_count::INT AS polypharmacy_count,
        -- Extract lab values from semi-structured column
        pt.lab_results:creatinine::FLOAT AS creatinine,
        pt.lab_results:alt::FLOAT AS alt_liver,
        pt.lab_results:platelets::FLOAT AS platelets
    FROM {database}.{schema}.prescriptions rx
    JOIN {database}.{schema}.patients pt
        ON rx.patient_id = pt.patient_id
    WHERE rx.prescribe_date >= %(start_date)s
      AND rx.prescribe_date < %(end_date)s
),
demographics_agg AS (
    SELECT
        drug_ndc,
        COUNT(DISTINCT patient_id) AS total_patients,
        AVG(age) AS mean_age,
        STDDEV(age) AS stddev_age,
        SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS pct_female,
        AVG(weight_kg) AS mean_weight,
        AVG(bmi) AS mean_bmi,
        AVG(num_comorbidities) AS avg_comorbidities,
        AVG(polypharmacy_count) AS avg_polypharmacy,
        AVG(renal_function_egfr) AS avg_egfr,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY age) AS age_p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY age) AS age_p75,
        AVG(creatinine) AS avg_creatinine,
        AVG(alt_liver) AS avg_alt
    FROM drug_patients
    GROUP BY drug_ndc
)
SELECT * FROM demographics_agg
"""

ADVERSE_EVENT_SQL = """
WITH pair_events AS (
    SELECT
        ae.drug_a_ndc,
        ae.drug_b_ndc,
        ae.event_type,
        ae.severity,
        ae.reported_date,
        ae.patient_id,
        ae.outcome,
        -- Window: running count of events for this pair
        COUNT(*) OVER (
            PARTITION BY ae.drug_a_ndc, ae.drug_b_ndc
            ORDER BY ae.reported_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_events,
        -- Window: days since previous event for same pair
        DATEDIFF('day',
            LAG(ae.reported_date) OVER (
                PARTITION BY ae.drug_a_ndc, ae.drug_b_ndc
                ORDER BY ae.reported_date
            ),
            ae.reported_date
        ) AS days_since_prev_event,
        -- Parse semi-structured event details
        ae.details:mechanism::STRING AS mechanism,
        ae.details:affected_organ::STRING AS affected_organ,
        ae.details:dechallenge_positive::BOOLEAN AS dechallenge_positive,
        ae.details:rechallenge_positive::BOOLEAN AS rechallenge_positive
    FROM {database}.{schema}.adverse_events ae
    WHERE ae.reported_date >= %(start_date)s
      AND ae.reported_date < %(end_date)s
),
pair_summary AS (
    SELECT
        drug_a_ndc,
        drug_b_ndc,
        COUNT(*) AS total_events,
        COUNT(DISTINCT patient_id) AS affected_patients,
        SUM(CASE WHEN severity = 'SEVERE' THEN 1 ELSE 0 END) AS severe_count,
        SUM(CASE WHEN severity = 'MODERATE' THEN 1 ELSE 0 END) AS moderate_count,
        SUM(CASE WHEN severity = 'MILD' THEN 1 ELSE 0 END) AS mild_count,
        SUM(CASE WHEN outcome = 'HOSPITALIZATION' THEN 1 ELSE 0 END) AS hospitalizations,
        SUM(CASE WHEN outcome = 'DEATH' THEN 1 ELSE 0 END) AS deaths,
        AVG(days_since_prev_event) AS avg_inter_event_days,
        MODE(mechanism) AS primary_mechanism,
        MODE(affected_organ) AS primary_organ,
        SUM(CASE WHEN dechallenge_positive THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0) AS dechallenge_rate,
        -- Proportional Reporting Ratio components
        MAX(cumulative_events) AS latest_cumulative
    FROM pair_events
    GROUP BY drug_a_ndc, drug_b_ndc
)
SELECT
    ps.*,
    -- Join with background event rate for PRR calculation
    bg.total_pair_events AS background_events,
    bg.total_pairs AS background_pairs,
    (ps.total_events::FLOAT / NULLIF(ps.affected_patients, 0))
        / NULLIF(bg.total_pair_events::FLOAT / NULLIF(bg.total_pairs, 0), 0) AS prr
FROM pair_summary ps
CROSS JOIN (
    SELECT COUNT(*) AS total_pair_events, COUNT(DISTINCT drug_a_ndc || drug_b_ndc) AS total_pairs
    FROM {database}.{schema}.adverse_events
    WHERE reported_date >= %(start_date)s AND reported_date < %(end_date)s
) bg
ORDER BY ps.total_events DESC
"""

TEMPORAL_PATTERNS_SQL = """
WITH weekly_prescriptions AS (
    SELECT
        drug_ndc,
        DATE_TRUNC('week', prescribe_date) AS week_start,
        COUNT(*) AS rx_count,
        COUNT(DISTINCT patient_id) AS unique_patients,
        AVG(dosage_mg) AS avg_dosage,
        -- 4-week moving average
        AVG(COUNT(*)) OVER (
            PARTITION BY drug_ndc
            ORDER BY DATE_TRUNC('week', prescribe_date)
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS ma_4w_rx_count,
        -- Week-over-week change
        LAG(COUNT(*)) OVER (
            PARTITION BY drug_ndc
            ORDER BY DATE_TRUNC('week', prescribe_date)
        ) AS prev_week_count,
        -- Rank within each week
        RANK() OVER (
            PARTITION BY DATE_TRUNC('week', prescribe_date)
            ORDER BY COUNT(*) DESC
        ) AS weekly_rank
    FROM {database}.{schema}.prescriptions
    WHERE prescribe_date >= %(start_date)s
      AND prescribe_date < %(end_date)s
    GROUP BY drug_ndc, DATE_TRUNC('week', prescribe_date)
),
seasonality AS (
    SELECT
        drug_ndc,
        EXTRACT(MONTH FROM week_start) AS month_num,
        AVG(rx_count) AS avg_monthly_rx,
        STDDEV(rx_count) AS stddev_monthly_rx,
        -- Seasonal index: ratio of month avg to overall avg
        AVG(rx_count) / NULLIF(AVG(AVG(rx_count)) OVER (PARTITION BY drug_ndc), 0) AS seasonal_index
    FROM weekly_prescriptions
    GROUP BY drug_ndc, EXTRACT(MONTH FROM week_start)
)
SELECT
    wp.drug_ndc,
    wp.week_start,
    wp.rx_count,
    wp.unique_patients,
    wp.avg_dosage,
    wp.ma_4w_rx_count,
    wp.prev_week_count,
    (wp.rx_count - wp.prev_week_count)::FLOAT / NULLIF(wp.prev_week_count, 0) AS wow_change_pct,
    wp.weekly_rank,
    s.seasonal_index
FROM weekly_prescriptions wp
LEFT JOIN seasonality s
    ON wp.drug_ndc = s.drug_ndc
    AND EXTRACT(MONTH FROM wp.week_start) = s.month_num
ORDER BY wp.drug_ndc, wp.week_start
"""


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


@dataclass
class SnowflakeFeatureExtractor:
    """Extract patient-level and prescription features from Snowflake.

    Parameters
    ----------
    config : SnowflakeConfig
        Connection parameters.
    """

    config: SnowflakeConfig
    _connection: Any = field(default=None, repr=False, init=False)

    @contextmanager
    def _connect(self) -> Generator[Any, None, None]:
        """Context manager for a Snowflake connection."""
        import snowflake.connector  # type: ignore[import-untyped]

        conn = snowflake.connector.connect(
            account=self.config.account,
            user=self.config.user,
            password=self.config.password,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema_name,
            role=self.config.role,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _execute_query(
        self,
        sql_template: str,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Execute a parameterised SQL query and return a DataFrame.

        The template is first formatted with database/schema names
        (safe string interpolation for identifiers), then parameterised
        values are bound through the connector's ``execute`` method.
        """
        sql = sql_template.format(
            database=self.config.database,
            schema=self.config.schema_name,
        )
        logger.info("Executing Snowflake query (%d chars)", len(sql))
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                logger.info("Query returned %d rows, %d columns", len(df), len(df.columns))
                return df
            finally:
                cursor.close()

    # -- public extraction methods ------------------------------------------

    def extract_co_prescription_features(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Extract co-prescription frequency and overlap features.

        Parameters
        ----------
        start_date, end_date : date
            Date range for prescriptions to consider.

        Returns
        -------
        pd.DataFrame
            Aggregated co-prescription features per drug pair.
        """
        logger.info(
            "Extracting co-prescription features from %s to %s",
            start_date,
            end_date,
        )
        return self._execute_query(
            CO_PRESCRIPTION_SQL,
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        )

    def extract_patient_demographics(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Extract patient demographics aggregated per drug.

        Includes semi-structured data parsing for comorbidities and lab values.
        """
        logger.info("Extracting patient demographics from %s to %s", start_date, end_date)
        return self._execute_query(
            PATIENT_DEMOGRAPHICS_SQL,
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        )

    def extract_adverse_event_rates(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Extract historical adverse event rates for drug pairs.

        Computes severity breakdown, Proportional Reporting Ratio (PRR),
        cumulative event counts, and mechanism/organ classification.
        """
        logger.info("Extracting adverse event rates from %s to %s", start_date, end_date)
        return self._execute_query(
            ADVERSE_EVENT_SQL,
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        )

    def extract_temporal_patterns(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Extract temporal prescription patterns with seasonality.

        Includes weekly aggregates, moving averages, week-over-week changes,
        and seasonal indices per drug.
        """
        logger.info("Extracting temporal patterns from %s to %s", start_date, end_date)
        return self._execute_query(
            TEMPORAL_PATTERNS_SQL,
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        )

    def extract_all_features(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Extract all feature sets and return as a dict of DataFrames.

        Returns
        -------
        dict
            Keys: ``co_prescription``, ``demographics``, ``adverse_events``,
            ``temporal_patterns``.
        """
        logger.info("Starting full feature extraction pipeline")
        return {
            "co_prescription": self.extract_co_prescription_features(start_date, end_date),
            "demographics": self.extract_patient_demographics(start_date, end_date),
            "adverse_events": self.extract_adverse_event_rates(start_date, end_date),
            "temporal_patterns": self.extract_temporal_patterns(start_date, end_date),
        }
