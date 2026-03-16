"""Feature drift detection using statistical tests.

Implements PSI (Population Stability Index), Kolmogorov-Smirnov test,
and chi-squared test for categorical features. Compares current feature
distributions against a stored baseline and generates severity-graded
drift reports. Uses SnowSQL to compute distribution statistics directly
in Snowflake for efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import snowflake.connector
from scipy import stats
from snowflake.connector import DictCursor, SnowflakeConnection

from feature_forge.extractors.structured_extractor import SnowflakeConfig

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DriftResult:
    """Result of a single feature drift test."""

    feature_name: str
    test_name: str
    statistic: float
    p_value: float | None
    threshold: float
    is_drifted: bool
    severity: DriftSeverity
    baseline_period: str
    current_period: str
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DriftReport:
    """Aggregated drift report across multiple features."""

    report_id: str
    generated_at: datetime
    total_features_checked: int
    drifted_features_count: int
    severity_counts: dict[str, int]
    results: list[DriftResult]
    overall_severity: DriftSeverity

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "total_features_checked": self.total_features_checked,
            "drifted_features_count": self.drifted_features_count,
            "severity_counts": self.severity_counts,
            "overall_severity": self.overall_severity.value,
            "results": [
                {
                    "feature_name": r.feature_name,
                    "test_name": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "is_drifted": r.is_drifted,
                    "severity": r.severity.value,
                }
                for r in self.results
            ],
        }


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift severity classification."""

    psi_low: float = 0.1
    psi_medium: float = 0.2
    psi_high: float = 0.3
    psi_critical: float = 0.5
    ks_p_value: float = 0.05
    chi2_p_value: float = 0.05


class FeatureDriftDetector:
    """Detect feature drift between baseline and current distributions.

    Connects to Snowflake to compute distribution histograms and
    summary statistics directly in the warehouse, then applies
    statistical tests locally.
    """

    DEFAULT_NUM_BINS = 20

    def __init__(
        self,
        config: SnowflakeConfig,
        thresholds: DriftThresholds | None = None,
    ) -> None:
        self._config = config
        self._thresholds = thresholds or DriftThresholds()
        self._conn: SnowflakeConnection | None = None
        logger.info("FeatureDriftDetector initialised")

    def connect(self) -> SnowflakeConnection:
        """Get or create Snowflake connection."""
        if self._conn is not None and not self._conn.is_closed():
            return self._conn
        self._conn = snowflake.connector.connect(
            account=self._config.account,
            user=self._config.user,
            password=self._config.password,
            warehouse=self._config.warehouse,
            database=self._config.database,
            schema=self._config.schema,
            role=self._config.role,
        )
        return self._conn

    def disconnect(self) -> None:
        """Close connection."""
        if self._conn and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def _execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            return list(cursor.fetchall())
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Distribution computation in Snowflake
    # ------------------------------------------------------------------

    def compute_numeric_distribution(
        self,
        table: str,
        column: str,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: str = "feature_ts",
        num_bins: int = DEFAULT_NUM_BINS,
    ) -> dict[str, Any]:
        """Compute histogram and summary statistics for a numeric feature in Snowflake.

        Uses WIDTH_BUCKET to create an equi-width histogram and computes
        mean, stddev, min, max, median, and percentiles.
        """
        sql = f"""
        WITH stats AS (
            SELECT
                MIN({column}) AS col_min,
                MAX({column}) AS col_max,
                AVG({column}) AS col_mean,
                STDDEV({column}) AS col_stddev,
                MEDIAN({column}) AS col_median,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) AS p75,
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {column}) AS p05,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {column}) AS p95,
                COUNT(*) AS total_count,
                COUNT({column}) AS non_null_count,
                COUNT(*) - COUNT({column}) AS null_count
            FROM {table}
            WHERE {timestamp_col} >= %(start_date)s
              AND {timestamp_col} < %(end_date)s
        ),
        binned AS (
            SELECT
                WIDTH_BUCKET(
                    {column},
                    (SELECT col_min FROM stats),
                    (SELECT col_max FROM stats) + 0.0001,
                    {num_bins}
                ) AS bin_id,
                COUNT(*) AS bin_count
            FROM {table}
            WHERE {timestamp_col} >= %(start_date)s
              AND {timestamp_col} < %(end_date)s
              AND {column} IS NOT NULL
            GROUP BY bin_id
            ORDER BY bin_id
        )
        SELECT
            s.*,
            ARRAY_AGG(OBJECT_CONSTRUCT('bin_id', b.bin_id, 'count', b.bin_count))
                WITHIN GROUP (ORDER BY b.bin_id) AS histogram
        FROM stats s
        CROSS JOIN binned b
        GROUP BY s.col_min, s.col_max, s.col_mean, s.col_stddev, s.col_median,
                 s.p25, s.p75, s.p05, s.p95, s.total_count, s.non_null_count, s.null_count
        """
        params = {"start_date": start_date, "end_date": end_date}
        rows = self._execute(sql, params)

        if not rows:
            return {"empty": True}

        row = rows[0]
        return {
            "min": row["COL_MIN"],
            "max": row["COL_MAX"],
            "mean": row["COL_MEAN"],
            "stddev": row["COL_STDDEV"],
            "median": row["COL_MEDIAN"],
            "p25": row["P25"],
            "p75": row["P75"],
            "p05": row["P05"],
            "p95": row["P95"],
            "total_count": row["TOTAL_COUNT"],
            "non_null_count": row["NON_NULL_COUNT"],
            "null_count": row["NULL_COUNT"],
            "histogram": row.get("HISTOGRAM", []),
        }

    def compute_categorical_distribution(
        self,
        table: str,
        column: str,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: str = "feature_ts",
    ) -> dict[str, int]:
        """Compute value counts for a categorical feature in Snowflake."""
        sql = f"""
        SELECT
            COALESCE({column}::STRING, '__NULL__') AS category,
            COUNT(*) AS category_count
        FROM {table}
        WHERE {timestamp_col} >= %(start_date)s
          AND {timestamp_col} < %(end_date)s
        GROUP BY category
        ORDER BY category_count DESC
        """
        rows = self._execute(sql, {"start_date": start_date, "end_date": end_date})
        return {r["CATEGORY"]: r["CATEGORY_COUNT"] for r in rows}

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def compute_psi(
        baseline_proportions: np.ndarray,
        current_proportions: np.ndarray,
        epsilon: float = 1e-6,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI = SUM( (current_i - baseline_i) * ln(current_i / baseline_i) )

        Interpretation:
        - PSI < 0.1  : no significant shift
        - 0.1 <= PSI < 0.2 : moderate shift
        - PSI >= 0.2 : significant shift
        """
        # Clip to avoid log(0)
        baseline = np.clip(baseline_proportions, epsilon, None)
        current = np.clip(current_proportions, epsilon, None)

        # Normalise
        baseline = baseline / baseline.sum()
        current = current / current.sum()

        psi = np.sum((current - baseline) * np.log(current / baseline))
        return float(psi)

    @staticmethod
    def ks_test(baseline_values: np.ndarray, current_values: np.ndarray) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test.

        Returns (statistic, p_value). Small p-values indicate the
        distributions differ significantly.
        """
        stat, p_value = stats.ks_2samp(baseline_values, current_values)
        return float(stat), float(p_value)

    @staticmethod
    def chi_squared_test(
        baseline_counts: dict[str, int],
        current_counts: dict[str, int],
    ) -> tuple[float, float]:
        """Chi-squared test for categorical feature drift.

        Aligns categories between baseline and current, fills missing
        categories with zero, and runs chi2 contingency test.
        """
        all_categories = sorted(set(baseline_counts) | set(current_counts))
        baseline_arr = np.array([baseline_counts.get(c, 0) for c in all_categories])
        current_arr = np.array([current_counts.get(c, 0) for c in all_categories])

        # Remove categories with zero counts in both
        mask = (baseline_arr + current_arr) > 0
        baseline_arr = baseline_arr[mask]
        current_arr = current_arr[mask]

        if len(baseline_arr) < 2:
            return 0.0, 1.0

        contingency = np.array([baseline_arr, current_arr])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        return float(chi2), float(p_value)

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------

    def classify_psi_severity(self, psi: float) -> DriftSeverity:
        """Map a PSI value to a severity level."""
        if psi >= self._thresholds.psi_critical:
            return DriftSeverity.CRITICAL
        if psi >= self._thresholds.psi_high:
            return DriftSeverity.HIGH
        if psi >= self._thresholds.psi_medium:
            return DriftSeverity.MEDIUM
        if psi >= self._thresholds.psi_low:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    # ------------------------------------------------------------------
    # High-level detection
    # ------------------------------------------------------------------

    def detect_numeric_drift(
        self,
        feature_name: str,
        table: str,
        column: str,
        baseline_start: datetime,
        baseline_end: datetime,
        current_start: datetime,
        current_end: datetime,
        timestamp_col: str = "feature_ts",
    ) -> DriftResult:
        """Run full drift detection for a numeric feature.

        Computes distributions in Snowflake and applies both PSI and KS tests.
        """
        logger.info("Detecting numeric drift for %s", feature_name)

        baseline_dist = self.compute_numeric_distribution(
            table, column, baseline_start, baseline_end, timestamp_col
        )
        current_dist = self.compute_numeric_distribution(
            table, column, current_start, current_end, timestamp_col
        )

        if baseline_dist.get("empty") or current_dist.get("empty"):
            logger.warning("Empty distribution for %s, cannot compute drift", feature_name)
            return DriftResult(
                feature_name=feature_name,
                test_name="PSI",
                statistic=0.0,
                p_value=None,
                threshold=self._thresholds.psi_medium,
                is_drifted=False,
                severity=DriftSeverity.NONE,
                baseline_period=f"{baseline_start.isoformat()} - {baseline_end.isoformat()}",
                current_period=f"{current_start.isoformat()} - {current_end.isoformat()}",
                details={"reason": "empty_distribution"},
            )

        # Extract histogram proportions
        baseline_hist = self._histogram_to_proportions(baseline_dist["histogram"])
        current_hist = self._histogram_to_proportions(current_dist["histogram"])

        # Align bin sizes
        max_bins = max(len(baseline_hist), len(current_hist))
        baseline_hist = np.pad(baseline_hist, (0, max_bins - len(baseline_hist)))
        current_hist = np.pad(current_hist, (0, max_bins - len(current_hist)))

        psi = self.compute_psi(baseline_hist, current_hist)
        severity = self.classify_psi_severity(psi)

        return DriftResult(
            feature_name=feature_name,
            test_name="PSI",
            statistic=psi,
            p_value=None,
            threshold=self._thresholds.psi_medium,
            is_drifted=psi >= self._thresholds.psi_low,
            severity=severity,
            baseline_period=f"{baseline_start.isoformat()} - {baseline_end.isoformat()}",
            current_period=f"{current_start.isoformat()} - {current_end.isoformat()}",
            details={
                "baseline_mean": baseline_dist["mean"],
                "current_mean": current_dist["mean"],
                "baseline_stddev": baseline_dist["stddev"],
                "current_stddev": current_dist["stddev"],
                "baseline_count": baseline_dist["total_count"],
                "current_count": current_dist["total_count"],
            },
        )

    def detect_categorical_drift(
        self,
        feature_name: str,
        table: str,
        column: str,
        baseline_start: datetime,
        baseline_end: datetime,
        current_start: datetime,
        current_end: datetime,
        timestamp_col: str = "feature_ts",
    ) -> DriftResult:
        """Run chi-squared drift detection for a categorical feature."""
        logger.info("Detecting categorical drift for %s", feature_name)

        baseline_counts = self.compute_categorical_distribution(
            table, column, baseline_start, baseline_end, timestamp_col
        )
        current_counts = self.compute_categorical_distribution(
            table, column, current_start, current_end, timestamp_col
        )

        if not baseline_counts or not current_counts:
            return DriftResult(
                feature_name=feature_name,
                test_name="chi_squared",
                statistic=0.0,
                p_value=1.0,
                threshold=self._thresholds.chi2_p_value,
                is_drifted=False,
                severity=DriftSeverity.NONE,
                baseline_period=f"{baseline_start.isoformat()} - {baseline_end.isoformat()}",
                current_period=f"{current_start.isoformat()} - {current_end.isoformat()}",
                details={"reason": "empty_distribution"},
            )

        chi2, p_value = self.chi_squared_test(baseline_counts, current_counts)
        is_drifted = p_value < self._thresholds.chi2_p_value

        if is_drifted and p_value < 0.001:
            severity = DriftSeverity.CRITICAL
        elif is_drifted and p_value < 0.01:
            severity = DriftSeverity.HIGH
        elif is_drifted:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.NONE

        return DriftResult(
            feature_name=feature_name,
            test_name="chi_squared",
            statistic=chi2,
            p_value=p_value,
            threshold=self._thresholds.chi2_p_value,
            is_drifted=is_drifted,
            severity=severity,
            baseline_period=f"{baseline_start.isoformat()} - {baseline_end.isoformat()}",
            current_period=f"{current_start.isoformat()} - {current_end.isoformat()}",
            details={
                "baseline_categories": len(baseline_counts),
                "current_categories": len(current_counts),
                "new_categories": list(set(current_counts) - set(baseline_counts)),
                "dropped_categories": list(set(baseline_counts) - set(current_counts)),
            },
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_drift_report(
        self,
        results: list[DriftResult],
        report_id: str | None = None,
    ) -> DriftReport:
        """Generate an aggregated drift report from individual results."""
        import uuid

        severity_counts: dict[str, int] = {s.value: 0 for s in DriftSeverity}
        for r in results:
            severity_counts[r.severity.value] += 1

        drifted_count = sum(1 for r in results if r.is_drifted)

        # Overall severity is the worst observed
        severity_order = list(DriftSeverity)
        overall = DriftSeverity.NONE
        for r in results:
            if severity_order.index(r.severity) > severity_order.index(overall):
                overall = r.severity

        report = DriftReport(
            report_id=report_id or str(uuid.uuid4())[:12],
            generated_at=datetime.utcnow(),
            total_features_checked=len(results),
            drifted_features_count=drifted_count,
            severity_counts=severity_counts,
            results=results,
            overall_severity=overall,
        )

        logger.info(
            "Drift report %s: %d/%d features drifted (overall=%s)",
            report.report_id,
            drifted_count,
            len(results),
            overall.value,
        )
        return report

    def store_drift_report(self, report: DriftReport) -> None:
        """Persist drift report results to Snowflake for historical tracking."""
        sql = """
        INSERT INTO DRIFT_REPORTS (
            report_id, generated_at, total_features, drifted_features,
            overall_severity, report_data
        ) VALUES (
            %(report_id)s, %(generated_at)s, %(total)s, %(drifted)s,
            %(severity)s, PARSE_JSON(%(data)s)
        )
        """
        import json

        self._execute(
            sql,
            {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "total": report.total_features_checked,
                "drifted": report.drifted_features_count,
                "severity": report.overall_severity.value,
                "data": json.dumps(report.to_dict()),
            },
        )
        logger.info("Stored drift report %s to Snowflake", report.report_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _histogram_to_proportions(histogram: Any) -> np.ndarray:
        """Convert a Snowflake histogram array to a numpy proportions array."""
        import json

        if isinstance(histogram, str):
            histogram = json.loads(histogram)

        if not histogram:
            return np.array([1.0])

        counts = []
        for entry in histogram:
            if isinstance(entry, dict):
                counts.append(entry.get("count", 0))
            elif isinstance(entry, str):
                parsed = json.loads(entry)
                counts.append(parsed.get("count", 0))
            else:
                counts.append(0)

        arr = np.array(counts, dtype=np.float64)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return arr

    def __enter__(self) -> FeatureDriftDetector:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
