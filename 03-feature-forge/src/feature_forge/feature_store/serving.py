"""Feature serving layer with point-in-time correct retrieval.

Provides batch and online serving modes, TTL-based caching, and
feature vector assembly for model inference. Ensures no data leakage
through temporal join logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector import DictCursor, SnowflakeConnection

from feature_forge.extractors.structured_extractor import SnowflakeConfig
from feature_forge.feature_store.registry import FeatureDefinition, FeatureRegistry

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached feature value with TTL tracking."""

    data: pd.DataFrame
    created_at: float  # time.monotonic()
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


@dataclass
class ServingConfig:
    """Configuration for the feature serving layer."""

    default_ttl_seconds: float = 300.0  # 5 minutes
    max_cache_entries: int = 1000
    batch_size: int = 10_000
    enable_cache: bool = True
    online_mode: bool = False


class FeatureServingLayer:
    """Serve features for model inference with point-in-time correctness.

    Supports two modes:
    - **Batch serving**: Retrieve feature vectors for a set of entities at
      a specific point in time. Used for training dataset construction.
    - **Online serving**: Low-latency retrieval of the latest features for
      a single entity. Used for real-time inference.

    Point-in-time correctness is enforced via temporal ASOF joins to
    prevent future data from leaking into historical feature vectors.
    """

    def __init__(
        self,
        sf_config: SnowflakeConfig,
        registry: FeatureRegistry,
        serving_config: ServingConfig | None = None,
    ) -> None:
        self._sf_config = sf_config
        self._registry = registry
        self._config = serving_config or ServingConfig()
        self._conn: SnowflakeConnection | None = None
        self._cache: dict[str, CacheEntry] = {}
        logger.info("FeatureServingLayer initialised (cache=%s)", self._config.enable_cache)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> SnowflakeConnection:
        """Get or create Snowflake connection."""
        if self._conn is not None and not self._conn.is_closed():
            return self._conn
        self._conn = snowflake.connector.connect(
            account=self._sf_config.account,
            user=self._sf_config.user,
            password=self._sf_config.password,
            warehouse=self._sf_config.warehouse,
            database=self._sf_config.database,
            schema=self._sf_config.schema,
            role=self._sf_config.role,
        )
        return self._conn

    def disconnect(self) -> None:
        """Close Snowflake connection."""
        if self._conn and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def _execute(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        conn = self.connect()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_key(
        self, feature_name: str, entity_ids: tuple[str, ...], as_of: datetime | None
    ) -> str:
        """Build a deterministic cache key."""
        ts_part = as_of.isoformat() if as_of else "latest"
        ids_hash = hash(entity_ids)
        return f"{feature_name}|{ids_hash}|{ts_part}"

    def _get_from_cache(self, key: str) -> pd.DataFrame | None:
        """Return cached data if valid, else None."""
        if not self._config.enable_cache:
            return None
        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            if entry is not None:
                del self._cache[key]
            return None
        entry.hit_count += 1
        return entry.data

    def _put_in_cache(self, key: str, data: pd.DataFrame, ttl: float | None = None) -> None:
        """Store data in cache with TTL."""
        if not self._config.enable_cache:
            return
        # Evict oldest if at capacity
        if len(self._cache) >= self._config.max_cache_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(
            data=data,
            created_at=time.monotonic(),
            ttl_seconds=ttl or self._config.default_ttl_seconds,
        )

    def invalidate_cache(self, feature_name: str | None = None) -> int:
        """Invalidate cache entries. If feature_name is given, only that feature."""
        if feature_name is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove = [k for k in self._cache if k.startswith(f"{feature_name}|")]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    # ------------------------------------------------------------------
    # Point-in-time correct retrieval
    # ------------------------------------------------------------------

    def get_point_in_time_features(
        self,
        feature_names: list[str],
        entity_ids: list[str],
        as_of_timestamps: list[datetime],
        entity_key: str = "patient_id",
    ) -> pd.DataFrame:
        """Retrieve features with point-in-time correctness.

        For each (entity_id, as_of_timestamp) pair, returns the latest
        feature values whose timestamps are strictly <= the as_of time.
        This prevents future data from leaking into training features.

        The spine (entity_id x timestamp) is joined against each feature
        table using a temporal ASOF-style join implemented via Snowflake
        window functions.
        """
        if len(entity_ids) != len(as_of_timestamps):
            raise ValueError("entity_ids and as_of_timestamps must have the same length")

        logger.info(
            "Point-in-time retrieval: %d features, %d entities",
            len(feature_names),
            len(entity_ids),
        )

        # Build a spine CTE from the entity/timestamp pairs
        spine_values = ", ".join(
            f"('{eid}', '{ts.isoformat()}'::TIMESTAMP_NTZ)"
            for eid, ts in zip(entity_ids, as_of_timestamps)
        )
        spine_cte = f"""
        spine AS (
            SELECT
                column1 AS {entity_key},
                column2 AS as_of_ts
            FROM VALUES {spine_values}
        )
        """

        # Build a temporal join for each feature
        feature_ctes: list[str] = []
        join_clauses: list[str] = []

        for i, fname in enumerate(feature_names):
            feature_def = self._registry.get_feature(fname)
            if feature_def is None:
                logger.warning("Feature '%s' not found in registry, skipping", fname)
                continue

            alias = f"f{i}"
            ts_col = feature_def.timestamp_column
            src_table = feature_def.source_table

            # ASOF join via QUALIFY ROW_NUMBER
            cte = f"""
            {alias}_ranked AS (
                SELECT
                    ft.{entity_key},
                    ft.*,
                    s.as_of_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY ft.{entity_key}, s.as_of_ts
                        ORDER BY ft.{ts_col} DESC
                    ) AS _rn
                FROM {src_table} ft
                INNER JOIN spine s
                    ON ft.{entity_key} = s.{entity_key}
                    AND ft.{ts_col} <= s.as_of_ts
            ),
            {alias} AS (
                SELECT * FROM {alias}_ranked WHERE _rn = 1
            )
            """
            feature_ctes.append(cte)
            join_clauses.append(
                f"LEFT JOIN {alias} ON s.{entity_key} = {alias}.{entity_key} "
                f"AND s.as_of_ts = {alias}.as_of_ts"
            )

        if not feature_ctes:
            logger.warning("No valid features to retrieve")
            return pd.DataFrame()

        all_ctes = ", ".join([spine_cte] + feature_ctes)
        joins = "\n".join(join_clauses)

        sql = f"""
        WITH {all_ctes}
        SELECT s.*
            {self._build_select_columns(feature_names)}
        FROM spine s
        {joins}
        ORDER BY s.{entity_key}, s.as_of_ts
        """

        return self._execute(sql)

    def _build_select_columns(self, feature_names: list[str]) -> str:
        """Build SELECT column list for joined features, avoiding duplicates."""
        parts: list[str] = []
        for i, fname in enumerate(feature_names):
            feature_def = self._registry.get_feature(fname)
            if feature_def is None:
                continue
            alias = f"f{i}"
            parts.append(f", {alias}.{fname}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Batch serving
    # ------------------------------------------------------------------

    def get_batch_features(
        self,
        feature_names: list[str],
        entity_ids: list[str] | None = None,
        as_of_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve the latest feature values in batch mode.

        Fetches the most recent feature values for all (or specified)
        entities. Supports caching for repeated reads.
        """
        effective_date = as_of_date or datetime.utcnow()
        cache_key = self._cache_key(
            "|".join(feature_names),
            tuple(entity_ids) if entity_ids else (),
            effective_date,
        )

        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit for batch features")
            return cached

        result_frames: list[pd.DataFrame] = []

        for fname in feature_names:
            feature_def = self._registry.get_feature(fname)
            if feature_def is None:
                logger.warning("Feature '%s' not found, skipping", fname)
                continue

            sql = self._build_batch_query(feature_def, entity_ids, effective_date)
            df = self._execute(sql)
            if not df.empty:
                result_frames.append(df)

        if not result_frames:
            return pd.DataFrame()

        # Merge all feature frames on entity key
        merged = result_frames[0]
        entity_key = "PATIENT_ID"  # Snowflake uppercases column names
        for df in result_frames[1:]:
            merged = merged.merge(df, on=entity_key, how="outer", suffixes=("", "_dup"))
            # Drop duplicate columns
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            merged = merged.drop(columns=dup_cols)

        self._put_in_cache(cache_key, merged)
        logger.info("Batch serving: %d rows, %d columns", len(merged), len(merged.columns))
        return merged

    def _build_batch_query(
        self,
        feature_def: FeatureDefinition,
        entity_ids: list[str] | None,
        as_of_date: datetime,
    ) -> str:
        """Build a batch retrieval query with latest-value semantics."""
        entity_key = feature_def.entity_key
        ts_col = feature_def.timestamp_column
        src_table = feature_def.source_table

        entity_filter = ""
        if entity_ids:
            id_list = ", ".join(f"'{eid}'" for eid in entity_ids)
            entity_filter = f"AND {entity_key} IN ({id_list})"

        return f"""
        SELECT * FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY {entity_key}
                    ORDER BY {ts_col} DESC
                ) AS _rn
            FROM {src_table}
            WHERE {ts_col} <= '{as_of_date.isoformat()}'::TIMESTAMP_NTZ
                {entity_filter}
        )
        WHERE _rn = 1
        """

    # ------------------------------------------------------------------
    # Online serving
    # ------------------------------------------------------------------

    def get_online_features(
        self,
        feature_names: list[str],
        entity_id: str,
        entity_key: str = "patient_id",
    ) -> dict[str, Any]:
        """Retrieve the latest feature values for a single entity (online mode).

        Optimised for low-latency single-entity lookups. Always checks
        cache first.
        """
        cache_key = self._cache_key("|".join(feature_names), (entity_id,), None)
        cached = self._get_from_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached.iloc[0].to_dict()

        feature_values: dict[str, Any] = {entity_key: entity_id}

        for fname in feature_names:
            feature_def = self._registry.get_feature(fname)
            if feature_def is None:
                feature_values[fname] = None
                continue

            sql = f"""
            SELECT {fname}
            FROM {feature_def.source_table}
            WHERE {entity_key} = %(entity_id)s
            ORDER BY {feature_def.timestamp_column} DESC
            LIMIT 1
            """
            df = self._execute(sql, {"entity_id": entity_id})
            if not df.empty:
                col_name = fname.upper()
                feature_values[fname] = df.iloc[0].get(col_name)
            else:
                feature_values[fname] = None

        # Cache the result
        result_df = pd.DataFrame([feature_values])
        self._put_in_cache(cache_key, result_df)

        return feature_values

    # ------------------------------------------------------------------
    # Feature vector assembly
    # ------------------------------------------------------------------

    def assemble_feature_vector(
        self,
        feature_names: list[str],
        entity_id: str,
        as_of: datetime | None = None,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Assemble a numeric feature vector for a single entity.

        Returns a 1-D numpy array with features in the order specified
        by feature_names. Missing values are filled with fill_value.
        """
        if as_of is not None:
            df = self.get_point_in_time_features(
                feature_names=feature_names,
                entity_ids=[entity_id],
                as_of_timestamps=[as_of],
            )
        else:
            values = self.get_online_features(feature_names, entity_id)
            df = pd.DataFrame([values])

        if df.empty:
            return np.full(len(feature_names), fill_value)

        vector = []
        for fname in feature_names:
            col = fname.upper()
            if col in df.columns:
                val = df.iloc[0][col]
                vector.append(float(val) if val is not None else fill_value)
            else:
                vector.append(fill_value)

        return np.array(vector, dtype=np.float64)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> FeatureServingLayer:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
