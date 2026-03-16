"""Feature registry and catalog backed by Snowflake metadata tables.

Provides feature registration with metadata (name, description, data type,
source, freshness SLA, owner), versioning, lineage tracking, and
dependency graph management. All metadata is persisted in Snowflake.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import snowflake.connector
from pydantic import BaseModel, Field
from snowflake.connector import DictCursor, SnowflakeConnection

from feature_forge.extractors.structured_extractor import SnowflakeConfig

logger = logging.getLogger(__name__)


class FeatureDataType(str, Enum):
    """Supported feature data types."""

    FLOAT = "FLOAT"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    TIMESTAMP = "TIMESTAMP"
    ARRAY = "ARRAY"
    EMBEDDING = "EMBEDDING"


class FeatureStatus(str, Enum):
    """Lifecycle status of a registered feature."""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class FeatureDefinition(BaseModel):
    """Complete definition of a registered feature."""

    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(default=1, ge=1)
    description: str = Field(default="")
    data_type: FeatureDataType = FeatureDataType.FLOAT
    source_table: str = Field(..., min_length=1)
    source_query: str = Field(default="")
    entity_key: str = Field(default="patient_id")
    timestamp_column: str = Field(default="feature_ts")
    freshness_sla_hours: int = Field(default=24, ge=1)
    owner: str = Field(default="data-engineering")
    tags: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: FeatureStatus = FeatureStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def feature_id(self) -> str:
        """Deterministic ID based on name and version."""
        raw = f"{self.name}:v{self.version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def qualified_name(self) -> str:
        """Full name including version."""
        return f"{self.name}:v{self.version}"


class LineageEdge(BaseModel):
    """Represents a lineage relationship between two features."""

    source_feature: str
    target_feature: str
    transformation: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureRegistry:
    """Feature registry backed by Snowflake metadata tables.

    Manages the lifecycle of feature definitions including registration,
    versioning, lineage tracking, dependency resolution, freshness
    monitoring, and search.
    """

    REGISTRY_TABLE = "FEATURE_REGISTRY"
    LINEAGE_TABLE = "FEATURE_LINEAGE"
    VERSION_HISTORY_TABLE = "FEATURE_VERSION_HISTORY"

    def __init__(self, config: SnowflakeConfig) -> None:
        self._config = config
        self._conn: SnowflakeConnection | None = None
        self._cache: dict[str, FeatureDefinition] = {}
        logger.info("FeatureRegistry initialised for %s.%s", config.database, config.schema)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> SnowflakeConnection:
        """Establish connection to Snowflake."""
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
        """Close Snowflake connection."""
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def _execute(
        self, sql: str, params: dict[str, Any] | None = None, fetch: bool = True
    ) -> list[dict[str, Any]]:
        """Execute SQL and optionally fetch results."""
        conn = self.connect()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params or {})
            if fetch:
                return list(cursor.fetchall())
            return []
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def ensure_tables_exist(self) -> None:
        """Create registry metadata tables if they do not exist."""
        ddl_registry = f"""
        CREATE TABLE IF NOT EXISTS {self.REGISTRY_TABLE} (
            feature_id      VARCHAR(16) PRIMARY KEY,
            name            VARCHAR(128) NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1,
            description     TEXT,
            data_type       VARCHAR(20),
            source_table    VARCHAR(256),
            source_query    TEXT,
            entity_key      VARCHAR(64),
            timestamp_column VARCHAR(64),
            freshness_sla_hours INTEGER DEFAULT 24,
            owner           VARCHAR(128),
            tags            VARIANT,
            dependencies    VARIANT,
            status          VARCHAR(20) DEFAULT 'DRAFT',
            created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            updated_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UNIQUE(name, version)
        )
        """
        ddl_lineage = f"""
        CREATE TABLE IF NOT EXISTS {self.LINEAGE_TABLE} (
            source_feature  VARCHAR(128) NOT NULL,
            target_feature  VARCHAR(128) NOT NULL,
            transformation  TEXT,
            created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (source_feature, target_feature)
        )
        """
        ddl_versions = f"""
        CREATE TABLE IF NOT EXISTS {self.VERSION_HISTORY_TABLE} (
            feature_id      VARCHAR(16),
            name            VARCHAR(128),
            version         INTEGER,
            change_type     VARCHAR(20),
            change_details  VARIANT,
            changed_by      VARCHAR(128),
            changed_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        for ddl in [ddl_registry, ddl_lineage, ddl_versions]:
            self._execute(ddl, fetch=False)
        logger.info("Registry metadata tables verified")

    # ------------------------------------------------------------------
    # Feature registration
    # ------------------------------------------------------------------

    def register_feature(self, feature: FeatureDefinition) -> FeatureDefinition:
        """Register a new feature or update an existing one.

        If the feature name already exists at the given version, it will be
        updated. Otherwise a new row is inserted and a version history
        entry is recorded.
        """
        existing = self.get_feature(feature.name, feature.version)
        feature.updated_at = datetime.utcnow()

        if existing is not None:
            return self._update_feature(feature)

        sql = f"""
        INSERT INTO {self.REGISTRY_TABLE} (
            feature_id, name, version, description, data_type,
            source_table, source_query, entity_key, timestamp_column,
            freshness_sla_hours, owner, tags, dependencies, status,
            created_at, updated_at
        ) VALUES (
            %(feature_id)s, %(name)s, %(version)s, %(description)s, %(data_type)s,
            %(source_table)s, %(source_query)s, %(entity_key)s, %(timestamp_column)s,
            %(freshness_sla_hours)s, %(owner)s, PARSE_JSON(%(tags)s),
            PARSE_JSON(%(dependencies)s), %(status)s,
            %(created_at)s, %(updated_at)s
        )
        """
        self._execute(sql, params=self._feature_to_params(feature), fetch=False)
        self._record_version_change(feature, "CREATE")
        self._cache[feature.qualified_name] = feature
        logger.info("Registered feature %s", feature.qualified_name)
        return feature

    def _update_feature(self, feature: FeatureDefinition) -> FeatureDefinition:
        """Update an existing feature registration."""
        sql = f"""
        UPDATE {self.REGISTRY_TABLE}
        SET description = %(description)s,
            data_type = %(data_type)s,
            source_table = %(source_table)s,
            source_query = %(source_query)s,
            freshness_sla_hours = %(freshness_sla_hours)s,
            owner = %(owner)s,
            tags = PARSE_JSON(%(tags)s),
            dependencies = PARSE_JSON(%(dependencies)s),
            status = %(status)s,
            updated_at = %(updated_at)s
        WHERE name = %(name)s AND version = %(version)s
        """
        self._execute(sql, params=self._feature_to_params(feature), fetch=False)
        self._record_version_change(feature, "UPDATE")
        self._cache[feature.qualified_name] = feature
        logger.info("Updated feature %s", feature.qualified_name)
        return feature

    @staticmethod
    def _feature_to_params(feature: FeatureDefinition) -> dict[str, Any]:
        """Convert FeatureDefinition to query parameters."""
        return {
            "feature_id": feature.feature_id,
            "name": feature.name,
            "version": feature.version,
            "description": feature.description,
            "data_type": feature.data_type.value,
            "source_table": feature.source_table,
            "source_query": feature.source_query,
            "entity_key": feature.entity_key,
            "timestamp_column": feature.timestamp_column,
            "freshness_sla_hours": feature.freshness_sla_hours,
            "owner": feature.owner,
            "tags": json.dumps(feature.tags),
            "dependencies": json.dumps(feature.dependencies),
            "status": feature.status.value,
            "created_at": feature.created_at,
            "updated_at": feature.updated_at,
        }

    def _record_version_change(self, feature: FeatureDefinition, change_type: str) -> None:
        """Write an entry to the version history table."""
        sql = f"""
        INSERT INTO {self.VERSION_HISTORY_TABLE}
            (feature_id, name, version, change_type, change_details, changed_by)
        VALUES
            (%(feature_id)s, %(name)s, %(version)s, %(change_type)s,
             PARSE_JSON(%(details)s), %(changed_by)s)
        """
        details = {
            "status": feature.status.value,
            "source_table": feature.source_table,
            "description": feature.description,
        }
        self._execute(
            sql,
            params={
                "feature_id": feature.feature_id,
                "name": feature.name,
                "version": feature.version,
                "change_type": change_type,
                "details": json.dumps(details),
                "changed_by": feature.owner,
            },
            fetch=False,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_feature(self, name: str, version: int | None = None) -> FeatureDefinition | None:
        """Retrieve a feature definition by name and optional version.

        If version is None, the latest active version is returned.
        """
        cache_key = f"{name}:v{version}" if version else name
        if cache_key in self._cache:
            return self._cache[cache_key]

        if version is not None:
            sql = f"""
            SELECT * FROM {self.REGISTRY_TABLE}
            WHERE name = %(name)s AND version = %(version)s
            """
            params: dict[str, Any] = {"name": name, "version": version}
        else:
            sql = f"""
            SELECT * FROM {self.REGISTRY_TABLE}
            WHERE name = %(name)s AND status = 'ACTIVE'
            ORDER BY version DESC LIMIT 1
            """
            params = {"name": name}

        rows = self._execute(sql, params)
        if not rows:
            return None
        return self._row_to_feature(rows[0])

    def list_features(
        self,
        status: FeatureStatus | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> list[FeatureDefinition]:
        """List features with optional filters."""
        conditions = ["1=1"]
        params: dict[str, Any] = {}

        if status is not None:
            conditions.append("status = %(status)s")
            params["status"] = status.value
        if owner is not None:
            conditions.append("owner = %(owner)s")
            params["owner"] = owner

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM {self.REGISTRY_TABLE} WHERE {where_clause} ORDER BY name, version"
        rows = self._execute(sql, params)

        features = [self._row_to_feature(r) for r in rows]

        # Client-side tag filtering (Snowflake VARIANT tag matching)
        if tags:
            tag_set = set(tags)
            features = [f for f in features if tag_set.intersection(f.tags)]

        return features

    def create_new_version(self, name: str) -> FeatureDefinition:
        """Create a new version of a feature by copying the latest version."""
        current = self.get_feature(name)
        if current is None:
            raise ValueError(f"Feature '{name}' not found")

        new_version = current.model_copy(
            update={
                "version": current.version + 1,
                "status": FeatureStatus.DRAFT,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )
        return self.register_feature(new_version)

    def deprecate_feature(self, name: str, version: int) -> None:
        """Mark a feature version as deprecated."""
        feature = self.get_feature(name, version)
        if feature is None:
            raise ValueError(f"Feature '{name}:v{version}' not found")
        feature.status = FeatureStatus.DEPRECATED
        feature.updated_at = datetime.utcnow()
        self._update_feature(feature)
        logger.info("Deprecated feature %s:v%d", name, version)

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def add_lineage(self, edge: LineageEdge) -> None:
        """Record a lineage edge between two features."""
        sql = f"""
        INSERT INTO {self.LINEAGE_TABLE} (source_feature, target_feature, transformation)
        VALUES (%(source)s, %(target)s, %(transformation)s)
        """
        self._execute(
            sql,
            params={
                "source": edge.source_feature,
                "target": edge.target_feature,
                "transformation": edge.transformation,
            },
            fetch=False,
        )
        logger.info("Lineage: %s -> %s", edge.source_feature, edge.target_feature)

    def get_upstream_features(self, feature_name: str) -> list[str]:
        """Get all upstream (source) features in the lineage graph."""
        sql = f"""
        WITH RECURSIVE upstream AS (
            SELECT source_feature, target_feature
            FROM {self.LINEAGE_TABLE}
            WHERE target_feature = %(name)s
            UNION ALL
            SELECT l.source_feature, l.target_feature
            FROM {self.LINEAGE_TABLE} l
            INNER JOIN upstream u ON l.target_feature = u.source_feature
        )
        SELECT DISTINCT source_feature FROM upstream
        """
        rows = self._execute(sql, {"name": feature_name})
        return [r["SOURCE_FEATURE"] for r in rows]

    def get_downstream_features(self, feature_name: str) -> list[str]:
        """Get all downstream (dependent) features in the lineage graph."""
        sql = f"""
        WITH RECURSIVE downstream AS (
            SELECT source_feature, target_feature
            FROM {self.LINEAGE_TABLE}
            WHERE source_feature = %(name)s
            UNION ALL
            SELECT l.source_feature, l.target_feature
            FROM {self.LINEAGE_TABLE} l
            INNER JOIN downstream d ON l.source_feature = d.target_feature
        )
        SELECT DISTINCT target_feature FROM downstream
        """
        rows = self._execute(sql, {"name": feature_name})
        return [r["TARGET_FEATURE"] for r in rows]

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Return the full dependency graph as an adjacency list."""
        sql = f"SELECT source_feature, target_feature FROM {self.LINEAGE_TABLE}"
        rows = self._execute(sql)
        graph: dict[str, list[str]] = {}
        for row in rows:
            src = row["SOURCE_FEATURE"]
            tgt = row["TARGET_FEATURE"]
            graph.setdefault(src, []).append(tgt)
        return graph

    # ------------------------------------------------------------------
    # Freshness monitoring
    # ------------------------------------------------------------------

    def check_freshness(self) -> list[dict[str, Any]]:
        """Check which active features are stale (exceed their freshness SLA).

        Queries the actual feature tables to find the max timestamp and
        compares against the SLA.
        """
        active_features = self.list_features(status=FeatureStatus.ACTIVE)
        stale: list[dict[str, Any]] = []

        for feature in active_features:
            try:
                sql = f"""
                SELECT MAX({feature.timestamp_column}) AS latest_ts
                FROM {feature.source_table}
                """
                rows = self._execute(sql)
                if not rows or rows[0]["LATEST_TS"] is None:
                    stale.append(
                        {
                            "feature": feature.qualified_name,
                            "reason": "no_data",
                            "sla_hours": feature.freshness_sla_hours,
                        }
                    )
                    continue

                latest = rows[0]["LATEST_TS"]
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest)

                age = datetime.utcnow() - latest
                if age > timedelta(hours=feature.freshness_sla_hours):
                    stale.append(
                        {
                            "feature": feature.qualified_name,
                            "reason": "stale",
                            "age_hours": age.total_seconds() / 3600,
                            "sla_hours": feature.freshness_sla_hours,
                        }
                    )
            except Exception:
                logger.exception("Error checking freshness for %s", feature.qualified_name)
                stale.append(
                    {
                        "feature": feature.qualified_name,
                        "reason": "error",
                    }
                )

        if stale:
            logger.warning("Found %d stale features", len(stale))
        else:
            logger.info("All active features are within freshness SLA")

        return stale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_feature(row: dict[str, Any]) -> FeatureDefinition:
        """Convert a Snowflake row dict to a FeatureDefinition."""
        tags_raw = row.get("TAGS")
        deps_raw = row.get("DEPENDENCIES")
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
        deps = json.loads(deps_raw) if isinstance(deps_raw, str) else (deps_raw or [])

        return FeatureDefinition(
            name=row["NAME"],
            version=row["VERSION"],
            description=row.get("DESCRIPTION", ""),
            data_type=FeatureDataType(row.get("DATA_TYPE", "FLOAT")),
            source_table=row["SOURCE_TABLE"],
            source_query=row.get("SOURCE_QUERY", ""),
            entity_key=row.get("ENTITY_KEY", "patient_id"),
            timestamp_column=row.get("TIMESTAMP_COLUMN", "feature_ts"),
            freshness_sla_hours=row.get("FRESHNESS_SLA_HOURS", 24),
            owner=row.get("OWNER", "data-engineering"),
            tags=tags,
            dependencies=deps,
            status=FeatureStatus(row.get("STATUS", "DRAFT")),
            created_at=row.get("CREATED_AT", datetime.utcnow()),
            updated_at=row.get("UPDATED_AT", datetime.utcnow()),
        )

    def __enter__(self) -> FeatureRegistry:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
