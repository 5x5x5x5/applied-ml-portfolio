"""
Informatica-style Mapping abstraction.

In Informatica PowerCenter, a Mapping defines the data flow from source to target
through a series of transformations. This module replicates that paradigm:

  - SourceDefinition: Represents a source (table, flat file, XML, API)
  - TargetDefinition: Represents a target (table, flat file)
  - MappingParameter: $$PARAM-style parameters resolved at runtime
  - SessionConfig: Commit interval, error threshold, buffer sizes
  - Mapping: The top-level container connecting sources -> transformations -> targets
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class SourceType(str, Enum):
    """Source connection types mirroring Informatica source definitions."""

    RELATIONAL = "relational"
    FLAT_FILE = "flat_file"
    XML = "xml"
    API = "api"


class TargetType(str, Enum):
    """Target connection types."""

    RELATIONAL = "relational"
    FLAT_FILE = "flat_file"


class CommitStrategy(str, Enum):
    """Session-level commit strategy (mirrors Informatica commit types)."""

    TARGET_BASED = "target"  # Commit at target after N rows
    SOURCE_BASED = "source"  # Commit when source provides a commit point


class LoadType(str, Enum):
    """Target load type."""

    INSERT = "insert"
    UPDATE = "update"
    UPSERT = "upsert"
    DELETE = "delete"
    TRUNCATE_INSERT = "truncate_insert"


@dataclass
class SourceDefinition:
    """
    Informatica Source Definition equivalent.

    Describes where data comes from -- a relational table, flat file,
    XML document, or API endpoint.
    """

    name: str
    source_type: SourceType
    connection_name: str = ""
    table_name: str = ""
    file_path: str = ""
    file_delimiter: str = ","
    file_has_header: bool = True
    sql_override: str = ""
    xml_root_element: str = ""
    api_url: str = ""
    api_method: str = "GET"
    api_headers: dict[str, str] = field(default_factory=dict)
    columns: list[ColumnDefinition] = field(default_factory=list)

    def read_data(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Read data from the source based on its type."""
        params = params or {}
        log = logger.bind(source=self.name, source_type=self.source_type.value)

        if self.source_type == SourceType.FLAT_FILE:
            resolved_path = _resolve_params(self.file_path, params)
            log.info("reading_flat_file", path=resolved_path)
            return pd.read_csv(
                resolved_path,
                delimiter=self.file_delimiter,
                header=0 if self.file_has_header else None,
            )

        if self.source_type == SourceType.XML:
            import lxml.etree as ET  # noqa: N812

            resolved_path = _resolve_params(self.file_path, params)
            log.info("reading_xml", path=resolved_path)
            tree = ET.parse(resolved_path)  # noqa: S320
            root = tree.getroot()
            element_tag = self.xml_root_element or root.tag
            records: list[dict[str, Any]] = []
            for elem in root.iter(element_tag):
                record: dict[str, Any] = {}
                for child in elem:
                    tag = child.tag
                    if "}" in tag:
                        tag = tag.split("}")[-1]
                    record[tag] = child.text
                if record:
                    records.append(record)
            return pd.DataFrame(records)

        if self.source_type == SourceType.RELATIONAL:
            from sqlalchemy import create_engine, text

            log.info("reading_relational", table=self.table_name)
            engine = create_engine(self.connection_name)
            query = self.sql_override or f"SELECT * FROM {self.table_name}"
            query = _resolve_params(query, params)
            with engine.connect() as conn:
                return pd.read_sql(text(query), conn)

        if self.source_type == SourceType.API:
            import json
            import urllib.request

            resolved_url = _resolve_params(self.api_url, params)
            log.info("reading_api", url=resolved_url)
            req = urllib.request.Request(  # noqa: S310
                resolved_url,
                method=self.api_method,
                headers=self.api_headers,
            )
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame([data])

        msg = f"Unsupported source type: {self.source_type}"
        raise ValueError(msg)


@dataclass
class ColumnDefinition:
    """Column metadata (mirrors Informatica port definition)."""

    name: str
    datatype: str = "string"
    precision: int = 0
    scale: int = 0
    nullable: bool = True
    key_type: str = ""  # "PK", "FK", or ""


@dataclass
class TargetDefinition:
    """
    Informatica Target Definition equivalent.

    Describes the target table or file where transformed data lands.
    """

    name: str
    target_type: TargetType
    connection_name: str = ""
    table_name: str = ""
    file_path: str = ""
    file_delimiter: str = ","
    load_type: LoadType = LoadType.INSERT
    update_columns: list[str] = field(default_factory=list)
    key_columns: list[str] = field(default_factory=list)
    pre_sql: str = ""
    post_sql: str = ""
    columns: list[ColumnDefinition] = field(default_factory=list)

    def write_data(
        self,
        df: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Write DataFrame to target. Returns row count written."""
        params = params or {}
        log = logger.bind(target=self.name, target_type=self.target_type.value)

        if self.target_type == TargetType.FLAT_FILE:
            resolved_path = _resolve_params(self.file_path, params)
            log.info("writing_flat_file", path=resolved_path, rows=len(df))
            df.to_csv(resolved_path, sep=self.file_delimiter, index=False)
            return len(df)

        if self.target_type == TargetType.RELATIONAL:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.connection_name)
            table = _resolve_params(self.table_name, params)

            with engine.begin() as conn:
                if self.pre_sql:
                    conn.execute(text(_resolve_params(self.pre_sql, params)))

                if self.load_type == LoadType.TRUNCATE_INSERT:
                    conn.execute(text(f"TRUNCATE TABLE {table}"))
                    df.to_sql(table, conn, if_exists="append", index=False)

                elif self.load_type == LoadType.INSERT:
                    df.to_sql(table, conn, if_exists="append", index=False)

                elif self.load_type in (LoadType.UPDATE, LoadType.UPSERT):
                    self._upsert(conn, df, table)

                elif self.load_type == LoadType.DELETE:
                    for _, row in df.iterrows():
                        where = " AND ".join(f"{k} = :{k}" for k in self.key_columns)
                        conn.execute(
                            text(f"DELETE FROM {table} WHERE {where}"),
                            {k: row[k] for k in self.key_columns},
                        )

                if self.post_sql:
                    conn.execute(text(_resolve_params(self.post_sql, params)))

            log.info("wrote_relational", table=table, rows=len(df))
            return len(df)

        msg = f"Unsupported target type: {self.target_type}"
        raise ValueError(msg)

    def _upsert(self, conn: Any, df: pd.DataFrame, table: str) -> None:
        """Perform upsert using merge-style logic."""
        from sqlalchemy import text

        for _, row in df.iterrows():
            key_where = " AND ".join(f"{k} = :{k}" for k in self.key_columns)
            check = conn.execute(
                text(f"SELECT 1 FROM {table} WHERE {key_where}"),
                {k: row[k] for k in self.key_columns},
            ).fetchone()

            if check:
                set_cols = self.update_columns or [
                    c for c in df.columns if c not in self.key_columns
                ]
                set_clause = ", ".join(f"{c} = :{c}" for c in set_cols)
                conn.execute(
                    text(f"UPDATE {table} SET {set_clause} WHERE {key_where}"),
                    row.to_dict(),
                )
            elif self.load_type == LoadType.UPSERT:
                cols = ", ".join(df.columns)
                vals = ", ".join(f":{c}" for c in df.columns)
                conn.execute(
                    text(f"INSERT INTO {table} ({cols}) VALUES ({vals})"),
                    row.to_dict(),
                )


@dataclass
class MappingParameter:
    """
    Informatica-style mapping parameter ($$PARAM_NAME).

    Parameters are resolved at session runtime, allowing the same mapping
    to be reused across environments or time periods.
    """

    name: str  # e.g., "$$SOURCE_DATE", "$$ENV"
    default_value: str = ""
    description: str = ""

    @property
    def token(self) -> str:
        """Return the $$-prefixed parameter token."""
        name = self.name if self.name.startswith("$$") else f"$${self.name}"
        return name


@dataclass
class SessionConfig:
    """
    Session-level configuration (mirrors Informatica Session properties).

    Controls runtime behavior like commit intervals, error handling,
    buffer sizes, and recovery settings.
    """

    commit_interval: int = 10_000
    commit_strategy: CommitStrategy = CommitStrategy.TARGET_BASED
    error_threshold: int = 0  # 0 = stop on first error
    dtm_buffer_size: int = 128_000_000  # 128 MB default
    line_sequential_buffer_size: int = 1_048_576
    enable_recovery: bool = True
    recovery_strategy: str = "restart"
    treat_source_rows_as: str = "insert"  # insert, update, delete, data-driven
    bad_file_path: str = ""
    log_level: str = "INFO"
    pre_session_command: str = ""
    post_session_command: str = ""

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings: list[str] = []
        if self.commit_interval < 1:
            warnings.append("commit_interval must be >= 1; defaulting to 10000")
            self.commit_interval = 10_000
        if self.error_threshold < 0:
            warnings.append("error_threshold must be >= 0; defaulting to 0")
            self.error_threshold = 0
        return warnings


@dataclass
class Mapping:
    """
    Informatica Mapping equivalent.

    A Mapping is the central design object that defines data flow:
      Source(s) -> Transformation pipeline -> Target(s)

    Supports multiple sources (joined via SourceQualifier or Joiner),
    an ordered pipeline of transformations, and one or more targets
    (via Router or direct connection).
    """

    name: str
    description: str = ""
    sources: list[SourceDefinition] = field(default_factory=list)
    targets: list[TargetDefinition] = field(default_factory=list)
    transformations: list[Any] = field(default_factory=list)
    parameters: list[MappingParameter] = field(default_factory=list)
    session_config: SessionConfig = field(default_factory=SessionConfig)
    _resolved_params: dict[str, str] = field(default_factory=dict, repr=False)

    def add_source(self, source: SourceDefinition) -> Mapping:
        """Add a source definition to the mapping."""
        self.sources.append(source)
        logger.info("source_added", mapping=self.name, source=source.name)
        return self

    def add_target(self, target: TargetDefinition) -> Mapping:
        """Add a target definition to the mapping."""
        self.targets.append(target)
        logger.info("target_added", mapping=self.name, target=target.name)
        return self

    def add_transformation(self, transform: Any) -> Mapping:
        """Add a transformation to the pipeline (order matters)."""
        self.transformations.append(transform)
        logger.info(
            "transformation_added",
            mapping=self.name,
            transform=type(transform).__name__,
        )
        return self

    def add_parameter(self, param: MappingParameter) -> Mapping:
        """Register a mapping parameter."""
        self.parameters.append(param)
        return self

    def resolve_parameters(self, runtime_params: dict[str, str] | None = None) -> dict[str, str]:
        """
        Resolve all mapping parameters.

        Runtime params override defaults. Returns the full resolved dict.
        """
        runtime_params = runtime_params or {}
        resolved: dict[str, str] = {}

        for p in self.parameters:
            token = p.token
            if token in runtime_params:
                resolved[token] = runtime_params[token]
            elif p.default_value:
                resolved[token] = p.default_value
            else:
                msg = f"Required parameter {token} has no value"
                raise ValueError(msg)

        self._resolved_params = resolved
        logger.info("parameters_resolved", mapping=self.name, params=resolved)
        return resolved

    def validate(self) -> list[str]:
        """
        Validate the mapping structure.

        Returns a list of validation errors (empty = valid).
        """
        errors: list[str] = []

        if not self.sources:
            errors.append("Mapping must have at least one source")
        if not self.targets:
            errors.append("Mapping must have at least one target")
        if not self.transformations:
            errors.append("Mapping must have at least one transformation")

        # Check parameter references in SQL overrides
        all_sql = " ".join(s.sql_override for s in self.sources if s.sql_override)
        param_refs = set(re.findall(r"\$\$\w+", all_sql))
        defined = {p.token for p in self.parameters}
        undefined = param_refs - defined
        if undefined:
            errors.append(f"Undefined parameters referenced: {undefined}")

        errors.extend(self.session_config.validate())

        if errors:
            logger.warning("mapping_validation_failed", mapping=self.name, errors=errors)
        else:
            logger.info("mapping_validated", mapping=self.name)

        return errors

    def get_lineage(self) -> dict[str, Any]:
        """
        Return data lineage metadata (source -> transform -> target).

        Useful for impact analysis and documentation.
        """
        return {
            "mapping": self.name,
            "sources": [s.name for s in self.sources],
            "transformations": [
                {"type": type(t).__name__, "name": getattr(t, "name", "")}
                for t in self.transformations
            ],
            "targets": [t.name for t in self.targets],
            "parameters": [p.token for p in self.parameters],
        }


def _resolve_params(text: str, params: dict[str, Any]) -> str:
    """Replace $$PARAM tokens in text with resolved values."""
    result = text
    for token, value in params.items():
        result = result.replace(token, str(value))
    return result
