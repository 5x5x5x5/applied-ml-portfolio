"""
Informatica PowerCenter transformation equivalents.

Each class mirrors an Informatica transformation type, operating on
pandas DataFrames. Transformations are chained in a Mapping pipeline
and executed sequentially by the Session runner.

Transformation catalog:
  - SourceQualifier: SQL override, filter, join sources at the source
  - Expression: Calculate derived/computed columns
  - Filter: Conditional row filtering
  - Aggregator: Group-by with aggregate functions
  - Joiner: Master-detail joins
  - Lookup: Cache-based lookup (connected / unconnected)
  - Router: Route rows to multiple output groups
  - Sorter: Sort by key columns
  - UpdateStrategy: Set row disposition (insert/update/delete/reject)
  - SequenceGenerator: Surrogate key generation
  - Normalizer: Normalize repeating groups into rows
  - Rank: Top/bottom N per group
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Transformation(ABC):
    """Abstract base for all transformations."""

    name: str

    @abstractmethod
    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        """Apply the transformation and return the result DataFrame."""
        ...

    def _log(self, event: str, **kwargs: Any) -> None:
        logger.info(event, transform=self.__class__.__name__, name=self.name, **kwargs)


# ---------------------------------------------------------------------------
# SourceQualifier
# ---------------------------------------------------------------------------


@dataclass
class SourceQualifier(Transformation):
    """
    Informatica Source Qualifier equivalent.

    Sits between the source and downstream transformations. Supports:
      - SQL override (custom query)
      - Source filter (WHERE clause applied in-memory)
      - Joining multiple source DataFrames
      - Column selection / projection
      - Distinct rows
    """

    name: str = "SQ"
    source_filter: str = ""
    sql_override: str = ""
    select_columns: list[str] = field(default_factory=list)
    distinct: bool = False
    join_condition: str = ""
    _additional_sources: list[pd.DataFrame] = field(default_factory=list, repr=False)

    def add_source(self, df: pd.DataFrame) -> SourceQualifier:
        """Add an additional source for joining."""
        self._additional_sources.append(df)
        return self

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        result = df.copy()

        # Join additional sources
        for extra_df in self._additional_sources:
            if self.join_condition:
                left_key, right_key = [k.strip() for k in self.join_condition.split("=")]
                result = result.merge(extra_df, left_on=left_key, right_on=right_key, how="inner")
            else:
                # Cross join if no condition
                result = result.merge(extra_df, how="cross")

        # Apply source filter
        if self.source_filter:
            result = result.query(self.source_filter)

        # Column projection
        if self.select_columns:
            available = [c for c in self.select_columns if c in result.columns]
            result = result[available]

        # Distinct
        if self.distinct:
            result = result.drop_duplicates()

        self._log("source_qualifier_applied", rows_out=len(result))
        return result


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------


@dataclass
class Expression(Transformation):
    """
    Informatica Expression transformation equivalent.

    Calculates derived columns using user-supplied functions.
    Each expression is a (column_name, callable) pair where the callable
    receives the entire DataFrame and returns a Series or scalar.
    """

    name: str = "EXP"
    expressions: list[tuple[str, Callable[[pd.DataFrame], Any]]] = field(default_factory=list)

    def add_expression(
        self,
        output_column: str,
        func: Callable[[pd.DataFrame], Any],
    ) -> Expression:
        """Register a derived column expression."""
        self.expressions.append((output_column, func))
        return self

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        result = df.copy()
        for col_name, func in self.expressions:
            try:
                result[col_name] = func(result)
            except Exception as exc:
                logger.error(
                    "expression_error",
                    column=col_name,
                    error=str(exc),
                )
                raise
        self._log("expression_applied", columns=[c for c, _ in self.expressions])
        return result


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass
class Filter(Transformation):
    """
    Informatica Filter transformation equivalent.

    Passes only rows meeting the filter condition. Rows that fail the
    condition are dropped (not routed elsewhere -- use Router for that).
    """

    name: str = "FIL"
    condition: str = ""
    condition_func: Callable[[pd.DataFrame], pd.Series] | None = None

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if self.condition_func is not None:
            mask = self.condition_func(df)
            result = df[mask].copy()
        elif self.condition:
            result = df.query(self.condition).copy()
        else:
            result = df.copy()

        self._log(
            "filter_applied",
            rows_in=len(df),
            rows_out=len(result),
            rows_rejected=len(df) - len(result),
        )
        return result


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class AggregatorMode(str, Enum):
    SORTED = "sorted"
    UNSORTED = "unsorted"


@dataclass
class Aggregator(Transformation):
    """
    Informatica Aggregator transformation equivalent.

    Groups data and computes aggregate values. Supports sorted input
    (memory-efficient) and unsorted (caches all data).
    """

    name: str = "AGG"
    group_by: list[str] = field(default_factory=list)
    aggregations: dict[str, str | Callable] = field(default_factory=dict)
    mode: AggregatorMode = AggregatorMode.UNSORTED

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if self.mode == AggregatorMode.SORTED:
            # Verify data is sorted by group keys
            for col in self.group_by:
                if not df[col].is_monotonic_increasing:
                    logger.warning(
                        "aggregator_unsorted_input",
                        column=col,
                        hint="SORTED mode expects pre-sorted input",
                    )
                    break

        if self.group_by:
            grouped = df.groupby(self.group_by, sort=False)
            result = grouped.agg(self.aggregations).reset_index()
        else:
            result = df.agg(self.aggregations).to_frame().T

        # Flatten multi-level columns if any
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ["_".join(str(c) for c in col) for col in result.columns]

        self._log("aggregator_applied", groups=len(result))
        return result


# ---------------------------------------------------------------------------
# Joiner
# ---------------------------------------------------------------------------


class JoinType(str, Enum):
    INNER = "inner"
    LEFT_OUTER = "left"
    RIGHT_OUTER = "right"
    FULL_OUTER = "outer"
    MASTER_ONLY = "master_only"


@dataclass
class Joiner(Transformation):
    """
    Informatica Joiner transformation equivalent.

    Joins a master (pipeline) DataFrame with a detail DataFrame.
    The detail is provided at construction or via context.
    """

    name: str = "JNR"
    detail_source: pd.DataFrame | None = None
    join_type: JoinType = JoinType.INNER
    master_keys: list[str] = field(default_factory=list)
    detail_keys: list[str] = field(default_factory=list)
    detail_source_name: str = ""

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        context = context or {}
        detail = self.detail_source
        if detail is None and self.detail_source_name:
            detail = context.get(self.detail_source_name)
        if detail is None:
            msg = f"Joiner '{self.name}': no detail source provided"
            raise ValueError(msg)

        if self.join_type == JoinType.MASTER_ONLY:
            # Anti-join: master rows with no match in detail
            merged = df.merge(
                detail[self.detail_keys],
                left_on=self.master_keys,
                right_on=self.detail_keys,
                how="left",
                indicator=True,
            )
            result = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            result = df.merge(
                detail,
                left_on=self.master_keys,
                right_on=self.detail_keys,
                how=self.join_type.value,
                suffixes=("", "_detail"),
            )

        self._log(
            "joiner_applied",
            join_type=self.join_type.value,
            master_rows=len(df),
            detail_rows=len(detail),
            result_rows=len(result),
        )
        return result


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class LookupMode(str, Enum):
    CONNECTED = "connected"  # In-pipeline, returns all lookup columns
    UNCONNECTED = "unconnected"  # Called from Expression, returns single value


@dataclass
class Lookup(Transformation):
    """
    Informatica Lookup transformation equivalent.

    Performs cache-based lookups against a reference dataset.
    Supports connected (in-pipeline) and unconnected (callable) modes.
    """

    name: str = "LKP"
    lookup_source: pd.DataFrame | None = None
    lookup_keys: list[str] = field(default_factory=list)
    return_columns: list[str] = field(default_factory=list)
    mode: LookupMode = LookupMode.CONNECTED
    default_values: dict[str, Any] = field(default_factory=dict)
    condition: str = ""
    _cache: dict[tuple, dict[str, Any]] = field(default_factory=dict, repr=False)

    def build_cache(self, source: pd.DataFrame | None = None) -> None:
        """Build an in-memory hash cache of the lookup data."""
        data = source if source is not None else self.lookup_source
        if data is None:
            msg = f"Lookup '{self.name}': no lookup source for cache build"
            raise ValueError(msg)

        self._cache.clear()
        for _, row in data.iterrows():
            key = tuple(row[k] for k in self.lookup_keys)
            self._cache[key] = {c: row[c] for c in self.return_columns if c in row.index}

        self._log("lookup_cache_built", cache_size=len(self._cache))

    def lookup_value(self, key_values: tuple) -> dict[str, Any]:
        """Unconnected lookup: return matching row dict or defaults."""
        if not self._cache:
            self.build_cache()
        return self._cache.get(key_values, self.default_values)

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if not self._cache:
            self.build_cache()

        if self.mode == LookupMode.CONNECTED:
            lookup_df = pd.DataFrame.from_dict(self._cache, orient="index")
            if lookup_df.empty:
                for col in self.return_columns:
                    df[col] = self.default_values.get(col)
                return df

            lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=self.lookup_keys)
            lookup_df = lookup_df.reset_index()

            result = df.merge(
                lookup_df,
                on=self.lookup_keys,
                how="left",
                suffixes=("", "_lkp"),
            )

            # Fill nulls with defaults
            for col, default in self.default_values.items():
                if col in result.columns:
                    result[col] = result[col].fillna(default)

            self._log(
                "lookup_connected",
                rows_in=len(df),
                rows_out=len(result),
                cache_hits=len(result.dropna(subset=self.return_columns[:1])),
            )
            return result
        else:
            # Unconnected mode: just build cache, used by Expression
            self._log("lookup_unconnected_ready", cache_size=len(self._cache))
            return df


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RouterGroup:
    """A named output group with its filter condition."""

    name: str
    condition: str = ""
    condition_func: Callable[[pd.DataFrame], pd.Series] | None = None


@dataclass
class Router(Transformation):
    """
    Informatica Router transformation equivalent.

    Routes rows to multiple output groups based on conditions.
    Rows not matching any user-defined group go to the DEFAULT group.
    """

    name: str = "RTR"
    groups: list[RouterGroup] = field(default_factory=list)
    _output_groups: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def add_group(
        self,
        group_name: str,
        condition: str = "",
        condition_func: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> Router:
        """Add a named output group."""
        self.groups.append(RouterGroup(group_name, condition, condition_func))
        return self

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        self._output_groups.clear()
        remaining = df.copy()

        for group in self.groups:
            if group.condition_func is not None:
                mask = group.condition_func(remaining)
            elif group.condition:
                mask = remaining.eval(group.condition)
            else:
                mask = pd.Series([True] * len(remaining), index=remaining.index)

            self._output_groups[group.name] = remaining[mask].copy()
            remaining = remaining[~mask]

        # DEFAULT group: rows that didn't match any condition
        self._output_groups["DEFAULT"] = remaining

        for gname, gdf in self._output_groups.items():
            self._log("router_group", group=gname, rows=len(gdf))

        # Return first group as the pipeline continuation
        if self.groups:
            return self._output_groups[self.groups[0].name]
        return df

    def get_group(self, group_name: str) -> pd.DataFrame:
        """Retrieve a specific output group's DataFrame."""
        if group_name not in self._output_groups:
            msg = f"Router group '{group_name}' not found"
            raise KeyError(msg)
        return self._output_groups[group_name]


# ---------------------------------------------------------------------------
# Sorter
# ---------------------------------------------------------------------------


@dataclass
class Sorter(Transformation):
    """
    Informatica Sorter transformation equivalent.

    Sorts data by specified key columns. Supports ascending/descending
    per column and case-sensitive/insensitive sorting.
    """

    name: str = "SRT"
    sort_keys: list[str] = field(default_factory=list)
    ascending: list[bool] = field(default_factory=list)
    case_sensitive: bool = True
    distinct: bool = False

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        asc = self.ascending if self.ascending else [True] * len(self.sort_keys)

        if not self.case_sensitive:
            # Create temporary lowercase columns for sorting
            temp_cols = {}
            for col in self.sort_keys:
                if df[col].dtype == object:
                    temp_name = f"_sort_{col}"
                    df[temp_name] = df[col].str.lower()
                    temp_cols[col] = temp_name

            sort_cols = [temp_cols.get(c, c) for c in self.sort_keys]
            result = df.sort_values(by=sort_cols, ascending=asc).copy()
            result = result.drop(columns=list(temp_cols.values()), errors="ignore")
        else:
            result = df.sort_values(by=self.sort_keys, ascending=asc).copy()

        if self.distinct:
            result = result.drop_duplicates(subset=self.sort_keys)

        result = result.reset_index(drop=True)
        self._log("sorter_applied", rows=len(result), keys=self.sort_keys)
        return result


# ---------------------------------------------------------------------------
# UpdateStrategy
# ---------------------------------------------------------------------------


class RowDisposition(IntEnum):
    """Row-level update strategy flags (matches Informatica constants)."""

    DD_INSERT = 0
    DD_UPDATE = 1
    DD_DELETE = 2
    DD_REJECT = 3


@dataclass
class UpdateStrategy(Transformation):
    """
    Informatica Update Strategy transformation equivalent.

    Assigns a row disposition (insert/update/delete/reject) to each row
    based on a user-defined function. The Session runner uses the
    disposition to determine how to write each row to the target.
    """

    name: str = "UPD"
    strategy_func: Callable[[pd.DataFrame], pd.Series] | None = None
    default_disposition: RowDisposition = RowDisposition.DD_INSERT
    disposition_column: str = "_disposition"

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        result = df.copy()

        if self.strategy_func is not None:
            result[self.disposition_column] = self.strategy_func(result)
        else:
            result[self.disposition_column] = self.default_disposition.value

        counts = result[self.disposition_column].value_counts().to_dict()
        self._log(
            "update_strategy_applied",
            inserts=counts.get(RowDisposition.DD_INSERT, 0),
            updates=counts.get(RowDisposition.DD_UPDATE, 0),
            deletes=counts.get(RowDisposition.DD_DELETE, 0),
            rejects=counts.get(RowDisposition.DD_REJECT, 0),
        )
        return result


# ---------------------------------------------------------------------------
# SequenceGenerator
# ---------------------------------------------------------------------------


@dataclass
class SequenceGenerator(Transformation):
    """
    Informatica Sequence Generator transformation equivalent.

    Generates surrogate keys (monotonically increasing integers).
    Supports start value, increment, and cycle options.
    """

    name: str = "SEQ"
    output_column: str = "surrogate_key"
    start_value: int = 1
    increment: int = 1
    end_value: int = 2_147_483_647
    cycle: bool = False
    _current_value: int = 0

    def __post_init__(self) -> None:
        self._current_value = self.start_value

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        result = df.copy()
        n = len(result)
        keys = []

        for _ in range(n):
            keys.append(self._current_value)
            self._current_value += self.increment
            if self._current_value > self.end_value:
                if self.cycle:
                    self._current_value = self.start_value
                else:
                    msg = f"Sequence '{self.name}' exceeded end_value {self.end_value}"
                    raise OverflowError(msg)

        result[self.output_column] = keys
        self._log("sequence_generated", count=n, last_value=self._current_value)
        return result

    @property
    def next_value(self) -> int:
        """Peek at the next value (for unconnected use)."""
        return self._current_value


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


@dataclass
class Normalizer(Transformation):
    """
    Informatica Normalizer transformation equivalent.

    Converts repeating groups (columns) into individual rows.
    For example, columns DRUG1, DRUG2, DRUG3 become separate rows
    with a single DRUG column plus an occurrence index.
    """

    name: str = "NRM"
    group_columns: list[str] = field(default_factory=list)  # Repeating cols
    normalized_column: str = ""  # Output column name
    index_column: str = "occurrence"  # 1-based occurrence counter
    id_columns: list[str] = field(default_factory=list)  # Kept as-is

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if not self.group_columns:
            return df

        id_cols = self.id_columns or [c for c in df.columns if c not in self.group_columns]

        melted = df.melt(
            id_vars=id_cols,
            value_vars=self.group_columns,
            var_name="_source_col",
            value_name=self.normalized_column or "value",
        )

        # Add occurrence index (1-based within each ID group)
        if id_cols:
            melted[self.index_column] = melted.groupby(id_cols).cumcount() + 1
        else:
            melted[self.index_column] = range(1, len(melted) + 1)

        # Drop rows where normalized value is null
        output_col = self.normalized_column or "value"
        melted = melted.dropna(subset=[output_col])
        melted = melted.drop(columns=["_source_col"])

        self._log(
            "normalizer_applied",
            rows_in=len(df),
            rows_out=len(melted),
            groups_normalized=len(self.group_columns),
        )
        return melted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


@dataclass
class Rank(Transformation):
    """
    Informatica Rank transformation equivalent.

    Selects the top or bottom N rows per group based on a rank column.
    Outputs a RANKINDEX column indicating the rank within each group.
    """

    name: str = "RNK"
    group_by: list[str] = field(default_factory=list)
    rank_column: str = ""
    top_n: int = 1
    ascending: bool = True  # True = bottom N, False = top N
    rank_index_column: str = "RANKINDEX"

    def execute(self, df: pd.DataFrame, context: dict[str, Any] | None = None) -> pd.DataFrame:
        if not self.rank_column:
            msg = f"Rank '{self.name}': rank_column is required"
            raise ValueError(msg)

        result = df.copy()

        if self.group_by:
            result[self.rank_index_column] = (
                result.groupby(self.group_by)[self.rank_column]
                .rank(method="first", ascending=self.ascending)
                .astype(int)
            )
        else:
            result[self.rank_index_column] = (
                result[self.rank_column].rank(method="first", ascending=self.ascending).astype(int)
            )

        result = result[result[self.rank_index_column] <= self.top_n]
        result = result.sort_values(by=self.group_by + [self.rank_index_column]).reset_index(
            drop=True
        )

        self._log(
            "rank_applied",
            rows_in=len(df),
            rows_out=len(result),
            top_n=self.top_n,
        )
        return result


# ---------------------------------------------------------------------------
# Utility: Row hash for SCD comparisons
# ---------------------------------------------------------------------------


def compute_row_hash(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Compute an MD5 hash of specified columns for each row.

    Used by SCD Type 2 logic to detect changes.
    """

    def _hash_row(row: pd.Series) -> str:
        concat = "|".join(str(row[c]) for c in columns)
        return hashlib.md5(concat.encode(), usedforsecurity=False).hexdigest()

    return df.apply(_hash_row, axis=1)
