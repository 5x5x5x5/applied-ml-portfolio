"""
Informatica Session runner equivalent.

A Session executes a Mapping at runtime. It handles:
  - Reading data from sources
  - Piping data through the transformation pipeline
  - Writing data to targets
  - Row-level error trapping and bad-file output
  - Commit strategies (target-based, source-based)
  - Recovery and restart logic
  - Performance statistics collection
"""

from __future__ import annotations

import csv
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from pharma_flow.framework.mapping import CommitStrategy, Mapping
from pharma_flow.framework.transformations import Router

logger = structlog.get_logger(__name__)


class SessionStatus(str, Enum):
    """Session execution states."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    RECOVERING = "recovering"


@dataclass
class PerformanceStats:
    """Session performance statistics (mirrors Informatica session log)."""

    session_name: str = ""
    mapping_name: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    source_rows_read: int = 0
    target_rows_written: int = 0
    target_rows_rejected: int = 0
    target_rows_updated: int = 0
    target_rows_deleted: int = 0
    error_count: int = 0
    throughput_rows_per_sec: float = 0.0
    status: SessionStatus = SessionStatus.NOT_STARTED

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def summary(self) -> dict[str, Any]:
        """Return summary dict for logging and metadata."""
        return {
            "session": self.session_name,
            "mapping": self.mapping_name,
            "status": self.status.value,
            "elapsed_sec": round(self.elapsed_seconds, 2),
            "source_rows": self.source_rows_read,
            "target_written": self.target_rows_written,
            "target_updated": self.target_rows_updated,
            "target_deleted": self.target_rows_deleted,
            "rejected": self.target_rows_rejected,
            "errors": self.error_count,
            "throughput_rps": round(self.throughput_rows_per_sec, 1),
        }


@dataclass
class ErrorRecord:
    """A single row-level error captured during session execution."""

    row_index: int
    source_row: dict[str, Any]
    transformation: str
    error_message: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()


@dataclass
class SessionCheckpoint:
    """Recovery checkpoint for restart capability."""

    last_committed_row: int = 0
    last_committed_target: str = ""
    checkpoint_time: str = ""
    params_snapshot: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Persist checkpoint to disk."""
        import json

        data = {
            "last_committed_row": self.last_committed_row,
            "last_committed_target": self.last_committed_target,
            "checkpoint_time": self.checkpoint_time,
            "params_snapshot": self.params_snapshot,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> SessionCheckpoint:
        """Load checkpoint from disk."""
        import json

        data = json.loads(path.read_text())
        return cls(**data)


@dataclass
class Session:
    """
    Informatica Session equivalent.

    Executes a Mapping with full runtime support including error handling,
    commit management, recovery, and statistics.
    """

    name: str
    mapping: Mapping
    runtime_params: dict[str, str] = field(default_factory=dict)
    stats: PerformanceStats = field(default_factory=PerformanceStats)
    errors: list[ErrorRecord] = field(default_factory=list)
    _checkpoint: SessionCheckpoint = field(default_factory=SessionCheckpoint)
    _bad_file_writer: Any = None
    _status: SessionStatus = SessionStatus.NOT_STARTED

    @property
    def status(self) -> SessionStatus:
        return self._status

    def execute(self) -> PerformanceStats:
        """
        Run the session end-to-end.

        Steps:
          1. Resolve parameters
          2. Validate mapping
          3. Read sources
          4. Execute transformation pipeline
          5. Write to targets
          6. Collect statistics

        Returns PerformanceStats with execution summary.
        """
        log = logger.bind(session=self.name, mapping=self.mapping.name)
        self.stats = PerformanceStats(
            session_name=self.name,
            mapping_name=self.mapping.name,
        )
        self._status = SessionStatus.RUNNING
        self.stats.status = SessionStatus.RUNNING
        self.stats.start_time = datetime.now(tz=UTC)
        self.errors.clear()

        try:
            # Pre-session command
            config = self.mapping.session_config
            if config.pre_session_command:
                log.info("pre_session_command", cmd=config.pre_session_command)

            # Step 1: Resolve parameters
            params = self.mapping.resolve_parameters(self.runtime_params)
            log.info("parameters_resolved", params=params)

            # Step 2: Validate
            validation_errors = self.mapping.validate()
            if validation_errors:
                msg = f"Mapping validation failed: {validation_errors}"
                raise ValueError(msg)

            # Step 3: Read sources
            source_dfs = self._read_sources(params)
            if not source_dfs:
                msg = "No source data read"
                raise ValueError(msg)

            # Combine sources (primary source is first)
            df = source_dfs[0]
            self.stats.source_rows_read = len(df)
            log.info("sources_read", total_rows=self.stats.source_rows_read)

            # Build context with extra sources for Joiners
            context: dict[str, Any] = {}
            for i, extra_df in enumerate(source_dfs[1:], start=1):
                src_name = self.mapping.sources[i].name
                context[src_name] = extra_df

            # Step 4: Execute transformations
            df = self._execute_pipeline(df, context, params)

            # Step 5: Write to targets
            self._write_targets(df, params)

            # Finalize
            self._status = SessionStatus.SUCCEEDED
            self.stats.status = SessionStatus.SUCCEEDED

        except Exception as exc:
            self._status = SessionStatus.FAILED
            self.stats.status = SessionStatus.FAILED
            log.error("session_failed", error=str(exc), traceback=traceback.format_exc())

            if self.mapping.session_config.enable_recovery:
                self._save_recovery_checkpoint(params if "params" in dir() else {})

            raise

        finally:
            self.stats.end_time = datetime.now(tz=UTC)
            elapsed = self.stats.elapsed_seconds
            if elapsed > 0:
                self.stats.throughput_rows_per_sec = self.stats.source_rows_read / elapsed

            # Write bad file if there are errors
            if self.errors:
                self._write_bad_file()

            # Post-session command
            if config.post_session_command:
                log.info("post_session_command", cmd=config.post_session_command)

            log.info("session_complete", **self.stats.summary())

        return self.stats

    def _read_sources(self, params: dict[str, Any]) -> list[pd.DataFrame]:
        """Read all source definitions and return list of DataFrames."""
        dfs: list[pd.DataFrame] = []
        for source in self.mapping.sources:
            try:
                df = source.read_data(params)
                logger.info(
                    "source_read",
                    source=source.name,
                    rows=len(df),
                    columns=list(df.columns),
                )
                dfs.append(df)
            except Exception as exc:
                logger.error("source_read_failed", source=source.name, error=str(exc))
                raise
        return dfs

    def _execute_pipeline(
        self,
        df: pd.DataFrame,
        context: dict[str, Any],
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Execute the transformation pipeline with error handling."""
        config = self.mapping.session_config
        error_threshold = config.error_threshold
        commit_interval = config.commit_interval

        result_df = df
        total_errors = 0

        for i, transform in enumerate(self.mapping.transformations):
            t_name = type(transform).__name__
            t_label = getattr(transform, "name", str(i))
            log = logger.bind(transform=t_name, label=t_label, step=i + 1)

            try:
                start = time.monotonic()

                if config.commit_strategy == CommitStrategy.TARGET_BASED:
                    # Process in chunks for target-based commit
                    if len(result_df) > commit_interval:
                        chunks = [
                            result_df.iloc[j : j + commit_interval]
                            for j in range(0, len(result_df), commit_interval)
                        ]
                        processed_chunks = []
                        for chunk_idx, chunk in enumerate(chunks):
                            try:
                                processed = transform.execute(chunk, context)
                                processed_chunks.append(processed)
                            except Exception as exc:
                                total_errors += len(chunk)
                                self._record_chunk_errors(
                                    chunk, t_name, str(exc), chunk_idx * commit_interval
                                )
                                if error_threshold > 0 and total_errors > error_threshold:
                                    msg = (
                                        f"Error threshold ({error_threshold}) "
                                        f"exceeded at transform {t_name}"
                                    )
                                    raise RuntimeError(msg) from exc
                        if processed_chunks:
                            result_df = pd.concat(processed_chunks, ignore_index=True)
                    else:
                        result_df = transform.execute(result_df, context)
                else:
                    result_df = transform.execute(result_df, context)

                elapsed = time.monotonic() - start
                log.info(
                    "transform_complete",
                    rows_out=len(result_df),
                    elapsed_ms=round(elapsed * 1000, 1),
                )

                # If Router, store output groups in context
                if isinstance(transform, Router):
                    for group in transform.groups:
                        context[f"RTR_{group.name}"] = transform.get_group(group.name)
                    if "DEFAULT" not in [g.name for g in transform.groups]:
                        context["RTR_DEFAULT"] = transform.get_group("DEFAULT")

            except RuntimeError:
                raise
            except Exception as exc:
                total_errors += 1
                log.error("transform_failed", error=str(exc))

                if error_threshold == 0:
                    raise

                if total_errors > error_threshold:
                    msg = f"Error threshold ({error_threshold}) exceeded"
                    raise RuntimeError(msg) from exc

        return result_df

    def _write_targets(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        """Write transformed data to all target definitions."""
        # Handle UpdateStrategy disposition if present
        disposition_col = "_disposition"
        has_disposition = disposition_col in df.columns

        for target in self.mapping.targets:
            try:
                if has_disposition:
                    self._write_with_disposition(target, df, disposition_col, params)
                else:
                    rows_written = target.write_data(df, params)
                    self.stats.target_rows_written += rows_written

                logger.info("target_written", target=target.name)

            except Exception as exc:
                logger.error(
                    "target_write_failed",
                    target=target.name,
                    error=str(exc),
                )
                raise

    def _write_with_disposition(
        self,
        target: Any,
        df: pd.DataFrame,
        disposition_col: str,
        params: dict[str, Any],
    ) -> None:
        """Write rows based on UpdateStrategy disposition flags."""
        from pharma_flow.framework.transformations import RowDisposition

        inserts = df[df[disposition_col] == RowDisposition.DD_INSERT].drop(
            columns=[disposition_col]
        )
        updates = df[df[disposition_col] == RowDisposition.DD_UPDATE].drop(
            columns=[disposition_col]
        )
        deletes = df[df[disposition_col] == RowDisposition.DD_DELETE].drop(
            columns=[disposition_col]
        )
        rejects = df[df[disposition_col] == RowDisposition.DD_REJECT].drop(
            columns=[disposition_col]
        )

        if len(inserts) > 0:
            from pharma_flow.framework.mapping import LoadType

            original_load = target.load_type
            target.load_type = LoadType.INSERT
            target.write_data(inserts, params)
            target.load_type = original_load
            self.stats.target_rows_written += len(inserts)

        if len(updates) > 0:
            from pharma_flow.framework.mapping import LoadType

            original_load = target.load_type
            target.load_type = LoadType.UPDATE
            target.write_data(updates, params)
            target.load_type = original_load
            self.stats.target_rows_updated += len(updates)

        if len(deletes) > 0:
            from pharma_flow.framework.mapping import LoadType

            original_load = target.load_type
            target.load_type = LoadType.DELETE
            target.write_data(deletes, params)
            target.load_type = original_load
            self.stats.target_rows_deleted += len(deletes)

        if len(rejects) > 0:
            self.stats.target_rows_rejected += len(rejects)
            for _, row in rejects.iterrows():
                self.errors.append(
                    ErrorRecord(
                        row_index=int(row.name) if hasattr(row, "name") else 0,
                        source_row=row.to_dict(),
                        transformation="UpdateStrategy",
                        error_message="Row rejected by update strategy",
                    )
                )

    def _record_chunk_errors(
        self,
        chunk: pd.DataFrame,
        transform_name: str,
        error_msg: str,
        offset: int,
    ) -> None:
        """Record errors for a failed chunk of rows."""
        for idx, (_, row) in enumerate(chunk.iterrows()):
            self.errors.append(
                ErrorRecord(
                    row_index=offset + idx,
                    source_row=row.to_dict(),
                    transformation=transform_name,
                    error_message=error_msg,
                )
            )

    def _write_bad_file(self) -> None:
        """Write rejected/errored rows to a bad file (like Informatica reject file)."""
        bad_path = self.mapping.session_config.bad_file_path
        if not bad_path:
            bad_path = (
                f"/tmp/pharmaflow_{self.name}_bad_{datetime.now(tz=UTC):%Y%m%d%H%M%S}.csv"
            )

        path = Path(bad_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["row_index", "transformation", "error_message", "timestamp"]
                + list(self.errors[0].source_row.keys() if self.errors else []),
            )
            writer.writeheader()
            for err in self.errors:
                row = {
                    "row_index": err.row_index,
                    "transformation": err.transformation,
                    "error_message": err.error_message,
                    "timestamp": err.timestamp,
                    **err.source_row,
                }
                writer.writerow(row)

        logger.info("bad_file_written", path=str(path), error_count=len(self.errors))

    def _save_recovery_checkpoint(self, params: dict[str, Any]) -> None:
        """Save a recovery checkpoint for restart."""
        self._checkpoint = SessionCheckpoint(
            last_committed_row=self.stats.target_rows_written,
            last_committed_target=(self.mapping.targets[0].name if self.mapping.targets else ""),
            checkpoint_time=datetime.now(tz=UTC).isoformat(),
            params_snapshot={k: str(v) for k, v in params.items()},
        )

        checkpoint_path = Path(f"/tmp/pharmaflow_{self.name}_checkpoint.json")
        self._checkpoint.save(checkpoint_path)
        logger.info("recovery_checkpoint_saved", path=str(checkpoint_path))

    def recover(self) -> PerformanceStats:
        """
        Attempt to recover and restart from the last checkpoint.

        Reads the checkpoint file and re-executes the session from
        the last committed row position.
        """
        checkpoint_path = Path(f"/tmp/pharmaflow_{self.name}_checkpoint.json")
        if not checkpoint_path.exists():
            logger.warning("no_checkpoint_found", session=self.name)
            return self.execute()

        self._status = SessionStatus.RECOVERING
        checkpoint = SessionCheckpoint.load(checkpoint_path)
        logger.info(
            "recovering_session",
            session=self.name,
            from_row=checkpoint.last_committed_row,
        )

        # Merge checkpoint params with runtime params
        merged_params = {**checkpoint.params_snapshot, **self.runtime_params}
        self.runtime_params = merged_params

        # Re-execute (in a real system, we'd skip already-committed rows)
        return self.execute()
