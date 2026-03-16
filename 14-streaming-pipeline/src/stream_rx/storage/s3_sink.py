"""
S3 data lake sink with Parquet writing, partitioning, and compaction.

Implements micro-batch buffering for efficient S3 writes, date/drug_class
partitioning for query optimization, schema evolution handling, and a
compaction job for merging small files into larger ones.
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from stream_rx.config import S3Config, get_config
from stream_rx.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Parquet schema definitions
# ---------------------------------------------------------------------------

PRESCRIPTION_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("patient_id", pa.string(), nullable=False),
        pa.field("drug_ndc", pa.string()),
        pa.field("drug_name", pa.string()),
        pa.field("drug_class", pa.string()),
        pa.field("prescriber_npi", pa.string()),
        pa.field("prescriber_name", pa.string()),
        pa.field("pharmacy_ncpdp", pa.string()),
        pa.field("pharmacy_name", pa.string()),
        pa.field("pharmacy_state", pa.string()),
        pa.field("quantity", pa.float64()),
        pa.field("days_supply", pa.int32()),
        pa.field("refill_number", pa.int32()),
        pa.field("diagnosis_codes", pa.list_(pa.string())),
        pa.field("plan_id", pa.string()),
    ]
)

ADVERSE_EVENT_SCHEMA = pa.schema(
    [
        pa.field("report_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("patient_id", pa.string(), nullable=False),
        pa.field("patient_age", pa.int32()),
        pa.field("patient_sex", pa.string()),
        pa.field("severity", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("hospitalized", pa.bool_()),
        pa.field("reporter_type", pa.string()),
        pa.field("reporter_country", pa.string()),
        pa.field("narrative", pa.string()),
        pa.field("suspect_drug_names", pa.list_(pa.string())),
        pa.field("reaction_terms", pa.list_(pa.string())),
    ]
)


# ---------------------------------------------------------------------------
# Micro-batch buffer
# ---------------------------------------------------------------------------


class MicroBatchBuffer:
    """
    Thread-safe buffer that accumulates records and flushes when thresholds
    are reached (size or time interval).
    """

    def __init__(
        self,
        max_size_mb: int = 128,
        max_interval_sec: int = 300,
    ) -> None:
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_interval_sec = max_interval_sec
        self._records: list[dict[str, Any]] = []
        self._current_size_bytes = 0
        self._last_flush_time = time.monotonic()
        self._lock = threading.Lock()
        self._flush_count = 0

    def add(self, record: dict[str, Any], estimated_size: int = 500) -> bool:
        """
        Add a record to the buffer.

        Args:
            record: The record dict to buffer.
            estimated_size: Estimated byte size of the record.

        Returns:
            True if the buffer should be flushed after this add.
        """
        with self._lock:
            self._records.append(record)
            self._current_size_bytes += estimated_size
            should_flush = (
                self._current_size_bytes >= self._max_size_bytes
                or (time.monotonic() - self._last_flush_time) >= self._max_interval_sec
            )
            return should_flush

    def drain(self) -> list[dict[str, Any]]:
        """Remove and return all buffered records."""
        with self._lock:
            records = list(self._records)
            self._records.clear()
            self._current_size_bytes = 0
            self._last_flush_time = time.monotonic()
            self._flush_count += 1
            return records

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def flush_count(self) -> int:
        return self._flush_count

    def should_flush(self) -> bool:
        """Check if the buffer should be flushed based on time."""
        with self._lock:
            if not self._records:
                return False
            return (time.monotonic() - self._last_flush_time) >= self._max_interval_sec


# ---------------------------------------------------------------------------
# S3 Parquet Sink
# ---------------------------------------------------------------------------


class S3ParquetSink:
    """
    Writes streaming data to S3 as partitioned Parquet files.

    Features:
    - Hive-style partitioning by date and drug_class
    - Micro-batch buffering for efficient writes
    - Schema evolution handling (additive columns)
    - Compaction of small files
    """

    def __init__(
        self,
        s3_config: S3Config | None = None,
        boto_session: boto3.Session | None = None,
        event_type: str = "prescriptions",
    ) -> None:
        cfg = s3_config or get_config().s3
        self._config = cfg
        session = boto_session or boto3.Session(region_name=cfg.region)
        self._s3_client = session.client("s3")
        self._event_type = event_type
        self._schema = (
            PRESCRIPTION_SCHEMA if event_type == "prescriptions" else ADVERSE_EVENT_SCHEMA
        )
        self._buffer = MicroBatchBuffer(
            max_size_mb=cfg.buffer_size_mb,
            max_interval_sec=cfg.buffer_interval_sec,
        )
        self._total_written = 0
        self._total_files = 0
        self._total_bytes = 0

        logger.info(
            "s3_sink_initialized",
            bucket=cfg.bucket,
            prefix=cfg.prefix,
            event_type=event_type,
            buffer_size_mb=cfg.buffer_size_mb,
        )

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_path(record: dict[str, Any], base_prefix: str, event_type: str) -> str:
        """
        Generate a Hive-style partition path for a record.

        Format: {prefix}/{event_type}/date=YYYY-MM-DD/drug_class={class}/
        """
        ts = record.get("timestamp")
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                dt = datetime.utcnow()
        elif isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.utcnow()

        date_str = dt.strftime("%Y-%m-%d")
        drug_class = record.get("drug_class", "unknown")

        return f"{base_prefix}/{event_type}/date={date_str}/drug_class={drug_class}"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_record(self, record: dict[str, Any]) -> bool:
        """
        Add a record to the buffer. Flushes if threshold exceeded.

        Returns:
            True if flush was triggered.
        """
        should_flush = self._buffer.add(record)
        if should_flush:
            self.flush()
            return True
        return False

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """
        Add multiple records, flushing as needed.

        Returns:
            Number of flushes performed.
        """
        flushes = 0
        for rec in records:
            if self.add_record(rec):
                flushes += 1
        return flushes

    def flush(self) -> int:
        """
        Flush buffered records to S3 as Parquet files.

        Groups records by partition and writes one Parquet file per partition.

        Returns:
            Number of files written.
        """
        records = self._buffer.drain()
        if not records:
            return 0

        # Group records by partition
        partitions: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            path = self._partition_path(rec, self._config.prefix, self._event_type)
            if path not in partitions:
                partitions[path] = []
            partitions[path].append(rec)

        files_written = 0
        for partition_path, partition_records in partitions.items():
            try:
                self._write_parquet(partition_path, partition_records)
                files_written += 1
                self._total_written += len(partition_records)
            except Exception as exc:
                logger.error(
                    "parquet_write_failed",
                    partition=partition_path,
                    record_count=len(partition_records),
                    error=str(exc),
                )

        self._total_files += files_written
        logger.info(
            "flush_completed",
            records=len(records),
            partitions=len(partitions),
            files_written=files_written,
        )
        return files_written

    def _write_parquet(self, partition_path: str, records: list[dict[str, Any]]) -> None:
        """Write records to a single Parquet file in S3."""
        df = self._records_to_dataframe(records)
        table = self._dataframe_to_arrow(df)

        # Write Parquet to buffer
        buf = io.BytesIO()
        pq.write_table(
            table,
            buf,
            compression="snappy",
            row_group_size=min(len(records), 100_000),
            write_statistics=True,
        )
        parquet_bytes = buf.getvalue()

        # Generate unique filename
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"part-{ts}-{self._total_files:06d}.parquet"
        s3_key = f"{partition_path}/{filename}"

        self._s3_client.put_object(
            Bucket=self._config.bucket,
            Key=s3_key,
            Body=parquet_bytes,
            ContentType="application/x-parquet",
            Metadata={
                "record_count": str(len(records)),
                "event_type": self._event_type,
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        self._total_bytes += len(parquet_bytes)
        logger.debug(
            "parquet_file_written",
            key=s3_key,
            records=len(records),
            bytes=len(parquet_bytes),
        )

    def _records_to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of record dicts to a pandas DataFrame with schema alignment."""
        df = pd.DataFrame(records)

        # Handle schema evolution: add missing columns with defaults
        for field_def in self._schema:
            col_name = field_def.name
            if col_name not in df.columns:
                if pa.types.is_string(field_def.type):
                    df[col_name] = ""
                elif pa.types.is_integer(field_def.type):
                    df[col_name] = 0
                elif pa.types.is_floating(field_def.type):
                    df[col_name] = 0.0
                elif pa.types.is_boolean(field_def.type):
                    df[col_name] = False
                elif pa.types.is_list(field_def.type):
                    df[col_name] = [[] for _ in range(len(df))]
                elif pa.types.is_timestamp(field_def.type):
                    df[col_name] = pd.NaT
                else:
                    df[col_name] = None

        # Select only schema columns (drop extra columns gracefully)
        schema_cols = [f.name for f in self._schema]
        available = [c for c in schema_cols if c in df.columns]
        return df[available]

    def _dataframe_to_arrow(self, df: pd.DataFrame) -> pa.Table:
        """Convert DataFrame to Arrow table, handling type coercion."""
        try:
            table = pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fallback: let Arrow infer the schema from the DataFrame
            logger.warning("schema_coercion_fallback", event_type=self._event_type)
            table = pa.Table.from_pandas(df, preserve_index=False)
        return table

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact_partition(self, partition_path: str) -> dict[str, Any]:
        """
        Compact small Parquet files in a partition into fewer, larger files.

        Lists all Parquet files in the given S3 partition prefix, reads them,
        merges them into target-sized files, writes the merged files, and
        deletes the originals.

        Args:
            partition_path: S3 prefix for the partition to compact.

        Returns:
            Summary of the compaction operation.
        """
        # List existing files
        files = self._list_partition_files(partition_path)
        if len(files) < self._config.compaction_threshold_files:
            return {
                "status": "skipped",
                "reason": "below_threshold",
                "file_count": len(files),
                "threshold": self._config.compaction_threshold_files,
            }

        logger.info(
            "compaction_started",
            partition=partition_path,
            file_count=len(files),
        )

        # Read all files
        all_tables: list[pa.Table] = []
        total_original_bytes = 0
        for file_key in files:
            try:
                obj = self._s3_client.get_object(Bucket=self._config.bucket, Key=file_key)
                data = obj["Body"].read()
                total_original_bytes += len(data)
                table = pq.read_table(io.BytesIO(data))
                all_tables.append(table)
            except Exception as exc:
                logger.warning("compaction_read_error", key=file_key, error=str(exc))

        if not all_tables:
            return {"status": "error", "reason": "no_readable_files"}

        # Merge all tables
        merged = pa.concat_tables(all_tables, promote_options="default")
        total_rows = merged.num_rows

        # Write compacted files (target size)
        target_bytes = self._config.target_file_size_mb * 1024 * 1024
        rows_per_file = max(
            1,
            int(total_rows * target_bytes / max(total_original_bytes, 1)),
        )

        new_files: list[str] = []
        for start in range(0, total_rows, rows_per_file):
            end = min(start + rows_per_file, total_rows)
            chunk = merged.slice(start, end - start)

            buf = io.BytesIO()
            pq.write_table(chunk, buf, compression="snappy", write_statistics=True)
            parquet_bytes = buf.getvalue()

            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            filename = f"compacted-{ts}-{len(new_files):04d}.parquet"
            new_key = f"{partition_path}/{filename}"

            self._s3_client.put_object(
                Bucket=self._config.bucket,
                Key=new_key,
                Body=parquet_bytes,
                ContentType="application/x-parquet",
                Metadata={
                    "compacted": "true",
                    "record_count": str(end - start),
                    "original_files": str(len(files)),
                },
            )
            new_files.append(new_key)

        # Delete original files
        deleted = 0
        for file_key in files:
            try:
                self._s3_client.delete_object(Bucket=self._config.bucket, Key=file_key)
                deleted += 1
            except ClientError as exc:
                logger.warning("compaction_delete_error", key=file_key, error=str(exc))

        result = {
            "status": "completed",
            "original_files": len(files),
            "original_bytes": total_original_bytes,
            "compacted_files": len(new_files),
            "total_rows": total_rows,
            "deleted": deleted,
        }
        logger.info("compaction_completed", **result)
        return result

    def _list_partition_files(self, partition_path: str) -> list[str]:
        """List all Parquet files in an S3 partition prefix."""
        files: list[str] = []
        paginator = self._s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.bucket, Prefix=partition_path):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    files.append(obj["Key"])
        return files

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_records_written": self._total_written,
            "total_files": self._total_files,
            "total_bytes": self._total_bytes,
            "buffer_size": self._buffer.size,
            "flush_count": self._buffer.flush_count,
        }
