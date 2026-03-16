"""
File Processing Module for PharmaDataVault.

Handles the ingestion of daily data feeds:
  - Parse CSV and fixed-width files from source systems
  - Validate file format, checksums, and row counts
  - Archive processed files with timestamps
  - Generate processing reports for audit trail

Supported file types:
  - drug_feed_YYYYMMDD.csv       (Drug master data)
  - patient_feed_YYYYMMDD.csv    (Patient demographics)
  - trial_feed_YYYYMMDD.csv      (Clinical trial data)
  - ae_feed_YYYYMMDD.csv         (Adverse event reports)
  - mfg_feed_YYYYMMDD.dat        (Manufacturing data, fixed-width)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pharma_vault._config import VaultConfig

logger = logging.getLogger(__name__)

# Expected columns per feed type
FEED_SCHEMAS: dict[str, list[str]] = {
    "drug_feed": [
        "ndc_code",
        "drug_name",
        "generic_name",
        "manufacturer",
        "dosage_form",
        "strength",
        "route",
        "dea_schedule",
        "approval_date",
        "therapeutic_class",
        "nda_number",
    ],
    "patient_feed": [
        "mrn",
        "date_of_birth",
        "age",
        "sex",
        "ethnicity",
        "race",
        "weight_kg",
        "height_cm",
        "country",
        "state",
        "smoking_status",
    ],
    "trial_feed": [
        "nct_id",
        "title",
        "phase",
        "status",
        "start_date",
        "est_end_date",
        "actual_end_date",
        "sponsor",
        "investigator",
        "protocol_number",
        "protocol_version",
        "target_enrollment",
        "actual_enrollment",
        "therapeutic_area",
        "primary_endpoint",
        "ind_number",
    ],
    "ae_feed": [
        "ae_report_id",
        "drug_ndc",
        "patient_mrn",
        "ae_term",
        "description",
        "severity",
        "seriousness",
        "causality",
        "outcome",
        "onset_date",
        "resolution_date",
        "report_date",
        "reporter_type",
        "meddra_code",
        "meddra_soc",
        "expectedness",
        "action_taken",
    ],
    "mfg_feed": [
        "drug_ndc",
        "lot_number",
        "batch_size",
        "batch_unit",
        "mfg_date",
        "expiry_date",
        "facility_id",
        "mfg_line",
        "qc_status",
        "qc_date",
        "qc_analyst",
        "yield_pct",
        "deviation_flag",
        "deviation_id",
        "release_date",
        "storage_conditions",
    ],
}

# Fixed-width field specifications for .dat files
MFG_FIXED_WIDTH_SPEC: list[tuple[str, int, int]] = [
    ("drug_ndc", 0, 13),
    ("lot_number", 13, 43),
    ("batch_size", 43, 55),
    ("batch_unit", 55, 75),
    ("mfg_date", 75, 85),
    ("expiry_date", 85, 95),
    ("facility_id", 95, 115),
    ("mfg_line", 115, 145),
    ("qc_status", 145, 165),
    ("qc_date", 165, 175),
    ("qc_analyst", 175, 275),
    ("yield_pct", 275, 281),
    ("deviation_flag", 281, 282),
    ("deviation_id", 282, 312),
    ("release_date", 312, 322),
    ("storage_conditions", 322, 422),
]


@dataclass
class FileManifest:
    """Metadata about an incoming data file."""

    file_path: str
    feed_type: str
    file_date: str  # YYYYMMDD from filename
    file_size_bytes: int
    expected_checksum: str | None = None
    actual_checksum: str | None = None
    row_count: int = 0
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class ProcessingReport:
    """Report generated after processing a file."""

    file_path: str
    feed_type: str
    start_time: datetime
    end_time: datetime | None = None
    rows_read: int = 0
    rows_processed: int = 0
    rows_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    checksum_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "file_path": self.file_path,
            "feed_type": self.feed_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "rows_read": self.rows_read,
            "rows_processed": self.rows_processed,
            "rows_rejected": self.rows_rejected,
            "rejection_reasons": self.rejection_reasons,
        }


class FileProcessor:
    """
    Processes incoming data files for the pharma data vault ETL pipeline.

    Handles file discovery, validation, parsing, and archival.
    """

    # Regex to extract feed type and date from filename
    FILENAME_PATTERN = re.compile(
        r"^(drug_feed|patient_feed|trial_feed|ae_feed|mfg_feed)"
        r"_(\d{8})\.(csv|dat)$"
    )

    def __init__(self, config: VaultConfig) -> None:
        self._config = config

    def discover_files(self, pattern: str = "*.csv") -> list[FileManifest]:
        """
        Discover incoming data files in the staging directory.

        Scans the staging directory for files matching the expected
        naming convention and creates manifests for each.

        Args:
            pattern: Glob pattern for file discovery.

        Returns:
            List of FileManifest objects for discovered files.
        """
        staging_dir = self._config.staging_dir
        if not staging_dir.exists():
            logger.warning("Staging directory does not exist: %s", staging_dir)
            return []

        manifests: list[FileManifest] = []

        # Scan for both CSV and DAT files
        for ext_pattern in ["*.csv", "*.dat"]:
            for filepath in sorted(staging_dir.glob(ext_pattern)):
                match = self.FILENAME_PATTERN.match(filepath.name)
                if not match:
                    logger.warning("Skipping file with unexpected name: %s", filepath.name)
                    continue

                feed_type = match.group(1)
                file_date = match.group(2)

                # Read checksum file if it exists (companion .md5 file)
                checksum_file = filepath.with_suffix(filepath.suffix + ".md5")
                expected_checksum = None
                if checksum_file.exists():
                    expected_checksum = checksum_file.read_text().strip().split()[0]

                manifest = FileManifest(
                    file_path=str(filepath),
                    feed_type=feed_type,
                    file_date=file_date,
                    file_size_bytes=filepath.stat().st_size,
                    expected_checksum=expected_checksum,
                )
                manifests.append(manifest)
                logger.info(
                    "Discovered file: %s (type=%s, date=%s, size=%d bytes)",
                    filepath.name,
                    feed_type,
                    file_date,
                    manifest.file_size_bytes,
                )

        return manifests

    def _compute_checksum(self, file_path: str) -> str:
        """Compute MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_file(self, manifest: FileManifest) -> bool:
        """
        Validate an incoming data file.

        Checks:
          - File exists and is non-empty
          - Checksum matches if expected checksum provided
          - Header row matches expected schema
          - File date is reasonable (not future, not too old)

        Args:
            manifest: The file manifest to validate.

        Returns:
            True if file is valid, False otherwise.
        """
        filepath = Path(manifest.file_path)
        errors: list[str] = []

        # Check file exists
        if not filepath.exists():
            errors.append(f"File not found: {manifest.file_path}")
            manifest.errors = errors
            manifest.is_valid = False
            return False

        # Check non-empty
        if manifest.file_size_bytes == 0:
            errors.append("File is empty")

        # Checksum validation
        manifest.actual_checksum = self._compute_checksum(manifest.file_path)
        if manifest.expected_checksum:
            if manifest.actual_checksum != manifest.expected_checksum:
                errors.append(
                    f"Checksum mismatch: expected={manifest.expected_checksum}, "
                    f"actual={manifest.actual_checksum}"
                )

        # Date validation
        try:
            file_date = datetime.strptime(manifest.file_date, "%Y%m%d")
            now = datetime.now()
            if file_date > now:
                errors.append(f"File date is in the future: {manifest.file_date}")
            days_old = (now - file_date).days
            if days_old > 30:
                errors.append(f"File is {days_old} days old (>30 days)")
        except ValueError:
            errors.append(f"Invalid date in filename: {manifest.file_date}")

        # Schema validation (CSV files only)
        if filepath.suffix == ".csv":
            try:
                with open(manifest.file_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)

                if header is None:
                    errors.append("CSV file has no header row")
                else:
                    expected_cols = FEED_SCHEMAS.get(manifest.feed_type, [])
                    header_clean = [col.strip().lower() for col in header]
                    if expected_cols and header_clean != expected_cols:
                        missing = set(expected_cols) - set(header_clean)
                        extra = set(header_clean) - set(expected_cols)
                        if missing:
                            errors.append(f"Missing columns: {missing}")
                        if extra:
                            logger.warning("Extra columns in %s: %s", filepath.name, extra)

                # Count rows
                with open(manifest.file_path, encoding="utf-8") as f:
                    manifest.row_count = sum(1 for _ in f) - 1  # Exclude header

            except (csv.Error, UnicodeDecodeError) as exc:
                errors.append(f"CSV parse error: {exc}")

        manifest.errors = errors
        manifest.is_valid = len(errors) == 0

        if not manifest.is_valid:
            logger.error("Validation failed for %s: %s", filepath.name, "; ".join(errors))
        else:
            logger.info("Validation passed for %s (%d rows)", filepath.name, manifest.row_count)

        return manifest.is_valid

    def process_file(self, manifest: FileManifest) -> dict[str, Any]:
        """
        Process a validated file: parse contents and prepare for staging.

        For CSV files, uses pandas for efficient parsing.
        For fixed-width files, uses positional parsing.

        Args:
            manifest: Validated file manifest.

        Returns:
            Processing report as a dictionary.
        """
        report = ProcessingReport(
            file_path=manifest.file_path,
            feed_type=manifest.feed_type,
            start_time=datetime.now(UTC),
        )

        filepath = Path(manifest.file_path)
        logger.info("Processing file: %s (type=%s)", filepath.name, manifest.feed_type)

        try:
            if filepath.suffix == ".csv":
                df = self._parse_csv(manifest)
            elif filepath.suffix == ".dat":
                df = self._parse_fixed_width(manifest)
            else:
                raise ValueError(f"Unsupported file extension: {filepath.suffix}")

            report.rows_read = len(df)

            # Apply basic cleansing
            df_clean, rejections = self._cleanse_dataframe(df, manifest.feed_type)
            report.rows_processed = len(df_clean)
            report.rows_rejected = report.rows_read - report.rows_processed
            report.rejection_reasons = rejections

            report.end_time = datetime.now(UTC)

            logger.info(
                "Processed %s: %d read, %d processed, %d rejected",
                filepath.name,
                report.rows_read,
                report.rows_processed,
                report.rows_rejected,
            )

        except Exception as exc:
            report.end_time = datetime.now(UTC)
            report.rejection_reasons["parse_error"] = 1
            logger.exception("Error processing %s: %s", filepath.name, exc)

        return report.to_dict()

    def _parse_csv(self, manifest: FileManifest) -> pd.DataFrame:
        """Parse a CSV file into a DataFrame."""
        expected_cols = FEED_SCHEMAS.get(manifest.feed_type, [])

        df = pd.read_csv(
            manifest.file_path,
            dtype=str,  # Read all as strings for staging
            na_values=["", "NULL", "N/A", "n/a"],
            keep_default_na=True,
            encoding="utf-8",
        )

        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]

        return df

    def _parse_fixed_width(self, manifest: FileManifest) -> pd.DataFrame:
        """Parse a fixed-width file into a DataFrame."""
        if manifest.feed_type != "mfg_feed":
            raise ValueError(f"Fixed-width parsing not supported for {manifest.feed_type}")

        records: list[dict[str, str]] = []

        with open(manifest.file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num == 1 and line.startswith("HDR"):
                    continue  # Skip header record
                if line.startswith("TRL"):
                    continue  # Skip trailer record

                record: dict[str, str] = {}
                for col_name, start, end in MFG_FIXED_WIDTH_SPEC:
                    value = line[start:end].strip() if len(line) >= end else ""
                    record[col_name] = value if value else None  # type: ignore[assignment]

                records.append(record)

        return pd.DataFrame(records)

    def _cleanse_dataframe(
        self,
        df: pd.DataFrame,
        feed_type: str,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Apply basic data cleansing to a DataFrame.

        - Strip whitespace from all string columns
        - Remove completely empty rows
        - Flag rows with critical missing fields
        - Standardize common values

        Returns:
            Tuple of (clean DataFrame, rejection reasons dict)
        """
        rejections: dict[str, int] = {}

        # Strip whitespace
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Remove empty rows
        initial_count = len(df)
        df = df.dropna(how="all")
        empty_rows = initial_count - len(df)
        if empty_rows > 0:
            rejections["empty_rows"] = empty_rows

        # Feed-specific required field checks
        required_fields: dict[str, list[str]] = {
            "drug_feed": ["ndc_code", "drug_name", "manufacturer"],
            "patient_feed": ["mrn", "sex"],
            "trial_feed": ["nct_id", "phase", "status", "sponsor"],
            "ae_feed": ["ae_report_id", "drug_ndc", "severity", "onset_date"],
            "mfg_feed": ["drug_ndc", "lot_number", "mfg_date", "facility_id"],
        }

        req_cols = required_fields.get(feed_type, [])
        for col in req_cols:
            if col in df.columns:
                null_mask = df[col].isna() | (df[col] == "")
                null_count = int(null_mask.sum())
                if null_count > 0:
                    rejections[f"missing_{col}"] = null_count
                    df = df[~null_mask]

        return df, rejections

    def archive_file(self, manifest: FileManifest) -> str:
        """
        Move a processed file to the archive directory.

        Files are archived with a timestamp suffix to prevent collisions.

        Args:
            manifest: The file manifest to archive.

        Returns:
            Path to the archived file.
        """
        filepath = Path(manifest.file_path)
        archive_dir = self._config.archive_dir / manifest.file_date[:6]  # YYYYMM subdirectory

        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%H%M%S")
        archive_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        archive_path = archive_dir / archive_name

        shutil.move(str(filepath), str(archive_path))
        logger.info("Archived %s -> %s", filepath.name, archive_path)

        # Also archive the checksum file if it exists
        checksum_file = filepath.with_suffix(filepath.suffix + ".md5")
        if checksum_file.exists():
            shutil.move(
                str(checksum_file),
                str(archive_dir / f"{archive_name}.md5"),
            )

        return str(archive_path)

    def generate_report(self, manifests: list[FileManifest]) -> dict[str, Any]:
        """Generate a summary report for a batch of processed files."""
        return {
            "report_time": datetime.now(UTC).isoformat(),
            "total_files": len(manifests),
            "valid_files": sum(1 for m in manifests if m.is_valid),
            "invalid_files": sum(1 for m in manifests if not m.is_valid),
            "total_rows": sum(m.row_count for m in manifests),
            "files": [
                {
                    "path": m.file_path,
                    "type": m.feed_type,
                    "date": m.file_date,
                    "rows": m.row_count,
                    "valid": m.is_valid,
                    "errors": m.errors,
                }
                for m in manifests
            ],
        }
