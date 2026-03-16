"""
Pseudo Record Keeping Service

Provides deterministic pseudonymization of regulatory record identifiers
with cryptographic hashing, bidirectional mapping, expiry management,
and controlled re-identification with full audit trails.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, select, update
from sqlalchemy.orm import Session

from reg_record.models.database import PseudoRecord, ReidentificationLog

logger = logging.getLogger(__name__)

# Prefix mapping for pseudo types
_PSEUDO_PREFIXES: dict[str, str] = {
    "SUBMISSION_ID": "PSB",
    "DOCUMENT_ID": "PDC",
    "PATIENT_ID": "PPT",
    "INVESTIGATOR_ID": "PIV",
    "SITE_ID": "PST",
    "BATCH_NUMBER": "PBN",
    "COMPOUND_CODE": "PCC",
    "PROTOCOL_NUMBER": "PPN",
}

VALID_PSEUDO_TYPES = set(_PSEUDO_PREFIXES.keys())
VALID_HASH_ALGORITHMS = {"SHA-256", "SHA-512", "HMAC-SHA256", "BLAKE2B"}


class PseudoRecordError(Exception):
    """Base exception for pseudo record operations."""


class MappingExistsError(PseudoRecordError):
    """Raised when a mapping already exists."""


class MappingExpiredError(PseudoRecordError):
    """Raised when a mapping has expired."""


class InvalidMappingKeyError(PseudoRecordError):
    """Raised when the mapping key is invalid."""


class UnauthorizedReidentificationError(PseudoRecordError):
    """Raised when re-identification authorization fails."""


class PseudoRecordService:
    """Service for managing pseudo record mappings with cryptographic security."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Core hash computation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_salt() -> str:
        """Generate a cryptographically secure random salt (32 bytes, hex encoded)."""
        return secrets.token_hex(32)

    @staticmethod
    def _compute_hash(
        input_data: str,
        salt: str,
        mapping_key: str,
        algorithm: str = "SHA-256",
    ) -> str:
        """
        Compute a deterministic hash for pseudo ID generation.

        The hash is computed over the concatenation of input_data, salt, and
        mapping_key, ensuring that:
          - Same input + same key + same salt = same output (deterministic)
          - Different salt = different output (per-record uniqueness)
          - Knowledge of the mapping_key is required for re-identification
        """
        combined = f"{input_data}|{salt}|{mapping_key}".encode()

        if algorithm == "SHA-256":
            digest = hashlib.sha256(combined).hexdigest()
        elif algorithm == "SHA-512":
            digest = hashlib.sha512(combined).hexdigest()
        elif algorithm == "HMAC-SHA256":
            digest = hmac.new(
                mapping_key.encode("utf-8"),
                f"{input_data}|{salt}".encode(),
                hashlib.sha256,
            ).hexdigest()
        elif algorithm == "BLAKE2B":
            digest = hashlib.blake2b(
                combined, digest_size=32, key=mapping_key.encode("utf-8")[:64]
            ).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        return digest

    @staticmethod
    def _format_pseudo_value(hash_hex: str, pseudo_type: str) -> str:
        """Format hash into a readable pseudo value with type prefix."""
        prefix = _PSEUDO_PREFIXES.get(pseudo_type, "PXX")
        # Use first 32 hex chars (128 bits) -- sufficient for uniqueness
        return f"{prefix}-{hash_hex[:32].upper()}"

    # ------------------------------------------------------------------
    # Generate pseudo ID
    # ------------------------------------------------------------------

    def generate_pseudo_id(
        self,
        original_record_id: int,
        source_table: str,
        pseudo_type: str,
        mapping_key: str,
        created_by: str,
        purpose: str | None = None,
        expiry_days: int = 365,
        algorithm: str = "SHA-256",
    ) -> str:
        """
        Generate a deterministic pseudo ID for a given record.

        If a mapping already exists for this combination, returns the existing
        pseudo value. Otherwise, creates a new mapping with a fresh salt.

        Args:
            original_record_id: The real record ID to pseudonymize.
            source_table: The table containing the original record.
            pseudo_type: Category of the pseudo ID (e.g., SUBMISSION_ID).
            mapping_key: Cryptographic key for deterministic hashing.
            created_by: User creating the mapping.
            purpose: Optional description of why the mapping was created.
            expiry_days: Number of days until the mapping expires.
            algorithm: Hash algorithm to use.

        Returns:
            The pseudo value string (e.g., "PSB-A1B2C3D4E5F6...").

        Raises:
            ValueError: If pseudo_type or algorithm is invalid.
            InvalidMappingKeyError: If mapping_key is too short.
        """
        if pseudo_type not in VALID_PSEUDO_TYPES:
            raise ValueError(f"Invalid pseudo type: {pseudo_type}")
        if algorithm not in VALID_HASH_ALGORITHMS:
            raise ValueError(f"Invalid hash algorithm: {algorithm}")
        if len(mapping_key) < 16:
            raise InvalidMappingKeyError(
                "Mapping key must be at least 16 characters for adequate security"
            )

        # Check for existing active mapping
        existing = self._db.execute(
            select(PseudoRecord).where(
                and_(
                    PseudoRecord.original_record_id == original_record_id,
                    PseudoRecord.source_table == source_table,
                    PseudoRecord.pseudo_type == pseudo_type,
                    PseudoRecord.mapping_key == mapping_key,
                    PseudoRecord.is_active == "Y",
                )
            )
        ).scalar_one_or_none()

        if existing is not None:
            logger.info(
                "Returning existing pseudo mapping for record %d in %s",
                original_record_id,
                source_table,
            )
            return existing.pseudo_value

        # Generate new mapping
        salt = self._generate_salt()
        input_data = f"{original_record_id}|{source_table}"
        hash_hex = self._compute_hash(input_data, salt, mapping_key, algorithm)
        pseudo_value = self._format_pseudo_value(hash_hex, pseudo_type)

        expiry_date = datetime.now(UTC) + timedelta(days=expiry_days)

        record = PseudoRecord(
            original_record_id=original_record_id,
            source_table=source_table,
            pseudo_type=pseudo_type,
            pseudo_value=pseudo_value,
            mapping_key=mapping_key,
            hash_algorithm=algorithm,
            salt=salt,
            is_active="Y",
            expiry_date=expiry_date,
            created_by=created_by,
            purpose=purpose,
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        logger.info(
            "Generated pseudo ID %s for record %d in %s (algo=%s)",
            pseudo_value,
            original_record_id,
            source_table,
            algorithm,
        )
        return pseudo_value

    # ------------------------------------------------------------------
    # Re-identification (pseudo -> real)
    # ------------------------------------------------------------------

    def map_pseudo_to_real(
        self,
        pseudo_value: str,
        requested_by: str,
        authorized_by: str,
        reason: str,
        ip_address: str | None = None,
    ) -> tuple[int, str]:
        """
        Map a pseudo value back to the original record ID.

        This requires two different users (requester and authorizer) to
        enforce separation of duties. All access is logged.

        Args:
            pseudo_value: The pseudo value to look up.
            requested_by: User requesting re-identification.
            authorized_by: User authorizing the request.
            reason: Justification for the re-identification.
            ip_address: Optional IP address of the requester.

        Returns:
            Tuple of (original_record_id, source_table).

        Raises:
            UnauthorizedReidentificationError: If the same user requests
                and authorizes.
            InvalidMappingKeyError: If the pseudo value is not found.
            MappingExpiredError: If the mapping has expired.
        """
        # Enforce separation of duties
        if requested_by == authorized_by:
            self._log_reid_attempt(
                pseudo_value,
                requested_by,
                authorized_by,
                reason,
                "DENIED",
                ip_address,
            )
            raise UnauthorizedReidentificationError(
                "Re-identification requires authorization from a different user than the requester"
            )

        # Look up the pseudo record
        record = self._db.execute(
            select(PseudoRecord).where(PseudoRecord.pseudo_value == pseudo_value)
        ).scalar_one_or_none()

        if record is None:
            raise InvalidMappingKeyError(f"Pseudo value not found: {pseudo_value}")

        # Check active status
        if record.is_active == "N":
            self._log_reid_attempt(
                pseudo_value,
                requested_by,
                authorized_by,
                reason,
                "DENIED",
                ip_address,
                record.id,
            )
            raise MappingExpiredError("Pseudo mapping has been deactivated")

        # Check expiry
        now = datetime.now(UTC)
        if record.expiry_date is not None:
            expiry = record.expiry_date
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry < now:
                record.is_active = "N"
                self._db.commit()
                self._log_reid_attempt(
                    pseudo_value,
                    requested_by,
                    authorized_by,
                    reason,
                    "DENIED",
                    ip_address,
                    record.id,
                )
                raise MappingExpiredError(
                    f"Pseudo mapping expired at {record.expiry_date.isoformat()}"
                )

        # Log successful re-identification
        reid_log = ReidentificationLog(
            pseudo_record_id=record.id,
            requested_by=requested_by,
            authorized_by=authorized_by,
            request_reason=reason,
            request_date=now,
            approval_date=now,
            status="APPROVED",
            ip_address=ip_address,
        )
        self._db.add(reid_log)

        # Update access tracking
        record.last_accessed = now
        record.access_count = (record.access_count or 0) + 1

        self._db.commit()

        logger.info(
            "Re-identification: %s -> record %d in %s (by %s, authorized by %s)",
            pseudo_value,
            record.original_record_id,
            record.source_table,
            requested_by,
            authorized_by,
        )

        return record.original_record_id, record.source_table

    def _log_reid_attempt(
        self,
        pseudo_value: str,
        requested_by: str,
        authorized_by: str,
        reason: str,
        status: str,
        ip_address: str | None = None,
        pseudo_record_id: int | None = None,
    ) -> None:
        """Log a re-identification attempt (success or failure)."""
        if pseudo_record_id is None:
            record = self._db.execute(
                select(PseudoRecord.id).where(PseudoRecord.pseudo_value == pseudo_value)
            ).scalar_one_or_none()
            pseudo_record_id = record if record else None

        if pseudo_record_id is not None:
            log_entry = ReidentificationLog(
                pseudo_record_id=pseudo_record_id,
                requested_by=requested_by,
                authorized_by=authorized_by,
                request_reason=f"{reason} [STATUS: {status}]",
                status=status,
                ip_address=ip_address,
            )
            self._db.add(log_entry)
            self._db.commit()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_pseudo_mapping(self, pseudo_value: str) -> bool:
        """
        Validate that a pseudo mapping is currently active and not expired.

        Returns True if the mapping exists, is active, and has not expired.
        """
        record = self._db.execute(
            select(PseudoRecord).where(PseudoRecord.pseudo_value == pseudo_value)
        ).scalar_one_or_none()

        if record is None:
            return False
        if record.is_active == "N":
            return False

        now = datetime.now(UTC)
        if record.expiry_date is not None:
            expiry = record.expiry_date
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry < now:
                return False

        return True

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_generate_pseudo_ids(
        self,
        record_ids: list[int],
        source_table: str,
        pseudo_type: str,
        mapping_key: str,
        created_by: str,
        purpose: str | None = None,
        expiry_days: int = 365,
    ) -> list[str]:
        """
        Generate pseudo IDs for multiple records in batch.

        Returns list of pseudo values in same order as input record_ids.
        """
        results: list[str] = []
        for rid in record_ids:
            pv = self.generate_pseudo_id(
                original_record_id=rid,
                source_table=source_table,
                pseudo_type=pseudo_type,
                mapping_key=mapping_key,
                created_by=created_by,
                purpose=purpose,
                expiry_days=expiry_days,
            )
            results.append(pv)

        logger.info(
            "Batch generated %d pseudo IDs for %s (%s)",
            len(results),
            source_table,
            pseudo_type,
        )
        return results

    # ------------------------------------------------------------------
    # Expiry and rotation
    # ------------------------------------------------------------------

    def purge_expired_mappings(
        self,
        purge_before: datetime | None = None,
        retention_days: int = 90,
    ) -> int:
        """
        Deactivate expired mappings and purge old inactive ones.

        First marks all expired-but-active mappings as inactive.
        Then deletes inactive mappings older than retention_days, unless
        they have re-identification log entries.

        Returns the number of records purged (deleted).
        """
        now = purge_before or datetime.now(UTC)

        # Deactivate expired
        self._db.execute(
            update(PseudoRecord)
            .where(
                and_(
                    PseudoRecord.expiry_date < now,
                    PseudoRecord.is_active == "Y",
                )
            )
            .values(is_active="N")
        )
        self._db.commit()

        # Find purgeable records (inactive, old, no reid logs)
        cutoff = now - timedelta(days=retention_days)
        subq = select(ReidentificationLog.pseudo_record_id).distinct()

        purgeable = (
            self._db.execute(
                select(PseudoRecord.id).where(
                    and_(
                        PseudoRecord.is_active == "N",
                        PseudoRecord.expiry_date < cutoff,
                        PseudoRecord.id.notin_(subq),
                    )
                )
            )
            .scalars()
            .all()
        )

        purged_count = 0
        for pid in purgeable:
            self._db.query(PseudoRecord).filter(PseudoRecord.id == pid).delete()
            purged_count += 1

        self._db.commit()
        logger.info("Purged %d expired pseudo mappings", purged_count)
        return purged_count

    def rotate_mapping_key(
        self,
        old_key: str,
        new_key: str,
        source_table: str,
        rotated_by: str,
    ) -> int:
        """
        Rotate mapping keys by deactivating old mappings and creating new ones.

        All active mappings with old_key in the specified source_table are
        deactivated, and new mappings with new_key are generated.

        Returns the number of mappings rotated.
        """
        if len(new_key) < 16:
            raise InvalidMappingKeyError("New mapping key must be at least 16 characters")

        old_mappings = (
            self._db.execute(
                select(PseudoRecord).where(
                    and_(
                        PseudoRecord.mapping_key == old_key,
                        PseudoRecord.source_table == source_table,
                        PseudoRecord.is_active == "Y",
                    )
                )
            )
            .scalars()
            .all()
        )

        count = 0
        for mapping in old_mappings:
            # Deactivate old mapping
            mapping.is_active = "N"
            mapping.expiry_date = datetime.now(UTC)

            # Generate new mapping with new key
            self.generate_pseudo_id(
                original_record_id=mapping.original_record_id,
                source_table=source_table,
                pseudo_type=mapping.pseudo_type,
                mapping_key=new_key,
                created_by=rotated_by,
                purpose=f"Key rotation from {old_key[:8]}...",
            )
            count += 1

        self._db.commit()
        logger.info(
            "Rotated %d mappings in %s from key %s... to %s...",
            count,
            source_table,
            old_key[:8],
            new_key[:8],
        )
        return count

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_pseudo_record(self, pseudo_value: str) -> PseudoRecord | None:
        """Look up a pseudo record by its pseudo value (no re-identification)."""
        return self._db.execute(
            select(PseudoRecord).where(PseudoRecord.pseudo_value == pseudo_value)
        ).scalar_one_or_none()

    def get_mappings_for_record(
        self,
        original_record_id: int,
        source_table: str,
        active_only: bool = True,
    ) -> list[PseudoRecord]:
        """Get all pseudo mappings for a given real record."""
        query = select(PseudoRecord).where(
            and_(
                PseudoRecord.original_record_id == original_record_id,
                PseudoRecord.source_table == source_table,
            )
        )
        if active_only:
            query = query.where(PseudoRecord.is_active == "Y")

        return list(self._db.execute(query).scalars().all())

    def get_coverage_stats(self) -> list[dict[str, object]]:
        """Get pseudo record coverage statistics by source table and type."""
        results = self._db.execute(
            select(
                PseudoRecord.source_table,
                PseudoRecord.pseudo_type,
                func.count().label("total"),
                func.sum(func.case((PseudoRecord.is_active == "Y", 1), else_=0)).label("active"),
                func.sum(func.case((PseudoRecord.is_active == "N", 1), else_=0)).label("inactive"),
                func.avg(PseudoRecord.access_count).label("avg_access_count"),
            ).group_by(PseudoRecord.source_table, PseudoRecord.pseudo_type)
        ).all()

        return [
            {
                "source_table": row.source_table,
                "pseudo_type": row.pseudo_type,
                "total": row.total,
                "active": row.active,
                "inactive": row.inactive,
                "avg_access_count": float(row.avg_access_count or 0),
            }
            for row in results
        ]
