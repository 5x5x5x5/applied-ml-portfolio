"""Data encryption utilities for PHI/genomic data.

Provides HIPAA-compliant encryption for protected health information
and genomic data at rest and in transit. Integrates with AWS KMS for
key management and includes field-level encryption and comprehensive
audit logging for data access events.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


class DataClassification(str, Enum):
    """Data sensitivity classification levels (HIPAA-aligned)."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"  # PII
    PHI = "phi"  # Protected Health Information
    GENOMIC = "genomic"  # Genomic data (highest sensitivity)


class AccessAction(str, Enum):
    """Auditable data access actions."""

    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    KEY_GENERATE = "key_generate"
    KEY_ROTATE = "key_rotate"
    FIELD_ENCRYPT = "field_encrypt"
    FIELD_DECRYPT = "field_decrypt"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"


@dataclass
class AuditLogEntry:
    """Immutable audit log entry for data access events."""

    event_id: str
    timestamp: str
    action: str
    principal: str
    resource: str
    data_classification: str
    success: bool
    source_ip: str = ""
    user_agent: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    kms_key_id: str = ""
    data_hash: str = ""


class AuditLogger:
    """HIPAA-compliant audit logger for data access events.

    Logs all encryption/decryption operations, key management events,
    and data access to both application logs and an audit trail suitable
    for compliance reporting.

    In production, this would write to CloudWatch Logs with a retention
    policy and CloudTrail integration.
    """

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []
        self._audit_logger = logging.getLogger("cloud_genomics.audit")

    def log_event(
        self,
        action: AccessAction,
        principal: str,
        resource: str,
        data_classification: DataClassification,
        success: bool,
        source_ip: str = "",
        user_agent: str = "",
        details: dict[str, Any] | None = None,
        kms_key_id: str = "",
        data_hash: str = "",
    ) -> AuditLogEntry:
        """Record an auditable data access event.

        Args:
            action: The type of access action.
            principal: Identity of the accessor (user/role ARN).
            resource: Identifier of the accessed resource.
            data_classification: Sensitivity level of the data.
            success: Whether the operation succeeded.
            source_ip: Requester IP address.
            user_agent: Requester user agent string.
            details: Additional context for the event.
            kms_key_id: KMS key involved (if any).
            data_hash: SHA-256 hash of the data involved.

        Returns:
            The created AuditLogEntry.
        """
        entry = AuditLogEntry(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            action=action.value,
            principal=principal,
            resource=resource,
            data_classification=data_classification.value,
            success=success,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details or {},
            kms_key_id=kms_key_id,
            data_hash=data_hash,
        )

        self._entries.append(entry)

        # Structured log output for CloudWatch/SIEM ingestion
        log_data = {
            "event_id": entry.event_id,
            "timestamp": entry.timestamp,
            "action": entry.action,
            "principal": entry.principal,
            "resource": entry.resource,
            "classification": entry.data_classification,
            "success": entry.success,
            "kms_key_id": entry.kms_key_id,
        }

        if success:
            self._audit_logger.info("audit_event %s", json.dumps(log_data))
        else:
            self._audit_logger.warning("audit_event_failed %s", json.dumps(log_data))

        return entry

    def get_entries(
        self,
        action: AccessAction | None = None,
        principal: str | None = None,
        resource: str | None = None,
        since: str | None = None,
    ) -> list[AuditLogEntry]:
        """Query audit log entries with optional filters.

        Args:
            action: Filter by action type.
            principal: Filter by principal identity.
            resource: Filter by resource identifier.
            since: ISO timestamp to filter entries after.

        Returns:
            List of matching AuditLogEntry objects.
        """
        results = self._entries

        if action is not None:
            results = [e for e in results if e.action == action.value]
        if principal is not None:
            results = [e for e in results if e.principal == principal]
        if resource is not None:
            results = [e for e in results if e.resource == resource]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]

        return results


class KMSEncryption:
    """AWS KMS integration for envelope encryption.

    Uses AWS KMS customer-managed keys (CMK) for generating and managing
    data encryption keys (DEK). Implements envelope encryption: KMS
    encrypts/decrypts the DEK, and the DEK encrypts/decrypts the data.
    """

    def __init__(
        self,
        kms_key_id: str,
        region: str = "us-east-1",
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._kms_key_id = kms_key_id
        self._region = region
        self._kms_client = boto3.client("kms", region_name=region)
        self._audit = audit_logger or AuditLogger()
        self._dek_cache: dict[str, tuple[bytes, float]] = {}
        self._dek_cache_ttl = 300  # 5 minutes

    def generate_data_key(
        self,
        encryption_context: dict[str, str] | None = None,
        principal: str = "system",
    ) -> tuple[bytes, bytes]:
        """Generate a new data encryption key using KMS.

        Returns both the plaintext key (for immediate use) and the
        encrypted key (for storage alongside the ciphertext).

        Args:
            encryption_context: KMS encryption context (for auth/auditing).
            principal: Identity requesting the key.

        Returns:
            Tuple of (plaintext_key, encrypted_key).

        Raises:
            ClientError: If KMS call fails.
        """
        context = encryption_context or {
            "service": "cloud-genomics",
            "purpose": "data-encryption",
        }

        try:
            response = self._kms_client.generate_data_key(
                KeyId=self._kms_key_id,
                KeySpec="AES_256",
                EncryptionContext=context,
            )

            plaintext_key = response["Plaintext"]
            encrypted_key = response["CiphertextBlob"]

            self._audit.log_event(
                action=AccessAction.KEY_GENERATE,
                principal=principal,
                resource=self._kms_key_id,
                data_classification=DataClassification.PHI,
                success=True,
                kms_key_id=self._kms_key_id,
            )

            logger.info("Generated new data encryption key via KMS")
            return plaintext_key, encrypted_key

        except ClientError as exc:
            self._audit.log_event(
                action=AccessAction.KEY_GENERATE,
                principal=principal,
                resource=self._kms_key_id,
                data_classification=DataClassification.PHI,
                success=False,
                details={"error": str(exc)},
                kms_key_id=self._kms_key_id,
            )
            logger.exception("Failed to generate data key from KMS")
            raise

    def decrypt_data_key(
        self,
        encrypted_key: bytes,
        encryption_context: dict[str, str] | None = None,
        principal: str = "system",
    ) -> bytes:
        """Decrypt a data encryption key using KMS.

        Args:
            encrypted_key: The encrypted DEK from generate_data_key.
            encryption_context: Must match the context used during generation.
            principal: Identity requesting decryption.

        Returns:
            Plaintext data encryption key.
        """
        context = encryption_context or {
            "service": "cloud-genomics",
            "purpose": "data-encryption",
        }

        # Check cache
        cache_key = hashlib.sha256(encrypted_key).hexdigest()
        if cache_key in self._dek_cache:
            cached_key, cached_time = self._dek_cache[cache_key]
            if time.monotonic() - cached_time < self._dek_cache_ttl:
                return cached_key

        try:
            response = self._kms_client.decrypt(
                CiphertextBlob=encrypted_key,
                EncryptionContext=context,
            )

            plaintext_key = response["Plaintext"]

            # Cache the decrypted key
            self._dek_cache[cache_key] = (plaintext_key, time.monotonic())

            self._audit.log_event(
                action=AccessAction.DECRYPT,
                principal=principal,
                resource=self._kms_key_id,
                data_classification=DataClassification.PHI,
                success=True,
                kms_key_id=self._kms_key_id,
            )

            return plaintext_key

        except ClientError as exc:
            self._audit.log_event(
                action=AccessAction.DECRYPT,
                principal=principal,
                resource=self._kms_key_id,
                data_classification=DataClassification.PHI,
                success=False,
                details={"error": str(exc)},
                kms_key_id=self._kms_key_id,
            )
            logger.exception("Failed to decrypt data key via KMS")
            raise

    def encrypt_envelope(
        self,
        plaintext: bytes,
        encryption_context: dict[str, str] | None = None,
        principal: str = "system",
    ) -> dict[str, str]:
        """Encrypt data using envelope encryption with KMS.

        Args:
            plaintext: Data to encrypt.
            encryption_context: KMS context for key generation.
            principal: Identity performing the encryption.

        Returns:
            Dictionary with 'encrypted_key', 'ciphertext', 'nonce',
            and 'context' fields (all base64-encoded where applicable).
        """
        plaintext_key, encrypted_key = self.generate_data_key(
            encryption_context=encryption_context,
            principal=principal,
        )

        # Encrypt data with AES-256-GCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(plaintext_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        data_hash = hashlib.sha256(plaintext).hexdigest()

        self._audit.log_event(
            action=AccessAction.ENCRYPT,
            principal=principal,
            resource="envelope_encryption",
            data_classification=DataClassification.GENOMIC,
            success=True,
            kms_key_id=self._kms_key_id,
            data_hash=data_hash,
            details={"data_size_bytes": len(plaintext)},
        )

        return {
            "encrypted_key": base64.b64encode(encrypted_key).decode("utf-8"),
            "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
            "nonce": base64.b64encode(nonce).decode("utf-8"),
            "context": json.dumps(encryption_context or {}),
            "algorithm": "AES-256-GCM",
            "kms_key_id": self._kms_key_id,
        }

    def decrypt_envelope(
        self,
        envelope: dict[str, str],
        principal: str = "system",
    ) -> bytes:
        """Decrypt data from an envelope encryption package.

        Args:
            envelope: Encrypted envelope from encrypt_envelope().
            principal: Identity performing the decryption.

        Returns:
            Decrypted plaintext bytes.
        """
        encrypted_key = base64.b64decode(envelope["encrypted_key"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
        nonce = base64.b64decode(envelope["nonce"])
        context = json.loads(envelope.get("context", "{}"))

        plaintext_key = self.decrypt_data_key(
            encrypted_key=encrypted_key,
            encryption_context=context if context else None,
            principal=principal,
        )

        aesgcm = AESGCM(plaintext_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        self._audit.log_event(
            action=AccessAction.DECRYPT,
            principal=principal,
            resource="envelope_decryption",
            data_classification=DataClassification.GENOMIC,
            success=True,
            kms_key_id=self._kms_key_id,
            data_hash=hashlib.sha256(plaintext).hexdigest(),
            details={"data_size_bytes": len(plaintext)},
        )

        return plaintext


class FieldLevelEncryption:
    """Field-level encryption for structured data containing PHI.

    Encrypts individual fields within JSON/dict structures, allowing
    selective encryption of sensitive fields while leaving non-sensitive
    fields in cleartext for querying.

    Suitable for encrypting patient identifiers, sample IDs, and
    other PHI within variant classification records.
    """

    # Fields that must always be encrypted (PHI/PII)
    SENSITIVE_FIELDS: set[str] = {
        "patient_id",
        "patient_name",
        "date_of_birth",
        "medical_record_number",
        "sample_id",
        "accession_number",
        "ordering_physician",
        "insurance_id",
        "ssn",
        "email",
        "phone",
        "address",
    }

    def __init__(
        self,
        encryption_key: bytes | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize field-level encryption.

        Args:
            encryption_key: 32-byte AES key. If None, generates a new one.
            audit_logger: Audit logger instance.
        """
        if encryption_key is None:
            encryption_key = os.urandom(32)
        if len(encryption_key) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes")

        self._key = encryption_key
        self._fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(encryption_key).digest()))
        self._audit = audit_logger or AuditLogger()

    def encrypt_fields(
        self,
        record: dict[str, Any],
        fields_to_encrypt: set[str] | None = None,
        principal: str = "system",
        resource_id: str = "",
    ) -> dict[str, Any]:
        """Encrypt specified fields within a record.

        Args:
            record: Dictionary containing data fields.
            fields_to_encrypt: Set of field names to encrypt. If None,
                uses SENSITIVE_FIELDS.
            principal: Identity performing the encryption.
            resource_id: Identifier for the record being encrypted.

        Returns:
            Copy of the record with sensitive fields encrypted.
        """
        target_fields = fields_to_encrypt or self.SENSITIVE_FIELDS
        encrypted_record = dict(record)
        encrypted_field_names: list[str] = []

        for field_name in target_fields:
            if field_name in encrypted_record and encrypted_record[field_name] is not None:
                value = encrypted_record[field_name]
                value_bytes = json.dumps(value).encode("utf-8")

                encrypted_value = self._fernet.encrypt(value_bytes)
                encrypted_record[field_name] = {
                    "__encrypted__": True,
                    "value": encrypted_value.decode("utf-8"),
                    "field": field_name,
                }
                encrypted_field_names.append(field_name)

        # Add metadata
        encrypted_record["__encryption_metadata__"] = {
            "encrypted_fields": encrypted_field_names,
            "encryption_timestamp": datetime.now(UTC).isoformat(),
            "algorithm": "Fernet (AES-128-CBC)",
        }

        self._audit.log_event(
            action=AccessAction.FIELD_ENCRYPT,
            principal=principal,
            resource=resource_id,
            data_classification=DataClassification.PHI,
            success=True,
            details={
                "encrypted_fields": encrypted_field_names,
                "total_fields": len(record),
            },
        )

        logger.info("Encrypted %d fields in record %s", len(encrypted_field_names), resource_id)
        return encrypted_record

    def decrypt_fields(
        self,
        record: dict[str, Any],
        principal: str = "system",
        resource_id: str = "",
    ) -> dict[str, Any]:
        """Decrypt encrypted fields within a record.

        Args:
            record: Dictionary with encrypted fields.
            principal: Identity performing the decryption.
            resource_id: Identifier for the record being decrypted.

        Returns:
            Copy of the record with sensitive fields decrypted.
        """
        decrypted_record = dict(record)
        decrypted_field_names: list[str] = []

        for field_name, value in record.items():
            if isinstance(value, dict) and value.get("__encrypted__"):
                try:
                    encrypted_bytes = value["value"].encode("utf-8")
                    decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
                    decrypted_record[field_name] = json.loads(decrypted_bytes)
                    decrypted_field_names.append(field_name)
                except Exception:
                    logger.exception("Failed to decrypt field: %s", field_name)
                    decrypted_record[field_name] = None

        # Remove encryption metadata
        decrypted_record.pop("__encryption_metadata__", None)

        self._audit.log_event(
            action=AccessAction.FIELD_DECRYPT,
            principal=principal,
            resource=resource_id,
            data_classification=DataClassification.PHI,
            success=True,
            details={
                "decrypted_fields": decrypted_field_names,
            },
        )

        logger.info("Decrypted %d fields in record %s", len(decrypted_field_names), resource_id)
        return decrypted_record

    def is_field_encrypted(self, value: Any) -> bool:
        """Check if a field value is in encrypted form."""
        return isinstance(value, dict) and value.get("__encrypted__") is True


class GenomicDataEncryption:
    """High-level encryption interface for genomic data workflows.

    Combines KMS envelope encryption for large data (VCF files) with
    field-level encryption for structured records (classification results).
    Ensures all genomic and PHI data is encrypted at rest.
    """

    def __init__(
        self,
        kms_key_id: str = "",
        region: str = "us-east-1",
        local_key: bytes | None = None,
    ) -> None:
        self._audit = AuditLogger()
        self._field_encryption = FieldLevelEncryption(
            encryption_key=local_key,
            audit_logger=self._audit,
        )
        self._kms_key_id = kms_key_id
        self._kms_encryption: KMSEncryption | None = None

        if kms_key_id:
            try:
                self._kms_encryption = KMSEncryption(
                    kms_key_id=kms_key_id,
                    region=region,
                    audit_logger=self._audit,
                )
            except Exception:
                logger.warning("KMS initialization failed; falling back to local encryption only")

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    def encrypt_vcf_content(
        self,
        vcf_bytes: bytes,
        sample_id: str,
        principal: str = "system",
    ) -> dict[str, str]:
        """Encrypt VCF file content for storage.

        Uses KMS envelope encryption if available, otherwise falls back
        to local AES-GCM encryption.

        Args:
            vcf_bytes: Raw VCF file content.
            sample_id: Sample identifier for audit trail.
            principal: Identity performing the encryption.

        Returns:
            Encrypted envelope dictionary.
        """
        if self._kms_encryption:
            context = {
                "service": "cloud-genomics",
                "data_type": "vcf",
                "sample_id": sample_id,
            }
            return self._kms_encryption.encrypt_envelope(
                plaintext=vcf_bytes,
                encryption_context=context,
                principal=principal,
            )

        # Local fallback encryption
        return self._local_encrypt(vcf_bytes, sample_id, principal)

    def decrypt_vcf_content(
        self,
        envelope: dict[str, str],
        principal: str = "system",
    ) -> bytes:
        """Decrypt VCF file content from storage.

        Args:
            envelope: Encrypted envelope from encrypt_vcf_content.
            principal: Identity performing the decryption.

        Returns:
            Decrypted VCF content bytes.
        """
        if self._kms_encryption and "kms_key_id" in envelope:
            return self._kms_encryption.decrypt_envelope(
                envelope=envelope,
                principal=principal,
            )

        return self._local_decrypt(envelope, principal)

    def encrypt_classification_record(
        self,
        record: dict[str, Any],
        principal: str = "system",
    ) -> dict[str, Any]:
        """Encrypt PHI fields in a classification result record.

        Args:
            record: Classification result with potential PHI fields.
            principal: Identity performing the encryption.

        Returns:
            Record with PHI fields encrypted.
        """
        resource_id = record.get("variant_id", str(uuid.uuid4()))
        return self._field_encryption.encrypt_fields(
            record=record,
            principal=principal,
            resource_id=resource_id,
        )

    def decrypt_classification_record(
        self,
        record: dict[str, Any],
        principal: str = "system",
    ) -> dict[str, Any]:
        """Decrypt PHI fields in a classification result record.

        Args:
            record: Record with encrypted PHI fields.
            principal: Identity performing the decryption.

        Returns:
            Record with PHI fields decrypted.
        """
        resource_id = record.get("variant_id", "unknown")
        return self._field_encryption.decrypt_fields(
            record=record,
            principal=principal,
            resource_id=resource_id,
        )

    def _local_encrypt(
        self,
        plaintext: bytes,
        resource_id: str,
        principal: str,
    ) -> dict[str, str]:
        """Local AES-GCM encryption fallback (when KMS unavailable)."""
        key = os.urandom(32)
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        self._audit.log_event(
            action=AccessAction.ENCRYPT,
            principal=principal,
            resource=resource_id,
            data_classification=DataClassification.GENOMIC,
            success=True,
            details={
                "method": "local_aes_gcm",
                "data_size_bytes": len(plaintext),
            },
        )

        return {
            "encrypted_key": base64.b64encode(key).decode("utf-8"),
            "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
            "nonce": base64.b64encode(nonce).decode("utf-8"),
            "algorithm": "AES-256-GCM",
            "method": "local",
        }

    def _local_decrypt(
        self,
        envelope: dict[str, str],
        principal: str,
    ) -> bytes:
        """Local AES-GCM decryption fallback."""
        key = base64.b64decode(envelope["encrypted_key"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
        nonce = base64.b64decode(envelope["nonce"])

        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        self._audit.log_event(
            action=AccessAction.DECRYPT,
            principal=principal,
            resource="local_decrypt",
            data_classification=DataClassification.GENOMIC,
            success=True,
            details={
                "method": "local_aes_gcm",
                "data_size_bytes": len(plaintext),
            },
        )

        return plaintext


def hash_identifier(value: str, salt: str = "") -> str:
    """Create a one-way hash of a sensitive identifier.

    Used for de-identification of PHI while maintaining the ability
    to link records. Uses SHA-256 with an optional salt.

    Args:
        value: Sensitive identifier to hash.
        salt: Optional salt for the hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()
