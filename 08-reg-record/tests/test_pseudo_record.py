"""Tests for the Pseudo Record Keeping Service."""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy.orm import Session

from reg_record.models.database import Drug, ReidentificationLog
from reg_record.services.pseudo_record_service import (
    InvalidMappingKeyError,
    MappingExpiredError,
    PseudoRecordService,
    UnauthorizedReidentificationError,
)

MAPPING_KEY = "test-mapping-key-1234567890abcdef"


class TestGeneratePseudoId:
    """Tests for pseudo ID generation."""

    def test_generate_pseudo_id_creates_mapping(self, db: Session, sample_drug: Drug) -> None:
        """Generating a pseudo ID should create a new mapping record."""
        svc = PseudoRecordService(db)

        pseudo_value = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
            purpose="Unit test",
            expiry_days=30,
        )

        assert pseudo_value is not None
        assert pseudo_value.startswith("PSB-")
        assert len(pseudo_value) == 36  # "PSB-" + 32 hex chars

    def test_generate_returns_existing_for_same_params(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """Same input parameters should return the same pseudo value."""
        svc = PseudoRecordService(db)

        pv1 = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )
        pv2 = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )

        assert pv1 == pv2

    def test_different_keys_produce_different_pseudo_values(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """Different mapping keys should produce different pseudo values."""
        svc = PseudoRecordService(db)

        pv1 = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )
        pv2 = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key="another-key-9876543210fedcba",
            created_by="test_user",
        )

        assert pv1 != pv2

    def test_different_pseudo_types_have_different_prefixes(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """Different pseudo types should have different prefixes."""
        svc = PseudoRecordService(db)

        pv_sub = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )
        pv_doc = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="DOCUMENT_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )

        assert pv_sub.startswith("PSB-")
        assert pv_doc.startswith("PDC-")

    def test_invalid_pseudo_type_raises_error(self, db: Session) -> None:
        """Invalid pseudo type should raise ValueError."""
        svc = PseudoRecordService(db)

        with pytest.raises(ValueError, match="Invalid pseudo type"):
            svc.generate_pseudo_id(
                original_record_id=1,
                source_table="DRUG",
                pseudo_type="INVALID_TYPE",
                mapping_key=MAPPING_KEY,
                created_by="test_user",
            )

    def test_short_mapping_key_raises_error(self, db: Session) -> None:
        """Mapping key shorter than 16 characters should be rejected."""
        svc = PseudoRecordService(db)

        with pytest.raises(InvalidMappingKeyError, match="at least 16"):
            svc.generate_pseudo_id(
                original_record_id=1,
                source_table="DRUG",
                pseudo_type="SUBMISSION_ID",
                mapping_key="short",
                created_by="test_user",
            )


class TestReidentification:
    """Tests for re-identification (pseudo -> real mapping)."""

    def _create_mapping(self, db: Session, record_id: int) -> str:
        svc = PseudoRecordService(db)
        return svc.generate_pseudo_id(
            original_record_id=record_id,
            source_table="DRUG",
            pseudo_type="SUBMISSION_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
            expiry_days=365,
        )

    def test_reidentification_returns_original_id(self, db: Session, sample_drug: Drug) -> None:
        """Re-identification should return the original record ID."""
        pseudo_value = self._create_mapping(db, sample_drug.drug_id)

        svc = PseudoRecordService(db)
        original_id, source_table = svc.map_pseudo_to_real(
            pseudo_value=pseudo_value,
            requested_by="requester",
            authorized_by="authorizer",
            reason="Unit test re-identification",
        )

        assert original_id == sample_drug.drug_id
        assert source_table == "DRUG"

    def test_reidentification_logs_access(self, db: Session, sample_drug: Drug) -> None:
        """Re-identification should create a log entry."""
        pseudo_value = self._create_mapping(db, sample_drug.drug_id)

        svc = PseudoRecordService(db)
        svc.map_pseudo_to_real(
            pseudo_value=pseudo_value,
            requested_by="requester",
            authorized_by="authorizer",
            reason="Unit test re-identification",
        )

        logs = (
            db.query(ReidentificationLog)
            .filter(ReidentificationLog.requested_by == "requester")
            .all()
        )

        assert len(logs) >= 1
        assert logs[0].status == "APPROVED"

    def test_same_user_cannot_request_and_authorize(self, db: Session, sample_drug: Drug) -> None:
        """Same user requesting and authorizing should be rejected."""
        pseudo_value = self._create_mapping(db, sample_drug.drug_id)

        svc = PseudoRecordService(db)
        with pytest.raises(UnauthorizedReidentificationError, match="different user"):
            svc.map_pseudo_to_real(
                pseudo_value=pseudo_value,
                requested_by="same_user",
                authorized_by="same_user",
                reason="Should fail - same user",
            )

    def test_nonexistent_pseudo_value_raises_error(self, db: Session) -> None:
        """Looking up a nonexistent pseudo value should raise an error."""
        svc = PseudoRecordService(db)

        with pytest.raises(InvalidMappingKeyError, match="not found"):
            svc.map_pseudo_to_real(
                pseudo_value="PSB-NONEXISTENT000000000000000000",
                requested_by="requester",
                authorized_by="authorizer",
                reason="Testing nonexistent value",
            )

    def test_expired_mapping_raises_error(self, db: Session, sample_drug: Drug) -> None:
        """Expired mapping should raise MappingExpiredError."""
        svc = PseudoRecordService(db)
        pseudo_value = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="PATIENT_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
            expiry_days=1,
        )

        # Manually expire the record
        record = svc.get_pseudo_record(pseudo_value)
        assert record is not None
        record.expiry_date = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        db.commit()

        with pytest.raises(MappingExpiredError):
            svc.map_pseudo_to_real(
                pseudo_value=pseudo_value,
                requested_by="requester",
                authorized_by="authorizer",
                reason="Testing expired mapping",
            )

    def test_deactivated_mapping_raises_error(self, db: Session, sample_drug: Drug) -> None:
        """Deactivated mapping should raise MappingExpiredError."""
        svc = PseudoRecordService(db)
        pseudo_value = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="SITE_ID",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )

        record = svc.get_pseudo_record(pseudo_value)
        assert record is not None
        record.is_active = "N"
        db.commit()

        with pytest.raises(MappingExpiredError, match="deactivated"):
            svc.map_pseudo_to_real(
                pseudo_value=pseudo_value,
                requested_by="requester",
                authorized_by="authorizer",
                reason="Testing deactivated mapping",
            )


class TestValidation:
    """Tests for pseudo mapping validation."""

    def test_active_mapping_is_valid(self, db: Session, sample_drug: Drug) -> None:
        svc = PseudoRecordService(db)
        pv = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="COMPOUND_CODE",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )
        assert svc.validate_pseudo_mapping(pv) is True

    def test_nonexistent_mapping_is_invalid(self, db: Session) -> None:
        svc = PseudoRecordService(db)
        assert svc.validate_pseudo_mapping("PSB-DOES_NOT_EXIST_0000000000") is False

    def test_deactivated_mapping_is_invalid(self, db: Session, sample_drug: Drug) -> None:
        svc = PseudoRecordService(db)
        pv = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="BATCH_NUMBER",
            mapping_key=MAPPING_KEY,
            created_by="test_user",
        )

        record = svc.get_pseudo_record(pv)
        assert record is not None
        record.is_active = "N"
        db.commit()

        assert svc.validate_pseudo_mapping(pv) is False


class TestBatchAndRotation:
    """Tests for batch operations and key rotation."""

    def test_batch_generate_returns_correct_count(
        self, db: Session, sample_drug: Drug, second_drug: Drug
    ) -> None:
        """Batch generation should return one pseudo value per record."""
        svc = PseudoRecordService(db)

        results = svc.batch_generate_pseudo_ids(
            record_ids=[sample_drug.drug_id, second_drug.drug_id],
            source_table="DRUG",
            pseudo_type="PROTOCOL_NUMBER",
            mapping_key=MAPPING_KEY,
            created_by="batch_user",
        )

        assert len(results) == 2
        assert results[0] != results[1]
        assert all(pv.startswith("PPN-") for pv in results)

    def test_key_rotation_deactivates_old_and_creates_new(
        self, db: Session, sample_drug: Drug
    ) -> None:
        """Key rotation should deactivate old mappings and create new ones."""
        svc = PseudoRecordService(db)
        old_key = "old-key-1234567890abcdef"
        new_key = "new-key-fedcba0987654321"

        # Create mapping with old key
        old_pv = svc.generate_pseudo_id(
            original_record_id=sample_drug.drug_id,
            source_table="DRUG",
            pseudo_type="INVESTIGATOR_ID",
            mapping_key=old_key,
            created_by="test_user",
        )

        # Rotate key
        rotated = svc.rotate_mapping_key(
            old_key=old_key,
            new_key=new_key,
            source_table="DRUG",
            rotated_by="admin_user",
        )

        assert rotated == 1

        # Old mapping should be deactivated
        assert svc.validate_pseudo_mapping(old_pv) is False

        # New mapping should exist
        new_mappings = svc.get_mappings_for_record(sample_drug.drug_id, "DRUG", active_only=True)
        new_inv_mappings = [m for m in new_mappings if m.pseudo_type == "INVESTIGATOR_ID"]
        assert len(new_inv_mappings) == 1
        assert new_inv_mappings[0].mapping_key == new_key


class TestHashComputation:
    """Tests for the hash computation internals."""

    def test_sha256_produces_deterministic_output(self) -> None:
        result1 = PseudoRecordService._compute_hash("input", "salt", "key")
        result2 = PseudoRecordService._compute_hash("input", "salt", "key")
        assert result1 == result2

    def test_different_inputs_produce_different_hashes(self) -> None:
        result1 = PseudoRecordService._compute_hash("input1", "salt", "key")
        result2 = PseudoRecordService._compute_hash("input2", "salt", "key")
        assert result1 != result2

    def test_different_salts_produce_different_hashes(self) -> None:
        result1 = PseudoRecordService._compute_hash("input", "salt1", "key")
        result2 = PseudoRecordService._compute_hash("input", "salt2", "key")
        assert result1 != result2

    def test_hmac_sha256_algorithm(self) -> None:
        result = PseudoRecordService._compute_hash("input", "salt", "key", algorithm="HMAC-SHA256")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_blake2b_algorithm(self) -> None:
        result = PseudoRecordService._compute_hash("input", "salt", "key", algorithm="BLAKE2B")
        assert isinstance(result, str)
        assert len(result) == 64  # 32 bytes = 64 hex chars

    def test_invalid_algorithm_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            PseudoRecordService._compute_hash("input", "salt", "key", algorithm="MD5")

    def test_salt_generation_is_random(self) -> None:
        salt1 = PseudoRecordService._generate_salt()
        salt2 = PseudoRecordService._generate_salt()
        assert salt1 != salt2
        assert len(salt1) == 64  # 32 bytes hex encoded
