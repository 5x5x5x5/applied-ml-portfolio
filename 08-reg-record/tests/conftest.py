"""Shared test fixtures for RegRecord tests."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from reg_record.models.database import Base, Drug


@pytest.fixture(scope="session")
def engine():
    """Create a test database engine (in-memory SQLite)."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture()
def db(engine):
    """Provide a clean database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def sample_drug(db: Session) -> Drug:
    """Create a sample drug record."""
    drug = Drug(
        drug_name="TestDrug-100",
        generic_name="testdruggeneric",
        ndc_code="12345-678-90",
        therapeutic_area="Oncology",
        manufacturer="TestPharma Inc.",
        active_flag="Y",
    )
    db.add(drug)
    db.flush()
    return drug


@pytest.fixture()
def second_drug(db: Session) -> Drug:
    """Create a second drug record for multi-drug tests."""
    drug = Drug(
        drug_name="TestDrug-200",
        generic_name="testdrugtwo",
        ndc_code="99999-888-77",
        therapeutic_area="Cardiology",
        manufacturer="TestPharma Inc.",
        active_flag="Y",
    )
    db.add(drug)
    db.flush()
    return drug
