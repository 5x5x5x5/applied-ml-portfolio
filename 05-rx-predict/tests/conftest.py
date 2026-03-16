"""Shared test fixtures for RxPredict."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rx_predict.models.drug_response_model import DrugResponseModel
from rx_predict.models.feature_processor import FeatureProcessor
from rx_predict.monitoring.performance import PerformanceMonitor


@pytest.fixture(scope="session")
def feature_processor() -> FeatureProcessor:
    """Shared feature processor instance."""
    return FeatureProcessor()


@pytest.fixture(scope="session")
def trained_model() -> DrugResponseModel:
    """Shared trained model instance (session-scoped for speed)."""
    model = DrugResponseModel(model_version="test-1.0.0")
    model.build_default_model()
    model.warm_up()
    return model


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Fresh performance monitor for each test."""
    return PerformanceMonitor()


@pytest.fixture
def sample_patient_data() -> dict[str, Any]:
    """Standard patient test data."""
    return {
        "genetic_profile": {
            "CYP2D6": ["*1", "*2"],
            "CYP2C19": ["*1"],
            "CYP3A4": ["*1"],
            "CYP2C9": ["*1"],
            "VKORC1": [],
            "DPYD": ["*1"],
            "TPMT": ["*1"],
            "UGT1A1": ["*1"],
            "SLCO1B1": ["*1A"],
            "HLA-B": [],
        },
        "metabolizer_phenotype": "normal",
        "demographics": {
            "age": 45,
            "weight_kg": 75.0,
            "height_cm": 170.0,
            "bmi": 26.0,
            "sex": "male",
            "ethnicity": "caucasian",
        },
        "drug": {
            "name": "Sertraline",
            "drug_class": "ssri",
            "dosage_mg": 50.0,
            "max_dosage_mg": 200.0,
        },
        "medical_history": {
            "num_current_medications": 2,
            "num_allergies": 0,
            "num_adverse_reactions": 0,
            "conditions": [],
            "pregnant": False,
            "age": 45,
        },
    }


@pytest.fixture
def sample_prediction_request() -> dict[str, Any]:
    """Sample API prediction request payload."""
    return {
        "genetic_profile": {
            "CYP2D6": ["*1", "*2"],
            "CYP2C19": ["*1"],
            "CYP3A4": ["*1"],
            "CYP2C9": ["*1"],
            "VKORC1": [],
            "DPYD": ["*1"],
            "TPMT": ["*1"],
            "UGT1A1": ["*1"],
            "SLCO1B1": ["*1A"],
            "HLA-B": [],
        },
        "metabolizer_phenotype": "normal",
        "demographics": {
            "age": 45,
            "weight_kg": 75.0,
            "height_cm": 170.0,
            "bmi": 26.0,
            "sex": "male",
            "ethnicity": "caucasian",
        },
        "drug": {
            "name": "Sertraline",
            "drug_class": "ssri",
            "dosage_mg": 50.0,
            "max_dosage_mg": 200.0,
        },
        "medical_history": {
            "num_current_medications": 2,
            "num_allergies": 0,
            "num_adverse_reactions": 0,
            "conditions": [],
            "pregnant": False,
            "age": 45,
        },
    }


@pytest.fixture
def poor_metabolizer_patient() -> dict[str, Any]:
    """Patient with poor metabolizer profile."""
    return {
        "genetic_profile": {
            "CYP2D6": ["*4", "*5"],
            "CYP2C19": ["*2"],
            "CYP3A4": ["*22"],
            "CYP2C9": ["*3"],
            "VKORC1": ["-1639G>A_AA"],
            "DPYD": ["*2A"],
            "TPMT": ["*3A"],
            "UGT1A1": ["*28"],
            "SLCO1B1": ["*5"],
            "HLA-B": ["*57:01_pos"],
        },
        "metabolizer_phenotype": "poor",
        "demographics": {
            "age": 72,
            "weight_kg": 60.0,
            "height_cm": 165.0,
            "bmi": 22.0,
            "sex": "female",
            "ethnicity": "caucasian",
        },
        "drug": {
            "name": "Codeine",
            "drug_class": "opioid",
            "dosage_mg": 30.0,
            "max_dosage_mg": 240.0,
        },
        "medical_history": {
            "num_current_medications": 8,
            "num_allergies": 3,
            "num_adverse_reactions": 2,
            "conditions": ["liver_disease", "kidney_disease", "diabetes"],
            "pregnant": False,
            "age": 72,
        },
    }


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for API testing."""
    from rx_predict.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
