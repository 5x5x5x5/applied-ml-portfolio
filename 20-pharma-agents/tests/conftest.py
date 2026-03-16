"""Shared test fixtures for PharmaAgents."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pharma_agents.agents.chemistry_agent import ChemistryAgent
from pharma_agents.agents.literature_agent import LiteratureAgent
from pharma_agents.agents.regulatory_agent import RegulatoryAgent
from pharma_agents.agents.safety_agent import SafetyAgent
from pharma_agents.orchestrator.coordinator import AgentCoordinator
from pharma_agents.orchestrator.workflow import WorkflowEngine


def _make_mock_client() -> MagicMock:
    """Create a mock Anthropic client that returns a minimal valid response."""
    client = MagicMock()

    # Build a mock response object
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "This is a mock agent response for testing purposes."

    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = "end_turn"

    client.messages.create.return_value = response
    return client


@pytest.fixture()
def mock_client() -> MagicMock:
    """Provide a mock Anthropic client."""
    return _make_mock_client()


@pytest.fixture()
def literature_agent(mock_client: MagicMock) -> LiteratureAgent:
    """Provide a LiteratureAgent with a mock client."""
    return LiteratureAgent(client=mock_client)


@pytest.fixture()
def safety_agent(mock_client: MagicMock) -> SafetyAgent:
    """Provide a SafetyAgent with a mock client."""
    return SafetyAgent(client=mock_client)


@pytest.fixture()
def chemistry_agent(mock_client: MagicMock) -> ChemistryAgent:
    """Provide a ChemistryAgent with a mock client."""
    return ChemistryAgent(client=mock_client)


@pytest.fixture()
def regulatory_agent(mock_client: MagicMock) -> RegulatoryAgent:
    """Provide a RegulatoryAgent with a mock client."""
    return RegulatoryAgent(client=mock_client)


@pytest.fixture()
def coordinator(mock_client: MagicMock) -> AgentCoordinator:
    """Provide an AgentCoordinator with mock clients on all agents."""
    coord = AgentCoordinator(client=mock_client)
    return coord


@pytest.fixture()
def workflow_engine(coordinator: AgentCoordinator) -> WorkflowEngine:
    """Provide a WorkflowEngine backed by a mocked coordinator."""
    return WorkflowEngine(coordinator=coordinator)


# Sample data fixtures

SAMPLE_SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
SAMPLE_DRUG_NAME = "imatinib"
SAMPLE_INDICATION = "chronic myeloid leukemia"


@pytest.fixture()
def sample_smiles() -> str:
    return SAMPLE_SMILES


@pytest.fixture()
def sample_drug_name() -> str:
    return SAMPLE_DRUG_NAME


@pytest.fixture()
def sample_indication() -> str:
    return SAMPLE_INDICATION
