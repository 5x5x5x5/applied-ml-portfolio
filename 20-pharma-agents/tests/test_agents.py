"""Unit tests for individual PharmaAgents."""

from __future__ import annotations

import pytest

from pharma_agents.agents.base_agent import AgentResponse, ToolDefinition
from pharma_agents.agents.chemistry_agent import ChemistryAgent
from pharma_agents.agents.literature_agent import LiteratureAgent
from pharma_agents.agents.regulatory_agent import RegulatoryAgent
from pharma_agents.agents.safety_agent import SafetyAgent

# ---------------------------------------------------------------------------
# Base Agent properties
# ---------------------------------------------------------------------------


class TestAgentIdentity:
    """Test that each agent has correct identity properties."""

    def test_literature_agent_identity(self, literature_agent: LiteratureAgent) -> None:
        assert literature_agent.name == "LiteratureAgent"
        assert "Literature" in literature_agent.role
        assert len(literature_agent.system_prompt) > 100

    def test_safety_agent_identity(self, safety_agent: SafetyAgent) -> None:
        assert safety_agent.name == "SafetyAgent"
        assert "Safety" in safety_agent.role
        assert "pharmacovigilance" in safety_agent.system_prompt.lower()

    def test_chemistry_agent_identity(self, chemistry_agent: ChemistryAgent) -> None:
        assert chemistry_agent.name == "ChemistryAgent"
        assert "Chemistry" in chemistry_agent.role
        assert "medicinal" in chemistry_agent.system_prompt.lower()

    def test_regulatory_agent_identity(self, regulatory_agent: RegulatoryAgent) -> None:
        assert regulatory_agent.name == "RegulatoryAgent"
        assert "Regulatory" in regulatory_agent.role
        assert "FDA" in regulatory_agent.system_prompt


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class TestAgentTools:
    """Test that each agent defines proper tools."""

    def test_literature_tools(self, literature_agent: LiteratureAgent) -> None:
        tools = literature_agent.get_tools()
        assert len(tools) >= 3
        assert all(isinstance(t, ToolDefinition) for t in tools)
        tool_names = {t.name for t in tools}
        assert "pubmed_search" in tool_names
        assert "get_paper_details" in tool_names
        assert "cross_reference" in tool_names

    def test_safety_tools(self, safety_agent: SafetyAgent) -> None:
        tools = safety_agent.get_tools()
        assert len(tools) >= 3
        tool_names = {t.name for t in tools}
        assert "get_adverse_events" in tool_names
        assert "compute_safety_metrics" in tool_names
        assert "detect_red_flags" in tool_names

    def test_chemistry_tools(self, chemistry_agent: ChemistryAgent) -> None:
        tools = chemistry_agent.get_tools()
        assert len(tools) >= 4
        tool_names = {t.name for t in tools}
        assert "analyze_molecule" in tool_names
        assert "assess_drug_likeness" in tool_names
        assert "predict_admet" in tool_names
        assert "suggest_modifications" in tool_names

    def test_regulatory_tools(self, regulatory_agent: RegulatoryAgent) -> None:
        tools = regulatory_agent.get_tools()
        assert len(tools) >= 3
        tool_names = {t.name for t in tools}
        assert "determine_pathway" in tool_names
        assert "get_required_studies" in tool_names
        assert "estimate_timeline" in tool_names


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    """Test that tool execution returns valid data."""

    def test_literature_pubmed_search(self, literature_agent: LiteratureAgent) -> None:
        result = literature_agent.execute_tool(
            "pubmed_search", {"query": "imatinib CML", "max_results": 5}
        )
        assert "imatinib" in result.lower() or "papers" in result.lower()

    def test_literature_unknown_tool(self, literature_agent: LiteratureAgent) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            literature_agent.execute_tool("nonexistent_tool", {})

    def test_safety_adverse_events(self, safety_agent: SafetyAgent) -> None:
        result = safety_agent.execute_tool("get_adverse_events", {"drug_name": "imatinib"})
        assert "imatinib" in result.lower()

    def test_safety_metrics(self, safety_agent: SafetyAgent) -> None:
        result = safety_agent.execute_tool(
            "compute_safety_metrics",
            {"drug_name": "imatinib", "event_name": "hepatotoxicity"},
        )
        assert "prr" in result.lower()
        assert "ror" in result.lower()

    def test_safety_red_flags(self, safety_agent: SafetyAgent) -> None:
        result = safety_agent.execute_tool("detect_red_flags", {"drug_name": "imatinib"})
        assert "red_flags_detected" in result

    def test_chemistry_analyze(self, chemistry_agent: ChemistryAgent, sample_smiles: str) -> None:
        result = chemistry_agent.execute_tool("analyze_molecule", {"smiles": sample_smiles})
        assert "molecular_weight" in result

    def test_chemistry_drug_likeness(
        self, chemistry_agent: ChemistryAgent, sample_smiles: str
    ) -> None:
        result = chemistry_agent.execute_tool("assess_drug_likeness", {"smiles": sample_smiles})
        assert "lipinski" in result.lower()

    def test_chemistry_admet(self, chemistry_agent: ChemistryAgent, sample_smiles: str) -> None:
        result = chemistry_agent.execute_tool("predict_admet", {"smiles": sample_smiles})
        assert "absorption" in result

    def test_regulatory_pathway(self, regulatory_agent: RegulatoryAgent) -> None:
        result = regulatory_agent.execute_tool(
            "determine_pathway",
            {"drug_class": "small_molecule", "has_reference_drug": False},
        )
        assert "NDA" in result or "pathway" in result.lower()

    def test_regulatory_timeline(self, regulatory_agent: RegulatoryAgent) -> None:
        result = regulatory_agent.execute_tool(
            "estimate_timeline",
            {"drug_class": "small_molecule", "pathway": "NDA"},
        )
        assert "total_months" in result


# ---------------------------------------------------------------------------
# Agent capabilities
# ---------------------------------------------------------------------------


class TestAgentCapabilities:
    """Test the get_capabilities method."""

    def test_capabilities_structure(self, literature_agent: LiteratureAgent) -> None:
        caps = literature_agent.get_capabilities()
        assert "name" in caps
        assert "role" in caps
        assert "tools" in caps
        assert "description" in caps
        assert isinstance(caps["tools"], list)
        assert len(caps["tools"]) > 0

    def test_capabilities_all_agents(
        self,
        literature_agent: LiteratureAgent,
        safety_agent: SafetyAgent,
        chemistry_agent: ChemistryAgent,
        regulatory_agent: RegulatoryAgent,
    ) -> None:
        agents = [literature_agent, safety_agent, chemistry_agent, regulatory_agent]
        names = {a.get_capabilities()["name"] for a in agents}
        assert len(names) == 4  # All names are unique


# ---------------------------------------------------------------------------
# Agent run (with mocked API)
# ---------------------------------------------------------------------------


class TestAgentRun:
    """Test the agent run method with mocked API calls."""

    @pytest.mark.asyncio
    async def test_literature_agent_run(self, literature_agent: LiteratureAgent) -> None:
        response = await literature_agent.run("Tell me about imatinib studies")
        assert isinstance(response, AgentResponse)
        assert response.agent_name == "LiteratureAgent"
        assert len(response.text) > 0
        assert response.processing_time_s >= 0

    @pytest.mark.asyncio
    async def test_safety_agent_run(self, safety_agent: SafetyAgent) -> None:
        response = await safety_agent.run("Assess imatinib safety profile")
        assert isinstance(response, AgentResponse)
        assert response.agent_name == "SafetyAgent"

    @pytest.mark.asyncio
    async def test_agent_reset_history(self, literature_agent: LiteratureAgent) -> None:
        await literature_agent.run("First message")
        assert len(literature_agent._history) == 2  # user + assistant
        literature_agent.reset_history()
        assert len(literature_agent._history) == 0
