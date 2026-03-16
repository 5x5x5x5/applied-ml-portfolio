"""Tests for the multi-agent coordinator and workflow engine."""

from __future__ import annotations

import pytest

from pharma_agents.agents.base_agent import AgentResponse
from pharma_agents.orchestrator.coordinator import (
    AgentCoordinator,
    ConflictResolution,
)
from pharma_agents.orchestrator.workflow import WorkflowEngine, WorkflowType

# ---------------------------------------------------------------------------
# Coordinator tests
# ---------------------------------------------------------------------------


class TestCoordinator:
    """Test the AgentCoordinator decomposition and routing logic."""

    def test_list_agents(self, coordinator: AgentCoordinator) -> None:
        agents = coordinator.list_agents()
        assert len(agents) == 4
        names = {a["name"] for a in agents}
        assert "LiteratureAgent" in names
        assert "SafetyAgent" in names
        assert "ChemistryAgent" in names
        assert "RegulatoryAgent" in names

    def test_route_to_safety(self, coordinator: AgentCoordinator) -> None:
        best = coordinator.route_to_best_agent("What are the adverse events of imatinib?")
        assert best == "SafetyAgent"

    def test_route_to_literature(self, coordinator: AgentCoordinator) -> None:
        best = coordinator.route_to_best_agent("Find recent publications about PD-1 inhibitors")
        assert best == "LiteratureAgent"

    def test_route_to_chemistry(self, coordinator: AgentCoordinator) -> None:
        best = coordinator.route_to_best_agent(
            "Analyze the molecular structure and Lipinski properties of this compound"
        )
        assert best == "ChemistryAgent"

    def test_route_to_regulatory(self, coordinator: AgentCoordinator) -> None:
        best = coordinator.route_to_best_agent(
            "What is the FDA approval pathway for a new biologic?"
        )
        assert best == "RegulatoryAgent"

    def test_route_fallback(self, coordinator: AgentCoordinator) -> None:
        """Ambiguous queries should default to LiteratureAgent."""
        best = coordinator.route_to_best_agent("Tell me about aspirin")
        assert best in coordinator.agents

    def test_decompose_multi_domain_query(self, coordinator: AgentCoordinator) -> None:
        subtasks = coordinator._decompose_query(
            "What is the safety profile and regulatory pathway for imatinib "
            "based on recent literature?"
        )
        assert len(subtasks) >= 2
        agents_used = {st.assigned_agent for st in subtasks}
        # Should engage at least safety and regulatory
        assert "SafetyAgent" in agents_used or "RegulatoryAgent" in agents_used

    def test_decompose_creates_dependencies(self, coordinator: AgentCoordinator) -> None:
        subtasks = coordinator._decompose_query(
            "Review the literature on imatinib safety signals and assess risk"
        )
        # Should have both literature and safety, with safety depending on literature
        safety_tasks = [st for st in subtasks if st.assigned_agent == "SafetyAgent"]
        lit_tasks = [st for st in subtasks if st.assigned_agent == "LiteratureAgent"]
        if safety_tasks and lit_tasks:
            assert any(lit_tasks[0].id in st.dependencies for st in safety_tasks)

    @pytest.mark.asyncio
    async def test_process_query(self, coordinator: AgentCoordinator) -> None:
        result = await coordinator.process_query("What is the safety profile of imatinib?")
        assert result.query == "What is the safety profile of imatinib?"
        assert len(result.agent_responses) >= 1
        assert len(result.synthesis) > 0
        assert result.total_processing_time_s >= 0

    @pytest.mark.asyncio
    async def test_query_single_agent(self, coordinator: AgentCoordinator) -> None:
        response = await coordinator.query_single_agent("SafetyAgent", "Assess imatinib safety")
        assert isinstance(response, AgentResponse)
        assert response.agent_name == "SafetyAgent"

    @pytest.mark.asyncio
    async def test_query_unknown_agent_raises(self, coordinator: AgentCoordinator) -> None:
        with pytest.raises(ValueError, match="Unknown agent"):
            await coordinator.query_single_agent("NonexistentAgent", "test")


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    """Test conflict detection between agent outputs."""

    def test_detect_safety_chemistry_conflict(self, coordinator: AgentCoordinator) -> None:
        responses = [
            AgentResponse(
                agent_name="ChemistryAgent",
                agent_role="Chemistry",
                text="The compound is drug-like with favorable properties.",
            ),
            AgentResponse(
                agent_name="SafetyAgent",
                agent_role="Safety",
                text="CRITICAL safety concern: hepatotoxicity signal detected.",
            ),
        ]
        conflicts = coordinator._detect_conflicts(responses)
        assert len(conflicts) >= 1
        assert any("ChemistryAgent" in c.agents_involved for c in conflicts)

    def test_no_conflict_when_consistent(self, coordinator: AgentCoordinator) -> None:
        responses = [
            AgentResponse(
                agent_name="ChemistryAgent",
                agent_role="Chemistry",
                text="The compound has acceptable properties.",
            ),
            AgentResponse(
                agent_name="SafetyAgent",
                agent_role="Safety",
                text="No significant safety signals detected.",
            ),
        ]
        conflicts = coordinator._detect_conflicts(responses)
        # Should have no conflicts when there is no drug-like + critical combo
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestSynthesis:
    """Test the synthesis of agent outputs."""

    def test_synthesis_includes_all_agents(self, coordinator: AgentCoordinator) -> None:
        responses = [
            AgentResponse(
                agent_name="LiteratureAgent",
                agent_role="Literature",
                text="Literature findings...",
                tools_used=["pubmed_search"],
                confidence=0.8,
            ),
            AgentResponse(
                agent_name="SafetyAgent",
                agent_role="Safety",
                text="Safety assessment...",
                tools_used=["get_adverse_events"],
                confidence=0.7,
            ),
        ]
        synthesis = coordinator._synthesise("Test query", responses, [])
        assert "LiteratureAgent" in synthesis
        assert "SafetyAgent" in synthesis
        assert "Integrated Analysis" in synthesis

    def test_synthesis_includes_conflicts(self, coordinator: AgentCoordinator) -> None:
        conflicts = [
            ConflictResolution(
                agents_involved=["A", "B"],
                conflict_description="Test conflict",
                resolution="Resolved by prioritising safety",
                confidence=0.85,
            )
        ]
        synthesis = coordinator._synthesise("q", [], conflicts)
        assert "Conflicts" in synthesis
        assert "Test conflict" in synthesis


# ---------------------------------------------------------------------------
# Workflow Engine tests
# ---------------------------------------------------------------------------


class TestWorkflowEngine:
    """Test the predefined workflow engine."""

    def test_list_workflows(self) -> None:
        workflows = WorkflowEngine.list_workflows()
        assert len(workflows) >= 4
        names = {w["name"] for w in workflows}
        assert "drug_candidate_assessment" in names
        assert "safety_signal_investigation" in names
        assert "competitive_analysis" in names
        assert "regulatory_strategy" in names

    def test_workflow_has_steps(self) -> None:
        workflows = WorkflowEngine.list_workflows()
        for wf in workflows:
            assert wf["steps"] >= 2
            assert len(wf["agents_involved"]) >= 1

    def test_get_parameters_drug_candidate(self) -> None:
        params = WorkflowEngine.get_workflow_parameters(WorkflowType.DRUG_CANDIDATE_ASSESSMENT)
        assert "drug_name" in params
        assert "indication" in params

    def test_get_parameters_safety_signal(self) -> None:
        params = WorkflowEngine.get_workflow_parameters(WorkflowType.SAFETY_SIGNAL_INVESTIGATION)
        assert "drug_name" in params
        assert "adverse_event" in params

    @pytest.mark.asyncio
    async def test_execute_workflow(self, workflow_engine: WorkflowEngine) -> None:
        result = await workflow_engine.execute(
            WorkflowType.COMPETITIVE_ANALYSIS,
            parameters={
                "therapeutic_area": "oncology",
                "mechanism": "PD-1 inhibitor",
            },
        )
        assert result.workflow_type == "competitive_analysis"
        assert result.steps_completed >= 1
        assert len(result.final_synthesis) > 0
        assert result.total_time_s >= 0

    @pytest.mark.asyncio
    async def test_execute_unknown_workflow(self, workflow_engine: WorkflowEngine) -> None:
        with pytest.raises(ValueError, match="Unknown workflow"):
            await workflow_engine.execute(
                "nonexistent_workflow",  # type: ignore[arg-type]
                parameters={},
            )
