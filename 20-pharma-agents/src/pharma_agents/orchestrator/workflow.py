"""Predefined pharmaceutical research workflows.

Each workflow defines a sequence of agent interactions with data flow
between steps, enabling reproducible multi-agent analyses.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from pharma_agents.agents.base_agent import AgentResponse
from pharma_agents.orchestrator.coordinator import AgentCoordinator

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------


class WorkflowType(str, Enum):
    """Available predefined workflows."""

    DRUG_CANDIDATE_ASSESSMENT = "drug_candidate_assessment"
    SAFETY_SIGNAL_INVESTIGATION = "safety_signal_investigation"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    REGULATORY_STRATEGY = "regulatory_strategy"


class WorkflowStep(BaseModel):
    """A single step in a workflow."""

    step_number: int
    agent_name: str
    prompt_template: str
    depends_on: list[int] = Field(default_factory=list)
    description: str = ""


class WorkflowResult(BaseModel):
    """Result of a completed workflow execution."""

    workflow_type: str
    parameters: dict[str, Any]
    steps_completed: int
    step_results: list[dict[str, Any]]
    final_synthesis: str
    total_time_s: float


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------

_WORKFLOWS: dict[WorkflowType, list[WorkflowStep]] = {
    WorkflowType.DRUG_CANDIDATE_ASSESSMENT: [
        WorkflowStep(
            step_number=1,
            agent_name="LiteratureAgent",
            prompt_template=(
                "Search the literature for studies related to {drug_name} "
                "({mechanism}) in {indication}. Summarise the current evidence "
                "base, including efficacy data, safety signals, and any head-to-head "
                "comparisons with existing therapies."
            ),
            description="Literature review of the drug candidate",
        ),
        WorkflowStep(
            step_number=2,
            agent_name="ChemistryAgent",
            prompt_template=(
                "Analyse the molecular properties of {drug_name} "
                "(SMILES: {smiles}). Assess drug-likeness using Lipinski, Veber, "
                "and Ghose rules. Predict ADMET properties and compare to known "
                "drugs in the {drug_class} class."
            ),
            description="Molecular and ADMET analysis",
        ),
        WorkflowStep(
            step_number=3,
            agent_name="SafetyAgent",
            prompt_template=(
                "Perform a comprehensive safety assessment of {drug_name}. "
                "Analyse adverse event data, compute PRR and ROR for key safety "
                "signals, perform risk-benefit analysis for {indication}, and scan "
                "for red-flag safety signals.\n\n"
                "Literature context:\n{step_1_summary}"
            ),
            depends_on=[1],
            description="Safety assessment with literature context",
        ),
        WorkflowStep(
            step_number=4,
            agent_name="RegulatoryAgent",
            prompt_template=(
                "Develop a regulatory strategy for {drug_name} ({drug_class}) "
                "targeting {indication}. Determine the appropriate FDA pathway, "
                "required studies, timeline estimate, and competitive landscape "
                "in {therapeutic_area}.\n\n"
                "Safety context:\n{step_3_summary}\n\n"
                "Chemistry context:\n{step_2_summary}"
            ),
            depends_on=[2, 3],
            description="Regulatory strategy with safety and chemistry context",
        ),
    ],
    WorkflowType.SAFETY_SIGNAL_INVESTIGATION: [
        WorkflowStep(
            step_number=1,
            agent_name="SafetyAgent",
            prompt_template=(
                "Investigate the safety signal for {drug_name} related to "
                "{adverse_event}. Retrieve adverse event data, compute "
                "disproportionality metrics (PRR, ROR), and determine if a "
                "signal is confirmed. Check for related red flags."
            ),
            description="Initial safety signal detection",
        ),
        WorkflowStep(
            step_number=2,
            agent_name="LiteratureAgent",
            prompt_template=(
                "Search the literature for published reports of {adverse_event} "
                "associated with {drug_name} or related drugs in the same class. "
                "Summarise case reports, cohort studies, and any mechanistic "
                "explanations.\n\n"
                "Safety signal data:\n{step_1_summary}"
            ),
            depends_on=[1],
            description="Literature corroboration of safety signal",
        ),
        WorkflowStep(
            step_number=3,
            agent_name="ChemistryAgent",
            prompt_template=(
                "Analyse the molecular structure of {drug_name} "
                "(SMILES: {smiles}) for structural alerts that could explain "
                "{adverse_event}. Suggest modifications that might mitigate "
                "the safety concern while maintaining efficacy."
            ),
            description="Structural basis for safety signal",
        ),
    ],
    WorkflowType.COMPETITIVE_ANALYSIS: [
        WorkflowStep(
            step_number=1,
            agent_name="RegulatoryAgent",
            prompt_template=(
                "Analyse the competitive landscape in {therapeutic_area} "
                "for drugs targeting {mechanism}. Identify approved drugs, "
                "pending applications, and recent FDA decisions."
            ),
            description="Regulatory competitive landscape",
        ),
        WorkflowStep(
            step_number=2,
            agent_name="LiteratureAgent",
            prompt_template=(
                "Review the clinical literature comparing different "
                "{mechanism} drugs in {therapeutic_area}. Focus on "
                "head-to-head trials, meta-analyses, and real-world evidence.\n\n"
                "Competitive landscape:\n{step_1_summary}"
            ),
            depends_on=[1],
            description="Literature-based competitive comparison",
        ),
        WorkflowStep(
            step_number=3,
            agent_name="SafetyAgent",
            prompt_template=(
                "Compare the safety profiles of drugs targeting {mechanism} "
                "in {therapeutic_area}. Which compounds have the best and "
                "worst safety records? Are there class-wide safety concerns?"
            ),
            description="Comparative safety analysis",
        ),
    ],
    WorkflowType.REGULATORY_STRATEGY: [
        WorkflowStep(
            step_number=1,
            agent_name="RegulatoryAgent",
            prompt_template=(
                "Determine the optimal regulatory pathway for {drug_name}, "
                "a {drug_class} targeting {indication}. Consider whether "
                "expedited programs (Fast Track, Breakthrough Therapy, "
                "Accelerated Approval, Priority Review) are applicable."
            ),
            description="Regulatory pathway determination",
        ),
        WorkflowStep(
            step_number=2,
            agent_name="RegulatoryAgent",
            prompt_template=(
                "Based on the recommended pathway, list all required "
                "studies and estimate the development timeline for "
                "{drug_name}. Include IND-enabling through approval.\n\n"
                "Pathway recommendation:\n{step_1_summary}"
            ),
            depends_on=[1],
            description="Detailed study requirements and timeline",
        ),
        WorkflowStep(
            step_number=3,
            agent_name="LiteratureAgent",
            prompt_template=(
                "Review published FDA guidance documents, advisory committee "
                "transcripts, and regulatory precedents for {drug_class} drugs "
                "in {indication}. What endpoints has FDA accepted? Are there "
                "known regulatory hurdles?\n\n"
                "Regulatory strategy:\n{step_2_summary}"
            ),
            depends_on=[2],
            description="Regulatory precedent research",
        ),
    ],
}


class WorkflowEngine:
    """Executes predefined multi-agent workflows.

    Each workflow is a directed acyclic graph of steps, where each step
    is assigned to a specific agent and may depend on outputs from
    earlier steps.
    """

    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        self._coordinator = coordinator or AgentCoordinator()
        self._log = logger.bind(component="workflow_engine")

    @staticmethod
    def list_workflows() -> list[dict[str, Any]]:
        """List all available workflows with their descriptions."""
        result = []
        for wf_type, steps in _WORKFLOWS.items():
            result.append(
                {
                    "name": wf_type.value,
                    "display_name": wf_type.value.replace("_", " ").title(),
                    "steps": len(steps),
                    "agents_involved": list({s.agent_name for s in steps}),
                    "step_descriptions": [
                        {"step": s.step_number, "agent": s.agent_name, "description": s.description}
                        for s in steps
                    ],
                }
            )
        return result

    @staticmethod
    def get_workflow_parameters(workflow_type: WorkflowType) -> list[str]:
        """Return the required parameters for a workflow."""
        steps = _WORKFLOWS.get(workflow_type, [])
        params: set[str] = set()
        import re

        for step in steps:
            # Extract {param_name} placeholders (excluding step references)
            found = re.findall(r"\{(\w+)\}", step.prompt_template)
            for p in found:
                if not p.startswith("step_"):
                    params.add(p)
        return sorted(params)

    async def execute(
        self,
        workflow_type: WorkflowType,
        parameters: dict[str, str],
        *,
        callback: Any | None = None,
    ) -> WorkflowResult:
        """Execute a workflow with the given parameters.

        Args:
            workflow_type: The workflow to execute.
            parameters: Dictionary of parameter values to fill into prompts.
            callback: Optional async callable for progress updates.
        """
        start = time.time()
        steps = _WORKFLOWS.get(workflow_type)
        if not steps:
            raise ValueError(f"Unknown workflow: {workflow_type}")

        self._log.info(
            "workflow.start",
            workflow=workflow_type.value,
            steps=len(steps),
        )

        step_results: dict[int, dict[str, Any]] = {}
        step_summaries: dict[int, str] = {}

        # Execute steps in dependency order
        for step in sorted(steps, key=lambda s: s.step_number):
            # Check dependencies
            for dep in step.depends_on:
                if dep not in step_results:
                    self._log.error(
                        "workflow.dependency_missing",
                        step=step.step_number,
                        missing_dep=dep,
                    )

            # Build the prompt by substituting parameters and step references
            prompt_params = dict(parameters)
            for dep_num, summary in step_summaries.items():
                prompt_params[f"step_{dep_num}_summary"] = summary[:2000]

            try:
                prompt = step.prompt_template.format(**prompt_params)
            except KeyError as exc:
                self._log.warning(
                    "workflow.param_missing",
                    step=step.step_number,
                    missing=str(exc),
                )
                # Fill missing params with placeholder
                prompt = step.prompt_template
                for key in parameters:
                    prompt = prompt.replace(f"{{{key}}}", parameters[key])
                # Remove remaining unfilled placeholders
                import re

                prompt = re.sub(r"\{step_\d+_summary\}", "(not available)", prompt)
                prompt = re.sub(r"\{\w+\}", "(unspecified)", prompt)

            if callback:
                await callback(
                    "workflow_step_start",
                    {
                        "step": step.step_number,
                        "agent": step.agent_name,
                        "description": step.description,
                    },
                )

            # Execute via the coordinator's single-agent query
            response = await self._coordinator.query_single_agent(step.agent_name, prompt)

            step_results[step.step_number] = {
                "step": step.step_number,
                "agent": step.agent_name,
                "description": step.description,
                "response": response.model_dump(),
            }
            step_summaries[step.step_number] = response.text

            if callback:
                await callback(
                    "workflow_step_complete",
                    {
                        "step": step.step_number,
                        "agent": step.agent_name,
                        "status": "completed",
                    },
                )

            self._log.info(
                "workflow.step.done",
                step=step.step_number,
                agent=step.agent_name,
            )

        # Generate final synthesis
        all_responses = [AgentResponse(**r["response"]) for r in step_results.values()]
        synthesis = self._coordinator._synthesise(
            f"Workflow: {workflow_type.value}",
            all_responses,
            conflicts=[],
        )

        elapsed = time.time() - start
        self._log.info("workflow.complete", elapsed=round(elapsed, 2))

        return WorkflowResult(
            workflow_type=workflow_type.value,
            parameters=parameters,
            steps_completed=len(step_results),
            step_results=list(step_results.values()),
            final_synthesis=synthesis,
            total_time_s=round(elapsed, 3),
        )
