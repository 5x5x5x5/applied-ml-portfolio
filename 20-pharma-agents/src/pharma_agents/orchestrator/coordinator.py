"""Multi-agent coordinator -- decomposes tasks, routes to agents, and synthesises results."""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

import anthropic
import structlog
from pydantic import BaseModel, Field

from pharma_agents.agents.base_agent import AgentResponse, BaseAgent
from pharma_agents.agents.chemistry_agent import ChemistryAgent
from pharma_agents.agents.literature_agent import LiteratureAgent
from pharma_agents.agents.regulatory_agent import RegulatoryAgent
from pharma_agents.agents.safety_agent import SafetyAgent

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """A decomposed sub-task assigned to a specific agent."""

    id: str
    description: str
    assigned_agent: str
    priority: int = 1  # 1 = highest
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed
    result: AgentResponse | None = None


class ConflictResolution(BaseModel):
    """Record of a conflict detected between agent outputs and how it was resolved."""

    agents_involved: list[str]
    conflict_description: str
    resolution: str
    confidence: float


class CoordinatorResponse(BaseModel):
    """Full response from the coordinator after processing a query."""

    query: str
    subtasks: list[SubTask]
    agent_responses: list[AgentResponse]
    synthesis: str
    conflicts: list[ConflictResolution] = Field(default_factory=list)
    total_processing_time_s: float = 0.0


class AgentCapability(str, Enum):
    """High-level capability categories used for routing."""

    LITERATURE = "literature"
    SAFETY = "safety"
    CHEMISTRY = "chemistry"
    REGULATORY = "regulatory"


# Keyword-to-capability mapping for routing
_ROUTING_KEYWORDS: dict[AgentCapability, list[str]] = {
    AgentCapability.LITERATURE: [
        "paper",
        "study",
        "publication",
        "literature",
        "research",
        "review",
        "evidence",
        "clinical trial",
        "meta-analysis",
        "pubmed",
        "journal",
        "citation",
        "abstract",
        "findings",
    ],
    AgentCapability.SAFETY: [
        "safety",
        "adverse",
        "toxicity",
        "side effect",
        "risk",
        "pharmacovigilance",
        "FAERS",
        "signal",
        "hepatotoxicity",
        "cardiac",
        "QT",
        "rhabdomyolysis",
        "contraindication",
        "warning",
        "black box",
        "REMS",
    ],
    AgentCapability.CHEMISTRY: [
        "molecule",
        "SMILES",
        "structure",
        "drug-like",
        "Lipinski",
        "ADMET",
        "LogP",
        "molecular weight",
        "scaffold",
        "SAR",
        "modification",
        "selectivity",
        "potency",
        "solubility",
        "permeability",
        "chemical",
        "compound",
        "pharmacophore",
    ],
    AgentCapability.REGULATORY: [
        "regulatory",
        "FDA",
        "EMA",
        "NDA",
        "BLA",
        "IND",
        "approval",
        "pathway",
        "submission",
        "clinical development",
        "phase",
        "timeline",
        "guidance",
        "510(k)",
        "505(b)(2)",
        "expedited",
        "breakthrough",
        "fast track",
        "priority review",
    ],
}


class AgentCoordinator:
    """Orchestrates multiple specialised agents to answer complex pharma queries.

    The coordinator:
    1. Decomposes a complex query into sub-tasks
    2. Routes each sub-task to the appropriate agent(s)
    3. Manages dependencies between sub-tasks
    4. Aggregates results and resolves conflicts
    5. Produces a unified synthesis
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()
        self._agents: dict[str, BaseAgent] = {
            "LiteratureAgent": LiteratureAgent(client=self._client),
            "SafetyAgent": SafetyAgent(client=self._client),
            "ChemistryAgent": ChemistryAgent(client=self._client),
            "RegulatoryAgent": RegulatoryAgent(client=self._client),
        }
        self._log = logger.bind(component="coordinator")

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Return the registry of available agents."""
        return self._agents

    def list_agents(self) -> list[dict[str, Any]]:
        """Return capabilities summary for all registered agents."""
        return [agent.get_capabilities() for agent in self._agents.values()]

    # -- main entry point ---------------------------------------------------

    async def process_query(
        self,
        query: str,
        *,
        callback: Any | None = None,
    ) -> CoordinatorResponse:
        """Process a complex research query through the multi-agent pipeline.

        Args:
            query: The user's research question.
            callback: Optional async callable for progress updates.
                      Signature: ``async def cb(event: str, data: dict) -> None``
        """
        start = time.time()
        self._log.info("coordinator.query.start", query=query[:120])

        # Step 1 -- Decompose into sub-tasks
        subtasks = self._decompose_query(query)
        if callback:
            await callback("decomposed", {"subtasks": [s.model_dump() for s in subtasks]})

        # Step 2 -- Execute sub-tasks (respecting dependencies)
        agent_responses = await self._execute_subtasks(subtasks, callback=callback)

        # Step 3 -- Detect and resolve conflicts
        conflicts = self._detect_conflicts(agent_responses)

        # Step 4 -- Synthesise results
        synthesis = self._synthesise(query, agent_responses, conflicts)

        elapsed = time.time() - start
        self._log.info("coordinator.query.complete", elapsed=round(elapsed, 2))

        return CoordinatorResponse(
            query=query,
            subtasks=subtasks,
            agent_responses=agent_responses,
            synthesis=synthesis,
            conflicts=conflicts,
            total_processing_time_s=round(elapsed, 3),
        )

    # -- task decomposition -------------------------------------------------

    def _decompose_query(self, query: str) -> list[SubTask]:
        """Break a complex query into sub-tasks and assign to agents.

        Uses keyword matching for deterministic routing.  For ambiguous
        queries, all relevant agents are engaged.
        """
        self._log.info("coordinator.decompose", query=query[:80])

        # Score each capability against the query
        scores: dict[AgentCapability, int] = {}
        query_lower = query.lower()
        for cap, keywords in _ROUTING_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                scores[cap] = score

        # If nothing matched, engage all agents
        if not scores:
            scores = {cap: 1 for cap in AgentCapability}

        # Map capabilities to agent names
        cap_to_agent: dict[AgentCapability, str] = {
            AgentCapability.LITERATURE: "LiteratureAgent",
            AgentCapability.SAFETY: "SafetyAgent",
            AgentCapability.CHEMISTRY: "ChemistryAgent",
            AgentCapability.REGULATORY: "RegulatoryAgent",
        }

        subtasks: list[SubTask] = []
        for idx, (cap, score) in enumerate(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ):
            agent_name = cap_to_agent[cap]
            description = self._generate_subtask_description(cap, query)
            subtasks.append(
                SubTask(
                    id=f"task-{idx + 1}",
                    description=description,
                    assigned_agent=agent_name,
                    priority=idx + 1,
                )
            )

        # Add dependency: safety should see literature results when both present
        agent_names_in_tasks = {st.assigned_agent for st in subtasks}
        if "SafetyAgent" in agent_names_in_tasks and "LiteratureAgent" in agent_names_in_tasks:
            safety_task = next(st for st in subtasks if st.assigned_agent == "SafetyAgent")
            lit_task = next(st for st in subtasks if st.assigned_agent == "LiteratureAgent")
            safety_task.dependencies.append(lit_task.id)

        self._log.info("coordinator.decompose.done", subtask_count=len(subtasks))
        return subtasks

    @staticmethod
    def _generate_subtask_description(cap: AgentCapability, query: str) -> str:
        """Create a focused sub-task prompt for a specific capability area."""
        templates: dict[AgentCapability, str] = {
            AgentCapability.LITERATURE: (
                f"Search and review the scientific literature relevant to the following "
                f"research question. Identify key publications, summarise evidence, and note "
                f"gaps in the literature:\n\n{query}"
            ),
            AgentCapability.SAFETY: (
                f"Perform a comprehensive safety assessment related to the following query. "
                f"Analyse adverse event data, compute safety metrics, detect red flags, and "
                f"provide a risk-benefit evaluation:\n\n{query}"
            ),
            AgentCapability.CHEMISTRY: (
                f"Analyse the medicinal chemistry aspects of the following query. Evaluate "
                f"molecular properties, drug-likeness, ADMET predictions, and suggest "
                f"structural optimisations:\n\n{query}"
            ),
            AgentCapability.REGULATORY: (
                f"Provide regulatory intelligence for the following query. Determine the "
                f"appropriate regulatory pathway, required studies, timeline estimates, and "
                f"competitive landscape:\n\n{query}"
            ),
        }
        return templates[cap]

    # -- task execution -----------------------------------------------------

    async def _execute_subtasks(
        self,
        subtasks: list[SubTask],
        *,
        callback: Any | None = None,
    ) -> list[AgentResponse]:
        """Execute sub-tasks, respecting dependency ordering.

        Independent tasks run concurrently via ``asyncio.gather``.
        """
        completed: dict[str, AgentResponse] = {}
        responses: list[AgentResponse] = []

        # Group by dependency level
        remaining = list(subtasks)

        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = [st for st in remaining if all(dep in completed for dep in st.dependencies)]
            if not ready:
                self._log.error("coordinator.deadlock", remaining=[s.id for s in remaining])
                break

            # Execute ready tasks concurrently
            async def _run_one(subtask: SubTask) -> AgentResponse:
                subtask.status = "running"
                if callback:
                    await callback(
                        "agent_start",
                        {
                            "task_id": subtask.id,
                            "agent": subtask.assigned_agent,
                        },
                    )

                agent = self._agents[subtask.assigned_agent]

                # Enrich prompt with dependency results
                prompt = subtask.description
                for dep_id in subtask.dependencies:
                    dep_result = completed.get(dep_id)
                    if dep_result:
                        prompt += (
                            f"\n\n--- Context from {dep_result.agent_name} ---\n"
                            f"{dep_result.text[:2000]}"
                        )

                try:
                    result = await agent.run(prompt)
                    subtask.status = "completed"
                    subtask.result = result
                except Exception as exc:
                    self._log.error(
                        "coordinator.subtask.failed",
                        task=subtask.id,
                        agent=subtask.assigned_agent,
                        error=str(exc),
                    )
                    subtask.status = "failed"
                    result = AgentResponse(
                        agent_name=subtask.assigned_agent,
                        agent_role="(failed)",
                        text=f"Agent failed: {exc}",
                    )
                    subtask.result = result

                if callback:
                    await callback(
                        "agent_complete",
                        {
                            "task_id": subtask.id,
                            "agent": subtask.assigned_agent,
                            "status": subtask.status,
                        },
                    )
                return result

            results = await asyncio.gather(*[_run_one(st) for st in ready])

            for st, res in zip(ready, results):
                completed[st.id] = res
                responses.append(res)
                remaining.remove(st)

        return responses

    # -- conflict detection -------------------------------------------------

    def _detect_conflicts(self, responses: list[AgentResponse]) -> list[ConflictResolution]:
        """Detect potential conflicts between agent outputs.

        Looks for contradictory safety assessments, differing timeline
        estimates, or conflicting recommendations.
        """
        conflicts: list[ConflictResolution] = []

        # Check safety vs. chemistry conflicts
        safety_resp = next((r for r in responses if r.agent_name == "SafetyAgent"), None)
        chem_resp = next((r for r in responses if r.agent_name == "ChemistryAgent"), None)

        if safety_resp and chem_resp:
            safety_text = safety_resp.text.lower()
            chem_text = chem_resp.text.lower()

            # Check if chemistry says drug-like but safety flags critical issues
            if "drug-like" in chem_text and "critical" in safety_text:
                conflicts.append(
                    ConflictResolution(
                        agents_involved=["ChemistryAgent", "SafetyAgent"],
                        conflict_description=(
                            "ChemistryAgent assessed the compound as drug-like, but SafetyAgent "
                            "identified critical safety concerns."
                        ),
                        resolution=(
                            "Safety concerns take precedence. The compound may be drug-like "
                            "from a physicochemical perspective but requires safety-driven "
                            "structural modifications before advancement."
                        ),
                        confidence=0.85,
                    )
                )

        # Check regulatory vs. literature timeline conflicts
        reg_resp = next((r for r in responses if r.agent_name == "RegulatoryAgent"), None)
        lit_resp = next((r for r in responses if r.agent_name == "LiteratureAgent"), None)

        if reg_resp and lit_resp:
            reg_data = reg_resp.structured_data
            if reg_data.get("overall_evidence_quality") == "LOW" and reg_resp.text:
                conflicts.append(
                    ConflictResolution(
                        agents_involved=["RegulatoryAgent", "LiteratureAgent"],
                        conflict_description=(
                            "Regulatory pathway assumes standard evidence package, but "
                            "literature review indicates limited existing evidence base."
                        ),
                        resolution=(
                            "Additional clinical studies may be required beyond the standard "
                            "regulatory package. Timeline should be adjusted upward by 12-24 months."
                        ),
                        confidence=0.70,
                    )
                )

        return conflicts

    # -- synthesis ----------------------------------------------------------

    def _synthesise(
        self,
        query: str,
        responses: list[AgentResponse],
        conflicts: list[ConflictResolution],
    ) -> str:
        """Produce a unified synthesis from all agent responses.

        Uses a deterministic template approach rather than another LLM call
        to keep latency low and avoid additional API costs.
        """
        sections: list[str] = []
        sections.append(f"# Integrated Analysis\n\n**Query:** {query}\n")

        for resp in responses:
            sections.append(f"## {resp.agent_name} ({resp.agent_role})")
            # Truncate very long responses for synthesis
            text = resp.text[:3000] if len(resp.text) > 3000 else resp.text
            sections.append(text)
            if resp.tools_used:
                sections.append(f"\n*Tools used: {', '.join(resp.tools_used)}*")
            sections.append(f"*Confidence: {resp.confidence:.0%}*\n")

        if conflicts:
            sections.append("## Conflicts & Resolutions")
            for c in conflicts:
                sections.append(
                    f"- **{' vs '.join(c.agents_involved)}:** {c.conflict_description}\n"
                    f"  - *Resolution:* {c.resolution} (confidence: {c.confidence:.0%})"
                )

        sections.append("## Summary")
        agent_names = [r.agent_name for r in responses]
        avg_confidence = sum(r.confidence for r in responses) / len(responses) if responses else 0
        sections.append(
            f"This analysis integrated inputs from {len(responses)} specialised agents "
            f"({', '.join(agent_names)}). Overall confidence: {avg_confidence:.0%}. "
            f"{len(conflicts)} conflict(s) were identified and resolved."
        )

        return "\n\n".join(sections)

    # -- single agent query (for simple routing) ----------------------------

    async def query_single_agent(self, agent_name: str, message: str) -> AgentResponse:
        """Send a message to a single named agent."""
        if agent_name not in self._agents:
            raise ValueError(
                f"Unknown agent '{agent_name}'. Available: {list(self._agents.keys())}"
            )
        return await self._agents[agent_name].run(message)

    def route_to_best_agent(self, query: str) -> str:
        """Determine the single best agent for a query (for simple routing)."""
        query_lower = query.lower()
        best_cap: AgentCapability | None = None
        best_score = 0

        for cap, keywords in _ROUTING_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > best_score:
                best_score = score
                best_cap = cap

        cap_to_agent = {
            AgentCapability.LITERATURE: "LiteratureAgent",
            AgentCapability.SAFETY: "SafetyAgent",
            AgentCapability.CHEMISTRY: "ChemistryAgent",
            AgentCapability.REGULATORY: "RegulatoryAgent",
        }

        if best_cap:
            return cap_to_agent[best_cap]
        return "LiteratureAgent"  # Default fallback
