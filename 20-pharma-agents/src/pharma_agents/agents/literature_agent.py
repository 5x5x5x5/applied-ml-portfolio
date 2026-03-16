"""Literature review agent -- searches, summarises, and cross-references research papers."""

from __future__ import annotations

import json
from typing import Any

import structlog

from pharma_agents.agents.base_agent import BaseAgent, ToolDefinition
from pharma_agents.tools.pubmed_tool import PubMedTool

logger = structlog.get_logger(__name__)


class LiteratureAgent(BaseAgent):
    """Agent specialised in pharmaceutical literature review.

    Capabilities:
    - Search and summarise research papers (simulated PubMed)
    - Extract key findings, methods, and conclusions
    - Cross-reference citations between papers
    - Generate structured literature reviews
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pubmed = PubMedTool()

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "LiteratureAgent"

    @property
    def role(self) -> str:
        return "Pharmaceutical Literature Review Specialist"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Dr. Lena Reeves, a meticulous pharmaceutical literature review specialist "
            "with 18 years of experience in systematic reviews and meta-analyses. You have a PhD "
            "in Pharmacology from Johns Hopkins and an MLIS in information science.\n\n"
            "Your responsibilities:\n"
            "- Search the biomedical literature for relevant papers on a given topic\n"
            "- Summarise key findings, methods, and conclusions from each paper\n"
            "- Cross-reference citations to map the evidence landscape\n"
            "- Produce structured literature reviews with evidence grading\n\n"
            "Communication style: scholarly yet accessible, always cites sources, highlights "
            "evidence quality (RCT > cohort > case series), and notes gaps in the literature.\n\n"
            "When responding, always structure your output with clear sections: "
            "Background, Search Strategy, Key Findings, Evidence Quality, Gaps & Limitations, "
            "and Conclusions. Include a JSON block with structured data."
        )

    # -- tools --------------------------------------------------------------

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="pubmed_search",
                description=(
                    "Search the PubMed database for biomedical research articles. "
                    "Returns titles, abstracts, authors, journal, year, and citation count."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g. drug name, disease, mechanism).",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum papers to return (default 10).",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="get_paper_details",
                description=(
                    "Retrieve full details of a paper by its PMID, including abstract, "
                    "methods summary, key findings, and references."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "pmid": {
                            "type": "string",
                            "description": "The PubMed ID of the paper.",
                        },
                    },
                    "required": ["pmid"],
                },
            ),
            ToolDefinition(
                name="cross_reference",
                description=(
                    "Find papers that cite a given PMID or are cited by it, "
                    "mapping the citation network."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "pmid": {
                            "type": "string",
                            "description": "The PubMed ID to cross-reference.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["cited_by", "references"],
                            "description": "Direction of citation lookup.",
                        },
                    },
                    "required": ["pmid"],
                },
            ),
            ToolDefinition(
                name="summarize_evidence",
                description=(
                    "Produce a structured evidence summary from a list of PMIDs, "
                    "grading each paper's evidence level."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "pmids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of PubMed IDs to summarise.",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Topic focus for the summary.",
                        },
                    },
                    "required": ["pmids", "focus"],
                },
            ),
        ]

    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("literature.tool.execute", tool=tool_name)
        if tool_name == "pubmed_search":
            results = self._pubmed.search(
                query=tool_input["query"],
                max_results=tool_input.get("max_results", 10),
            )
            return json.dumps(results, indent=2)

        if tool_name == "get_paper_details":
            details = self._pubmed.get_paper_details(pmid=tool_input["pmid"])
            return json.dumps(details, indent=2)

        if tool_name == "cross_reference":
            refs = self._pubmed.cross_reference(
                pmid=tool_input["pmid"],
                direction=tool_input.get("direction", "cited_by"),
            )
            return json.dumps(refs, indent=2)

        if tool_name == "summarize_evidence":
            summary = self._pubmed.summarize_evidence(
                pmids=tool_input["pmids"],
                focus=tool_input["focus"],
            )
            return json.dumps(summary, indent=2)

        raise ValueError(f"Unknown tool: {tool_name}")
