"""Medicinal chemistry agent -- analyses molecular properties and drug-likeness."""

from __future__ import annotations

import json
from typing import Any

import structlog

from pharma_agents.agents.base_agent import BaseAgent, ToolDefinition
from pharma_agents.tools.molecule_tool import MoleculeTool

logger = structlog.get_logger(__name__)


class ChemistryAgent(BaseAgent):
    """Agent specialised in medicinal chemistry and molecular analysis.

    Capabilities:
    - Analyse molecular properties from SMILES notation
    - Assess drug-likeness (Lipinski Rule of Five, Veber rules)
    - Predict ADMET properties
    - Suggest structural modifications for optimisation
    - Compare molecules to known drug classes
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mol_tool = MoleculeTool()

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "ChemistryAgent"

    @property
    def role(self) -> str:
        return "Medicinal Chemistry & Molecular Analysis Specialist"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Dr. Anika Patel, a principal medicinal chemist with 20 years of experience "
            "in drug design and optimisation. You have a PhD in organic chemistry from MIT and "
            "led multiple programs from hit identification to clinical candidates.\n\n"
            "Your responsibilities:\n"
            "- Analyse molecular structures provided as SMILES strings\n"
            "- Evaluate drug-likeness using Lipinski, Veber, and related rules\n"
            "- Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)\n"
            "- Suggest structural modifications to improve properties\n"
            "- Compare compounds to known drug scaffolds and pharmacophores\n\n"
            "Communication style: technically precise, uses proper chemical nomenclature, "
            "provides actionable SAR (structure-activity relationship) insights. Always "
            "considers synthetic feasibility of proposed modifications.\n\n"
            "Structure your output as: Molecular Profile, Drug-Likeness Assessment, "
            "ADMET Predictions, Structural Recommendations, Comparison to Known Drugs. "
            "Include a JSON block with computed properties."
        )

    # -- tools --------------------------------------------------------------

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="analyze_molecule",
                description=(
                    "Analyse a molecule from its SMILES string. Returns molecular weight, "
                    "LogP, HBD/HBA, TPSA, rotatable bonds, and other descriptors."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "smiles": {
                            "type": "string",
                            "description": "SMILES notation of the molecule.",
                        },
                    },
                    "required": ["smiles"],
                },
            ),
            ToolDefinition(
                name="assess_drug_likeness",
                description=(
                    "Evaluate drug-likeness of a molecule using Lipinski Rule of Five, "
                    "Veber rules, Ghose filter, and lead-likeness criteria."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string"},
                    },
                    "required": ["smiles"],
                },
            ),
            ToolDefinition(
                name="predict_admet",
                description=(
                    "Predict ADMET properties including oral bioavailability, CYP inhibition, "
                    "blood-brain barrier permeability, hERG liability, and hepatotoxicity risk."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string"},
                    },
                    "required": ["smiles"],
                },
            ),
            ToolDefinition(
                name="suggest_modifications",
                description=(
                    "Suggest structural modifications to improve drug-like properties, "
                    "considering synthetic feasibility and SAR."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string"},
                        "optimise_for": {
                            "type": "string",
                            "enum": [
                                "potency",
                                "selectivity",
                                "solubility",
                                "metabolic_stability",
                                "permeability",
                                "safety",
                            ],
                            "description": "Property to optimise for.",
                        },
                    },
                    "required": ["smiles", "optimise_for"],
                },
            ),
            ToolDefinition(
                name="compare_to_known_drugs",
                description=(
                    "Compare the molecule to known approved drugs in the same structural "
                    "or pharmacological class."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string"},
                        "drug_class": {
                            "type": "string",
                            "description": "Pharmacological class (e.g., 'kinase_inhibitor').",
                        },
                    },
                    "required": ["smiles"],
                },
            ),
        ]

    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("chemistry.tool.execute", tool=tool_name)

        if tool_name == "analyze_molecule":
            result = self._mol_tool.analyze(tool_input["smiles"])
            return json.dumps(result, indent=2)

        if tool_name == "assess_drug_likeness":
            result = self._mol_tool.assess_drug_likeness(tool_input["smiles"])
            return json.dumps(result, indent=2)

        if tool_name == "predict_admet":
            result = self._mol_tool.predict_admet(tool_input["smiles"])
            return json.dumps(result, indent=2)

        if tool_name == "suggest_modifications":
            result = self._mol_tool.suggest_modifications(
                tool_input["smiles"],
                tool_input["optimise_for"],
            )
            return json.dumps(result, indent=2)

        if tool_name == "compare_to_known_drugs":
            result = self._mol_tool.compare_to_known_drugs(
                tool_input["smiles"],
                tool_input.get("drug_class"),
            )
            return json.dumps(result, indent=2)

        raise ValueError(f"Unknown tool: {tool_name}")
