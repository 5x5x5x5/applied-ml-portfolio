"""Regulatory intelligence agent -- provides regulatory pathway and strategy guidance."""

from __future__ import annotations

import json
from typing import Any

import structlog

from pharma_agents.agents.base_agent import BaseAgent, ToolDefinition

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regulatory knowledge base (deterministic, no external API needed)
# ---------------------------------------------------------------------------

REGULATORY_PATHWAYS = {
    "NDA": {
        "name": "New Drug Application (NDA)",
        "section": "505(b)(1)",
        "description": "Full new drug application with complete safety and efficacy data.",
        "typical_timeline_months": 60,
        "requirements": [
            "Phase I/II/III clinical trials",
            "CMC (Chemistry, Manufacturing, Controls) data",
            "Non-clinical pharmacology/toxicology",
            "Clinical pharmacology (PK/PD)",
            "Statistical analysis of pivotal trials",
        ],
    },
    "505(b)(2)": {
        "name": "505(b)(2) Application",
        "section": "505(b)(2)",
        "description": "Application relying partly on FDA's findings for a previously approved drug.",
        "typical_timeline_months": 36,
        "requirements": [
            "Bridge studies (bioequivalence or comparative PK)",
            "Literature references for safety/efficacy of reference drug",
            "CMC data for new formulation",
            "Any additional clinical studies for new indication/route",
        ],
    },
    "BLA": {
        "name": "Biologics License Application (BLA)",
        "section": "351(a)",
        "description": "Application for a new biological product.",
        "typical_timeline_months": 72,
        "requirements": [
            "Phase I/II/III clinical trials",
            "Manufacturing process characterisation",
            "Lot release specifications",
            "Immunogenicity assessment",
            "Biosimilar comparability (if applicable)",
        ],
    },
    "ANDA": {
        "name": "Abbreviated New Drug Application (ANDA)",
        "section": "505(j)",
        "description": "Generic drug application demonstrating bioequivalence to RLD.",
        "typical_timeline_months": 24,
        "requirements": [
            "Bioequivalence study vs. Reference Listed Drug",
            "CMC data",
            "Labeling consistent with RLD",
            "Patent certification (Paragraph IV if applicable)",
        ],
    },
}

DRUG_CLASS_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "small_molecule": {
        "pathway": "NDA",
        "additional_studies": [
            "QT/QTc study (ICH E14)",
            "Drug-drug interaction studies",
            "Renal/hepatic impairment studies",
            "Carcinogenicity (2-year rodent)",
            "Reproductive toxicology",
        ],
    },
    "biologic": {
        "pathway": "BLA",
        "additional_studies": [
            "Immunogenicity program",
            "Comparability studies (process changes)",
            "Post-marketing safety study (REMS if needed)",
        ],
    },
    "gene_therapy": {
        "pathway": "BLA",
        "additional_studies": [
            "Long-term follow-up (15 years recommended)",
            "Biodistribution study",
            "Shedding study",
            "Integration site analysis",
            "Germline transmission assessment",
        ],
    },
    "cell_therapy": {
        "pathway": "BLA",
        "additional_studies": [
            "Potency assay development",
            "Tumorigenicity assessment",
            "Donor eligibility/screening",
            "Manufacturing consistency (Process Performance Qualification)",
        ],
    },
    "combination_product": {
        "pathway": "NDA",
        "additional_studies": [
            "Device component testing (biocompatibility, EMC)",
            "Human factors study",
            "Combination product stability",
            "Cross-labeling alignment",
        ],
    },
}


class RegulatoryAgent(BaseAgent):
    """Agent specialised in regulatory intelligence and strategy.

    Capabilities:
    - Determine the appropriate regulatory pathway (NDA, BLA, 505(b)(2), ANDA)
    - Identify required studies based on drug class
    - Estimate development and approval timelines
    - Analyse competitive landscape and regulatory precedents
    """

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "RegulatoryAgent"

    @property
    def role(self) -> str:
        return "Regulatory Intelligence & Strategy Specialist"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Dr. Evelyn Marsh, a senior regulatory affairs strategist with 22 years of "
            "experience navigating FDA, EMA, and PMDA submissions. You were formerly a Review "
            "Division Director at the FDA CDER. You have an MD and a JD.\n\n"
            "Your responsibilities:\n"
            "- Determine the optimal regulatory pathway for a product\n"
            "- Identify all required studies and data packages\n"
            "- Estimate realistic timelines and milestones\n"
            "- Analyse regulatory precedents and competitive landscape\n"
            "- Identify potential regulatory risks and mitigation strategies\n\n"
            "Communication style: authoritative, strategic, regulatory-jargon-fluent but "
            "explains implications clearly. Always references relevant FDA guidance documents "
            "and precedent decisions.\n\n"
            "Structure output as: Regulatory Pathway Recommendation, Required Studies, "
            "Timeline Estimate, Competitive Landscape, Regulatory Risks, Strategic "
            "Recommendations. Include a JSON block."
        )

    # -- tools --------------------------------------------------------------

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="determine_pathway",
                description=(
                    "Determine the appropriate FDA regulatory pathway (NDA, BLA, 505(b)(2), "
                    "ANDA) based on drug type, novelty, and existing approvals."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_class": {
                            "type": "string",
                            "enum": list(DRUG_CLASS_REQUIREMENTS.keys()),
                            "description": "Classification of the drug product.",
                        },
                        "has_reference_drug": {
                            "type": "boolean",
                            "description": "Whether a reference listed drug exists.",
                        },
                        "novel_mechanism": {
                            "type": "boolean",
                            "description": "Whether this is a first-in-class mechanism.",
                        },
                    },
                    "required": ["drug_class"],
                },
            ),
            ToolDefinition(
                name="get_required_studies",
                description=(
                    "List all regulatory studies required for a given drug class and pathway, "
                    "including non-clinical, clinical, and CMC requirements."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_class": {
                            "type": "string",
                            "enum": list(DRUG_CLASS_REQUIREMENTS.keys()),
                        },
                        "pathway": {
                            "type": "string",
                            "enum": list(REGULATORY_PATHWAYS.keys()),
                        },
                    },
                    "required": ["drug_class", "pathway"],
                },
            ),
            ToolDefinition(
                name="estimate_timeline",
                description=(
                    "Estimate the development and regulatory review timeline, broken into "
                    "phases: IND, Phase I-III, NDA/BLA submission, review, approval."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_class": {
                            "type": "string",
                            "enum": list(DRUG_CLASS_REQUIREMENTS.keys()),
                        },
                        "pathway": {
                            "type": "string",
                            "enum": list(REGULATORY_PATHWAYS.keys()),
                        },
                        "expedited_programs": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "fast_track",
                                    "breakthrough_therapy",
                                    "accelerated_approval",
                                    "priority_review",
                                    "rtor",
                                ],
                            },
                            "description": "Any FDA expedited programs that may apply.",
                        },
                    },
                    "required": ["drug_class", "pathway"],
                },
            ),
            ToolDefinition(
                name="competitive_landscape",
                description=(
                    "Analyse the competitive regulatory landscape: approved drugs in the "
                    "same class, pending applications, and recent FDA advisory committee "
                    "decisions."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "therapeutic_area": {
                            "type": "string",
                            "description": "Therapeutic area (e.g. 'oncology', 'neurology').",
                        },
                        "mechanism": {
                            "type": "string",
                            "description": "Mechanism of action (e.g. 'PD-1 inhibitor').",
                        },
                    },
                    "required": ["therapeutic_area"],
                },
            ),
        ]

    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("regulatory.tool.execute", tool=tool_name)

        if tool_name == "determine_pathway":
            return json.dumps(
                self._determine_pathway(
                    drug_class=tool_input["drug_class"],
                    has_reference=tool_input.get("has_reference_drug", False),
                    novel_mechanism=tool_input.get("novel_mechanism", True),
                ),
                indent=2,
            )

        if tool_name == "get_required_studies":
            return json.dumps(
                self._get_required_studies(
                    drug_class=tool_input["drug_class"],
                    pathway=tool_input["pathway"],
                ),
                indent=2,
            )

        if tool_name == "estimate_timeline":
            return json.dumps(
                self._estimate_timeline(
                    drug_class=tool_input["drug_class"],
                    pathway=tool_input["pathway"],
                    expedited=tool_input.get("expedited_programs", []),
                ),
                indent=2,
            )

        if tool_name == "competitive_landscape":
            return json.dumps(
                self._competitive_landscape(
                    therapeutic_area=tool_input["therapeutic_area"],
                    mechanism=tool_input.get("mechanism"),
                ),
                indent=2,
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    # -- tool implementations -----------------------------------------------

    def _determine_pathway(
        self,
        drug_class: str,
        has_reference: bool,
        novel_mechanism: bool,
    ) -> dict[str, Any]:
        class_info = DRUG_CLASS_REQUIREMENTS.get(drug_class)
        if not class_info:
            return {"error": f"Unknown drug class: {drug_class}"}

        base_pathway = class_info["pathway"]

        # Adjust pathway based on reference drug availability
        if has_reference and not novel_mechanism and base_pathway == "NDA":
            recommended_pathway = "505(b)(2)"
        else:
            recommended_pathway = base_pathway

        pathway_info = REGULATORY_PATHWAYS[recommended_pathway]

        # Determine eligible expedited programs
        expedited_eligible: list[str] = []
        if novel_mechanism:
            expedited_eligible.extend(["fast_track", "breakthrough_therapy"])
        expedited_eligible.append("priority_review")
        if drug_class in ("gene_therapy", "cell_therapy"):
            expedited_eligible.append("rtor")  # Real-Time Oncology Review if applicable

        return {
            "drug_class": drug_class,
            "recommended_pathway": recommended_pathway,
            "pathway_details": pathway_info,
            "rationale": (
                f"Based on drug class '{drug_class}', "
                f"{'presence' if has_reference else 'absence'} of a reference drug, "
                f"and {'novel' if novel_mechanism else 'known'} mechanism of action, "
                f"the recommended pathway is {pathway_info['name']}."
            ),
            "expedited_programs_eligible": expedited_eligible,
            "pre_submission_meetings": [
                "Pre-IND meeting (Type B)",
                "End-of-Phase 2 meeting (Type B)",
                "Pre-NDA/BLA meeting (Type A)",
            ],
        }

    def _get_required_studies(self, drug_class: str, pathway: str) -> dict[str, Any]:
        class_info = DRUG_CLASS_REQUIREMENTS.get(drug_class, {})
        pathway_info = REGULATORY_PATHWAYS.get(pathway, {})

        if not class_info or not pathway_info:
            return {"error": "Invalid drug class or pathway."}

        base_requirements = pathway_info.get("requirements", [])
        additional = class_info.get("additional_studies", [])

        nonclinical = [
            "In vitro pharmacology (primary & secondary)",
            "Single-dose toxicology (2 species)",
            "Repeat-dose toxicology (2 species, GLP)",
            "Safety pharmacology core battery (CV, CNS, respiratory)",
            "Genotoxicity battery (Ames, in vitro/in vivo)",
        ]
        if drug_class == "small_molecule":
            nonclinical.append("Carcinogenicity (2-year rodent study)")
            nonclinical.append("Reproductive & developmental toxicology (Segments I-III)")

        clinical = [
            "Phase I: First-in-Human, dose-escalation, SAD/MAD, PK/PD",
            "Phase II: Proof-of-concept, dose-finding, preliminary efficacy",
            "Phase III: Pivotal efficacy trials (2 adequate & well-controlled)",
        ]

        cmc = [
            "Drug substance characterisation and specifications",
            "Manufacturing process description and validation",
            "Stability studies (ICH Q1A/Q1B conditions)",
            "Container closure system qualification",
        ]

        return {
            "drug_class": drug_class,
            "pathway": pathway,
            "regulatory_requirements": base_requirements,
            "nonclinical_studies": nonclinical,
            "clinical_studies": clinical,
            "cmc_requirements": cmc,
            "additional_class_specific": additional,
            "estimated_total_studies": (
                len(nonclinical) + len(clinical) + len(cmc) + len(additional)
            ),
        }

    def _estimate_timeline(
        self,
        drug_class: str,
        pathway: str,
        expedited: list[str],
    ) -> dict[str, Any]:
        pathway_info = REGULATORY_PATHWAYS.get(pathway, {})
        base_months = pathway_info.get("typical_timeline_months", 60)

        # Phase-by-phase breakdown
        phases = {
            "IND_preparation": {"months": 6, "description": "IND-enabling studies + submission"},
            "Phase_I": {"months": 12, "description": "First-in-human, dose escalation"},
            "Phase_II": {"months": 18, "description": "Proof-of-concept, dose-finding"},
            "Phase_III": {"months": 24, "description": "Pivotal efficacy trials"},
            "NDA_BLA_preparation": {"months": 6, "description": "Dossier compilation"},
            "FDA_review": {
                "months": 12,
                "description": "Standard 10-month review + 2-month filing",
            },
        }

        # Adjust for expedited programs
        time_savings_months = 0
        expedited_details: list[dict[str, Any]] = []

        if "fast_track" in expedited:
            time_savings_months += 3
            expedited_details.append(
                {
                    "program": "Fast Track",
                    "benefit": "Rolling review, more frequent FDA interactions",
                    "time_saved_months": 3,
                }
            )
        if "breakthrough_therapy" in expedited:
            time_savings_months += 6
            expedited_details.append(
                {
                    "program": "Breakthrough Therapy",
                    "benefit": "Intensive FDA guidance, organizational commitment, rolling review",
                    "time_saved_months": 6,
                }
            )
        if "accelerated_approval" in expedited:
            time_savings_months += 12
            phases["Phase_III"]["months"] = 12
            expedited_details.append(
                {
                    "program": "Accelerated Approval",
                    "benefit": "Approval based on surrogate endpoint, confirmatory trial post-approval",
                    "time_saved_months": 12,
                }
            )
        if "priority_review" in expedited:
            phases["FDA_review"]["months"] = 8
            time_savings_months += 4
            expedited_details.append(
                {
                    "program": "Priority Review",
                    "benefit": "6-month review instead of 10-month",
                    "time_saved_months": 4,
                }
            )

        total_months = sum(p["months"] for p in phases.values())

        return {
            "drug_class": drug_class,
            "pathway": pathway,
            "phases": phases,
            "total_months": total_months,
            "total_years": round(total_months / 12, 1),
            "expedited_programs": expedited_details,
            "time_savings_months": time_savings_months,
            "key_milestones": [
                {"milestone": "IND filing", "month": phases["IND_preparation"]["months"]},
                {
                    "milestone": "End of Phase II meeting",
                    "month": (
                        phases["IND_preparation"]["months"]
                        + phases["Phase_I"]["months"]
                        + phases["Phase_II"]["months"]
                    ),
                },
                {
                    "milestone": "NDA/BLA submission",
                    "month": total_months - phases["FDA_review"]["months"],
                },
                {"milestone": "Target approval", "month": total_months},
            ],
            "risks_to_timeline": [
                "Clinical hold (adds 3-12 months)",
                "Refuse-to-file (adds 6 months)",
                "Complete Response Letter (adds 12-24 months)",
                "Advisory committee review (adds 2-3 months)",
                "Manufacturing delays / site inspection issues",
            ],
        }

    def _competitive_landscape(
        self,
        therapeutic_area: str,
        mechanism: str | None = None,
    ) -> dict[str, Any]:
        """Simulated competitive landscape data."""
        # A realistic but simulated landscape for common therapeutic areas
        landscapes: dict[str, dict[str, Any]] = {
            "oncology": {
                "approved_drugs": [
                    {"name": "Pembrolizumab", "mechanism": "PD-1 inhibitor", "year": 2014},
                    {"name": "Nivolumab", "mechanism": "PD-1 inhibitor", "year": 2014},
                    {"name": "Atezolizumab", "mechanism": "PD-L1 inhibitor", "year": 2016},
                    {"name": "Osimertinib", "mechanism": "EGFR T790M inhibitor", "year": 2015},
                    {"name": "Sotorasib", "mechanism": "KRAS G12C inhibitor", "year": 2021},
                ],
                "pending_applications": 12,
                "recent_approvals_12mo": 8,
                "market_size_bn": 200,
            },
            "neurology": {
                "approved_drugs": [
                    {"name": "Lecanemab", "mechanism": "Anti-amyloid beta", "year": 2023},
                    {"name": "Donanemab", "mechanism": "Anti-amyloid beta", "year": 2024},
                    {"name": "Erenumab", "mechanism": "CGRP receptor antagonist", "year": 2018},
                ],
                "pending_applications": 6,
                "recent_approvals_12mo": 3,
                "market_size_bn": 45,
            },
            "cardiology": {
                "approved_drugs": [
                    {"name": "Inclisiran", "mechanism": "PCSK9 siRNA", "year": 2021},
                    {"name": "Mavacamten", "mechanism": "Cardiac myosin inhibitor", "year": 2022},
                    {"name": "Empagliflozin", "mechanism": "SGLT2 inhibitor", "year": 2014},
                ],
                "pending_applications": 4,
                "recent_approvals_12mo": 2,
                "market_size_bn": 65,
            },
        }

        area_lower = therapeutic_area.lower()
        landscape = landscapes.get(
            area_lower,
            {
                "approved_drugs": [],
                "pending_applications": 0,
                "recent_approvals_12mo": 0,
                "market_size_bn": 0,
                "note": f"Limited data available for '{therapeutic_area}'.",
            },
        )

        # Filter by mechanism if provided
        if mechanism and "approved_drugs" in landscape:
            mechanism_lower = mechanism.lower()
            same_mechanism = [
                d
                for d in landscape["approved_drugs"]
                if mechanism_lower in d.get("mechanism", "").lower()
            ]
            landscape["same_mechanism_drugs"] = same_mechanism
            landscape["differentiation_needed"] = len(same_mechanism) > 0

        landscape["therapeutic_area"] = therapeutic_area
        landscape["regulatory_considerations"] = [
            "Review division workload and recent precedent decisions",
            "Advisory committee trends for this therapeutic area",
            "Patient population unmet need assessment",
            "Available endpoints and FDA-accepted surrogate endpoints",
        ]

        return landscape
