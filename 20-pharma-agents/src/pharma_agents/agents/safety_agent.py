"""Drug safety assessment agent -- analyses adverse events and computes safety metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import structlog

from pharma_agents.agents.base_agent import BaseAgent, ToolDefinition
from pharma_agents.tools.database_tool import DrugDatabaseTool

logger = structlog.get_logger(__name__)


class SafetyAgent(BaseAgent):
    """Agent specialised in pharmacovigilance and drug safety assessment.

    Capabilities:
    - Analyse adverse event (AE) data for a drug
    - Compute disproportionality metrics (PRR, ROR, EBGM)
    - Perform risk-benefit analysis
    - Detect red-flag safety signals
    - Generate structured safety summary reports
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._db = DrugDatabaseTool()

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "SafetyAgent"

    @property
    def role(self) -> str:
        return "Drug Safety & Pharmacovigilance Specialist"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Dr. Marcus Okafor, a senior pharmacovigilance scientist with 15 years "
            "of experience at the FDA and in biopharma safety departments. You hold an MD/PhD "
            "from the University of Pennsylvania.\n\n"
            "Your responsibilities:\n"
            "- Analyse adverse event databases for safety signals\n"
            "- Compute disproportionality metrics (PRR, ROR, EBGM)\n"
            "- Perform quantitative risk-benefit analyses\n"
            "- Identify red-flag safety signals that require urgent action\n"
            "- Generate structured safety summary reports\n\n"
            "Communication style: precise, evidence-driven, conservative. Always err on the "
            "side of patient safety. Flag any signal that crosses established thresholds. "
            "Clearly distinguish between statistical signals and confirmed risks.\n\n"
            "When reporting, structure output as: Executive Summary, Signal Detection Results, "
            "Risk-Benefit Assessment, Red Flags, Recommendations. Include a JSON block."
        )

    # -- tools --------------------------------------------------------------

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_adverse_events",
                description=(
                    "Retrieve adverse event reports for a drug from the FAERS-like database. "
                    "Returns event types, counts, seriousness, and outcomes."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "Generic or brand name of the drug.",
                        },
                        "serious_only": {
                            "type": "boolean",
                            "description": "If true, return only serious AEs.",
                        },
                    },
                    "required": ["drug_name"],
                },
            ),
            ToolDefinition(
                name="compute_safety_metrics",
                description=(
                    "Compute disproportionality metrics (PRR, ROR) for a drug-event pair "
                    "using a 2x2 contingency table."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                        "event_name": {
                            "type": "string",
                            "description": "The adverse event MedDRA preferred term.",
                        },
                    },
                    "required": ["drug_name", "event_name"],
                },
            ),
            ToolDefinition(
                name="risk_benefit_analysis",
                description=(
                    "Perform a quantitative risk-benefit analysis comparing efficacy "
                    "endpoints against safety endpoints for the drug."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                        "indication": {
                            "type": "string",
                            "description": "Target indication for the analysis.",
                        },
                    },
                    "required": ["drug_name", "indication"],
                },
            ),
            ToolDefinition(
                name="detect_red_flags",
                description=(
                    "Scan a drug's safety profile for red-flag signals such as hepatotoxicity, "
                    "QT prolongation, rhabdomyolysis, or anaphylaxis above threshold."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                    },
                    "required": ["drug_name"],
                },
            ),
        ]

    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("safety.tool.execute", tool=tool_name)

        if tool_name == "get_adverse_events":
            return json.dumps(
                self._get_adverse_events(
                    tool_input["drug_name"],
                    tool_input.get("serious_only", False),
                ),
                indent=2,
            )

        if tool_name == "compute_safety_metrics":
            return json.dumps(
                self._compute_safety_metrics(
                    tool_input["drug_name"],
                    tool_input["event_name"],
                ),
                indent=2,
            )

        if tool_name == "risk_benefit_analysis":
            return json.dumps(
                self._risk_benefit_analysis(
                    tool_input["drug_name"],
                    tool_input["indication"],
                ),
                indent=2,
            )

        if tool_name == "detect_red_flags":
            return json.dumps(
                self._detect_red_flags(tool_input["drug_name"]),
                indent=2,
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    # -- tool implementations -----------------------------------------------

    def _get_adverse_events(self, drug_name: str, serious_only: bool = False) -> dict[str, Any]:
        """Retrieve simulated adverse event data."""
        drug_info = self._db.lookup_drug(drug_name)
        if not drug_info:
            return {"error": f"Drug '{drug_name}' not found in database."}

        # Simulated FAERS-like adverse event data
        rng = np.random.default_rng(hash(drug_name.lower()) % 2**31)
        ae_terms = [
            ("Headache", False),
            ("Nausea", False),
            ("Dizziness", False),
            ("Fatigue", False),
            ("Rash", False),
            ("Diarrhea", False),
            ("Hepatotoxicity", True),
            ("QT prolongation", True),
            ("Anaphylaxis", True),
            ("Thrombocytopenia", True),
            ("Myalgia", False),
            ("Insomnia", False),
        ]

        events = []
        total_reports = int(rng.integers(500, 5000))
        for term, is_serious in ae_terms:
            if serious_only and not is_serious:
                continue
            count = int(rng.integers(5, total_reports // 3))
            events.append(
                {
                    "preferred_term": term,
                    "count": count,
                    "serious": is_serious,
                    "proportion": round(count / total_reports, 4),
                    "outcomes": {
                        "recovered": int(count * rng.uniform(0.5, 0.9)),
                        "not_recovered": int(count * rng.uniform(0.05, 0.2)),
                        "fatal": int(count * rng.uniform(0.0, 0.03)) if is_serious else 0,
                        "unknown": int(count * rng.uniform(0.05, 0.15)),
                    },
                }
            )

        return {
            "drug_name": drug_name,
            "total_reports": total_reports,
            "reporting_period": "2018-01-01 to 2025-12-31",
            "events": sorted(events, key=lambda e: e["count"], reverse=True),
        }

    def _compute_safety_metrics(self, drug_name: str, event_name: str) -> dict[str, Any]:
        """Compute PRR and ROR from a simulated 2x2 contingency table."""
        rng = np.random.default_rng(hash(f"{drug_name}:{event_name}".lower()) % 2**31)

        # Simulated 2x2 table: [drug+event, drug+no_event, no_drug+event, no_drug+no_event]
        a = int(rng.integers(20, 300))  # drug & event
        b = int(rng.integers(500, 5000))  # drug & no event
        c = int(rng.integers(100, 2000))  # no drug & event
        d = int(rng.integers(10000, 100000))  # no drug & no event

        # Proportional Reporting Ratio
        prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else 0.0
        # Reporting Odds Ratio
        ror = (a * d) / (b * c) if (b * c) > 0 else 0.0

        # 95% CI for ROR (Woolf's method)
        import math

        if a > 0 and b > 0 and c > 0 and d > 0:
            se_ln_ror = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            ror_lower = math.exp(math.log(ror) - 1.96 * se_ln_ror)
            ror_upper = math.exp(math.log(ror) + 1.96 * se_ln_ror)
        else:
            ror_lower = ror_upper = 0.0

        signal_detected = prr >= 2.0 and a >= 3 and ror_lower > 1.0

        return {
            "drug_name": drug_name,
            "event_name": event_name,
            "contingency_table": {"a": a, "b": b, "c": c, "d": d},
            "prr": round(prr, 3),
            "ror": round(ror, 3),
            "ror_95ci": [round(ror_lower, 3), round(ror_upper, 3)],
            "signal_detected": signal_detected,
            "interpretation": (
                f"PRR={prr:.2f}, ROR={ror:.2f} (95% CI: {ror_lower:.2f}-{ror_upper:.2f}). "
                + (
                    "SIGNAL DETECTED: meets threshold criteria."
                    if signal_detected
                    else "No signal detected at standard thresholds."
                )
            ),
        }

    def _risk_benefit_analysis(self, drug_name: str, indication: str) -> dict[str, Any]:
        """Simulated risk-benefit analysis."""
        rng = np.random.default_rng(hash(f"{drug_name}:{indication}".lower()) % 2**31)

        nnt = int(rng.integers(3, 25))  # Number Needed to Treat
        nnh = int(rng.integers(10, 200))  # Number Needed to Harm

        efficacy_score = round(rng.uniform(0.3, 0.95), 2)
        safety_score = round(rng.uniform(0.4, 0.95), 2)
        rb_ratio = round(nnh / nnt, 2)

        return {
            "drug_name": drug_name,
            "indication": indication,
            "efficacy": {
                "nnt": nnt,
                "response_rate": round(rng.uniform(0.3, 0.8), 2),
                "effect_size": round(rng.uniform(0.2, 1.5), 2),
                "score": efficacy_score,
            },
            "safety": {
                "nnh": nnh,
                "discontinuation_rate": round(rng.uniform(0.05, 0.25), 2),
                "serious_ae_rate": round(rng.uniform(0.01, 0.1), 3),
                "score": safety_score,
            },
            "risk_benefit_ratio": rb_ratio,
            "overall_assessment": (
                "FAVORABLE"
                if rb_ratio > 5
                else "ACCEPTABLE"
                if rb_ratio > 2
                else "MARGINAL"
                if rb_ratio > 1
                else "UNFAVORABLE"
            ),
            "recommendation": (
                f"Risk-benefit ratio of {rb_ratio} for {indication}. "
                f"NNT={nnt}, NNH={nnh}. "
                + (
                    "Benefits clearly outweigh risks."
                    if rb_ratio > 5
                    else "Benefits outweigh risks with appropriate monitoring."
                    if rb_ratio > 2
                    else "Close monitoring required; consider alternatives."
                    if rb_ratio > 1
                    else "Risks may outweigh benefits; alternative therapies preferred."
                )
            ),
        }

    def _detect_red_flags(self, drug_name: str) -> dict[str, Any]:
        """Scan for critical safety signals."""
        rng = np.random.default_rng(hash(drug_name.lower()) % 2**31)

        red_flag_categories = [
            {
                "category": "Hepatotoxicity",
                "markers": ["ALT > 3x ULN", "AST > 3x ULN", "Hy's Law cases"],
                "threshold": 0.35,
            },
            {
                "category": "Cardiac (QT prolongation)",
                "markers": ["QTcF > 500ms", "Torsades de Pointes", "Sudden cardiac death"],
                "threshold": 0.25,
            },
            {
                "category": "Rhabdomyolysis",
                "markers": ["CK > 10x ULN", "Myoglobinuria", "Acute renal failure"],
                "threshold": 0.20,
            },
            {
                "category": "Severe hypersensitivity",
                "markers": ["Anaphylaxis", "Stevens-Johnson syndrome", "DRESS syndrome"],
                "threshold": 0.30,
            },
            {
                "category": "Suicidality",
                "markers": ["Suicidal ideation", "Suicide attempt", "Completed suicide"],
                "threshold": 0.15,
            },
        ]

        flags: list[dict[str, Any]] = []
        for cat in red_flag_categories:
            score = float(rng.uniform(0.0, 0.6))
            detected = score > cat["threshold"]
            if detected:
                flags.append(
                    {
                        "category": cat["category"],
                        "severity": "HIGH" if score > 0.5 else "MODERATE",
                        "score": round(score, 3),
                        "threshold": cat["threshold"],
                        "markers_detected": [m for m in cat["markers"] if rng.random() > 0.4],
                        "action_required": (
                            "URGENT: Requires immediate safety review and possible REMS."
                            if score > 0.5
                            else "Monitor closely; consider additional safety studies."
                        ),
                    }
                )

        return {
            "drug_name": drug_name,
            "red_flags_detected": len(flags),
            "overall_safety_concern": (
                "CRITICAL"
                if any(f["severity"] == "HIGH" for f in flags)
                else "ELEVATED"
                if flags
                else "LOW"
            ),
            "flags": flags,
            "recommendation": (
                "Immediate safety committee review recommended."
                if any(f["severity"] == "HIGH" for f in flags)
                else "Enhanced pharmacovigilance monitoring recommended."
                if flags
                else "Standard pharmacovigilance monitoring sufficient."
            ),
        }
