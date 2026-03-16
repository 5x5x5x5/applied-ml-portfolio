"""Safety guardrails for pharmaceutical AI assistant.

Implements medical disclaimer injection, off-topic detection, confidence scoring,
harmful content filtering, and professional consultation triggers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class GuardrailCategory(str, Enum):
    """Categories of guardrail checks."""

    SAFE = "safe"
    OFF_TOPIC = "off_topic"
    HARMFUL = "harmful"
    LOW_CONFIDENCE = "low_confidence"
    CONSULT_PROFESSIONAL = "consult_professional"
    EMERGENCY = "emergency"


@dataclass
class GuardrailResult:
    """Result of a guardrail check on user input or model output."""

    category: GuardrailCategory
    blocked: bool
    confidence: float  # 0.0 to 1.0
    block_reason: str | None = None
    response_override: str | None = None
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Medical disclaimer appended to all responses
MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "**Medical Disclaimer:** This information is provided for educational and informational "
    "purposes only and is not intended as medical advice. It should not be used as a substitute "
    "for professional medical advice, diagnosis, or treatment. Always seek the advice of a "
    "qualified healthcare provider with any questions you may have regarding a medical condition "
    "or medication. Never disregard professional medical advice or delay in seeking it because "
    "of something you have read here. If you think you may have a medical emergency, call your "
    "doctor or 911 immediately."
)

# Patterns that indicate harmful intent
HARMFUL_PATTERNS = [
    # Overdose / self-harm related
    r"(?i)\bhow\s+(?:to|can\s+I|do\s+I)\s+(?:overdose|OD|kill\s+myself|end\s+my\s+life)",
    r"(?i)\blethal\s+dose\b.*\b(?:human|person|myself|someone)\b",
    r"(?i)\bhow\s+much\s+(?:\w+\s+){0,3}(?:to\s+die|to\s+kill|fatal|lethal)\b",
    r"(?i)\bsuicid(?:e|al)\b.*\b(?:dose|medication|drug|pill)\b",
    r"(?i)\b(?:poison|poisoning)\s+(?:someone|a\s+person|my)\b",
    # Drug abuse / misuse
    r"(?i)\bhow\s+to\s+(?:get\s+high|abuse|misuse|snort|inject)\b.*\b(?:drug|medication|pill)\b",
    r"(?i)\b(?:crush|snort|inject)\b.*\b(?:pill|tablet|capsule|medication)\b",
    r"(?i)\brecreational\s+(?:use|dose|dosing)\b",
    # Manufacturing / synthesis
    r"(?i)\bhow\s+to\s+(?:make|synthesize|manufacture|produce)\b.*\b(?:drug|narcotic|opiate|fentanyl|meth)\b",
    r"(?i)\b(?:synthesis|recipe|formula)\b.*\b(?:illegal|controlled|narcotic)\b",
]

# Patterns that indicate off-topic questions
OFF_TOPIC_PATTERNS = [
    r"(?i)\b(?:recipe|cooking|baking|dinner|lunch|breakfast)\b(?!.*(?:drug|pharma|medical|dose))",
    r"(?i)\b(?:weather|forecast|temperature|climate)\b(?!.*(?:drug|storage|stability))",
    r"(?i)\b(?:sports|football|basketball|soccer|baseball|hockey)\b",
    r"(?i)\b(?:movie|film|tv\s+show|actor|actress|celebrity)\b",
    r"(?i)\b(?:stock\s+market|invest|bitcoin|crypto|trading)\b(?!.*(?:pharma|biotech))",
    r"(?i)\b(?:tell\s+me\s+a\s+joke|sing|poem|story)\b",
    r"(?i)\b(?:write\s+(?:me\s+)?(?:a\s+)?(?:code|program|script|essay))\b",
    r"(?i)\b(?:translate|translation)\b(?!.*(?:drug|medical|pharma))",
    r"(?i)\b(?:math|calculate|equation|algebra)\b(?!.*(?:dose|dosing|pharma))",
]

# Triggers that should prompt a "consult your healthcare provider" recommendation
CONSULT_TRIGGERS = [
    r"(?i)\bshould\s+I\s+(?:take|stop|start|switch|change|increase|decrease|combine)\b",
    r"(?i)\bcan\s+I\s+(?:take|use|mix|combine)\b.*\b(?:while|with|during)\b",
    r"(?i)\bis\s+(?:it|this)\s+(?:safe|ok|okay|fine)\s+(?:to|for\s+me)\b",
    r"(?i)\bwhat\s+(?:dose|dosage|amount)\s+should\s+I\b",
    r"(?i)\bI\s+(?:am|have\s+been)\s+(?:taking|using|on)\b.*\b(?:and|but|however)\b",
    r"(?i)\bpregnant|pregnancy|breastfeeding|nursing\b.*\b(?:take|safe|medication)\b",
    r"(?i)\b(?:my\s+)?(?:child|baby|infant|toddler|kid)\b.*\b(?:take|give|medication|medicine)\b",
    r"(?i)\bdiagnos(?:e|is|ed)\b",
    r"(?i)\b(?:symptom|symptoms|feeling|experiencing)\b.*\b(?:after|since|while)\s+(?:taking|using)\b",
]

# Emergency-related patterns
EMERGENCY_PATTERNS = [
    r"(?i)\b(?:overdos(?:e|ed|ing)|OD['d]?)\b",
    r"(?i)\b(?:swallow(?:ed)?|ingest(?:ed)?|ate)\b.*\b(?:too\s+many|whole\s+bottle|all)\b",
    r"(?i)\b(?:not\s+breathing|unconscious|seizure|anaphylaxis|anaphylactic)\b",
    r"(?i)\b(?:severe\s+(?:allergic|reaction|bleeding|pain))\b.*\b(?:medication|drug|pill)\b",
    r"(?i)\b(?:poison\s+control|emergency\s+room|ER|ambulance)\b",
]


class SafetyGuardrails:
    """Comprehensive safety guardrail system for pharmaceutical AI.

    Checks user input for harmful intent, off-topic content, emergency situations,
    and questions that require professional medical consultation. Also post-processes
    model output to inject disclaimers and validate safety.
    """

    def __init__(
        self,
        strict_mode: bool = True,
        inject_disclaimer: bool = True,
        off_topic_threshold: float = 0.7,
    ) -> None:
        self._strict_mode = strict_mode
        self._inject_disclaimer = inject_disclaimer
        self._off_topic_threshold = off_topic_threshold
        logger.info(
            "guardrails.initialized",
            strict_mode=strict_mode,
            disclaimer=inject_disclaimer,
        )

    def check(self, user_input: str) -> GuardrailResult:
        """Run all guardrail checks on user input.

        Returns a GuardrailResult indicating whether the input is safe to process.
        """
        if not user_input or not user_input.strip():
            return GuardrailResult(
                category=GuardrailCategory.SAFE,
                blocked=False,
                confidence=1.0,
            )

        # Check for emergency situations first
        emergency_result = self._check_emergency(user_input)
        if emergency_result.category == GuardrailCategory.EMERGENCY:
            return emergency_result

        # Check for harmful content
        harmful_result = self._check_harmful(user_input)
        if harmful_result.blocked:
            return harmful_result

        # Check for off-topic content
        off_topic_result = self._check_off_topic(user_input)
        if off_topic_result.blocked:
            return off_topic_result

        # Check for "consult professional" triggers
        consult_result = self._check_consult_triggers(user_input)

        # Build final result
        flags: list[str] = []
        if consult_result.category == GuardrailCategory.CONSULT_PROFESSIONAL:
            flags.append("consult_recommended")

        confidence = self._estimate_pharma_relevance(user_input)

        return GuardrailResult(
            category=consult_result.category if flags else GuardrailCategory.SAFE,
            blocked=False,
            confidence=confidence,
            flags=flags,
        )

    def _check_harmful(self, text: str) -> GuardrailResult:
        """Check for harmful intent in user input."""
        matched_patterns: list[str] = []

        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, text):
                matched_patterns.append(pattern)

        if matched_patterns:
            logger.warning(
                "guardrails.harmful_detected",
                pattern_count=len(matched_patterns),
                input_preview=text[:50],
            )
            return GuardrailResult(
                category=GuardrailCategory.HARMFUL,
                blocked=True,
                confidence=1.0,
                block_reason=(
                    "This question appears to involve potentially harmful use of medications. "
                    "I cannot provide information that could be used to cause harm."
                ),
                response_override=(
                    "I'm not able to provide information that could be used to cause harm "
                    "to yourself or others.\n\n"
                    "If you or someone you know is in crisis:\n"
                    "- **National Suicide Prevention Lifeline:** 988 (call or text)\n"
                    "- **Crisis Text Line:** Text HOME to 741741\n"
                    "- **Poison Control:** 1-800-222-1222\n"
                    "- **Emergency Services:** 911\n\n"
                    "Please reach out to a healthcare professional or one of these resources "
                    "for help."
                ),
                flags=["harmful_intent"],
                metadata={"matched_patterns": len(matched_patterns)},
            )

        return GuardrailResult(
            category=GuardrailCategory.SAFE,
            blocked=False,
            confidence=1.0,
        )

    def _check_off_topic(self, text: str) -> GuardrailResult:
        """Check if the question is off-topic (not pharmaceutical/medical)."""
        off_topic_matches = 0
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, text):
                off_topic_matches += 1

        # Also check for pharmaceutical relevance keywords
        pharma_keywords = {
            "drug",
            "medication",
            "medicine",
            "dose",
            "dosage",
            "side effect",
            "adverse",
            "interaction",
            "prescription",
            "otc",
            "fda",
            "clinical",
            "trial",
            "indication",
            "contraindication",
            "pharmacology",
            "tablet",
            "capsule",
            "injection",
            "generic",
            "brand",
            "therapeutic",
            "treatment",
            "symptom",
            "diagnosis",
            "patient",
            "healthcare",
            "pharmaceutical",
            "pharmacy",
            "pharmacist",
            "doctor",
            "physician",
            "medical",
            "health",
            "disease",
            "condition",
            "therapy",
            "efficacy",
            "safety",
            "warning",
            "precaution",
            "toxicity",
            "absorption",
            "metabolism",
            "excretion",
            "bioavailability",
            "half-life",
            "receptor",
            "enzyme",
            "protein",
        }

        text_lower = text.lower()
        has_pharma_keyword = any(kw in text_lower for kw in pharma_keywords)

        if off_topic_matches > 0 and not has_pharma_keyword:
            relevance = self._estimate_pharma_relevance(text)
            if relevance < self._off_topic_threshold:
                logger.info(
                    "guardrails.off_topic_detected",
                    relevance=relevance,
                    input_preview=text[:50],
                )
                return GuardrailResult(
                    category=GuardrailCategory.OFF_TOPIC,
                    blocked=True,
                    confidence=1.0 - relevance,
                    block_reason="This question does not appear to be related to pharmaceutical or medical topics.",
                    response_override=(
                        "I'm a pharmaceutical knowledge assistant, and I can only help with "
                        "questions about medications, drug interactions, dosing, side effects, "
                        "clinical guidelines, and related pharmaceutical topics.\n\n"
                        "Could you please rephrase your question to focus on a pharmaceutical "
                        "or medical topic? For example:\n"
                        "- What are the side effects of aspirin?\n"
                        "- What are the drug interactions for metformin?\n"
                        "- What are the dosing guidelines for amoxicillin?"
                    ),
                    flags=["off_topic"],
                )

        return GuardrailResult(
            category=GuardrailCategory.SAFE,
            blocked=False,
            confidence=1.0,
        )

    def _check_emergency(self, text: str) -> GuardrailResult:
        """Check for medical emergency indicators."""
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, text):
                logger.warning(
                    "guardrails.emergency_detected",
                    input_preview=text[:50],
                )
                return GuardrailResult(
                    category=GuardrailCategory.EMERGENCY,
                    blocked=False,  # Don't block, but prepend emergency info
                    confidence=1.0,
                    response_override=None,
                    flags=["emergency_detected"],
                )

        return GuardrailResult(
            category=GuardrailCategory.SAFE,
            blocked=False,
            confidence=1.0,
        )

    def _check_consult_triggers(self, text: str) -> GuardrailResult:
        """Check for questions that should recommend professional consultation."""
        for pattern in CONSULT_TRIGGERS:
            if re.search(pattern, text):
                return GuardrailResult(
                    category=GuardrailCategory.CONSULT_PROFESSIONAL,
                    blocked=False,
                    confidence=0.9,
                    flags=["consult_recommended"],
                )

        return GuardrailResult(
            category=GuardrailCategory.SAFE,
            blocked=False,
            confidence=1.0,
        )

    @staticmethod
    def _estimate_pharma_relevance(text: str) -> float:
        """Estimate how likely a question is pharmaceutical/medical in nature.

        Returns a score from 0.0 (completely off-topic) to 1.0 (clearly pharmaceutical).
        """
        pharma_indicators = [
            r"(?i)\b(?:drug|medication|medicine|pharmaceutical|pharma)\b",
            r"(?i)\b(?:dose|dosage|dosing|mg|mcg|ml|tablet|capsule|pill)\b",
            r"(?i)\b(?:side\s+effect|adverse\s+(?:reaction|event|effect))\b",
            r"(?i)\b(?:interact(?:ion)?|contraindic(?:ation|ated))\b",
            r"(?i)\b(?:prescri(?:be|ption|bing)|OTC|over[\s-]the[\s-]counter)\b",
            r"(?i)\b(?:FDA|EMA|pharmacology|pharmacokinetics|pharmacodynamics)\b",
            r"(?i)\b(?:clinical\s+trial|efficacy|bioavailability|half[\s-]life)\b",
            r"(?i)\b(?:patient|treatment|therapy|therapeutic|indication)\b",
            r"(?i)\b(?:symptom|diagnosis|disease|condition|syndrome|disorder)\b",
            r"(?i)\b(?:aspirin|ibuprofen|acetaminophen|metformin|insulin)\b",
            r"(?i)\b(?:antibiotic|antiviral|antifungal|analgesic|anti[\s-]?inflam)\b",
            r"(?i)\b(?:warning|precaution|black\s+box|boxed\s+warning)\b",
        ]

        matches = sum(1 for pattern in pharma_indicators if re.search(pattern, text))
        # Normalize: 3+ matches = fully relevant
        return min(1.0, matches / 3.0)

    def post_process(self, response: str, guardrail_result: GuardrailResult) -> str:
        """Post-process model output to inject disclaimers and safety messages."""
        parts: list[str] = []

        # Emergency header
        if "emergency_detected" in guardrail_result.flags:
            parts.append(
                "**IMPORTANT: If this is a medical emergency, please call 911 or contact "
                "your local emergency services immediately. You can also call Poison Control "
                "at 1-800-222-1222.**\n\n"
            )

        parts.append(response)

        # Consult professional recommendation
        if "consult_recommended" in guardrail_result.flags:
            # Only add if not already present in the response
            if "consult" not in response.lower() or "healthcare" not in response.lower():
                parts.append(
                    "\n\n**Important:** The question you asked involves personalized medical "
                    "decisions. Please consult with your doctor, pharmacist, or other qualified "
                    "healthcare provider for advice specific to your situation."
                )

        # Medical disclaimer
        if self._inject_disclaimer:
            if "Medical Disclaimer" not in response:
                parts.append(MEDICAL_DISCLAIMER)

        return "".join(parts)

    @staticmethod
    def compute_answer_confidence(
        answer: str,
        retrieval_score: float,
        citation_count: int,
    ) -> float:
        """Compute a confidence score for the generated answer.

        Considers retrieval relevance, number of citations, and answer characteristics.
        """
        # Base score from retrieval relevance
        score = retrieval_score * 0.5

        # Citation bonus
        if citation_count >= 3:
            score += 0.2
        elif citation_count >= 1:
            score += 0.1

        # Penalty for hedging language
        hedging = [
            "i'm not sure",
            "i don't know",
            "it's unclear",
            "may or may not",
            "cannot determine",
            "insufficient information",
            "no relevant information",
            "not found in",
        ]
        answer_lower = answer.lower()
        hedging_count = sum(1 for h in hedging if h in answer_lower)
        score -= hedging_count * 0.1

        # Penalty for very short answers
        word_count = len(answer.split())
        if word_count < 20:
            score -= 0.15
        elif word_count < 50:
            score -= 0.05

        return max(0.0, min(1.0, score))
