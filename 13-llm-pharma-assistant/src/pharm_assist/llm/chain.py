"""LLM chain for pharmaceutical question answering with RAG context.

Orchestrates the system prompt, context injection from the retriever,
citation formatting, hallucination guardrails, and conversation memory.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from pharm_assist.llm.guardrails import GuardrailResult, SafetyGuardrails
from pharm_assist.rag.retriever import PharmaceuticalRetriever, RetrievalResult

logger = structlog.get_logger(__name__)

# System prompt that enforces pharmaceutical domain behavior
PHARMA_SYSTEM_PROMPT = """\
You are PharmAssistAI, a pharmaceutical knowledge assistant powered by a \
Retrieval-Augmented Generation (RAG) system. Your purpose is to provide \
accurate, well-cited information about pharmaceutical drugs, their interactions, \
dosing guidelines, adverse effects, and regulatory requirements.

CRITICAL RULES:
1. ONLY answer based on the provided context from the knowledge base. \
   If the context does not contain enough information to answer the question, \
   say so explicitly. NEVER fabricate drug information.

2. ALWAYS cite your sources using the citation markers [1], [2], etc. that \
   correspond to the source documents provided in the context.

3. ALWAYS include the following disclaimer at the end of every response:
   "This information is for educational purposes only and should not replace \
   professional medical advice. Always consult a qualified healthcare provider \
   before making any medication decisions."

4. If asked about drug dosing, ALWAYS mention that dosing should be \
   individualized by a healthcare provider based on patient-specific factors.

5. If you detect that a question could lead to harm (e.g., how to overdose, \
   how to misuse medications), refuse to answer and recommend contacting \
   emergency services or a healthcare professional.

6. Use precise medical terminology but also provide plain-language explanations.

7. When discussing adverse reactions, categorize them by frequency \
   (common, uncommon, rare, very rare) when this information is available \
   in the context.

8. For drug interactions, specify the severity (major, moderate, minor) \
   and the mechanism of interaction when available.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Support with evidence from the provided sources
- Include inline citations [1], [2], etc.
- End with the medical disclaimer
- Use bullet points or numbered lists for clarity when appropriate
"""


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """Complete result from the LLM chain."""

    question: str
    answer: str
    citations_text: str
    retrieval_result: RetrievalResult | None
    guardrail_result: GuardrailResult | None
    model: str
    latency_ms: float
    token_usage: dict[str, int] = field(default_factory=dict)


class ConversationMemory:
    """Manages conversation history for multi-turn interactions.

    Maintains a sliding window of conversation turns to provide
    context for follow-up questions without exceeding token limits.
    """

    def __init__(self, max_turns: int = 10, max_tokens: int = 4000) -> None:
        self._history: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._max_tokens = max_tokens

    def add_turn(self, role: str, content: str, **metadata: Any) -> None:
        """Add a conversation turn to history."""
        self._history.append(ConversationTurn(role=role, content=content, metadata=metadata))

    def get_messages(self) -> list[dict[str, str]]:
        """Get conversation history formatted for the Anthropic API.

        Trims older messages if the total would exceed the token budget.
        """
        messages: list[dict[str, str]] = []
        estimated_tokens = 0

        # Work backwards from most recent to stay within budget
        for turn in reversed(self._history):
            turn_tokens = len(turn.content.split()) * 1.3  # rough estimate
            if estimated_tokens + turn_tokens > self._max_tokens:
                break
            messages.insert(0, {"role": turn.role, "content": turn.content})
            estimated_tokens += turn_tokens

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of turns in history."""
        return len(self._history)

    def get_last_user_query(self) -> str | None:
        """Get the most recent user question for follow-up detection."""
        for turn in reversed(self._history):
            if turn.role == "user":
                return turn.content
        return None


class PharmaceuticalChain:
    """LLM chain that combines retrieval, guardrails, and response generation.

    This is the main orchestrator that:
    1. Checks safety guardrails on the input
    2. Retrieves relevant context from the vector store
    3. Injects context into the prompt
    4. Generates an answer using Claude
    5. Formats citations and adds disclaimers
    6. Maintains conversation memory for follow-ups
    """

    def __init__(
        self,
        retriever: PharmaceuticalRetriever,
        anthropic_api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        enable_guardrails: bool = True,
    ) -> None:
        if anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            # Will use ANTHROPIC_API_KEY env var
            self._client = anthropic.Anthropic()

        self._retriever = retriever
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._guardrails = SafetyGuardrails() if enable_guardrails else None
        self._memory = ConversationMemory()

        logger.info(
            "chain.initialized",
            model=model,
            guardrails=enable_guardrails,
        )

    @property
    def memory(self) -> ConversationMemory:
        """Access the conversation memory."""
        return self._memory

    async def ask(
        self,
        question: str,
        drug_name: str | None = None,
        section_type: str | None = None,
        n_results: int = 5,
    ) -> AnswerResult:
        """Ask a pharmaceutical question and get a comprehensive answer.

        Full pipeline: guardrails -> retrieval -> context injection -> generation.
        """
        start_time = time.monotonic()

        # Step 1: Safety guardrails
        guardrail_result: GuardrailResult | None = None
        if self._guardrails:
            guardrail_result = self._guardrails.check(question)
            if guardrail_result.blocked:
                elapsed = (time.monotonic() - start_time) * 1000
                return AnswerResult(
                    question=question,
                    answer=guardrail_result.response_override
                    or (
                        "I'm unable to answer this question. "
                        + (guardrail_result.block_reason or "")
                        + "\n\nIf you have a medical emergency, please call 911 or "
                        "contact your local emergency services immediately."
                    ),
                    citations_text="",
                    retrieval_result=None,
                    guardrail_result=guardrail_result,
                    model=self._model,
                    latency_ms=elapsed,
                )

        # Step 2: Detect follow-up and enrich query if needed
        enriched_question = self._enrich_followup(question)

        # Step 3: Retrieve relevant context
        retrieval_result = await self._retriever.retrieve(
            query=enriched_question,
            n_results=n_results,
            drug_name=drug_name,
            section_type=section_type,
        )

        # Step 4: Build messages with context
        messages = self._build_messages(enriched_question, retrieval_result)

        # Step 5: Generate response
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=PHARMA_SYSTEM_PROMPT,
                messages=messages,
            )

            answer_text = response.content[0].text
            token_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        except anthropic.APIError as e:
            logger.exception("chain.api_error")
            answer_text = (
                f"I encountered an error while processing your question. "
                f"Please try again later. (Error: {e.message})"
            )
            token_usage = {}

        # Step 6: Apply post-processing guardrails
        if self._guardrails and guardrail_result:
            answer_text = self._guardrails.post_process(answer_text, guardrail_result)

        # Step 7: Update conversation memory
        self._memory.add_turn("user", question)
        self._memory.add_turn("assistant", answer_text)

        # Step 8: Format citations
        citations_text = retrieval_result.format_citations_block()

        elapsed = (time.monotonic() - start_time) * 1000

        logger.info(
            "chain.answer_generated",
            question_preview=question[:60],
            answer_length=len(answer_text),
            citations=len(retrieval_result.citations),
            latency_ms=round(elapsed, 1),
        )

        return AnswerResult(
            question=question,
            answer=answer_text,
            citations_text=citations_text,
            retrieval_result=retrieval_result,
            guardrail_result=guardrail_result,
            model=self._model,
            latency_ms=elapsed,
            token_usage=token_usage,
        )

    async def ask_stream(
        self,
        question: str,
        drug_name: str | None = None,
        n_results: int = 5,
    ) -> AsyncIterator[str]:
        """Stream the answer token by token for real-time display.

        Yields text chunks as they are generated by the LLM.
        """
        # Safety guardrails
        if self._guardrails:
            guardrail_result = self._guardrails.check(question)
            if guardrail_result.blocked:
                msg = guardrail_result.response_override or (
                    "I'm unable to answer this question. " + (guardrail_result.block_reason or "")
                )
                yield msg
                return

        # Retrieve context
        enriched_question = self._enrich_followup(question)
        retrieval_result = await self._retriever.retrieve(
            query=enriched_question,
            n_results=n_results,
            drug_name=drug_name,
        )

        messages = self._build_messages(enriched_question, retrieval_result)

        # Stream response
        full_response: list[str] = []
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=PHARMA_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response.append(text)
                    yield text

        except anthropic.APIError as e:
            error_msg = f"\n\n[Error: {e.message}]"
            yield error_msg
            full_response.append(error_msg)

        # Update memory with full response
        self._memory.add_turn("user", question)
        self._memory.add_turn("assistant", "".join(full_response))

        # Yield citations at the end
        if retrieval_result.citations:
            yield "\n\n---\n"
            yield retrieval_result.format_citations_block()

    def _build_messages(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> list[dict[str, str]]:
        """Build the message list with conversation history and context."""
        messages: list[dict[str, str]] = []

        # Add conversation history (excluding the current question)
        history = self._memory.get_messages()
        messages.extend(history)

        # Build the current user message with context
        context = retrieval_result.format_context_with_citations()

        user_message = (
            f"RETRIEVED CONTEXT FROM KNOWLEDGE BASE:\n"
            f"{context}\n\n"
            f"---\n\n"
            f"USER QUESTION: {question}\n\n"
            f"Please answer the question based ONLY on the context provided above. "
            f"Cite sources using [1], [2], etc. Include the medical disclaimer."
        )

        if not retrieval_result.has_results:
            user_message = (
                f"USER QUESTION: {question}\n\n"
                f"NOTE: No relevant information was found in the knowledge base for this question. "
                f"Please inform the user that you don't have information on this topic in your "
                f"current knowledge base, and suggest they consult a healthcare professional "
                f"or pharmacist for accurate information."
            )

        messages.append({"role": "user", "content": user_message})
        return messages

    def _enrich_followup(self, question: str) -> str:
        """Detect and enrich follow-up questions with prior context.

        If the question appears to be a follow-up (uses pronouns like 'it',
        'this drug', etc.), prepend context from the previous turn.
        """
        followup_indicators = [
            "it",
            "this drug",
            "that medication",
            "the same",
            "also",
            "what about",
            "and what",
            "how about",
            "its",
            "this medication",
            "the drug",
        ]

        lower_q = question.lower().strip()
        is_followup = any(
            lower_q.startswith(ind) or f" {ind} " in lower_q for ind in followup_indicators
        )

        if is_followup:
            last_query = self._memory.get_last_user_query()
            if last_query:
                enriched = f"(Follow-up to: '{last_query}') {question}"
                logger.debug("chain.followup_detected", enriched=enriched[:100])
                return enriched

        return question

    def reset_conversation(self) -> None:
        """Reset the conversation memory."""
        self._memory.clear()
        logger.info("chain.conversation_reset")
