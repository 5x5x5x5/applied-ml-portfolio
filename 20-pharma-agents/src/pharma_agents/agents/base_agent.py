"""Base agent class providing shared infrastructure for all specialized agents."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for agent communication
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Schema for a tool available to an agent."""

    name: str
    description: str
    input_schema: dict[str, Any]


class AgentMessage(BaseModel):
    """A single message in the agent conversation history."""

    role: str  # "user" | "assistant"
    content: Any  # str or list of content blocks
    timestamp: float = Field(default_factory=time.time)


class ToolResult(BaseModel):
    """Result returned after executing a tool."""

    tool_use_id: str
    content: str
    is_error: bool = False


class AgentResponse(BaseModel):
    """Structured response produced by an agent."""

    agent_name: str
    agent_role: str
    text: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    processing_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all PharmaAgents.

    Provides:
    - Claude API integration with tool use
    - Conversation history management
    - Automatic tool execution loop
    - Structured output parsing
    - Retry logic with exponential back-off
    """

    MAX_RETRIES: int = 3
    MAX_TOOL_ROUNDS: int = 10
    MODEL: str = "claude-sonnet-4-20250514"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()
        self._history: list[AgentMessage] = []
        self._log = logger.bind(agent=self.name)

    # -- abstract properties each subclass must define ----------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable name for this agent."""

    @property
    @abstractmethod
    def role(self) -> str:
        """Short description of the agent's role."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt that establishes the agent's persona and expertise."""

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """Return the tool definitions this agent may invoke."""

    @abstractmethod
    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""

    # -- public API ---------------------------------------------------------

    async def run(self, user_message: str) -> AgentResponse:
        """Process a user message through the agent, executing tools as needed.

        Returns a structured ``AgentResponse``.
        """
        start = time.time()
        self._history.append(AgentMessage(role="user", content=user_message))
        self._log.info("agent.run.start", message_preview=user_message[:120])

        messages = self._build_messages()
        tools_api = self._tools_to_api_format()
        tools_used: list[str] = []

        # Agentic loop: keep running until the model stops requesting tools
        response_text = ""
        structured_data: dict[str, Any] = {}

        for round_num in range(1, self.MAX_TOOL_ROUNDS + 1):
            api_response = self._call_api_with_retry(messages, tools_api)

            # Collect text blocks and tool-use blocks
            text_parts: list[str] = []
            tool_use_blocks: list[Any] = []

            for block in api_response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            response_text = "\n".join(text_parts)

            if not tool_use_blocks:
                # No more tool calls -- we are done
                self._log.info("agent.run.complete", rounds=round_num)
                break

            # Build the assistant message exactly as the API returned it
            messages.append({"role": "assistant", "content": api_response.content})

            # Execute each tool and build tool-result message
            tool_results: list[dict[str, Any]] = []
            for tb in tool_use_blocks:
                self._log.info("agent.tool.call", tool=tb.name, round=round_num)
                tools_used.append(tb.name)
                try:
                    result_str = self.execute_tool(tb.name, tb.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": result_str,
                        }
                    )
                except Exception as exc:
                    self._log.error("agent.tool.error", tool=tb.name, error=str(exc))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": f"Error executing tool: {exc}",
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
        else:
            self._log.warning("agent.run.max_rounds_reached")

        # Try to parse structured data from response text
        structured_data = self._extract_structured_data(response_text)

        # Persist final assistant text in history
        self._history.append(AgentMessage(role="assistant", content=response_text))

        elapsed = time.time() - start
        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            text=response_text,
            structured_data=structured_data,
            tools_used=tools_used,
            confidence=self._estimate_confidence(response_text, tools_used),
            processing_time_s=round(elapsed, 3),
        )

    def reset_history(self) -> None:
        """Clear the agent's conversation history."""
        self._history.clear()

    def get_capabilities(self) -> dict[str, Any]:
        """Return a summary of this agent's capabilities for the coordinator."""
        return {
            "name": self.name,
            "role": self.role,
            "tools": [t.name for t in self.get_tools()],
            "description": self.system_prompt[:300],
        }

    # -- internal helpers ---------------------------------------------------

    def _build_messages(self) -> list[dict[str, Any]]:
        """Convert internal history into the API message format."""
        return [{"role": m.role, "content": m.content} for m in self._history]

    def _tools_to_api_format(self) -> list[dict[str, Any]]:
        """Convert ``ToolDefinition`` list to the Anthropic API schema."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.get_tools()
        ]

    def _call_api_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the Claude API with exponential-back-off retry."""
        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.MODEL,
                    "max_tokens": 4096,
                    "system": self.system_prompt,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                return self._client.messages.create(**kwargs)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                wait = 2**attempt
                self._log.warning("api.rate_limit", attempt=attempt, wait=wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                last_exc = exc
                if attempt == self.MAX_RETRIES:
                    break
                wait = 2**attempt
                self._log.warning("api.error", attempt=attempt, error=str(exc), wait=wait)
                time.sleep(wait)

        raise RuntimeError(f"API call failed after {self.MAX_RETRIES} retries: {last_exc}")

    @staticmethod
    def _extract_structured_data(text: str) -> dict[str, Any]:
        """Attempt to extract a JSON block from the response text."""
        # Look for ```json ... ``` fenced blocks
        import re

        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _estimate_confidence(text: str, tools_used: list[str]) -> float:
        """Heuristic confidence score based on response richness."""
        score = 0.5
        if tools_used:
            score += min(len(tools_used) * 0.1, 0.3)
        if len(text) > 500:
            score += 0.1
        if len(text) > 1500:
            score += 0.1
        return min(round(score, 2), 1.0)
