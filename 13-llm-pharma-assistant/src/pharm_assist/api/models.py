"""Pydantic request and response models for the PharmAssistAI API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request Models ──────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The pharmaceutical question to answer.",
        examples=["What are the common side effects of metformin?"],
    )
    drug_name: str | None = Field(
        default=None,
        description="Optional: filter retrieval to a specific drug.",
        examples=["metformin"],
    )
    section_type: str | None = Field(
        default=None,
        description="Optional: filter to a specific FDA label section.",
        examples=["ADVERSE REACTIONS"],
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve.",
    )
    stream: bool = Field(
        default=False,
        description="If true, stream the response via SSE.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity.",
    )


class DocumentIngestRequest(BaseModel):
    """Request body for the /documents/ingest endpoint."""

    collection: str = Field(
        default="general_pharma",
        description="Collection to ingest documents into.",
        examples=["drug_labels", "clinical_guidelines"],
    )
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: fixed_size, recursive, semantic, section_based.",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target chunk size in tokens.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks in tokens.",
    )


class DocumentSearchRequest(BaseModel):
    """Query parameters for the /documents/search endpoint."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Search query text.",
    )
    collection: str | None = Field(
        default=None,
        description="Collection to search in. None searches all collections.",
    )
    n_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )
    drug_name: str | None = Field(
        default=None,
        description="Filter results to a specific drug name.",
    )
    use_mmr: bool = Field(
        default=False,
        description="Use MMR for result diversity.",
    )


# ── Response Models ─────────────────────────────────────────────────────────


class CitationResponse(BaseModel):
    """A single source citation in an answer response."""

    citation_id: int
    source_file: str
    drug_name: str | None = None
    section_type: str | None = None
    relevance_score: float
    excerpt: str = Field(description="Brief excerpt from the source chunk.")


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer.",
    )
    model: str
    retrieval_strategy: str | None = None
    latency_ms: float
    session_id: str | None = None
    guardrail_flags: list[str] = Field(default_factory=list)


class DocumentChunkResponse(BaseModel):
    """A single document chunk in search results."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)
    relevance_score: float
    collection: str


class DocumentSearchResponse(BaseModel):
    """Response body for the /documents/search endpoint."""

    query: str
    results: list[DocumentChunkResponse]
    total_results: int


class DocumentIngestResponse(BaseModel):
    """Response body for the /documents/ingest endpoint."""

    files_processed: int
    chunks_created: int
    collection: str
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "healthy"
    version: str
    collections: dict[str, int] = Field(
        default_factory=dict,
        description="Document counts per collection.",
    )
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str | None = None
    status_code: int


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication."""

    type: str = Field(
        description="Message type: 'question', 'answer_chunk', 'answer_complete', 'error', 'citation'."
    )
    content: str = ""
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)
