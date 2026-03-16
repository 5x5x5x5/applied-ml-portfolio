"""FastAPI application for PharmAssistAI.

Provides REST and WebSocket endpoints for pharmaceutical question answering,
document ingestion, and knowledge base search.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from pharm_assist import __app_name__, __version__
from pharm_assist.api.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DocumentChunkResponse,
    DocumentIngestResponse,
    DocumentSearchResponse,
    ErrorResponse,
    HealthResponse,
    WebSocketMessage,
)
from pharm_assist.llm.chain import PharmaceuticalChain
from pharm_assist.llm.guardrails import SafetyGuardrails
from pharm_assist.rag.document_processor import ChunkStrategy, DocumentProcessor
from pharm_assist.rag.retriever import PharmaceuticalRetriever
from pharm_assist.rag.vector_store import CollectionName, SearchQuery, VectorStore

logger = structlog.get_logger(__name__)

# ── Application state ───────────────────────────────────────────────────────

_start_time: float = 0.0
_vector_store: VectorStore | None = None
_retriever: PharmaceuticalRetriever | None = None
_chain: PharmaceuticalChain | None = None
_sessions: dict[str, PharmaceuticalChain] = {}


def _get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_directory="./data/chromadb")
    return _vector_store


def _get_retriever() -> PharmaceuticalRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = PharmaceuticalRetriever(
            vector_store=_get_vector_store(),
            use_multi_query=True,
            use_mmr=True,
        )
    return _retriever


def _get_chain(session_id: str | None = None) -> PharmaceuticalChain:
    """Get a chain instance, optionally per-session for conversation memory."""
    global _chain
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    if _chain is None:
        _chain = PharmaceuticalChain(
            retriever=_get_retriever(),
            model="claude-sonnet-4-20250514",
            enable_guardrails=True,
        )

    if session_id:
        # Create a session-specific chain with its own memory
        session_chain = PharmaceuticalChain(
            retriever=_get_retriever(),
            model="claude-sonnet-4-20250514",
            enable_guardrails=True,
        )
        _sessions[session_id] = session_chain
        return session_chain

    return _chain


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan: initialize and tear down resources."""
    global _start_time
    _start_time = time.time()

    logger.info("app.starting", version=__version__)

    # Pre-initialize components
    _get_vector_store()
    _get_retriever()

    # Ingest sample data if collections are empty
    vs = _get_vector_store()
    stats = vs.get_collection_stats()
    if not stats or all(v == 0 for v in stats.values()):
        sample_dir = Path("./data/sample_drug_labels")
        if sample_dir.exists():
            logger.info("app.ingesting_sample_data")
            processor = DocumentProcessor(
                chunk_strategy=ChunkStrategy.RECURSIVE,
                chunk_size=512,
                chunk_overlap=64,
            )
            chunks = processor.process_directory(sample_dir)
            if chunks:
                vs.index_chunks(chunks, CollectionName.DRUG_LABELS)

    logger.info("app.started")
    yield
    logger.info("app.shutdown")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=__app_name__,
    description=(
        "LLM-powered Pharmaceutical Knowledge Assistant using RAG. "
        "Ask questions about drugs, interactions, clinical guidelines, "
        "and regulatory requirements."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if available
frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint with system status."""
    vs = _get_vector_store()
    stats = vs.get_collection_stats()
    uptime = time.time() - _start_time

    return HealthResponse(
        status="healthy",
        version=__version__,
        collections=stats,
        uptime_seconds=round(uptime, 1),
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest) -> AskResponse | StreamingResponse:
    """Ask a pharmaceutical question and get an AI-generated answer.

    Uses RAG to retrieve relevant information from the knowledge base
    and generates a cited answer using Claude.

    If `stream=true`, returns a Server-Sent Events stream.
    """
    session_id = request.session_id or str(uuid.uuid4())

    if request.stream:
        return StreamingResponse(
            _stream_answer(request, session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    chain = _get_chain(session_id)

    try:
        result = await chain.ask(
            question=request.question,
            drug_name=request.drug_name,
            section_type=request.section_type,
            n_results=request.n_results,
        )
    except Exception as e:
        logger.exception("api.ask_error")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Build citation responses
    citations: list[CitationResponse] = []
    if result.retrieval_result:
        for c in result.retrieval_result.citations:
            citations.append(
                CitationResponse(
                    citation_id=c.citation_id,
                    source_file=c.source_file,
                    drug_name=c.drug_name,
                    section_type=c.section_type,
                    relevance_score=round(c.relevance_score, 3),
                    excerpt=c.chunk_text[:200],
                )
            )

    # Compute confidence
    confidence = 0.5
    if result.retrieval_result and result.retrieval_result.search_results:
        avg_relevance = sum(
            r.relevance_score for r in result.retrieval_result.search_results
        ) / len(result.retrieval_result.search_results)
        confidence = SafetyGuardrails.compute_answer_confidence(
            result.answer, avg_relevance, len(citations)
        )

    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=citations,
        confidence=round(confidence, 3),
        model=result.model,
        retrieval_strategy=(
            result.retrieval_result.retrieval_strategy if result.retrieval_result else None
        ),
        latency_ms=round(result.latency_ms, 1),
        session_id=session_id,
        guardrail_flags=result.guardrail_result.flags if result.guardrail_result else [],
    )


async def _stream_answer(request: AskRequest, session_id: str):  # type: ignore[no-untyped-def]
    """Generator for SSE streaming responses."""
    chain = _get_chain(session_id)

    try:
        async for chunk in chain.ask_stream(
            question=request.question,
            drug_name=request.drug_name,
            n_results=request.n_results,
        ):
            data = json.dumps({"type": "chunk", "content": chunk})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)  # yield control

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        error_data = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {error_data}\n\n"


@app.post("/documents/ingest", response_model=DocumentIngestResponse, tags=["Documents"])
async def ingest_documents(
    files: list[UploadFile],
    collection: str = "general_pharma",
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> DocumentIngestResponse:
    """Ingest new documents into the knowledge base.

    Accepts PDF and text file uploads. Documents are chunked, embedded,
    and indexed into the specified collection.
    """
    try:
        strategy = ChunkStrategy(chunk_strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunk strategy: {chunk_strategy}. "
            f"Valid options: {[s.value for s in ChunkStrategy]}",
        )

    processor = DocumentProcessor(
        chunk_strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    vs = _get_vector_store()
    total_chunks = 0
    errors: list[str] = []
    files_processed = 0

    for upload_file in files:
        if not upload_file.filename:
            continue

        try:
            # Save uploaded file to temp location
            suffix = Path(upload_file.filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await upload_file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Process and index
            chunks = processor.process_file(tmp_path)
            if chunks:
                # Override source_file with original filename
                for chunk in chunks:
                    chunk.metadata.source_file = upload_file.filename

                indexed = vs.index_chunks(chunks, collection)
                total_chunks += indexed
                files_processed += 1

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.exception("api.ingest_error", filename=upload_file.filename)
            errors.append(f"{upload_file.filename}: {e!s}")

    return DocumentIngestResponse(
        files_processed=files_processed,
        chunks_created=total_chunks,
        collection=collection,
        errors=errors,
    )


@app.get("/documents/search", response_model=DocumentSearchResponse, tags=["Documents"])
async def search_documents(
    query: str,
    collection: str | None = None,
    n_results: int = 10,
    drug_name: str | None = None,
    use_mmr: bool = False,
) -> DocumentSearchResponse:
    """Search the document knowledge base.

    Returns relevant document chunks with metadata and relevance scores.
    """
    vs = _get_vector_store()

    collection_names: list[CollectionName] | None = None
    if collection:
        try:
            collection_names = [CollectionName(collection)]
        except ValueError:
            # Use as raw string collection name
            collection_names = None

    search_query = SearchQuery(
        query_text=query,
        collection_names=collection_names,
        n_results=n_results,
        drug_name=drug_name,
        use_mmr=use_mmr,
        use_hybrid=True,
    )

    results = vs.search(search_query)

    chunks = [
        DocumentChunkResponse(
            chunk_id=r.chunk_id,
            text=r.text,
            metadata={k: v for k, v in r.metadata.items()},
            relevance_score=round(r.relevance_score, 4),
            collection=r.collection,
        )
        for r in results
    ]

    return DocumentSearchResponse(
        query=query,
        results=chunks,
        total_results=len(chunks),
    )


# ── WebSocket chat ──────────────────────────────────────────────────────────


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time chat interaction.

    Protocol:
    - Client sends: {"type": "question", "content": "...", "metadata": {...}}
    - Server sends: {"type": "answer_chunk", "content": "..."}
    - Server sends: {"type": "citation", "content": "...", "metadata": {...}}
    - Server sends: {"type": "answer_complete", "content": "", "metadata": {...}}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    chain = _get_chain(session_id)

    logger.info("websocket.connected", session_id=session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = WebSocketMessage.model_validate_json(raw)
            except Exception:
                await websocket.send_json({"type": "error", "content": "Invalid message format."})
                continue

            if message.type != "question":
                await websocket.send_json(
                    {"type": "error", "content": f"Unknown message type: {message.type}"}
                )
                continue

            question = message.content
            drug_name = message.metadata.get("drug_name") if message.metadata else None
            drug_name_str = str(drug_name) if drug_name else None

            try:
                full_answer: list[str] = []
                async for chunk in chain.ask_stream(
                    question=question,
                    drug_name=drug_name_str,
                ):
                    full_answer.append(chunk)
                    await websocket.send_json({"type": "answer_chunk", "content": chunk})

                # Send completion with metadata
                await websocket.send_json(
                    {
                        "type": "answer_complete",
                        "content": "",
                        "metadata": {
                            "session_id": session_id,
                            "answer_length": len("".join(full_answer)),
                        },
                    }
                )

            except Exception as e:
                logger.exception("websocket.error", session_id=session_id)
                await websocket.send_json({"type": "error", "content": f"Error: {e!s}"})

    except WebSocketDisconnect:
        logger.info("websocket.disconnected", session_id=session_id)
        _sessions.pop(session_id, None)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "pharm_assist.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
