# PharmAssistAI - LLM-Powered Pharmaceutical Knowledge Assistant

A production-grade RAG (Retrieval-Augmented Generation) pipeline for answering pharmaceutical questions about drugs, interactions, clinical guidelines, and regulatory requirements.

## Architecture

```
User Question
    |
    v
[Safety Guardrails] ---- Block harmful/off-topic queries
    |
    v
[Multi-Query Generator] - Rephrase for better recall
    |
    v
[Vector Store Search] --- ChromaDB hybrid search (semantic + keyword)
    |                      with MMR diversity re-ranking
    v
[Contextual Compression] - Extract relevant portions
    |
    v
[Parent Doc Retrieval] -- Expand to full section context
    |
    v
[LLM Chain (Claude)] ---- Generate cited answer with
    |                      hallucination guardrails
    v
[Post-Processing] ------- Inject disclaimers, confidence score
    |
    v
Cited Answer + Sources
```

## Features

- **Document Ingestion Pipeline**: Parse PDF drug labels, clinical guidelines, FDA documents with multiple chunking strategies (semantic, recursive, fixed-size, section-based)
- **ChromaDB Vector Store**: Hybrid search combining semantic similarity with keyword matching, MMR for result diversity, metadata filtering by drug name and section
- **Advanced Retrieval**: Multi-query retrieval, parent document expansion, contextual compression, source attribution with inline citations
- **LLM Chain**: Claude-powered answer generation with pharmaceutical domain system prompt, conversation memory for follow-up questions, hallucination guardrails
- **Safety Guardrails**: Harmful content detection, off-topic filtering, emergency situation handling, medical disclaimer injection, confidence scoring
- **FastAPI Service**: REST + WebSocket endpoints for Q&A, document ingestion, and knowledge base search
- **Chat Frontend**: Professional dark-themed UI with streaming responses, citation expansion, document upload

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key

### Setup

```bash
cd 13-llm-pharma-assistant

# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Run the application
uv run uvicorn pharm_assist.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080/static/index.html to access the chat interface.

### Docker

```bash
# Set your API key in .env
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Start app + ChromaDB
docker compose up --build
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a pharmaceutical question (supports streaming via SSE) |
| POST | `/documents/ingest` | Upload and ingest new documents |
| GET | `/documents/search` | Search the document knowledge base |
| GET | `/health` | Health check with collection stats |
| WS | `/ws/chat` | WebSocket endpoint for real-time chat |

## Project Structure

```
13-llm-pharma-assistant/
  src/pharm_assist/
    rag/
      document_processor.py  # Chunking, metadata extraction, normalization
      embeddings.py          # Embedding generation with caching
      vector_store.py        # ChromaDB indexing, hybrid search, MMR
      retriever.py           # Multi-query, parent doc, compression
    llm/
      chain.py               # Claude LLM chain with memory
      guardrails.py          # Safety checks, disclaimers, confidence
    api/
      main.py                # FastAPI application
      models.py              # Pydantic schemas
  frontend/                  # Chat UI (HTML/CSS/JS)
  data/sample_drug_labels/   # Sample FDA drug labels
  tests/                     # pytest test suite
```

## Sample Data

Includes real FDA-style drug labels for Aspirin and Metformin covering indications, dosing, contraindications, adverse reactions, drug interactions, and clinical pharmacology.

## Key Technologies

- **Anthropic Claude** (via `anthropic` SDK) - LLM for answer generation and query expansion
- **ChromaDB** - Vector database for document embeddings and similarity search
- **LangChain Text Splitters** - Document chunking utilities
- **FastAPI** - Async web framework with WebSocket support
- **Pydantic** - Request/response validation
- **structlog** - Structured logging
- **tiktoken** - Token counting for chunk size optimization
