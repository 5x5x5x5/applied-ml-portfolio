## PharmAssistAI
- Python project using `uv` for package management
- Linting: ruff. Type checking: mypy (strict). Formatting: ruff format
- Tests: pytest in tests/ directory (async with pytest-asyncio)
- Run tests: `uv run pytest tests/`
- Run app: `uv run uvicorn pharm_assist.api.main:app --reload`

## Key Architecture
- RAG pipeline: document processing + ChromaDB vector store in `src/pharm_assist/rag/`
- Claude API integration with LangChain chains + guardrails in `llm/`
- FastAPI service with WebSocket support in `api/`, frontend in `frontend/`

## Conventions
- Type hints on all public functions
- Use structlog, never print()
- Default branch: main
