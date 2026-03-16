## PharmaAgents

Multi-Agent AI System for pharmaceutical research workflows using Claude API with tool use.

### Setup

```bash
cd /home/danny/projects/AI_ML_projects/20-pharma-agents
uv sync --all-extras
```

### Running

```bash
# API server
ANTHROPIC_API_KEY=sk-... uv run uvicorn pharma_agents.api.main:app --reload

# With Docker
docker compose up --build
```

### Environment

- `ANTHROPIC_API_KEY` - Required for Claude API calls
- `REDIS_URL` - Optional Redis connection (default: redis://localhost:6379/0)

### Testing

```bash
uv run pytest tests/ -v
```

### Project Structure

- `src/pharma_agents/agents/` - Specialized agents (literature, safety, chemistry, regulatory)
- `src/pharma_agents/orchestrator/` - Multi-agent coordination and predefined workflows
- `src/pharma_agents/tools/` - Simulated tools (PubMed, molecule analysis, drug database)
- `src/pharma_agents/api/` - FastAPI application with REST and WebSocket endpoints
- `frontend/` - Browser-based multi-agent chat dashboard

### Conventions

- Python 3.11+, type hints on all public functions
- Logging via structlog (no print statements)
- Linting: ruff. Formatting: ruff format. Type checking: mypy
- Tests in tests/ using pytest with pytest-asyncio
- uv for dependency management
- Default branch: main
