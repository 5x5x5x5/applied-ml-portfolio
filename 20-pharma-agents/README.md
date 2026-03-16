# PharmaAgents

A Multi-Agent AI System for pharmaceutical research workflows. PharmaAgents uses the Anthropic Claude API with tool use to orchestrate multiple specialized AI agents that collaborate on drug research tasks.

## Architecture

PharmaAgents consists of four specialized agents, each with a distinct persona and expertise domain:

| Agent | Persona | Expertise |
|-------|---------|-----------|
| **LiteratureAgent** | Dr. Lena Reeves | Systematic literature review, evidence grading, citation analysis |
| **SafetyAgent** | Dr. Marcus Okafor | Pharmacovigilance, adverse event analysis, PRR/ROR metrics, risk-benefit |
| **ChemistryAgent** | Dr. Anika Patel | Molecular analysis, drug-likeness (Lipinski/Veber), ADMET prediction |
| **RegulatoryAgent** | Dr. Evelyn Marsh | FDA pathways (NDA/BLA/505(b)(2)), timeline estimation, competitive landscape |

### Multi-Agent Orchestration

The **AgentCoordinator** decomposes complex research queries into subtasks, routes them to appropriate agents, manages dependencies between tasks, detects conflicts between agent outputs, and produces a unified synthesis.

### Predefined Workflows

- **Drug Candidate Assessment** -- All four agents collaborate to evaluate a drug candidate
- **Safety Signal Investigation** -- Safety, literature, and chemistry agents investigate a safety signal
- **Competitive Analysis** -- Regulatory, literature, and safety agents map the competitive landscape
- **Regulatory Strategy** -- Regulatory and literature agents develop an approval strategy

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the server
uv run uvicorn pharma_agents.api.main:app --reload --port 8000

# Open http://localhost:8000 for the web dashboard
```

### Docker

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Submit a research question (auto-routed or agent-specific) |
| `POST` | `/workflow/{name}` | Execute a predefined multi-agent workflow |
| `GET` | `/agents` | List available agents and their capabilities |
| `WebSocket` | `/ws/session` | Interactive multi-agent session with streaming |

## Testing

```bash
uv run pytest tests/ -v
```

## Technology Stack

- **AI**: Anthropic Claude API (claude-sonnet-4-20250514) with tool use
- **Backend**: FastAPI, uvicorn, Pydantic v2, structlog
- **Frontend**: Vanilla HTML/CSS/JS with WebSocket
- **Data**: NumPy, Pandas, simulated pharmaceutical databases
- **Infrastructure**: Docker, Redis, uv
