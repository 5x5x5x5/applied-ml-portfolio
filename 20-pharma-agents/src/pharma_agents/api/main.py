"""FastAPI application for PharmaAgents multi-agent system.

Provides REST endpoints and WebSocket for interactive multi-agent sessions.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pharma_agents.orchestrator.coordinator import AgentCoordinator
from pharma_agents.orchestrator.workflow import WorkflowEngine, WorkflowType

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_coordinator: AgentCoordinator | None = None
_workflow_engine: WorkflowEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources on startup."""
    global _coordinator, _workflow_engine
    logger.info("api.startup")
    _coordinator = AgentCoordinator()
    _workflow_engine = WorkflowEngine(coordinator=_coordinator)
    yield
    logger.info("api.shutdown")


app = FastAPI(
    title="PharmaAgents",
    description="Multi-Agent AI System for pharmaceutical research workflows",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
import pathlib

_FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    question: str
    agent: str | None = None  # If set, route to specific agent; otherwise auto-route


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""

    query: str
    agent_responses: list[dict[str, Any]]
    synthesis: str
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    processing_time_s: float


class WorkflowRequest(BaseModel):
    """Request body for /workflow endpoint."""

    parameters: dict[str, str]


class WorkflowResponse(BaseModel):
    """Response body for /workflow endpoint."""

    workflow_type: str
    steps_completed: int
    step_results: list[dict[str, Any]]
    synthesis: str
    processing_time_s: float


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> HTMLResponse:
    """Serve the frontend."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(
        content="<h1>PharmaAgents API</h1><p>Frontend not found. Use /docs for API docs.</p>"
    )


@app.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all available agents and their capabilities."""
    assert _coordinator is not None
    agents = _coordinator.list_agents()
    workflows = WorkflowEngine.list_workflows()
    return {
        "agents": agents,
        "workflows": workflows,
        "total_agents": len(agents),
        "total_workflows": len(workflows),
    }


@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    """Submit a research question to be processed by the multi-agent system.

    If ``agent`` is specified, the query is routed to that single agent.
    Otherwise, the coordinator auto-decomposes and routes the query.
    """
    assert _coordinator is not None
    start = time.time()

    try:
        if request.agent:
            # Single-agent mode
            response = await _coordinator.query_single_agent(request.agent, request.question)
            return QueryResponse(
                query=request.question,
                agent_responses=[response.model_dump()],
                synthesis=response.text,
                processing_time_s=round(time.time() - start, 3),
            )
        else:
            # Multi-agent orchestration mode
            result = await _coordinator.process_query(request.question)
            return QueryResponse(
                query=request.question,
                agent_responses=[r.model_dump() for r in result.agent_responses],
                synthesis=result.synthesis,
                conflicts=[c.model_dump() for c in result.conflicts],
                processing_time_s=result.total_processing_time_s,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("api.query.error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")


@app.post("/workflow/{workflow_name}", response_model=WorkflowResponse)
async def execute_workflow(workflow_name: str, request: WorkflowRequest) -> WorkflowResponse:
    """Execute a predefined multi-agent workflow."""
    assert _workflow_engine is not None

    try:
        wf_type = WorkflowType(workflow_name)
    except ValueError:
        available = [wt.value for wt in WorkflowType]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow '{workflow_name}'. Available: {available}",
        )

    try:
        result = await _workflow_engine.execute(wf_type, request.parameters)
        return WorkflowResponse(
            workflow_type=result.workflow_type,
            steps_completed=result.steps_completed,
            step_results=result.step_results,
            synthesis=result.final_synthesis,
            processing_time_s=result.total_time_s,
        )
    except Exception as exc:
        logger.error("api.workflow.error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")


@app.get("/workflow/{workflow_name}/parameters")
async def get_workflow_parameters(workflow_name: str) -> dict[str, Any]:
    """Get required parameters for a workflow."""
    try:
        wf_type = WorkflowType(workflow_name)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown workflow: {workflow_name}")

    params = WorkflowEngine.get_workflow_parameters(wf_type)
    return {"workflow": workflow_name, "required_parameters": params}


# ---------------------------------------------------------------------------
# WebSocket for interactive sessions
# ---------------------------------------------------------------------------


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    """Interactive multi-agent session over WebSocket.

    Client sends JSON messages:
        {"type": "query", "question": "...", "agent": "..."}
        {"type": "workflow", "workflow_name": "...", "parameters": {...}}

    Server sends JSON events:
        {"type": "status", "message": "..."}
        {"type": "agent_start", "agent": "...", "task_id": "..."}
        {"type": "agent_complete", "agent": "...", "response": {...}}
        {"type": "synthesis", "text": "..."}
        {"type": "error", "message": "..."}
    """
    assert _coordinator is not None
    assert _workflow_engine is not None

    await websocket.accept()
    logger.info("ws.session.connected")

    async def send_event(event_type: str, data: dict[str, Any]) -> None:
        await websocket.send_json({"type": event_type, **data})

    async def progress_callback(event: str, data: dict[str, Any]) -> None:
        await send_event(event, data)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send_event("error", {"message": "Invalid JSON"})
                continue

            msg_type = message.get("type")

            if msg_type == "query":
                question = message.get("question", "")
                agent = message.get("agent")
                await send_event("status", {"message": f"Processing query: {question[:80]}..."})

                try:
                    if agent:
                        response = await _coordinator.query_single_agent(agent, question)
                        await send_event(
                            "agent_complete",
                            {
                                "agent": response.agent_name,
                                "response": response.model_dump(),
                            },
                        )
                        await send_event("synthesis", {"text": response.text})
                    else:
                        result = await _coordinator.process_query(
                            question, callback=progress_callback
                        )
                        for resp in result.agent_responses:
                            await send_event(
                                "agent_complete",
                                {
                                    "agent": resp.agent_name,
                                    "response": resp.model_dump(),
                                },
                            )
                        await send_event(
                            "synthesis",
                            {
                                "text": result.synthesis,
                                "conflicts": [c.model_dump() for c in result.conflicts],
                            },
                        )
                except Exception as exc:
                    await send_event("error", {"message": str(exc)})

            elif msg_type == "workflow":
                wf_name = message.get("workflow_name", "")
                params = message.get("parameters", {})
                await send_event("status", {"message": f"Starting workflow: {wf_name}"})

                try:
                    wf_type = WorkflowType(wf_name)
                    result = await _workflow_engine.execute(
                        wf_type, params, callback=progress_callback
                    )
                    await send_event(
                        "workflow_complete",
                        {
                            "workflow_type": result.workflow_type,
                            "steps_completed": result.steps_completed,
                            "synthesis": result.final_synthesis,
                            "processing_time_s": result.total_time_s,
                        },
                    )
                except Exception as exc:
                    await send_event("error", {"message": str(exc)})

            elif msg_type == "list_agents":
                agents = _coordinator.list_agents()
                await send_event("agents_list", {"agents": agents})

            else:
                await send_event("error", {"message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("ws.session.disconnected")
    except Exception as exc:
        logger.error("ws.session.error", error=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "pharma_agents.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
