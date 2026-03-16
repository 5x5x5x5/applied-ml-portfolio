"""FastAPI application for BiomarkerDash.

Provides REST endpoints for submitting and querying biomarker data,
WebSocket endpoints for real-time streaming, and health checks.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from biomarker_dash import __version__
from biomarker_dash.alerts.alert_engine import AlertEngine
from biomarker_dash.data.biomarker_store import BiomarkerStore
from biomarker_dash.models.anomaly_detector import AnomalyDetector
from biomarker_dash.models.trend_analyzer import TrendAnalyzer
from biomarker_dash.schemas import (
    BiomarkerReading,
    BiomarkerType,
    HealthStatus,
    PatientContext,
)
from biomarker_dash.streaming.event_processor import EventProcessor, WebSocketManager

logger = logging.getLogger(__name__)

# Module-level state (initialized during lifespan)
_state: dict[str, Any] = {}

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend")

START_TIME = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and tear down resources."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting BiomarkerDash v%s", __version__)

    # Initialize Redis
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)
    store = BiomarkerStore(redis_client)

    # Initialize ML components
    anomaly_detector = AnomalyDetector()
    trend_analyzer = TrendAnalyzer()
    alert_engine = AlertEngine()

    # Initialize WebSocket manager and event processor
    ws_manager = WebSocketManager()
    processor = EventProcessor(
        store=store,
        anomaly_detector=anomaly_detector,
        trend_analyzer=trend_analyzer,
        alert_engine=alert_engine,
        ws_manager=ws_manager,
    )
    await processor.start()

    _state["redis"] = redis_client
    _state["store"] = store
    _state["anomaly_detector"] = anomaly_detector
    _state["trend_analyzer"] = trend_analyzer
    _state["alert_engine"] = alert_engine
    _state["ws_manager"] = ws_manager
    _state["processor"] = processor

    logger.info("All components initialized successfully")
    yield

    # Shutdown
    logger.info("Shutting down BiomarkerDash...")
    await processor.stop()
    await redis_client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="BiomarkerDash",
    description="Real-time Biomarker Monitoring Dashboard with ML Anomaly Detection",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_store() -> BiomarkerStore:
    return _state["store"]


def _get_processor() -> EventProcessor:
    return _state["processor"]


def _get_ws_manager() -> WebSocketManager:
    return _state["ws_manager"]


def _get_alert_engine() -> AlertEngine:
    return _state["alert_engine"]


def _get_anomaly_detector() -> AnomalyDetector:
    return _state["anomaly_detector"]


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Application health check endpoint."""
    store = _get_store()
    redis_ok = await store.ping()
    ws_mgr = _get_ws_manager()

    return HealthStatus(
        status="healthy" if redis_ok else "degraded",
        version=__version__,
        redis_connected=redis_ok,
        uptime_seconds=round(time.monotonic() - START_TIME, 1),
        active_ws_connections=ws_mgr.active_connection_count,
    )


# ------------------------------------------------------------------
# Biomarker readings
# ------------------------------------------------------------------


@app.post("/api/biomarkers", status_code=201)
async def submit_biomarker(reading: BiomarkerReading) -> dict[str, Any]:
    """Submit a new biomarker reading for processing.

    The reading will be:
    1. Stored in the time-series database
    2. Evaluated by the ML anomaly detector
    3. Checked against clinical alert rules
    4. Broadcast to WebSocket subscribers
    """
    processor = _get_processor()
    accepted = await processor.submit(reading)

    if not accepted:
        raise HTTPException(
            status_code=503,
            detail="Processing queue full. Please retry.",
        )

    return {
        "status": "accepted",
        "reading_id": reading.reading_id,
        "patient_id": reading.patient_id,
        "biomarker_type": reading.biomarker_type.value,
    }


@app.get("/api/patients/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    hours: int = Query(default=24, ge=1, le=720),
    biomarker_type: str | None = Query(default=None),
) -> dict[str, Any]:
    """Get historical biomarker data for a patient.

    Returns time-series data grouped by biomarker type.
    """
    store = _get_store()

    if biomarker_type:
        try:
            bt = BiomarkerType(biomarker_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown biomarker type: {biomarker_type}",
            )
        from datetime import timedelta

        start = datetime.utcnow() - timedelta(hours=hours)
        readings = await store.get_readings(patient_id, bt, start=start)
        return {
            "patient_id": patient_id,
            "hours": hours,
            "biomarkers": {bt.value: [r.model_dump(mode="json") for r in readings]},
        }

    history = await store.get_patient_history(patient_id, hours=hours)
    return {
        "patient_id": patient_id,
        "hours": hours,
        "biomarkers": {
            bt: [r.model_dump(mode="json") for r in readings] for bt, readings in history.items()
        },
    }


@app.get("/api/patients/{patient_id}/anomalies")
async def get_patient_anomalies(
    patient_id: str,
    hours: int = Query(default=24, ge=1, le=720),
) -> dict[str, Any]:
    """Get detected anomalies for a patient.

    Returns anomaly detection results from the ML models.
    """
    store = _get_store()
    anomaly_detector = _get_anomaly_detector()

    # Get baseline stats for all biomarker types this patient has data for
    baselines: dict[str, Any] = {}
    for bt in BiomarkerType:
        stats = anomaly_detector.get_baseline_stats(patient_id, bt.value)
        if stats.get("count", 0) > 0:
            baselines[bt.value] = stats

    # Get recent alerts triggered by anomaly detection
    alerts = await store.get_patient_alerts(patient_id, hours=hours)
    anomaly_alerts = [
        a.model_dump(mode="json") for a in alerts if a.detection_source in ("anomaly_ml", "trend")
    ]

    return {
        "patient_id": patient_id,
        "baselines": baselines,
        "anomaly_alerts": anomaly_alerts,
        "total_anomalies": len(anomaly_alerts),
    }


# ------------------------------------------------------------------
# Alerts
# ------------------------------------------------------------------


@app.get("/api/alerts")
async def get_active_alerts(
    patient_id: str | None = Query(default=None),
    min_severity: str = Query(default="info"),
) -> dict[str, Any]:
    """Get active clinical alerts.

    Optionally filter by patient_id and minimum severity level.
    """
    alert_engine = _get_alert_engine()

    from biomarker_dash.schemas import AlertSeverity

    try:
        sev = AlertSeverity(min_severity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid severity: {min_severity}",
        )

    alerts = alert_engine.get_active_alerts(patient_id=patient_id, min_severity=sev)
    return {
        "alerts": [a.model_dump(mode="json") for a in alerts],
        "total": len(alerts),
        "stats": alert_engine.get_alert_stats(),
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user: str = Query(..., description="User acknowledging the alert"),
) -> dict[str, str]:
    """Acknowledge a clinical alert."""
    alert_engine = _get_alert_engine()
    store = _get_store()

    success = alert_engine.acknowledge_alert(alert_id, user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    await store.acknowledge_alert(alert_id, user)
    return {"status": "acknowledged", "alert_id": alert_id, "by": user}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> dict[str, str]:
    """Resolve and deactivate a clinical alert."""
    alert_engine = _get_alert_engine()
    store = _get_store()

    success = alert_engine.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    await store.resolve_alert(alert_id)
    return {"status": "resolved", "alert_id": alert_id}


# ------------------------------------------------------------------
# Patient context
# ------------------------------------------------------------------


@app.post("/api/patients", status_code=201)
async def register_patient(ctx: PatientContext) -> dict[str, str]:
    """Register or update a patient's clinical context."""
    store = _get_store()
    await store.save_patient_context(ctx)
    return {"status": "registered", "patient_id": ctx.patient_id}


@app.get("/api/patients")
async def list_patients() -> dict[str, Any]:
    """List all known patient IDs."""
    store = _get_store()
    patient_ids = await store.list_patient_ids()
    return {"patients": patient_ids, "total": len(patient_ids)}


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


@app.get("/api/metrics")
async def get_metrics() -> dict[str, Any]:
    """Return processing and system metrics."""
    processor = _get_processor()
    return processor.get_metrics()


# ------------------------------------------------------------------
# WebSocket endpoint
# ------------------------------------------------------------------


@app.websocket("/ws/biomarkers/{patient_id}")
async def websocket_biomarkers(websocket: WebSocket, patient_id: str) -> None:
    """Real-time biomarker stream for a specific patient.

    Sends JSON messages with types: 'reading', 'alert', 'trend'.
    """
    ws_manager = _get_ws_manager()
    await websocket.accept()
    await ws_manager.connect(patient_id, websocket)

    logger.info("WebSocket client connected for patient %s", patient_id)

    try:
        # Send initial data snapshot
        store = _get_store()
        history = await store.get_patient_history(patient_id, hours=1)
        import json

        initial_data: dict[str, Any] = {
            "type": "snapshot",
            "data": {
                "patient_id": patient_id,
                "biomarkers": {},
            },
        }
        for bt, readings in history.items():
            initial_data["data"]["biomarkers"][bt] = [
                {
                    "value": r.value,
                    "unit": r.unit,
                    "timestamp": r.timestamp.isoformat(),
                    "biomarker_type": r.biomarker_type.value,
                }
                for r in readings[-50:]  # Last 50 per biomarker
            ]
        await websocket.send_json(initial_data)

        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Client can send ping or configuration messages
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except (json.JSONDecodeError, AttributeError):
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for patient %s", patient_id)
    except Exception:
        logger.exception("WebSocket error for patient %s", patient_id)
    finally:
        await ws_manager.disconnect(patient_id, websocket)


# ------------------------------------------------------------------
# Static frontend files
# ------------------------------------------------------------------


# Serve frontend at root
@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve the main dashboard page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return FileResponse(os.path.join(os.path.dirname(__file__), "fallback.html"))


# Mount static files for CSS/JS
try:
    if os.path.isdir(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
except Exception:
    logger.warning("Could not mount frontend static directory at %s", FRONTEND_DIR)
