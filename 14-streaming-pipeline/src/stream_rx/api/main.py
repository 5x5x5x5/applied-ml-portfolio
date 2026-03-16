"""
FastAPI dashboard API for the StreamRx pipeline.

Exposes REST endpoints for metrics, consumer lag, active alerts, and a
WebSocket endpoint for live event streaming to a monitoring dashboard.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stream_rx.config import get_config
from stream_rx.consumers.prescription_processor import PrescriptionProcessor
from stream_rx.consumers.signal_detector import SignalDetector
from stream_rx.logging_setup import configure_logging, get_logger
from stream_rx.monitoring.stream_metrics import MetricsFlusher, StreamMetricsCollector

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialized on startup)
# ---------------------------------------------------------------------------

_metrics_collector: StreamMetricsCollector | None = None
_signal_detector: SignalDetector | None = None
_prescription_processor: PrescriptionProcessor | None = None
_metrics_flusher: MetricsFlusher | None = None
_websocket_clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and tear down shared components."""
    global _metrics_collector, _signal_detector, _prescription_processor, _metrics_flusher

    cfg = get_config()
    configure_logging(level=cfg.log_level, json_output=cfg.environment != "development")

    _metrics_collector = StreamMetricsCollector()
    _signal_detector = SignalDetector()
    _prescription_processor = PrescriptionProcessor()
    _metrics_flusher = MetricsFlusher(_metrics_collector)
    _metrics_flusher.start()

    logger.info("api_started", host=cfg.api_host, port=cfg.api_port)
    yield

    # Shutdown
    if _metrics_flusher:
        _metrics_flusher.stop()
    logger.info("api_shutdown")


app = FastAPI(
    title="StreamRx Dashboard API",
    description="Real-time monitoring and alerting for the pharmaceutical event streaming pipeline.",
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


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ThroughputResponse(BaseModel):
    events_per_sec: float
    total_in_window: int
    window_seconds: int
    topics: dict[str, int]
    timestamp: str


class LagResponse(BaseModel):
    total_lag: int
    partition_count: int
    partitions: list[dict[str, Any]]
    timestamp: str


class ActiveAlertsResponse(BaseModel):
    interaction_alerts: list[dict[str, Any]]
    safety_signals: list[dict[str, Any]]
    total_count: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    environment: str
    version: str


class MetricsSummaryResponse(BaseModel):
    throughput: dict[str, Any]
    latency: dict[str, Any]
    consumer_lag: dict[str, Any]
    error_rate: dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

_startup_time = time.monotonic()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    cfg = get_config()
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.monotonic() - _startup_time, 1),
        environment=cfg.environment,
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# Metrics endpoints
# ---------------------------------------------------------------------------


@app.get("/metrics/throughput", response_model=ThroughputResponse)
async def get_throughput(
    window: int = Query(default=300, ge=10, le=3600, description="Window in seconds"),
) -> ThroughputResponse:
    """Get current processing rates for the pipeline."""
    if _metrics_collector is None:
        return ThroughputResponse(
            events_per_sec=0.0,
            total_in_window=0,
            window_seconds=window,
            topics={},
            timestamp=datetime.utcnow().isoformat(),
        )

    data = _metrics_collector.get_throughput(window_seconds=window)
    return ThroughputResponse(
        events_per_sec=data["events_per_sec"],
        total_in_window=data["total_in_window"],
        window_seconds=data["window_seconds"],
        topics=data["topics"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics/lag", response_model=LagResponse)
async def get_consumer_lag() -> LagResponse:
    """Get consumer lag per partition across all tracked topics."""
    if _metrics_collector is None:
        return LagResponse(
            total_lag=0,
            partition_count=0,
            partitions=[],
            timestamp=datetime.utcnow().isoformat(),
        )

    data = _metrics_collector.get_consumer_lag()
    return LagResponse(
        total_lag=data["total_lag"],
        partition_count=data["partition_count"],
        partitions=data["partitions"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics/latency")
async def get_latency(
    window: int = Query(default=300, ge=10, le=3600),
) -> JSONResponse:
    """Get processing latency percentiles."""
    if _metrics_collector is None:
        return JSONResponse({"p50": 0, "p95": 0, "p99": 0, "avg": 0, "count": 0})
    data = _metrics_collector.get_latency_percentiles(window_seconds=window)
    return JSONResponse({**data, "timestamp": datetime.utcnow().isoformat()})


@app.get("/metrics/errors")
async def get_error_rate(
    window: int = Query(default=300, ge=10, le=3600),
) -> JSONResponse:
    """Get error rate and breakdown."""
    if _metrics_collector is None:
        return JSONResponse({"error_count": 0, "error_rate": 0.0})
    data = _metrics_collector.get_error_rate(window_seconds=window)
    return JSONResponse({**data, "timestamp": datetime.utcnow().isoformat()})


@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary() -> MetricsSummaryResponse:
    """Get a comprehensive metrics summary."""
    if _metrics_collector is None:
        empty: dict[str, Any] = {}
        return MetricsSummaryResponse(
            throughput=empty,
            latency=empty,
            consumer_lag=empty,
            error_rate=empty,
            timestamp=datetime.utcnow().isoformat(),
        )

    summary = _metrics_collector.get_summary()
    return MetricsSummaryResponse(
        throughput=summary["throughput"],
        latency=summary["latency"],
        consumer_lag=summary["consumer_lag"],
        error_rate=summary["error_rate"],
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Alerts endpoints
# ---------------------------------------------------------------------------


@app.get("/alerts/active", response_model=ActiveAlertsResponse)
async def get_active_alerts() -> ActiveAlertsResponse:
    """Get all currently active safety signals and interaction alerts."""
    interaction_alerts: list[dict[str, Any]] = []
    safety_signals: list[dict[str, Any]] = []

    if _prescription_processor is not None:
        pending = _prescription_processor.get_pending_alerts()
        interaction_alerts = [a.model_dump() for a in pending]

    if _signal_detector is not None:
        active = _signal_detector.get_active_signals()
        safety_signals = [s.model_dump() for s in active]

    return ActiveAlertsResponse(
        interaction_alerts=interaction_alerts,
        safety_signals=safety_signals,
        total_count=len(interaction_alerts) + len(safety_signals),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> JSONResponse:
    """Acknowledge and dismiss an active alert."""
    if _signal_detector is not None:
        if _signal_detector.acknowledge_signal(alert_id):
            return JSONResponse({"status": "acknowledged", "alert_id": alert_id})
    return JSONResponse(
        {"status": "not_found", "alert_id": alert_id},
        status_code=404,
    )


@app.get("/alerts/dlq")
async def get_dead_letter_queue(
    limit: int = Query(default=50, ge=1, le=500),
) -> JSONResponse:
    """Retrieve entries from the dead letter queue."""
    if _signal_detector is None:
        return JSONResponse({"entries": [], "count": 0})
    entries = _signal_detector.get_dlq_entries(limit=limit)
    return JSONResponse({"entries": entries, "count": len(entries)})


# ---------------------------------------------------------------------------
# Processor stats
# ---------------------------------------------------------------------------


@app.get("/processors/prescription/stats")
async def get_prescription_processor_stats() -> JSONResponse:
    """Get statistics from the prescription processor."""
    if _prescription_processor is None:
        return JSONResponse({"status": "not_initialized"})
    return JSONResponse(_prescription_processor.stats)


@app.get("/processors/signal/stats")
async def get_signal_detector_stats() -> JSONResponse:
    """Get statistics from the signal detector."""
    if _signal_detector is None:
        return JSONResponse({"status": "not_initialized"})
    return JSONResponse(_signal_detector.stats)


# ---------------------------------------------------------------------------
# WebSocket for live event streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live event streaming to dashboard clients.

    Sends periodic metrics snapshots and any new alerts as they occur.
    """
    await websocket.accept()
    _websocket_clients.add(websocket)
    logger.info("websocket_client_connected", total_clients=len(_websocket_clients))

    try:
        while True:
            # Send periodic metrics snapshot
            snapshot: dict[str, Any] = {
                "type": "metrics_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {},
            }

            if _metrics_collector is not None:
                summary = _metrics_collector.get_summary()
                snapshot["data"] = {
                    "throughput": summary["throughput"],
                    "latency": summary["latency"],
                    "total_lag": summary["consumer_lag"]["total_lag"],
                    "error_rate": summary["error_rate"]["error_rate"],
                    "total_processed": summary["total_processed"],
                }

            # Include active alerts
            if _signal_detector is not None:
                active_signals = _signal_detector.get_active_signals()
                if active_signals:
                    snapshot["alerts"] = [
                        {
                            "alert_id": s.alert_id,
                            "drug": s.drug_name,
                            "reaction": s.reaction_term,
                            "priority": s.priority.value,
                            "metric": s.metric_name,
                            "value": s.metric_value,
                        }
                        for s in active_signals[:10]  # Limit to latest 10
                    ]

            await websocket.send_text(json.dumps(snapshot, default=str))

            # Wait for next update interval or client message
            try:
                # Accept pings from client (non-blocking wait)
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            except TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected")
    except Exception as exc:
        logger.error("websocket_error", error=str(exc))
    finally:
        _websocket_clients.discard(websocket)


async def broadcast_event(event_data: dict[str, Any]) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    if not _websocket_clients:
        return
    message = json.dumps(event_data, default=str)
    disconnected: set[WebSocket] = set()
    for ws in _websocket_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    _websocket_clients.difference_update(disconnected)


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the FastAPI server via uvicorn."""
    cfg = get_config()
    configure_logging(level=cfg.log_level)
    uvicorn.run(
        "stream_rx.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.environment == "development",
        log_level=cfg.log_level.lower(),
        workers=1,
    )


if __name__ == "__main__":
    run_server()
