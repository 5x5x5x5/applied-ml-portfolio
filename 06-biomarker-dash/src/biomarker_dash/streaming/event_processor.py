"""Real-time event processing pipeline for biomarker readings.

Processes incoming biomarker readings through the ML anomaly detection
and alert engines, publishes updates via WebSocket, and handles
backpressure with buffering and batching.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from biomarker_dash.alerts.alert_engine import AlertEngine
from biomarker_dash.data.biomarker_store import BiomarkerStore
from biomarker_dash.models.anomaly_detector import AnomalyDetector
from biomarker_dash.models.trend_analyzer import TrendAnalyzer
from biomarker_dash.schemas import (
    BiomarkerReading,
    ClinicalAlert,
    TrendResult,
)

logger = logging.getLogger(__name__)

# Processing configuration
BATCH_SIZE = 10
BATCH_TIMEOUT_SECONDS = 1.0
MAX_QUEUE_SIZE = 5000
TREND_ANALYSIS_INTERVAL = 60  # Seconds between trend analyses per patient/biomarker
ESCALATION_CHECK_INTERVAL = 30  # Seconds between escalation checks


class WebSocketManager:
    """Manages WebSocket connections for real-time client updates.

    Groups connections by patient_id so updates are routed only to
    interested subscribers.
    """

    def __init__(self) -> None:
        # patient_id -> set of WebSocket connections
        self._connections: dict[str, set[Any]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, patient_id: str, websocket: Any) -> None:
        """Register a WebSocket connection for a patient."""
        async with self._lock:
            self._connections[patient_id].add(websocket)
        logger.info(
            "WebSocket connected for patient %s (total: %d)",
            patient_id,
            len(self._connections[patient_id]),
        )

    async def disconnect(self, patient_id: str, websocket: Any) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections[patient_id].discard(websocket)
            if not self._connections[patient_id]:
                del self._connections[patient_id]
        logger.info("WebSocket disconnected for patient %s", patient_id)

    async def broadcast(self, patient_id: str, message: dict[str, Any]) -> None:
        """Send a message to all WebSocket connections for a patient."""
        connections = self._connections.get(patient_id, set())
        if not connections:
            return

        import json

        payload = json.dumps(message, default=str)
        dead_connections: list[Any] = []

        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead_connections.append(ws)
                logger.debug("Removing dead WebSocket for patient %s", patient_id)

        # Clean up dead connections
        for ws in dead_connections:
            await self.disconnect(patient_id, ws)

    async def broadcast_alert(self, alert: ClinicalAlert) -> None:
        """Broadcast an alert to the specific patient's subscribers."""
        message = {
            "type": "alert",
            "data": {
                "alert_id": alert.alert_id,
                "patient_id": alert.patient_id,
                "biomarker_type": alert.biomarker_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "value": alert.value,
                "created_at": alert.created_at.isoformat(),
            },
        }
        await self.broadcast(alert.patient_id, message)

    @property
    def active_connection_count(self) -> int:
        """Total number of active WebSocket connections."""
        return sum(len(conns) for conns in self._connections.values())


class EventProcessor:
    """Asynchronous event processor for biomarker readings.

    Ingests readings, runs them through the anomaly detector and alert
    engine, stores results, and pushes updates to WebSocket clients.

    Uses an internal asyncio.Queue with batching for efficiency and
    backpressure handling.
    """

    def __init__(
        self,
        store: BiomarkerStore,
        anomaly_detector: AnomalyDetector,
        trend_analyzer: TrendAnalyzer,
        alert_engine: AlertEngine,
        ws_manager: WebSocketManager,
    ) -> None:
        self._store = store
        self._anomaly_detector = anomaly_detector
        self._trend_analyzer = trend_analyzer
        self._alert_engine = alert_engine
        self._ws_manager = ws_manager

        self._queue: asyncio.Queue[BiomarkerReading] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

        # Track last trend analysis time per (patient_id, biomarker_type)
        self._last_trend_analysis: dict[str, float] = {}

        # Metrics
        self._processed_count = 0
        self._dropped_count = 0
        self._error_count = 0
        self._start_time: float = 0.0

    async def start(self) -> None:
        """Start the event processing loop."""
        if self._running:
            logger.warning("EventProcessor already running")
            return

        self._running = True
        self._start_time = time.monotonic()
        self._tasks = [
            asyncio.create_task(self._processing_loop(), name="event-processor"),
            asyncio.create_task(self._escalation_loop(), name="escalation-checker"),
        ]
        logger.info("EventProcessor started with batch_size=%d", BATCH_SIZE)

    async def stop(self) -> None:
        """Gracefully stop the event processor."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info(
            "EventProcessor stopped. Processed: %d, Dropped: %d, Errors: %d",
            self._processed_count,
            self._dropped_count,
            self._error_count,
        )

    async def submit(self, reading: BiomarkerReading) -> bool:
        """Submit a reading for processing.

        Returns True if accepted, False if backpressure forced a drop.
        """
        if not reading.reading_id:
            reading.reading_id = str(uuid.uuid4())

        try:
            self._queue.put_nowait(reading)
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning(
                "Queue full (%d). Dropped reading %s for patient %s",
                MAX_QUEUE_SIZE,
                reading.reading_id,
                reading.patient_id,
            )
            return False

    async def _processing_loop(self) -> None:
        """Main processing loop: collects batches and processes them."""
        while self._running:
            batch: list[BiomarkerReading] = []
            try:
                # Wait for at least one item
                first = await asyncio.wait_for(self._queue.get(), timeout=BATCH_TIMEOUT_SECONDS)
                batch.append(first)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Collect more items up to batch size (non-blocking)
            while len(batch) < BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

            # Process the batch
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[BiomarkerReading]) -> None:
        """Process a batch of readings through the pipeline."""
        for reading in batch:
            try:
                await self._process_single(reading)
                self._processed_count += 1
            except Exception:
                self._error_count += 1
                logger.exception(
                    "Error processing reading %s for patient %s",
                    reading.reading_id,
                    reading.patient_id,
                )

    async def _process_single(self, reading: BiomarkerReading) -> None:
        """Process a single biomarker reading through all stages."""
        # Stage 1: Store the raw reading
        await self._store.store_reading(reading)

        # Stage 2: Get patient context (if available)
        patient_ctx = await self._store.get_patient_context(reading.patient_id)

        # Stage 3: Anomaly detection
        anomaly_result = self._anomaly_detector.detect(reading, patient_ctx)

        # Stage 4: Rule-based alerts
        rule_alerts = self._alert_engine.evaluate_reading(reading)

        # Stage 5: ML-based alerts (from anomaly detection)
        ml_alerts = self._alert_engine.evaluate_anomaly(anomaly_result)

        # Stage 6: Trend analysis (throttled)
        trend_result = await self._maybe_analyze_trend(reading)
        trend_alerts: list[ClinicalAlert] = []
        if trend_result:
            trend_alerts = self._alert_engine.evaluate_trend(trend_result)

        # Stage 7: Publish updates to WebSocket subscribers
        all_alerts = rule_alerts + ml_alerts + trend_alerts

        # Store alerts
        for alert in all_alerts:
            await self._store.store_alert(alert)

        # Broadcast reading update
        ws_message: dict[str, Any] = {
            "type": "reading",
            "data": {
                "reading_id": reading.reading_id,
                "patient_id": reading.patient_id,
                "biomarker_type": reading.biomarker_type.value,
                "value": reading.value,
                "unit": reading.unit,
                "timestamp": reading.timestamp.isoformat(),
                "anomaly": {
                    "is_anomaly": anomaly_result.is_anomaly,
                    "score": round(anomaly_result.anomaly_score, 3),
                    "severity": anomaly_result.severity.value,
                    "explanation": anomaly_result.explanation,
                    "normal_range": list(anomaly_result.normal_range),
                },
            },
        }

        if trend_result:
            ws_message["data"]["trend"] = {
                "direction": trend_result.direction.value,
                "rate_of_change": trend_result.rate_of_change,
                "predicted_value_24h": trend_result.predicted_value_24h,
                "predicted_exit_normal": trend_result.predicted_exit_normal,
                "confidence": trend_result.confidence,
            }

        await self._ws_manager.broadcast(reading.patient_id, ws_message)

        # Broadcast alerts
        for alert in all_alerts:
            await self._ws_manager.broadcast_alert(alert)

    async def _maybe_analyze_trend(self, reading: BiomarkerReading) -> TrendResult | None:
        """Run trend analysis if enough time has passed since last analysis."""
        key = f"{reading.patient_id}:{reading.biomarker_type.value}"
        now = time.monotonic()
        last = self._last_trend_analysis.get(key, 0.0)

        if now - last < TREND_ANALYSIS_INTERVAL:
            return None

        # Get recent readings for trend analysis
        readings = await self._store.get_readings(
            reading.patient_id, reading.biomarker_type, limit=100
        )

        if len(readings) < 5:
            return None

        self._last_trend_analysis[key] = now
        return self._trend_analyzer.analyze(readings, reading.biomarker_type, window_hours=24)

    async def _escalation_loop(self) -> None:
        """Periodically check for alerts that need escalation."""
        while self._running:
            try:
                await asyncio.sleep(ESCALATION_CHECK_INTERVAL)
                escalated = self._alert_engine.check_escalations()
                for alert in escalated:
                    await self._ws_manager.broadcast_alert(alert)
                    await self._store.store_alert(alert)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in escalation check loop")

    def get_metrics(self) -> dict[str, Any]:
        """Return processing metrics."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        return {
            "processed": self._processed_count,
            "dropped": self._dropped_count,
            "errors": self._error_count,
            "queue_size": self._queue.qsize(),
            "queue_capacity": MAX_QUEUE_SIZE,
            "uptime_seconds": round(uptime, 1),
            "ws_connections": self._ws_manager.active_connection_count,
        }
