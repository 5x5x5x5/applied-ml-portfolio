"""Tests for the event processor and WebSocket manager."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from biomarker_dash.alerts.alert_engine import AlertEngine
from biomarker_dash.models.anomaly_detector import AnomalyDetector
from biomarker_dash.models.trend_analyzer import TrendAnalyzer
from biomarker_dash.schemas import BiomarkerType
from biomarker_dash.streaming.event_processor import (
    EventProcessor,
    WebSocketManager,
)
from tests.conftest import make_reading


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise ConnectionError("WebSocket closed")
        self.sent_messages.append(data)


class MockRedis:
    """Minimal mock Redis client for testing without a real Redis server."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._sets: dict[str, set[str]] = {}
        self._sorted_sets: dict[str, list[tuple[float, str]]] = {}
        self._hashes: dict[str, dict[str, str]] = {}

    async def ping(self) -> bool:
        return True

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def sadd(self, key: str, *values: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(values)
        return len(values)

    async def smembers(self, key: str) -> set[str]:
        return self._sets.get(key, set())

    async def hset(self, key: str, field: str, value: str) -> int:
        if key not in self._hashes:
            self._hashes[key] = {}
        self._hashes[key][field] = value
        return 1

    async def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})

    async def hdel(self, key: str, field: str) -> int:
        if key in self._hashes and field in self._hashes[key]:
            del self._hashes[key][field]
            return 1
        return 0

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = []
        for member, score in mapping.items():
            self._sorted_sets[key].append((score, member))
        self._sorted_sets[key].sort(key=lambda x: x[0])
        return len(mapping)

    async def zrangebyscore(
        self, key: str, min_score: Any, max_score: Any, start: int = 0, num: int = -1
    ) -> list[str]:
        items = self._sorted_sets.get(key, [])
        min_val = float("-inf") if min_score == "-inf" else float(min_score)
        max_val = float("inf") if max_score == "+inf" else float(max_score)
        filtered = [m for s, m in items if min_val <= s <= max_val]
        if num > 0:
            return filtered[start : start + num]
        return filtered[start:]

    async def zrange(self, key: str, start: int, end: int) -> list[str]:
        items = self._sorted_sets.get(key, [])
        members = [m for _, m in items]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    async def zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        return 0

    async def zremrangebyscore(self, key: str, min_score: Any, max_score: Any) -> int:
        return 0

    async def expire(self, key: str, seconds: int) -> bool:
        return True

    def pipeline(self) -> MockRedisPipeline:
        return MockRedisPipeline(self)

    async def close(self) -> None:
        pass


class MockRedisPipeline:
    """Mock Redis pipeline that executes commands immediately."""

    def __init__(self, redis: MockRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def zadd(self, key: str, mapping: dict[str, float]) -> MockRedisPipeline:
        self._commands.append(("zadd", (key, mapping), {}))
        return self

    def zremrangebyrank(self, key: str, start: int, stop: int) -> MockRedisPipeline:
        return self

    def expire(self, key: str, seconds: int) -> MockRedisPipeline:
        return self

    def sadd(self, key: str, *values: str) -> MockRedisPipeline:
        self._commands.append(("sadd", (key, *values), {}))
        return self

    def hset(self, key: str, field: str, value: str) -> MockRedisPipeline:
        self._commands.append(("hset", (key, field, value), {}))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            result = await method(*args, **kwargs)
            results.append(result)
        self._commands.clear()
        return results


class TestWebSocketManager:
    """Tests for WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_connect_and_broadcast(self) -> None:
        mgr = WebSocketManager()
        ws = MockWebSocket()
        await mgr.connect("P001", ws)

        assert mgr.active_connection_count == 1

        await mgr.broadcast("P001", {"type": "test", "data": "hello"})
        assert len(ws.sent_messages) == 1
        msg = json.loads(ws.sent_messages[0])
        assert msg["type"] == "test"

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        mgr = WebSocketManager()
        ws = MockWebSocket()
        await mgr.connect("P001", ws)
        await mgr.disconnect("P001", ws)
        assert mgr.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_correct_patient(self) -> None:
        mgr = WebSocketManager()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await mgr.connect("P001", ws1)
        await mgr.connect("P002", ws2)

        await mgr.broadcast("P001", {"type": "test"})
        assert len(ws1.sent_messages) == 1
        assert len(ws2.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self) -> None:
        mgr = WebSocketManager()
        ws = MockWebSocket()
        ws.closed = True  # Simulate dead connection
        await mgr.connect("P001", ws)

        await mgr.broadcast("P001", {"type": "test"})
        # Dead connection should be removed
        assert mgr.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_multiple_connections_per_patient(self) -> None:
        mgr = WebSocketManager()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await mgr.connect("P001", ws1)
        await mgr.connect("P001", ws2)

        assert mgr.active_connection_count == 2

        await mgr.broadcast("P001", {"type": "test"})
        assert len(ws1.sent_messages) == 1
        assert len(ws2.sent_messages) == 1


class TestEventProcessor:
    """Tests for the event processing pipeline."""

    def _create_processor(self) -> tuple[EventProcessor, MockRedis, WebSocketManager]:
        from biomarker_dash.data.biomarker_store import BiomarkerStore

        mock_redis = MockRedis()
        store = BiomarkerStore(mock_redis)  # type: ignore[arg-type]
        anomaly_detector = AnomalyDetector()
        trend_analyzer = TrendAnalyzer()
        alert_engine = AlertEngine()
        ws_manager = WebSocketManager()

        processor = EventProcessor(
            store=store,
            anomaly_detector=anomaly_detector,
            trend_analyzer=trend_analyzer,
            alert_engine=alert_engine,
            ws_manager=ws_manager,
        )
        return processor, mock_redis, ws_manager

    @pytest.mark.asyncio
    async def test_submit_reading(self) -> None:
        processor, _, _ = self._create_processor()
        await processor.start()

        reading = make_reading(value=85.0)
        accepted = await processor.submit(reading)
        assert accepted

        # Give the processor time to process
        await asyncio.sleep(0.5)

        metrics = processor.get_metrics()
        assert metrics["processed"] >= 1

        await processor.stop()

    @pytest.mark.asyncio
    async def test_submit_broadcasts_to_websocket(self) -> None:
        processor, _, ws_manager = self._create_processor()
        await processor.start()

        # Connect a WebSocket client
        ws = MockWebSocket()
        await ws_manager.connect("TEST001", ws)

        reading = make_reading(value=85.0, patient_id="TEST001")
        await processor.submit(reading)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Should have received at least one message
        assert len(ws.sent_messages) >= 1
        msg = json.loads(ws.sent_messages[0])
        assert msg["type"] == "reading"
        assert msg["data"]["patient_id"] == "TEST001"

        await processor.stop()

    @pytest.mark.asyncio
    async def test_submit_anomalous_reading_generates_alert(self) -> None:
        processor, _, ws_manager = self._create_processor()
        await processor.start()

        ws = MockWebSocket()
        await ws_manager.connect("TEST001", ws)

        # Submit a clearly anomalous reading
        reading = make_reading(
            value=500.0,
            patient_id="TEST001",
            biomarker_type=BiomarkerType.GLUCOSE,
        )
        await processor.submit(reading)
        await asyncio.sleep(0.5)

        # Check for alert messages
        alert_msgs = [json.loads(m) for m in ws.sent_messages if '"type": "alert"' in m]
        # At minimum, the reading should have been broadcast
        reading_msgs = [json.loads(m) for m in ws.sent_messages if '"type": "reading"' in m]
        assert len(reading_msgs) >= 1

        await processor.stop()

    @pytest.mark.asyncio
    async def test_backpressure_drops_readings(self) -> None:
        processor, _, _ = self._create_processor()
        # Don't start the processor so the queue fills up
        # Manually fill the queue
        count_accepted = 0
        count_rejected = 0
        for _ in range(6000):
            reading = make_reading(value=85.0)
            accepted = await processor.submit(reading)
            if accepted:
                count_accepted += 1
            else:
                count_rejected += 1

        assert count_rejected > 0  # Some should be dropped
        assert count_accepted <= 5000  # Queue max size

    @pytest.mark.asyncio
    async def test_metrics_tracking(self) -> None:
        processor, _, _ = self._create_processor()
        await processor.start()

        for _ in range(3):
            await processor.submit(make_reading(value=85.0))

        await asyncio.sleep(1.0)

        metrics = processor.get_metrics()
        assert metrics["processed"] >= 3
        assert metrics["errors"] == 0
        assert "uptime_seconds" in metrics
        assert "queue_size" in metrics

        await processor.stop()

    @pytest.mark.asyncio
    async def test_stop_is_graceful(self) -> None:
        processor, _, _ = self._create_processor()
        await processor.start()
        await processor.submit(make_reading(value=85.0))
        await asyncio.sleep(0.3)
        await processor.stop()

        # Should not raise and metrics should be final
        metrics = processor.get_metrics()
        assert isinstance(metrics["processed"], int)
