"""Custom middleware for latency tracking, correlation IDs, and caching.

Designed for minimal overhead to maintain sub-100ms latency target.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import orjson
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add correlation IDs to every request for distributed tracing.

    Checks for incoming X-Request-ID header, generates one if absent.
    Adds it to response headers and structlog context.
    """

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        # Bind to structlog context for all downstream log calls
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Store on request state for access in handlers
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LatencyTracker(BaseHTTPMiddleware):
    """Track per-endpoint latency and add timing headers.

    Records timing to the performance monitor and adds
    X-Response-Time header to every response.
    """

    def __init__(self, app: Any, performance_monitor: Any = None) -> None:
        super().__init__(app)
        self._monitor = performance_monitor

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{elapsed_ms:.3f}ms"
        response.headers["X-Latency-Target"] = "100ms"

        # Record to performance monitor
        if self._monitor is not None:
            self._monitor.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                latency_ms=elapsed_ms,
            )

        # Log slow requests
        if elapsed_ms > 100:
            logger.warning(
                "slow_request",
                path=request.url.path,
                method=request.method,
                latency_ms=round(elapsed_ms, 3),
                status=response.status_code,
            )

        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """Redis-based response caching for identical prediction requests.

    Only caches POST /predict responses to avoid caching health checks.
    Uses the request body hash as the cache key.
    """

    CACHEABLE_PATHS = {"/predict"}

    def __init__(self, app: Any, redis_cache: Any = None) -> None:
        super().__init__(app)
        self._cache = redis_cache

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        # Only cache specific endpoints
        if (
            self._cache is None
            or request.method != "POST"
            or request.url.path not in self.CACHEABLE_PATHS
        ):
            return await call_next(request)

        # Read and cache the body for cache key generation
        body = await request.body()
        if not body:
            return await call_next(request)

        try:
            patient_data = orjson.loads(body)
        except (orjson.JSONDecodeError, ValueError):
            return await call_next(request)

        # Check cache
        cached_result = self._cache.get_prediction(patient_data)
        if cached_result is not None:
            cached_result["cache_hit"] = True
            return Response(
                content=orjson.dumps(cached_result),
                media_type="application/json",
                headers={
                    "X-Cache": "HIT",
                    "Content-Type": "application/json",
                },
            )

        # Cache miss - proceed with request
        response = await call_next(request)

        # Cache the response if successful
        if response.status_code == 200:
            try:
                # Read the response body
                response_body = b""
                async for chunk in response.body_iterator:
                    if isinstance(chunk, str):
                        response_body += chunk.encode()
                    else:
                        response_body += chunk

                # Cache the result
                try:
                    result = orjson.loads(response_body)
                    self._cache.set_prediction(patient_data, result)
                except (orjson.JSONDecodeError, ValueError):
                    pass

                # Return new response with the body
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception as exc:
                logger.warning("cache_middleware_error", error=str(exc))

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter.

    Uses a sliding window counter per client IP.
    For production, this should be backed by Redis for distributed rate limiting.
    """

    def __init__(
        self,
        app: Any,
        max_requests_per_minute: int = 600,
    ) -> None:
        super().__init__(app)
        self._max_rpm = max_requests_per_minute
        self._window_seconds = 60.0
        self._requests: dict[str, list[float]] = {}

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        cutoff = now - self._window_seconds

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Remove old entries
        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if ts > cutoff]

        if len(self._requests[client_ip]) >= self._max_rpm:
            return True

        self._requests[client_ip].append(now)
        return False

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        client_ip = self._get_client_ip(request)

        if self._is_rate_limited(client_ip):
            logger.warning("rate_limited", client_ip=client_ip)
            return Response(
                content=orjson.dumps(
                    {
                        "error": "rate_limit_exceeded",
                        "message": f"Max {self._max_rpm} requests per minute",
                        "retry_after_seconds": 60,
                    }
                ),
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
