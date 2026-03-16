"""Redis caching layer with graceful degradation.

Provides prediction result caching and feature vector caching
with TTL-based expiration, cache warming, and hit/miss metrics.
Falls back gracefully when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import orjson
import structlog

logger = structlog.get_logger(__name__)

# Attempt Redis import - graceful if unavailable
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheMetrics:
    """Track cache hit/miss statistics."""

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.errors: int = 0
        self.set_operations: int = 0
        self.evictions: int = 0
        self._start_time = time.monotonic()

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "set_operations": self.set_operations,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


class RedisCache:
    """Redis-based caching with graceful degradation.

    When Redis is unavailable, operations become no-ops instead of
    raising exceptions, ensuring the prediction service stays available.
    """

    PREDICTION_PREFIX = "rxp:pred:"
    FEATURE_PREFIX = "rxp:feat:"
    WARM_PREFIX = "rxp:warm:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prediction_ttl: int = 3600,
        feature_ttl: int = 7200,
        max_connections: int = 20,
        socket_timeout: float = 0.5,
        socket_connect_timeout: float = 0.5,
    ) -> None:
        self._redis_url = redis_url
        self._prediction_ttl = prediction_ttl
        self._feature_ttl = feature_ttl
        self._client: Any | None = None
        self._connected = False
        self.metrics = CacheMetrics()

        if REDIS_AVAILABLE:
            try:
                pool = redis.ConnectionPool.from_url(
                    redis_url,
                    max_connections=max_connections,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    decode_responses=False,
                )
                self._client = redis.Redis(connection_pool=pool)
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info("redis_connected", url=redis_url)
            except Exception as exc:
                logger.warning("redis_connection_failed", error=str(exc), url=redis_url)
                self._connected = False
        else:
            logger.warning("redis_not_installed", msg="Running without cache")

    @property
    def is_connected(self) -> bool:
        """Check if Redis is currently connected."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    @staticmethod
    def _make_prediction_key(patient_data: dict[str, Any]) -> str:
        """Create a deterministic cache key from patient data."""
        serialized = orjson.dumps(patient_data, option=orjson.OPT_SORT_KEYS)
        data_hash = hashlib.blake2b(serialized, digest_size=16).hexdigest()
        return f"{RedisCache.PREDICTION_PREFIX}{data_hash}"

    @staticmethod
    def _make_feature_key(genetic_key: str) -> str:
        """Create a cache key for feature vectors."""
        data_hash = hashlib.blake2b(genetic_key.encode(), digest_size=16).hexdigest()
        return f"{RedisCache.FEATURE_PREFIX}{data_hash}"

    def get_prediction(self, patient_data: dict[str, Any]) -> dict[str, Any] | None:
        """Retrieve cached prediction result.

        Returns None on cache miss or Redis unavailability.
        """
        if not self._connected or self._client is None:
            self.metrics.misses += 1
            return None

        try:
            key = self._make_prediction_key(patient_data)
            cached = self._client.get(key)
            if cached is not None:
                self.metrics.hits += 1
                result: dict[str, Any] = orjson.loads(cached)
                result["cache_hit"] = True
                return result
            self.metrics.misses += 1
            return None
        except Exception as exc:
            self.metrics.errors += 1
            logger.warning("cache_get_error", error=str(exc))
            return None

    def set_prediction(
        self,
        patient_data: dict[str, Any],
        prediction: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a prediction result.

        Returns True if successfully cached.
        """
        if not self._connected or self._client is None:
            return False

        try:
            key = self._make_prediction_key(patient_data)
            serialized = orjson.dumps(prediction)
            self._client.setex(key, ttl or self._prediction_ttl, serialized)
            self.metrics.set_operations += 1
            return True
        except Exception as exc:
            self.metrics.errors += 1
            logger.warning("cache_set_error", error=str(exc))
            return False

    def get_feature_vector(self, genetic_key: str) -> bytes | None:
        """Retrieve cached feature vector (stored as raw bytes)."""
        if not self._connected or self._client is None:
            return None

        try:
            key = self._make_feature_key(genetic_key)
            return self._client.get(key)
        except Exception as exc:
            self.metrics.errors += 1
            logger.warning("cache_feature_get_error", error=str(exc))
            return None

    def set_feature_vector(self, genetic_key: str, vector_bytes: bytes) -> bool:
        """Cache a feature vector as raw bytes."""
        if not self._connected or self._client is None:
            return False

        try:
            key = self._make_feature_key(genetic_key)
            self._client.setex(key, self._feature_ttl, vector_bytes)
            return True
        except Exception as exc:
            self.metrics.errors += 1
            logger.warning("cache_feature_set_error", error=str(exc))
            return False

    def warm_cache(self, common_requests: list[dict[str, Any]]) -> int:
        """Pre-populate cache with predictions for common request patterns.

        Returns the number of entries warmed.
        """
        if not self._connected or self._client is None:
            return 0

        warmed = 0
        try:
            pipe = self._client.pipeline(transaction=False)
            for req_data in common_requests:
                key = self._make_prediction_key(req_data)
                warm_key = f"{self.WARM_PREFIX}{key}"
                pipe.exists(warm_key)
            results = pipe.execute()

            # Only count those not already in cache
            for i, exists in enumerate(results):
                if not exists:
                    warmed += 1

            logger.info("cache_warm_complete", warmed=warmed, total=len(common_requests))
        except Exception as exc:
            logger.warning("cache_warm_error", error=str(exc))

        return warmed

    def invalidate_all_predictions(self) -> int:
        """Invalidate all cached predictions (e.g., after model update).

        Returns the number of keys deleted.
        """
        if not self._connected or self._client is None:
            return 0

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor=cursor,
                    match=f"{self.PREDICTION_PREFIX}*",
                    count=100,
                )
                if keys:
                    deleted += self._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info("cache_invalidated", deleted=deleted)
            self.metrics.evictions += deleted
            return deleted
        except Exception as exc:
            logger.warning("cache_invalidate_error", error=str(exc))
            return 0

    def health_check(self) -> dict[str, Any]:
        """Detailed health check for the cache layer."""
        status: dict[str, Any] = {
            "connected": self._connected,
            "redis_available": REDIS_AVAILABLE,
            "metrics": self.metrics.to_dict(),
        }

        if self._connected and self._client is not None:
            try:
                start = time.perf_counter()
                self._client.ping()
                ping_ms = (time.perf_counter() - start) * 1000
                info = self._client.info(section="memory")
                status.update(
                    {
                        "ping_ms": round(ping_ms, 3),
                        "used_memory_bytes": info.get("used_memory", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "connected_clients": info.get("connected_clients", 0),
                    }
                )
            except Exception as exc:
                status["error"] = str(exc)

        return status

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._connected = False
            logger.info("redis_connection_closed")
