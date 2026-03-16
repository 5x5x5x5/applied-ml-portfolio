"""Cache layer tests for RxPredict."""

from __future__ import annotations

from unittest.mock import MagicMock

from rx_predict.cache.redis_cache import RedisCache


class TestRedisCacheNoConnection:
    """Test cache behavior when Redis is unavailable (graceful degradation)."""

    def setup_method(self) -> None:
        """Create a cache instance without Redis connection."""
        # Use an invalid URL to ensure no connection
        self.cache = RedisCache(
            redis_url="redis://nonexistent:9999/0",
            socket_connect_timeout=0.1,
        )

    def test_not_connected(self) -> None:
        """Cache should report not connected."""
        assert not self.cache.is_connected

    def test_get_prediction_returns_none(self) -> None:
        """Get should return None gracefully."""
        result = self.cache.get_prediction({"test": "data"})
        assert result is None

    def test_set_prediction_returns_false(self) -> None:
        """Set should return False gracefully."""
        result = self.cache.set_prediction({"test": "data"}, {"result": 1})
        assert result is False

    def test_get_feature_vector_returns_none(self) -> None:
        """Feature vector get should return None."""
        result = self.cache.get_feature_vector("test_key")
        assert result is None

    def test_invalidate_returns_zero(self) -> None:
        """Invalidation should return 0 deleted."""
        result = self.cache.invalidate_all_predictions()
        assert result == 0

    def test_health_check_reports_disconnected(self) -> None:
        """Health check should report disconnected state."""
        health = self.cache.health_check()
        assert health["connected"] is False
        assert "metrics" in health

    def test_metrics_tracking(self) -> None:
        """Metrics should track operations even when disconnected."""
        self.cache.get_prediction({"a": 1})
        self.cache.get_prediction({"b": 2})
        assert self.cache.metrics.misses == 2
        assert self.cache.metrics.hits == 0


class TestRedisCacheMocked:
    """Test cache behavior with mocked Redis connection."""

    def setup_method(self) -> None:
        """Create a cache with mocked Redis client."""
        self.cache = RedisCache.__new__(RedisCache)
        self.cache._redis_url = "redis://localhost:6379/0"
        self.cache._prediction_ttl = 3600
        self.cache._feature_ttl = 7200
        self.cache._connected = True
        self.cache._client = MagicMock()
        self.cache.metrics = __import__(
            "rx_predict.cache.redis_cache", fromlist=["CacheMetrics"]
        ).CacheMetrics()

    def test_prediction_key_deterministic(self) -> None:
        """Same data should produce same cache key."""
        data = {"age": 30, "drug": "aspirin"}
        key1 = RedisCache._make_prediction_key(data)
        key2 = RedisCache._make_prediction_key(data)
        assert key1 == key2
        assert key1.startswith("rxp:pred:")

    def test_different_data_different_keys(self) -> None:
        """Different data should produce different keys."""
        key1 = RedisCache._make_prediction_key({"age": 30})
        key2 = RedisCache._make_prediction_key({"age": 31})
        assert key1 != key2

    def test_cache_hit(self) -> None:
        """Cache hit should return data and increment hit counter."""
        import orjson

        cached_data = {"predicted_class": "good_response", "probability": 0.85}
        self.cache._client.get.return_value = orjson.dumps(cached_data)

        result = self.cache.get_prediction({"test": "data"})
        assert result is not None
        assert result["predicted_class"] == "good_response"
        assert result["cache_hit"] is True
        assert self.cache.metrics.hits == 1

    def test_cache_miss(self) -> None:
        """Cache miss should return None and increment miss counter."""
        self.cache._client.get.return_value = None

        result = self.cache.get_prediction({"test": "data"})
        assert result is None
        assert self.cache.metrics.misses == 1

    def test_set_prediction(self) -> None:
        """Setting a prediction should call Redis setex."""
        data = {"test": "data"}
        prediction = {"predicted_class": "good_response"}

        result = self.cache.set_prediction(data, prediction)
        assert result is True
        self.cache._client.setex.assert_called_once()
        assert self.cache.metrics.set_operations == 1

    def test_set_prediction_custom_ttl(self) -> None:
        """Custom TTL should be used when specified."""
        data = {"test": "data"}
        prediction = {"predicted_class": "good_response"}

        self.cache.set_prediction(data, prediction, ttl=60)
        args = self.cache._client.setex.call_args
        assert args[0][1] == 60  # TTL argument

    def test_cache_error_handling(self) -> None:
        """Redis errors should be caught and counted."""
        self.cache._client.get.side_effect = Exception("Connection lost")

        result = self.cache.get_prediction({"test": "data"})
        assert result is None
        assert self.cache.metrics.errors == 1

    def test_feature_vector_cache(self) -> None:
        """Feature vectors should be cached as raw bytes."""
        test_bytes = b"\x00\x01\x02\x03"
        self.cache._client.get.return_value = test_bytes

        result = self.cache.get_feature_vector("test_key")
        assert result == test_bytes

    def test_cache_warm(self) -> None:
        """Cache warming should process request list."""
        pipe_mock = MagicMock()
        pipe_mock.execute.return_value = [False, False, True]
        self.cache._client.pipeline.return_value = pipe_mock

        requests = [{"a": 1}, {"b": 2}, {"c": 3}]
        warmed = self.cache.warm_cache(requests)
        assert warmed == 2  # 2 were not already cached

    def test_hit_rate_calculation(self) -> None:
        """Hit rate should be calculated correctly."""
        self.cache.metrics.hits = 75
        self.cache.metrics.misses = 25
        assert self.cache.metrics.hit_rate == 0.75
        assert self.cache.metrics.total_requests == 100

    def test_close(self) -> None:
        """Close should clean up the connection."""
        self.cache.close()
        assert self.cache._connected is False
        self.cache._client.close.assert_called_once()
