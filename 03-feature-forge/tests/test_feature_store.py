"""Tests for feature store registry and serving layer."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from feature_forge.extractors.structured_extractor import SnowflakeConfig
from feature_forge.feature_store.registry import (
    FeatureDataType,
    FeatureDefinition,
    FeatureRegistry,
    FeatureStatus,
    LineageEdge,
)
from feature_forge.feature_store.serving import (
    CacheEntry,
    FeatureServingLayer,
    ServingConfig,
)


class TestFeatureDefinition:
    """Tests for the FeatureDefinition Pydantic model."""

    def test_valid_feature(self) -> None:
        """Valid feature definition is created successfully."""
        feature = FeatureDefinition(
            name="hemoglobin_mean",
            description="Mean hemoglobin",
            data_type=FeatureDataType.FLOAT,
            source_table="LAB_FEATURES",
        )
        assert feature.name == "hemoglobin_mean"
        assert feature.version == 1
        assert feature.status == FeatureStatus.DRAFT

    def test_feature_id_deterministic(self) -> None:
        """Feature ID is deterministic based on name and version."""
        f1 = FeatureDefinition(name="test_feat", source_table="T")
        f2 = FeatureDefinition(name="test_feat", source_table="T")
        assert f1.feature_id == f2.feature_id

    def test_feature_id_changes_with_version(self) -> None:
        """Different versions produce different feature IDs."""
        f1 = FeatureDefinition(name="test_feat", version=1, source_table="T")
        f2 = FeatureDefinition(name="test_feat", version=2, source_table="T")
        assert f1.feature_id != f2.feature_id

    def test_qualified_name(self) -> None:
        """Qualified name includes version."""
        f = FeatureDefinition(name="age", version=3, source_table="T")
        assert f.qualified_name == "age:v3"

    def test_invalid_name_rejected(self) -> None:
        """Invalid feature names are rejected by Pydantic validation."""
        with pytest.raises(Exception):
            FeatureDefinition(name="123_invalid", source_table="T")

    def test_invalid_name_uppercase_rejected(self) -> None:
        """Uppercase names are rejected."""
        with pytest.raises(Exception):
            FeatureDefinition(name="Invalid_Name", source_table="T")

    def test_tags_default_empty(self) -> None:
        """Tags default to empty list."""
        f = FeatureDefinition(name="test", source_table="T")
        assert f.tags == []


class TestFeatureRegistry:
    """Tests for FeatureRegistry."""

    def test_init(self, snowflake_config: SnowflakeConfig) -> None:
        """Registry initialises with config."""
        registry = FeatureRegistry(snowflake_config)
        assert registry._conn is None
        assert len(registry._cache) == 0

    def test_connect(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
    ) -> None:
        """Registry connects to Snowflake."""
        registry = FeatureRegistry(snowflake_config)
        conn = registry.connect()
        assert conn is not None

    def test_register_feature_inserts(
        self,
        snowflake_config: SnowflakeConfig,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Registering a new feature issues an INSERT."""
        cursor = mock_snowflake_connection.cursor.return_value
        cursor.fetchall.return_value = []  # get_feature returns None (not found)

        registry = FeatureRegistry(snowflake_config)
        feature = FeatureDefinition(
            name="test_feature",
            source_table="TEST_TABLE",
            status=FeatureStatus.ACTIVE,
        )
        result = registry.register_feature(feature)

        assert result.name == "test_feature"
        assert feature.qualified_name in registry._cache
        # Verify SQL was executed (INSERT for register + INSERT for version history)
        assert cursor.execute.call_count >= 2

    def test_get_feature_from_cache(
        self,
        snowflake_config: SnowflakeConfig,
    ) -> None:
        """Cached features are returned without querying Snowflake."""
        registry = FeatureRegistry(snowflake_config)
        feature = FeatureDefinition(
            name="cached_feat",
            version=1,
            source_table="T",
        )
        registry._cache["cached_feat:v1"] = feature

        result = registry.get_feature("cached_feat", version=1)
        assert result is feature

    def test_row_to_feature(self, sample_feature_rows: list[dict[str, Any]]) -> None:
        """Snowflake row dict is correctly converted to FeatureDefinition."""
        feature = FeatureRegistry._row_to_feature(sample_feature_rows[0])
        assert feature.name == "hemoglobin_mean"
        assert feature.version == 1
        assert feature.data_type == FeatureDataType.FLOAT
        assert feature.status == FeatureStatus.ACTIVE
        assert "lab" in feature.tags

    def test_feature_to_params(self) -> None:
        """Feature is correctly serialised to query parameters."""
        feature = FeatureDefinition(
            name="test",
            source_table="T",
            tags=["a", "b"],
            dependencies=["dep1"],
        )
        params = FeatureRegistry._feature_to_params(feature)
        assert params["name"] == "test"
        assert '"a"' in params["tags"]
        assert '"dep1"' in params["dependencies"]


class TestLineageEdge:
    """Tests for LineageEdge model."""

    def test_create_edge(self) -> None:
        edge = LineageEdge(
            source_feature="raw_hemoglobin",
            target_feature="hemoglobin_mean",
            transformation="AVG(raw_hemoglobin) OVER 365d window",
        )
        assert edge.source_feature == "raw_hemoglobin"
        assert edge.target_feature == "hemoglobin_mean"


class TestCacheEntry:
    """Tests for the serving layer cache."""

    def test_cache_not_expired(self) -> None:
        """Fresh cache entry is not expired."""
        entry = CacheEntry(
            data=pd.DataFrame(),
            created_at=time.monotonic(),
            ttl_seconds=300.0,
        )
        assert not entry.is_expired

    def test_cache_expired(self) -> None:
        """Old cache entry is expired."""
        entry = CacheEntry(
            data=pd.DataFrame(),
            created_at=time.monotonic() - 400,
            ttl_seconds=300.0,
        )
        assert entry.is_expired


class TestServingConfig:
    """Tests for ServingConfig defaults."""

    def test_defaults(self) -> None:
        config = ServingConfig()
        assert config.default_ttl_seconds == 300.0
        assert config.max_cache_entries == 1000
        assert config.batch_size == 10_000
        assert config.enable_cache is True


class TestFeatureServingLayer:
    """Tests for FeatureServingLayer."""

    @pytest.fixture()
    def mock_registry(self) -> MagicMock:
        registry = MagicMock(spec=FeatureRegistry)
        return registry

    def test_init(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Serving layer initialises with config and registry."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        assert layer._conn is None
        assert len(layer._cache) == 0

    def test_cache_put_and_get(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Cache put and get work correctly."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        df = pd.DataFrame({"a": [1, 2, 3]})
        layer._put_in_cache("key1", df)

        result = layer._get_from_cache("key1")
        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_cache_miss(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Cache miss returns None."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        assert layer._get_from_cache("nonexistent") is None

    def test_cache_eviction(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Cache evicts oldest entry when at capacity."""
        config = ServingConfig(max_cache_entries=3)
        layer = FeatureServingLayer(snowflake_config, mock_registry, config)

        for i in range(4):
            layer._put_in_cache(f"key{i}", pd.DataFrame({"v": [i]}))

        # key0 should have been evicted
        assert layer._get_from_cache("key0") is None
        assert layer._get_from_cache("key3") is not None

    def test_cache_disabled(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Cache disabled means always miss."""
        config = ServingConfig(enable_cache=False)
        layer = FeatureServingLayer(snowflake_config, mock_registry, config)

        layer._put_in_cache("key", pd.DataFrame({"a": [1]}))
        assert layer._get_from_cache("key") is None

    def test_invalidate_all_cache(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Invalidating all cache clears everything."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        layer._put_in_cache("a", pd.DataFrame())
        layer._put_in_cache("b", pd.DataFrame())
        count = layer.invalidate_cache()
        assert count == 2
        assert len(layer._cache) == 0

    def test_invalidate_by_feature(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Invalidating by feature name only removes matching entries."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        layer._cache["feature_a|123|latest"] = CacheEntry(
            data=pd.DataFrame(), created_at=time.monotonic(), ttl_seconds=300
        )
        layer._cache["feature_b|456|latest"] = CacheEntry(
            data=pd.DataFrame(), created_at=time.monotonic(), ttl_seconds=300
        )
        count = layer.invalidate_cache("feature_a")
        assert count == 1
        assert len(layer._cache) == 1

    def test_assemble_feature_vector_empty(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
        mock_snowflake_connect: Any,
        mock_snowflake_connection: MagicMock,
    ) -> None:
        """Assembling a vector with no data returns fill values."""
        cursor = mock_snowflake_connection.cursor.return_value
        cursor.fetchall.return_value = []
        mock_registry.get_feature.return_value = None

        layer = FeatureServingLayer(snowflake_config, mock_registry)
        vector = layer.assemble_feature_vector(
            feature_names=["feat_a", "feat_b"],
            entity_id="P001",
            fill_value=-1.0,
        )

        assert len(vector) == 2
        np.testing.assert_array_equal(vector, [-1.0, -1.0])

    def test_point_in_time_validates_length(
        self,
        snowflake_config: SnowflakeConfig,
        mock_registry: MagicMock,
    ) -> None:
        """Point-in-time retrieval raises if entity_ids and timestamps differ in length."""
        layer = FeatureServingLayer(snowflake_config, mock_registry)
        with pytest.raises(ValueError, match="same length"):
            layer.get_point_in_time_features(
                feature_names=["f1"],
                entity_ids=["P1", "P2"],
                as_of_timestamps=[datetime(2025, 1, 1)],
            )
