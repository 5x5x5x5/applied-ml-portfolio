"""
Centralized configuration for the StreamRx pipeline.

All settings are loaded from environment variables with sensible defaults
for local development. Production deployments should set these via ECS
task definitions or Kubernetes ConfigMaps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KafkaConfig:
    """Kafka broker and topic configuration."""

    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    schema_registry_url: str = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    prescription_topic: str = os.getenv("KAFKA_PRESCRIPTION_TOPIC", "rx.prescriptions")
    adverse_event_topic: str = os.getenv("KAFKA_ADVERSE_EVENT_TOPIC", "rx.adverse_events")
    alerts_topic: str = os.getenv("KAFKA_ALERTS_TOPIC", "rx.alerts")
    dlq_topic: str = os.getenv("KAFKA_DLQ_TOPIC", "rx.dead_letter_queue")
    consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "streamrx-processors")
    num_partitions: int = int(os.getenv("KAFKA_NUM_PARTITIONS", "12"))
    replication_factor: int = int(os.getenv("KAFKA_REPLICATION_FACTOR", "3"))
    session_timeout_ms: int = 30_000
    max_poll_interval_ms: int = 300_000
    enable_idempotence: bool = True
    acks: str = "all"
    compression_type: str = "lz4"
    batch_size: int = 65_536
    linger_ms: int = 10


@dataclass(frozen=True)
class KinesisConfig:
    """AWS Kinesis configuration."""

    region: str = os.getenv("AWS_REGION", "us-east-1")
    prescription_stream: str = os.getenv("KINESIS_RX_STREAM", "streamrx-prescriptions")
    adverse_event_stream: str = os.getenv("KINESIS_AE_STREAM", "streamrx-adverse-events")
    firehose_delivery_stream: str = os.getenv("KINESIS_FIREHOSE_STREAM", "streamrx-s3-delivery")
    checkpoint_table: str = os.getenv("DYNAMO_CHECKPOINT_TABLE", "streamrx-checkpoints")
    shard_count: int = int(os.getenv("KINESIS_SHARD_COUNT", "4"))
    enhanced_fanout_consumer: str = os.getenv("KINESIS_EFO_CONSUMER", "streamrx-efo-consumer")
    iterator_type: str = "LATEST"
    max_records_per_get: int = 1000


@dataclass(frozen=True)
class S3Config:
    """S3 data lake sink configuration."""

    bucket: str = os.getenv("S3_DATA_LAKE_BUCKET", "streamrx-data-lake")
    prefix: str = os.getenv("S3_PREFIX", "pharma-events")
    region: str = os.getenv("AWS_REGION", "us-east-1")
    buffer_size_mb: int = int(os.getenv("S3_BUFFER_SIZE_MB", "128"))
    buffer_interval_sec: int = int(os.getenv("S3_BUFFER_INTERVAL_SEC", "300"))
    compaction_threshold_files: int = int(os.getenv("S3_COMPACTION_THRESHOLD", "50"))
    target_file_size_mb: int = int(os.getenv("S3_TARGET_FILE_SIZE_MB", "256"))


@dataclass(frozen=True)
class RedisConfig:
    """Redis configuration for state and caching."""

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str | None = os.getenv("REDIS_PASSWORD")
    key_prefix: str = "streamrx:"
    state_ttl_sec: int = 86_400  # 24 hours
    metrics_ttl_sec: int = 3_600  # 1 hour


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring and alerting configuration."""

    datadog_api_key: str = os.getenv("DD_API_KEY", "")
    datadog_app_key: str = os.getenv("DD_APP_KEY", "")
    metrics_prefix: str = "streamrx"
    lag_warning_threshold: int = int(os.getenv("LAG_WARNING_THRESHOLD", "10000"))
    lag_critical_threshold: int = int(os.getenv("LAG_CRITICAL_THRESHOLD", "100000"))
    error_rate_threshold: float = float(os.getenv("ERROR_RATE_THRESHOLD", "0.01"))
    metrics_flush_interval_sec: int = 10


@dataclass(frozen=True)
class SignalDetectionConfig:
    """Configuration for pharmacovigilance signal detection."""

    prr_threshold: float = float(os.getenv("PRR_THRESHOLD", "2.0"))
    ror_threshold: float = float(os.getenv("ROR_THRESHOLD", "2.0"))
    min_case_count: int = int(os.getenv("MIN_CASE_COUNT", "3"))
    sliding_window_hours: int = int(os.getenv("SIGNAL_WINDOW_HOURS", "24"))
    baseline_window_days: int = int(os.getenv("BASELINE_WINDOW_DAYS", "90"))
    chi_square_threshold: float = float(os.getenv("CHI_SQUARE_THRESHOLD", "3.84"))


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    kinesis: KinesisConfig = field(default_factory=KinesisConfig)
    s3: S3Config = field(default_factory=S3Config)
    redis: RedisConfig = field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    signal_detection: SignalDetectionConfig = field(default_factory=SignalDetectionConfig)
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")


def get_config() -> PipelineConfig:
    """Return the pipeline configuration singleton."""
    return PipelineConfig()
