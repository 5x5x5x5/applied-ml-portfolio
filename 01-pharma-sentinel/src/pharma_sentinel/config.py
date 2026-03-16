"""Application configuration using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults
for local development. Secrets are never hardcoded.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class AWSSettings(BaseSettings):
    """AWS service configuration."""

    model_config = SettingsConfigDict(env_prefix="AWS_")

    region: str = Field(default="us-east-1", description="AWS region")
    access_key_id: str | None = Field(default=None, description="AWS access key ID")
    secret_access_key: str | None = Field(default=None, description="AWS secret access key")
    endpoint_url: str | None = Field(
        default=None,
        description="Custom endpoint URL for LocalStack or other S3-compatible services",
    )

    s3_input_bucket: str = Field(
        default="pharma-sentinel-input",
        description="S3 bucket for raw FAERS data",
    )
    s3_output_bucket: str = Field(
        default="pharma-sentinel-output",
        description="S3 bucket for processed results",
    )
    s3_model_bucket: str = Field(
        default="pharma-sentinel-models",
        description="S3 bucket for ML model artifacts",
    )

    sqs_critical_queue_url: str = Field(
        default="",
        description="SQS queue URL for critical event notifications",
    )
    sqs_processing_queue_url: str = Field(
        default="",
        description="SQS queue URL for file processing tasks",
    )


class DataDogSettings(BaseSettings):
    """DataDog monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="DD_")

    api_key: str | None = Field(default=None, description="DataDog API key")
    app_key: str | None = Field(default=None, description="DataDog application key")
    env: str = Field(default="local", description="DataDog environment tag")
    service: str = Field(default="pharma-sentinel", description="DataDog service name")
    version: str = Field(default="1.0.0", description="Application version for DD tagging")
    agent_host: str = Field(default="localhost", description="DataDog agent host")
    agent_port: int = Field(default=8126, description="DataDog agent port")
    trace_enabled: bool = Field(default=False, description="Enable APM tracing")
    logs_injection: bool = Field(default=True, description="Inject trace IDs into logs")
    statsd_host: str = Field(default="localhost", description="StatsD host for custom metrics")
    statsd_port: int = Field(default=8125, description="StatsD port for custom metrics")


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    artifact_path: str = Field(
        default="models/adverse_event_classifier.joblib",
        description="Local path to model artifact",
    )
    s3_artifact_key: str = Field(
        default="models/adverse_event_classifier/latest/model.joblib",
        description="S3 key for model artifact",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for predictions",
    )
    max_text_length: int = Field(
        default=10000,
        gt=0,
        description="Maximum input text length for classification",
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        le=1024,
        description="Batch size for batch prediction",
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure confidence threshold is reasonable."""
        if v < 0.1:
            logging.getLogger(__name__).warning(
                "Confidence threshold %.2f is very low; predictions may be unreliable", v
            )
        return v


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL for Redis connection")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")


class AppSettings(BaseSettings):
    """Root application configuration aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    environment: Environment = Field(
        default=Environment.LOCAL,
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    host: str = Field(default="0.0.0.0", description="API server host")  # noqa: S104
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of uvicorn workers")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication",
    )

    aws: AWSSettings = Field(default_factory=AWSSettings)
    datadog: DataDogSettings = Field(default_factory=DataDogSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_local(self) -> bool:
        """Check if running in local development environment."""
        return self.environment == Environment.LOCAL


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Get cached application settings singleton.

    Returns:
        AppSettings: The application configuration instance.
    """
    return AppSettings()
