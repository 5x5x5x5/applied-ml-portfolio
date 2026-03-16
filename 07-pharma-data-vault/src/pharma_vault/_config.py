"""Central configuration for PharmaDataVault."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VaultConfig:
    """Immutable configuration for the data vault environment."""

    db_host: str = field(default_factory=lambda: os.getenv("PHARMA_DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("PHARMA_DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("PHARMA_DB_NAME", "pharma_vault"))
    db_user: str = field(default_factory=lambda: os.getenv("PHARMA_DB_USER", "etl_user"))
    db_password: str = field(default_factory=lambda: os.getenv("PHARMA_DB_PASSWORD", ""))

    staging_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PHARMA_STAGING_DIR", "/data/pharma/staging"))
    )
    archive_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PHARMA_ARCHIVE_DIR", "/data/pharma/archive"))
    )
    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PHARMA_LOG_DIR", "/var/log/pharma_vault"))
    )

    batch_size: int = 5000
    max_retries: int = 3
    retry_delay_seconds: int = 30

    @property
    def connection_string(self) -> str:
        """Build SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
