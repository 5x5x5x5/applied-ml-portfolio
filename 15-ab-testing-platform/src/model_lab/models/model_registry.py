"""Model version management with champion/challenger designation and rollback.

Provides a registry for ML model versions with metadata tracking, automated
promotion based on experiment results, and SageMaker endpoint integration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ModelStatus(str, Enum):
    """Lifecycle status of a registered model version."""

    REGISTERED = "registered"
    VALIDATING = "validating"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


class ModelFramework(str, Enum):
    """ML framework used to create the model."""

    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelVersion(BaseModel):
    """A registered model version with its metadata and deployment info."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    version: str
    status: ModelStatus = ModelStatus.REGISTERED
    framework: ModelFramework = ModelFramework.CUSTOM
    artifact_uri: str = ""
    description: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    created_by: str = ""
    sagemaker_endpoint: str | None = None
    sagemaker_variant_name: str | None = None
    parent_version_id: str | None = None


class PromotionCriteria(BaseModel):
    """Criteria for automatically promoting a model based on experiment results."""

    min_sample_size: int = Field(default=1000, ge=10)
    min_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    min_relative_improvement: float = Field(default=0.0, ge=0.0)
    max_latency_p99_ms: float | None = None
    max_error_rate: float | None = None
    required_guardrail_pass: bool = True


class ModelRegistry:
    """Registry for ML model versions with champion/challenger management.

    Tracks model versions, manages promotions and rollbacks, and integrates
    with AWS SageMaker for endpoint management.
    """

    def __init__(self) -> None:
        self._versions: dict[str, ModelVersion] = {}
        self._model_history: dict[str, list[str]] = {}  # model_name -> [version_ids]
        self._rollback_stack: dict[str, list[str]] = {}  # model_name -> [previous champion ids]

    def register(self, model_version: ModelVersion) -> ModelVersion:
        """Register a new model version.

        Args:
            model_version: The model version to register.

        Returns:
            The registered model version.

        Raises:
            ValueError: If a version with the same model_name and version exists.
        """
        for existing in self._versions.values():
            if (
                existing.model_name == model_version.model_name
                and existing.version == model_version.version
            ):
                msg = (
                    f"Version {model_version.version} already exists "
                    f"for model {model_version.model_name}"
                )
                raise ValueError(msg)

        self._versions[model_version.id] = model_version

        if model_version.model_name not in self._model_history:
            self._model_history[model_version.model_name] = []
        self._model_history[model_version.model_name].append(model_version.id)

        logger.info(
            "model_registered",
            model_id=model_version.id,
            model_name=model_version.model_name,
            version=model_version.version,
        )
        return model_version

    def get_version(self, version_id: str) -> ModelVersion:
        """Get a model version by ID.

        Raises:
            KeyError: If version not found.
        """
        if version_id not in self._versions:
            msg = f"Model version {version_id} not found"
            raise KeyError(msg)
        return self._versions[version_id]

    def get_champion(self, model_name: str) -> ModelVersion | None:
        """Get the current champion version for a model.

        Args:
            model_name: The model name to look up.

        Returns:
            The champion ModelVersion, or None if no champion exists.
        """
        for version in self._versions.values():
            if version.model_name == model_name and version.status == ModelStatus.CHAMPION:
                return version
        return None

    def get_challengers(self, model_name: str) -> list[ModelVersion]:
        """Get all challenger versions for a model."""
        return [
            v
            for v in self._versions.values()
            if v.model_name == model_name and v.status == ModelStatus.CHALLENGER
        ]

    def list_versions(
        self,
        model_name: str | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        """List model versions with optional filtering."""
        versions = list(self._versions.values())
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        if status:
            versions = [v for v in versions if v.status == status]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def set_challenger(self, version_id: str) -> ModelVersion:
        """Designate a model version as a challenger.

        Args:
            version_id: The version to promote to challenger.

        Returns:
            The updated ModelVersion.
        """
        version = self.get_version(version_id)
        version.status = ModelStatus.CHALLENGER
        version.updated_at = datetime.now(UTC)
        logger.info(
            "model_set_challenger",
            model_id=version_id,
            model_name=version.model_name,
        )
        return version

    def promote_to_champion(
        self,
        version_id: str,
        criteria: PromotionCriteria | None = None,
        experiment_results: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Promote a model version to champion.

        If criteria and experiment_results are provided, validates that the
        results meet the promotion criteria before promoting.

        Args:
            version_id: Version to promote.
            criteria: Optional promotion criteria to validate.
            experiment_results: Optional experiment results to check against criteria.

        Returns:
            The promoted ModelVersion.

        Raises:
            ValueError: If promotion criteria are not met.
        """
        version = self.get_version(version_id)

        # Validate promotion criteria if provided
        if criteria and experiment_results:
            self._validate_promotion(criteria, experiment_results)

        # Demote current champion
        current_champion = self.get_champion(version.model_name)
        if current_champion:
            # Push to rollback stack
            if version.model_name not in self._rollback_stack:
                self._rollback_stack[version.model_name] = []
            self._rollback_stack[version.model_name].append(current_champion.id)

            current_champion.status = ModelStatus.ARCHIVED
            current_champion.updated_at = datetime.now(UTC)
            logger.info(
                "champion_demoted",
                model_id=current_champion.id,
                model_name=current_champion.model_name,
            )

        # Promote new champion
        version.status = ModelStatus.CHAMPION
        version.promoted_at = datetime.now(UTC)
        version.updated_at = datetime.now(UTC)

        logger.info(
            "model_promoted_to_champion",
            model_id=version_id,
            model_name=version.model_name,
            version=version.version,
        )
        return version

    def rollback(self, model_name: str) -> ModelVersion:
        """Rollback to the previous champion version.

        Args:
            model_name: The model to rollback.

        Returns:
            The restored champion ModelVersion.

        Raises:
            ValueError: If no rollback history exists.
        """
        stack = self._rollback_stack.get(model_name, [])
        if not stack:
            msg = f"No rollback history for model {model_name}"
            raise ValueError(msg)

        # Demote current champion
        current_champion = self.get_champion(model_name)
        if current_champion:
            current_champion.status = ModelStatus.ROLLED_BACK
            current_champion.updated_at = datetime.now(UTC)

        # Restore previous champion
        previous_id = stack.pop()
        previous = self.get_version(previous_id)
        previous.status = ModelStatus.CHAMPION
        previous.updated_at = datetime.now(UTC)

        logger.info(
            "model_rolled_back",
            model_name=model_name,
            restored_version_id=previous_id,
            rolled_back_version_id=current_champion.id if current_champion else None,
        )
        return previous

    def configure_sagemaker_endpoint(
        self,
        version_id: str,
        endpoint_name: str,
        variant_name: str | None = None,
    ) -> ModelVersion:
        """Configure SageMaker endpoint details for a model version.

        This stores the endpoint configuration. Actual deployment would be
        handled by a separate deployment service using boto3.

        Args:
            version_id: Model version to configure.
            endpoint_name: SageMaker endpoint name.
            variant_name: Optional production variant name.

        Returns:
            Updated ModelVersion.
        """
        version = self.get_version(version_id)
        version.sagemaker_endpoint = endpoint_name
        version.sagemaker_variant_name = variant_name or f"variant-{version.version}"
        version.updated_at = datetime.now(UTC)

        logger.info(
            "sagemaker_configured",
            model_id=version_id,
            endpoint=endpoint_name,
            variant=version.sagemaker_variant_name,
        )
        return version

    def get_sagemaker_endpoint_config(self, model_name: str) -> dict[str, Any]:
        """Generate SageMaker endpoint configuration for champion + challengers.

        Creates a production variant config suitable for use with
        boto3 SageMaker create_endpoint_config.

        Args:
            model_name: The model to generate config for.

        Returns:
            Dict with endpoint configuration including production variants.
        """
        champion = self.get_champion(model_name)
        challengers = self.get_challengers(model_name)

        variants = []
        if champion and champion.sagemaker_endpoint:
            variants.append(
                {
                    "VariantName": champion.sagemaker_variant_name or "champion",
                    "ModelName": f"{model_name}-{champion.version}",
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "InitialVariantWeight": 0.9,
                }
            )

        for challenger in challengers:
            if challenger.sagemaker_endpoint:
                weight = 0.1 / len(challengers) if challengers else 0.1
                variants.append(
                    {
                        "VariantName": challenger.sagemaker_variant_name
                        or f"challenger-{challenger.version}",
                        "ModelName": f"{model_name}-{challenger.version}",
                        "InitialInstanceCount": 1,
                        "InstanceType": "ml.m5.xlarge",
                        "InitialVariantWeight": weight,
                    }
                )

        return {
            "EndpointConfigName": f"{model_name}-ab-config",
            "ProductionVariants": variants,
        }

    def _validate_promotion(
        self,
        criteria: PromotionCriteria,
        results: dict[str, Any],
    ) -> None:
        """Validate experiment results against promotion criteria.

        Raises:
            ValueError: If any criterion is not met.
        """
        sample_size = results.get("sample_size", 0)
        if sample_size < criteria.min_sample_size:
            msg = f"Insufficient sample size: {sample_size} < {criteria.min_sample_size}"
            raise ValueError(msg)

        confidence = results.get("confidence", 0.0)
        if confidence < criteria.min_confidence:
            msg = f"Insufficient confidence: {confidence} < {criteria.min_confidence}"
            raise ValueError(msg)

        relative_improvement = results.get("relative_improvement", 0.0)
        if relative_improvement < criteria.min_relative_improvement:
            msg = (
                f"Insufficient improvement: {relative_improvement} "
                f"< {criteria.min_relative_improvement}"
            )
            raise ValueError(msg)

        if criteria.max_latency_p99_ms is not None:
            latency = results.get("latency_p99_ms", float("inf"))
            if latency > criteria.max_latency_p99_ms:
                msg = f"Latency too high: {latency}ms > {criteria.max_latency_p99_ms}ms"
                raise ValueError(msg)

        if criteria.max_error_rate is not None:
            error_rate = results.get("error_rate", 1.0)
            if error_rate > criteria.max_error_rate:
                msg = f"Error rate too high: {error_rate} > {criteria.max_error_rate}"
                raise ValueError(msg)

        if criteria.required_guardrail_pass:
            guardrails_passed = results.get("guardrails_passed", False)
            if not guardrails_passed:
                msg = "Guardrail metrics not met"
                raise ValueError(msg)
