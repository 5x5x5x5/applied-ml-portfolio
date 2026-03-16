"""MLflow-based model registry for drug interaction models.

Handles experiment tracking, model versioning, performance comparison,
and stage promotion (staging -> production).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ModelStage(str, Enum):
    """MLflow model stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class RegisteredModelInfo(BaseModel):
    """Summary of a registered model version."""

    name: str
    version: int
    stage: ModelStage
    run_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class ComparisonResult(BaseModel):
    """Result of comparing a candidate model against production."""

    candidate_version: int
    production_version: int | None
    candidate_metrics: dict[str, float]
    production_metrics: dict[str, float]
    metric_deltas: dict[str, float]
    is_improvement: bool
    recommendation: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class DrugInteractionModelRegistry:
    """Manage drug interaction models in MLflow.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking server URI.
    experiment_name : str
        Name of the MLflow experiment.
    registered_model_name : str
        Name under which models are registered.
    """

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "drug-interaction-prediction"
    registered_model_name: str = "drug-interaction-xgboost"
    _client: MlflowClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._client = MlflowClient(self.tracking_uri)
        logger.info(
            "MLflow registry initialised: tracking=%s, experiment=%s",
            self.tracking_uri,
            self.experiment_name,
        )

    # -- Experiment tracking ------------------------------------------------

    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start an MLflow run and return the run ID.

        Parameters
        ----------
        run_name : str
            Human-readable name for the run.
        tags : dict, optional
            Additional tags to set on the run.

        Returns
        -------
        str
            The MLflow run ID.
        """
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        run_id = run.info.run_id
        logger.info("Started MLflow run: %s (id=%s)", run_name, run_id)
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active run."""
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        *,
        input_example: Any | None = None,
    ) -> str:
        """Log an XGBoost model as an MLflow artifact.

        Returns
        -------
        str
            The model URI.
        """
        info = mlflow.xgboost.log_model(
            model,
            artifact_path=artifact_path,
            input_example=input_example,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        logger.info("Logged model artifact: %s", model_uri)
        return model_uri

    def log_feature_importance(self, importances: dict[str, float]) -> None:
        """Log feature importance as a JSON artifact."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(importances, f, indent=2)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="feature_importance")

    def end_run(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()

    # -- Model registration -------------------------------------------------

    def register_model(
        self,
        model_uri: str,
        description: str = "",
    ) -> RegisteredModelInfo:
        """Register a model version from a run artifact.

        Parameters
        ----------
        model_uri : str
            URI of the logged model (e.g. ``runs:/<run_id>/model``).
        description : str
            Description for the model version.

        Returns
        -------
        RegisteredModelInfo
        """
        result = mlflow.register_model(model_uri, self.registered_model_name)

        # Set description
        self._client.update_model_version(
            name=self.registered_model_name,
            version=int(result.version),
            description=description,
        )

        info = RegisteredModelInfo(
            name=self.registered_model_name,
            version=int(result.version),
            stage=ModelStage.NONE,
            run_id=result.run_id or "",
            description=description,
        )
        logger.info(
            "Registered model %s version %d",
            self.registered_model_name,
            info.version,
        )
        return info

    # -- Stage promotion ----------------------------------------------------

    def transition_stage(
        self,
        version: int,
        stage: ModelStage,
        *,
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to a new stage.

        Parameters
        ----------
        version : int
            Model version number.
        stage : ModelStage
            Target stage.
        archive_existing : bool
            If True, archive any existing model in the target stage.
        """
        self._client.transition_model_version_stage(
            name=self.registered_model_name,
            version=str(version),
            stage=stage.value,
            archive_existing_versions=archive_existing,
        )
        logger.info(
            "Transitioned %s v%d -> %s",
            self.registered_model_name,
            version,
            stage.value,
        )

    def promote_to_staging(self, version: int) -> None:
        """Promote a model version to Staging."""
        self.transition_stage(version, ModelStage.STAGING)

    def promote_to_production(self, version: int) -> None:
        """Promote a model version to Production (archives previous)."""
        self.transition_stage(version, ModelStage.PRODUCTION, archive_existing=True)

    # -- Model comparison ---------------------------------------------------

    def get_production_model_info(self) -> RegisteredModelInfo | None:
        """Get info about the current production model, if any."""
        versions = self._client.get_latest_versions(
            self.registered_model_name, stages=["Production"]
        )
        if not versions:
            return None
        v = versions[0]
        run = self._client.get_run(v.run_id)
        return RegisteredModelInfo(
            name=self.registered_model_name,
            version=int(v.version),
            stage=ModelStage.PRODUCTION,
            run_id=v.run_id,
            metrics={k: float(val) for k, val in run.data.metrics.items()},
            tags=dict(run.data.tags),
            description=v.description or "",
        )

    def compare_with_production(
        self,
        candidate_run_id: str,
        primary_metric: str = "f1_macro",
        min_improvement: float = 0.005,
    ) -> ComparisonResult:
        """Compare a candidate run's metrics against the production model.

        Parameters
        ----------
        candidate_run_id : str
            Run ID of the candidate model.
        primary_metric : str
            The metric to use for the improvement check.
        min_improvement : float
            Minimum improvement required to recommend promotion.

        Returns
        -------
        ComparisonResult
        """
        candidate_run = self._client.get_run(candidate_run_id)
        candidate_metrics = {k: float(v) for k, v in candidate_run.data.metrics.items()}

        prod_info = self.get_production_model_info()
        if prod_info is None:
            return ComparisonResult(
                candidate_version=0,
                production_version=None,
                candidate_metrics=candidate_metrics,
                production_metrics={},
                metric_deltas=candidate_metrics,
                is_improvement=True,
                recommendation="No production model exists. Candidate should be promoted.",
            )

        prod_metrics = prod_info.metrics
        deltas = {
            k: candidate_metrics.get(k, 0.0) - prod_metrics.get(k, 0.0)
            for k in set(candidate_metrics) | set(prod_metrics)
        }

        candidate_primary = candidate_metrics.get(primary_metric, 0.0)
        prod_primary = prod_metrics.get(primary_metric, 0.0)
        is_improvement = (candidate_primary - prod_primary) >= min_improvement

        recommendation = (
            f"Candidate improves {primary_metric} by "
            f"{candidate_primary - prod_primary:.4f}. Promote to production."
            if is_improvement
            else f"Candidate does not meet minimum improvement threshold "
            f"({min_improvement}) on {primary_metric}. Keep current production model."
        )

        result = ComparisonResult(
            candidate_version=0,
            production_version=prod_info.version,
            candidate_metrics=candidate_metrics,
            production_metrics=prod_metrics,
            metric_deltas=deltas,
            is_improvement=is_improvement,
            recommendation=recommendation,
        )
        logger.info("Comparison result: %s", result.recommendation)
        return result

    # -- Querying -----------------------------------------------------------

    def list_model_versions(
        self,
        stages: list[str] | None = None,
    ) -> list[RegisteredModelInfo]:
        """List all versions of the registered model.

        Parameters
        ----------
        stages : list[str], optional
            Filter by stages (e.g. ``["Production", "Staging"]``).
        """
        if stages:
            versions = self._client.get_latest_versions(self.registered_model_name, stages=stages)
        else:

            filter_str = f"name='{self.registered_model_name}'"
            results = self._client.search_model_versions(filter_str)
            versions = list(results)

        infos: list[RegisteredModelInfo] = []
        for v in versions:
            run = self._client.get_run(v.run_id)
            infos.append(
                RegisteredModelInfo(
                    name=self.registered_model_name,
                    version=int(v.version),
                    stage=ModelStage(v.current_stage),
                    run_id=v.run_id,
                    metrics={k: float(val) for k, val in run.data.metrics.items()},
                    description=v.description or "",
                )
            )
        return infos

    def load_production_model(self) -> Any:
        """Load the current production model for inference."""
        model_uri = f"models:/{self.registered_model_name}/Production"
        model = mlflow.xgboost.load_model(model_uri)
        logger.info("Loaded production model from %s", model_uri)
        return model
