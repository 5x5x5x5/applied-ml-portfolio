"""FastAPI application for ModelLab A/B testing platform.

Provides REST endpoints for experiment management, traffic routing,
event logging, and result analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_lab.analysis.metrics_calculator import MetricsCalculator
from model_lab.analysis.statistical_engine import StatisticalEngine
from model_lab.experiments.experiment_manager import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentState,
    RolloutStage,
    SuccessMetric,
    Variant,
)
from model_lab.experiments.traffic_router import RoutingContext, TrafficRouter
from model_lab.models.model_registry import ModelRegistry
from model_lab.monitoring.experiment_monitor import ExperimentMonitor, GuardrailConfig

logger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# Application setup
# ----------------------------------------------------------------

app = FastAPI(
    title="ModelLab",
    description="A/B Testing Platform for ML Models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core services
experiment_manager = ExperimentManager()
traffic_router = TrafficRouter(experiment_manager)
statistical_engine = StatisticalEngine()
metrics_calculator = MetricsCalculator()
model_registry = ModelRegistry()
experiment_monitor = ExperimentMonitor(
    guardrails=[
        GuardrailConfig(metric_name="latency_p99", max_degradation_pct=10.0),
        GuardrailConfig(metric_name="error_rate", max_degradation_pct=1.0),
    ]
)

# In-memory event store (would be a database in production)
event_store: dict[str, list[dict[str, Any]]] = {}


# ----------------------------------------------------------------
# Request/Response Models
# ----------------------------------------------------------------


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""

    name: str
    hypothesis: str = ""
    description: str = ""
    variants: list[dict[str, Any]]
    success_metrics: list[dict[str, Any]]
    traffic_allocation: str = "random"
    mutual_exclusion_group: str | None = None
    rollout_stages: list[dict[str, Any]] | None = None
    max_duration_days: int = 30
    min_sample_size_per_variant: int = 1000
    owner: str = ""
    tags: list[str] = Field(default_factory=list)


class RouteRequest(BaseModel):
    """Request to route a user to a model variant."""

    user_id: str
    experiment_id: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    force_variant: str | None = None
    session_id: str | None = None


class LogEventRequest(BaseModel):
    """Request to log an experiment event."""

    experiment_id: str
    variant_id: str
    user_id: str
    event_type: str  # "conversion", "outcome", "latency", "error"
    value: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None


class ExperimentSummary(BaseModel):
    """Summary view of an experiment."""

    id: str
    name: str
    state: str
    variants: list[dict[str, Any]]
    created_at: str
    started_at: str | None
    sample_counts: dict[str, int]
    traffic_allocation: str


class AnalysisResult(BaseModel):
    """Result of experiment analysis."""

    experiment_id: str
    experiment_name: str
    state: str
    frequentist: dict[str, Any] | None = None
    bayesian: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    recommendation: str = ""


# ----------------------------------------------------------------
# Experiment Endpoints
# ----------------------------------------------------------------


@app.post("/experiments", response_model=ExperimentSummary)
def create_experiment(request: CreateExperimentRequest) -> ExperimentSummary:
    """Create a new A/B experiment."""
    try:
        variants = [
            Variant(
                name=v["name"],
                traffic_percentage=v["traffic_percentage"],
                is_control=v.get("is_control", False),
                model_version_id=v.get("model_version_id"),
                description=v.get("description", ""),
            )
            for v in request.variants
        ]

        success_metrics = [
            SuccessMetric(
                name=m["name"],
                metric_type=m.get("metric_type", "conversion"),
                minimum_detectable_effect=m.get("minimum_detectable_effect", 0.01),
                primary=m.get("primary", True),
                direction=m.get("direction", "increase"),
            )
            for m in request.success_metrics
        ]

        rollout_stages = None
        if request.rollout_stages:
            rollout_stages = [
                RolloutStage(
                    percentage=s["percentage"],
                    min_duration_hours=s.get("min_duration_hours", 24.0),
                    min_sample_size=s.get("min_sample_size", 100),
                    auto_advance=s.get("auto_advance", False),
                )
                for s in request.rollout_stages
            ]

        config = ExperimentConfig(
            name=request.name,
            hypothesis=request.hypothesis,
            description=request.description,
            variants=variants,
            success_metrics=success_metrics,
            traffic_allocation=request.traffic_allocation,
            mutual_exclusion_group=request.mutual_exclusion_group,
            rollout_stages=rollout_stages,
            max_duration_days=request.max_duration_days,
            min_sample_size_per_variant=request.min_sample_size_per_variant,
            owner=request.owner,
            tags=request.tags,
        )

        experiment = experiment_manager.create_experiment(config)
        event_store[experiment.id] = []

        return ExperimentSummary(
            id=experiment.id,
            name=experiment.config.name,
            state=experiment.state.value,
            variants=[
                {
                    "id": v.id,
                    "name": v.name,
                    "traffic_percentage": v.traffic_percentage,
                    "is_control": v.is_control,
                }
                for v in experiment.config.variants
            ],
            created_at=experiment.created_at.isoformat(),
            started_at=None,
            sample_counts=experiment.variant_sample_counts,
            traffic_allocation=experiment.config.traffic_allocation.value,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/experiments/{experiment_id}/start")
def start_experiment(experiment_id: str) -> dict[str, Any]:
    """Start an experiment, moving it to RUNNING state."""
    try:
        experiment = experiment_manager.start_experiment(experiment_id)
        return {
            "status": "started",
            "experiment_id": experiment.id,
            "state": experiment.state.value,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/experiments/{experiment_id}/pause")
def pause_experiment(experiment_id: str) -> dict[str, Any]:
    """Pause a running experiment."""
    try:
        experiment = experiment_manager.pause_experiment(experiment_id)
        return {"status": "paused", "experiment_id": experiment.id, "state": experiment.state.value}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/experiments/{experiment_id}/stop")
def stop_experiment(experiment_id: str) -> dict[str, Any]:
    """Stop a running experiment and move to ANALYZING."""
    try:
        experiment = experiment_manager.stop_experiment(experiment_id)
        return {
            "status": "analyzing",
            "experiment_id": experiment.id,
            "state": experiment.state.value,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/experiments/{experiment_id}/complete")
def complete_experiment(experiment_id: str) -> dict[str, Any]:
    """Mark experiment as COMPLETED."""
    try:
        experiment = experiment_manager.complete_experiment(experiment_id)
        return {
            "status": "completed",
            "experiment_id": experiment.id,
            "state": experiment.state.value,
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/experiments")
def list_experiments(
    state: str | None = Query(None),
    tag: str | None = Query(None),
) -> list[ExperimentSummary]:
    """List all experiments with optional filtering."""
    state_enum = ExperimentState(state) if state else None
    experiments = experiment_manager.list_experiments(state=state_enum, tag=tag)
    return [
        ExperimentSummary(
            id=e.id,
            name=e.config.name,
            state=e.state.value,
            variants=[
                {
                    "id": v.id,
                    "name": v.name,
                    "traffic_percentage": v.traffic_percentage,
                    "is_control": v.is_control,
                }
                for v in e.config.variants
            ],
            created_at=e.created_at.isoformat(),
            started_at=e.started_at.isoformat() if e.started_at else None,
            sample_counts=e.variant_sample_counts,
            traffic_allocation=e.config.traffic_allocation.value,
        )
        for e in experiments
    ]


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Get full experiment details."""
    try:
        e = experiment_manager.get_experiment(experiment_id)
        return {
            "id": e.id,
            "name": e.config.name,
            "hypothesis": e.config.hypothesis,
            "description": e.config.description,
            "state": e.state.value,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "traffic_percentage": v.traffic_percentage,
                    "is_control": v.is_control,
                    "model_version_id": v.model_version_id,
                }
                for v in e.config.variants
            ],
            "success_metrics": [
                {
                    "name": m.name,
                    "metric_type": m.metric_type,
                    "minimum_detectable_effect": m.minimum_detectable_effect,
                    "primary": m.primary,
                    "direction": m.direction,
                }
                for m in e.config.success_metrics
            ],
            "traffic_allocation": e.config.traffic_allocation.value,
            "created_at": e.created_at.isoformat(),
            "started_at": e.started_at.isoformat() if e.started_at else None,
            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
            "sample_counts": e.variant_sample_counts,
            "current_rollout_percentage": e.current_rollout_percentage,
            "event_count": len(event_store.get(e.id, [])),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ----------------------------------------------------------------
# Traffic Routing
# ----------------------------------------------------------------


@app.post("/route")
def route_request(request: RouteRequest) -> dict[str, Any]:
    """Route a request to the appropriate model variant."""
    context = RoutingContext(
        user_id=request.user_id,
        experiment_id=request.experiment_id,
        features=request.features,
        force_variant=request.force_variant,
        session_id=request.session_id,
    )

    decision = traffic_router.route(context)
    if decision is None:
        raise HTTPException(status_code=404, detail="No applicable experiment found")

    return {
        "experiment_id": decision.experiment_id,
        "variant_id": decision.variant_id,
        "variant_name": decision.variant_name,
        "model_version_id": decision.model_version_id,
        "is_override": decision.is_override,
        "reason": decision.reason,
    }


# ----------------------------------------------------------------
# Event Logging
# ----------------------------------------------------------------


@app.post("/events")
def log_event(request: LogEventRequest) -> dict[str, Any]:
    """Log an experiment event (conversion, outcome, latency measurement)."""
    try:
        experiment_manager.get_experiment(request.experiment_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    event = {
        "experiment_id": request.experiment_id,
        "variant_id": request.variant_id,
        "user_id": request.user_id,
        "event_type": request.event_type,
        "value": request.value,
        "metadata": request.metadata,
        "timestamp": request.timestamp or datetime.now(UTC).isoformat(),
    }

    if request.experiment_id not in event_store:
        event_store[request.experiment_id] = []
    event_store[request.experiment_id].append(event)

    # Record reward for bandit experiments
    if request.event_type == "conversion" and request.value is not None:
        experiment_manager.record_reward(
            request.experiment_id,
            request.variant_id,
            request.value,
        )

    return {"status": "recorded", "event_count": len(event_store[request.experiment_id])}


# ----------------------------------------------------------------
# Analysis Results
# ----------------------------------------------------------------


@app.get("/experiments/{experiment_id}/results")
def get_results(experiment_id: str) -> AnalysisResult:
    """Get statistical analysis results for an experiment."""
    try:
        experiment = experiment_manager.get_experiment(experiment_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    events = event_store.get(experiment_id, [])
    if not events:
        return AnalysisResult(
            experiment_id=experiment_id,
            experiment_name=experiment.config.name,
            state=experiment.state.value,
            recommendation="Insufficient data for analysis",
        )

    # Organize events by variant
    variant_events: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        vid = event["variant_id"]
        if vid not in variant_events:
            variant_events[vid] = []
        variant_events[vid].append(event)

    # Find control and treatment
    control_variant = next((v for v in experiment.config.variants if v.is_control), None)
    treatment_variants = [v for v in experiment.config.variants if not v.is_control]

    if not control_variant or not treatment_variants:
        return AnalysisResult(
            experiment_id=experiment_id,
            experiment_name=experiment.config.name,
            state=experiment.state.value,
            recommendation="Missing control or treatment variant",
        )

    # Compute per-variant sample sizes
    sample_sizes = {vid: len(evts) for vid, evts in variant_events.items()}

    # Analyze conversion events
    frequentist_result = None
    bayesian_result = None

    control_events = variant_events.get(control_variant.id, [])
    control_conversions = [e for e in control_events if e["event_type"] == "conversion"]

    for treatment in treatment_variants:
        treatment_events = variant_events.get(treatment.id, [])
        treatment_conversions = [e for e in treatment_events if e["event_type"] == "conversion"]

        if control_conversions and treatment_conversions:
            # Count successes (value > 0.5 counts as conversion)
            c_successes = sum(1 for e in control_conversions if (e.get("value") or 0) > 0.5)
            c_total = len(control_conversions)
            t_successes = sum(1 for e in treatment_conversions if (e.get("value") or 0) > 0.5)
            t_total = len(treatment_conversions)

            if c_total > 0 and t_total > 0:
                # Frequentist z-test
                freq = statistical_engine.z_test_proportions(
                    c_successes, c_total, t_successes, t_total
                )
                frequentist_result = {
                    "test_type": freq.test_type.value,
                    "test_statistic": round(freq.test_statistic, 4),
                    "p_value": round(freq.p_value, 6),
                    "is_significant": freq.is_significant,
                    "confidence_interval": [
                        round(freq.confidence_interval[0], 6),
                        round(freq.confidence_interval[1], 6),
                    ],
                    "effect_size": round(freq.effect_size, 6),
                    "relative_effect": round(freq.relative_effect, 4),
                    "control_rate": round(freq.control_mean, 4),
                    "treatment_rate": round(freq.treatment_mean, 4),
                }

                # Bayesian
                bayes = statistical_engine.bayesian_beta_binomial(
                    c_successes, c_total, t_successes, t_total
                )
                bayesian_result = {
                    "probability_treatment_better": round(bayes.probability_b_better, 4),
                    "expected_loss_treatment": round(bayes.expected_loss_b, 6),
                    "expected_loss_control": round(bayes.expected_loss_a, 6),
                    "credible_interval": [
                        round(bayes.credible_interval[0], 6),
                        round(bayes.credible_interval[1], 6),
                    ],
                    "risk_threshold_met": bayes.risk_threshold_met,
                    "posterior_control": {k: round(v, 4) for k, v in bayes.posterior_a.items()},
                    "posterior_treatment": {k: round(v, 4) for k, v in bayes.posterior_b.items()},
                }

    # Generate recommendation
    recommendation = _generate_recommendation(frequentist_result, bayesian_result, sample_sizes)

    # Compute additional metrics
    metrics: dict[str, Any] = {}
    for vid, evts in variant_events.items():
        latency_events = [e for e in evts if e["event_type"] == "latency" and e.get("value")]
        if latency_events:
            latencies = np.array([e["value"] for e in latency_events])
            quantiles = metrics_calculator.quantile_metrics(latencies, vid)
            metrics[f"latency_{vid}"] = {
                "p50": round(quantiles.p50, 2),
                "p95": round(quantiles.p95, 2),
                "p99": round(quantiles.p99, 2),
                "mean": round(quantiles.mean, 2),
            }

    return AnalysisResult(
        experiment_id=experiment_id,
        experiment_name=experiment.config.name,
        state=experiment.state.value,
        frequentist=frequentist_result,
        bayesian=bayesian_result,
        metrics=metrics,
        sample_sizes=sample_sizes,
        recommendation=recommendation,
    )


# ----------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------


@app.get("/experiments/{experiment_id}/health")
def experiment_health(experiment_id: str) -> dict[str, Any]:
    """Run health checks on an experiment."""
    try:
        experiment = experiment_manager.get_experiment(experiment_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    events = event_store.get(experiment_id, [])

    # Build variant data for monitoring
    variant_data: dict[str, dict[str, Any]] = {}
    for variant in experiment.config.variants:
        v_events = [e for e in events if e["variant_id"] == variant.id]
        conversions = [e for e in v_events if e["event_type"] == "conversion"]
        latencies = [
            e["value"] for e in v_events if e["event_type"] == "latency" and e.get("value")
        ]
        errors = [e for e in v_events if e["event_type"] == "error"]

        variant_data[variant.id] = {
            "sample_size": len(v_events),
            "conversions": sum(1 for e in conversions if (e.get("value") or 0) > 0.5),
            "latency_values": latencies,
            "error_count": len(errors),
        }

    report = experiment_monitor.check_health(experiment, variant_data)

    return {
        "experiment_id": report.experiment_id,
        "is_healthy": report.is_healthy,
        "alerts": [
            {
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "message": a.message,
                "actionable": a.actionable,
            }
            for a in report.alerts
        ],
        "srm_check": report.srm_check,
        "data_quality": report.data_quality,
        "guardrail_status": report.guardrail_status,
        "early_stopping": report.early_stopping_recommendation,
    }


# ----------------------------------------------------------------
# Power Analysis
# ----------------------------------------------------------------


@app.get("/power-analysis")
def power_analysis(
    baseline_rate: float = Query(..., description="Expected baseline conversion rate"),
    mde: float = Query(..., description="Minimum detectable effect (absolute)"),
    alpha: float = Query(0.05, description="Significance level"),
    power: float = Query(0.8, description="Desired statistical power"),
) -> dict[str, Any]:
    """Calculate required sample size for an experiment."""
    result = statistical_engine.power_analysis_proportions(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=mde,
        alpha=alpha,
        power=power,
    )
    return {
        "required_sample_size_per_group": result.required_sample_size_per_group,
        "total_required": result.required_sample_size_per_group * 2,
        "power": result.power,
        "alpha": result.alpha,
        "mde": result.minimum_detectable_effect,
        "baseline_rate": result.baseline_rate,
    }


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _generate_recommendation(
    frequentist: dict[str, Any] | None,
    bayesian: dict[str, Any] | None,
    sample_sizes: dict[str, int],
) -> str:
    """Generate a human-readable recommendation based on analysis results."""
    total_samples = sum(sample_sizes.values())

    if total_samples < 100:
        return "Insufficient data. Continue collecting observations."

    parts = []

    if frequentist:
        if frequentist["is_significant"]:
            direction = "positive" if frequentist["effect_size"] > 0 else "negative"
            parts.append(
                f"Statistically significant {direction} effect "
                f"(p={frequentist['p_value']:.4f}, "
                f"relative effect={frequentist['relative_effect']:.1%})"
            )
        else:
            parts.append(
                f"Not statistically significant (p={frequentist['p_value']:.4f}). "
                "Consider increasing sample size."
            )

    if bayesian:
        prob = bayesian["probability_treatment_better"]
        if prob > 0.95:
            parts.append(
                f"Bayesian analysis: {prob:.1%} probability treatment is better. "
                "Consider promoting."
            )
        elif prob < 0.05:
            parts.append(
                f"Bayesian analysis: Only {prob:.1%} probability treatment is better. "
                "Consider reverting."
            )
        else:
            parts.append(
                f"Bayesian analysis: {prob:.1%} probability treatment is better. "
                "Continue experiment."
            )

    return " | ".join(parts) if parts else "Analysis pending."


# ----------------------------------------------------------------
# App lifecycle
# ----------------------------------------------------------------


@app.get("/health")
def health_check() -> dict[str, str]:
    """Application health check."""
    return {"status": "healthy", "service": "model-lab"}
