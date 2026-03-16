"""Shared test fixtures for ModelLab tests."""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.analysis.metrics_calculator import MetricsCalculator
from model_lab.analysis.statistical_engine import StatisticalEngine
from model_lab.experiments.experiment_manager import (
    Experiment,
    ExperimentConfig,
    ExperimentManager,
    SuccessMetric,
    Variant,
)
from model_lab.experiments.traffic_router import TrafficRouter
from model_lab.models.model_registry import ModelRegistry
from model_lab.monitoring.experiment_monitor import ExperimentMonitor, GuardrailConfig


@pytest.fixture
def experiment_manager() -> ExperimentManager:
    """Fresh experiment manager for each test."""
    return ExperimentManager()


@pytest.fixture
def traffic_router(experiment_manager: ExperimentManager) -> TrafficRouter:
    """Traffic router backed by the experiment manager fixture."""
    return TrafficRouter(experiment_manager)


@pytest.fixture
def statistical_engine() -> StatisticalEngine:
    """Statistical engine with default settings."""
    return StatisticalEngine(default_alpha=0.05)


@pytest.fixture
def metrics_calculator() -> MetricsCalculator:
    """Metrics calculator with default settings."""
    return MetricsCalculator(alpha=0.05)


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Fresh model registry."""
    return ModelRegistry()


@pytest.fixture
def experiment_monitor() -> ExperimentMonitor:
    """Experiment monitor with standard guardrails."""
    return ExperimentMonitor(
        srm_alpha=0.001,
        early_stopping_threshold=0.99,
        guardrails=[
            GuardrailConfig(metric_name="latency_p99", max_degradation_pct=10.0),
            GuardrailConfig(metric_name="error_rate", max_degradation_pct=1.0),
        ],
    )


@pytest.fixture
def simple_experiment_config() -> ExperimentConfig:
    """A standard 50/50 A/B experiment configuration."""
    return ExperimentConfig(
        name="Test Experiment",
        hypothesis="Treatment improves conversion by 5%",
        variants=[
            Variant(
                id="control",
                name="Control",
                traffic_percentage=50.0,
                is_control=True,
                model_version_id="model-v1",
            ),
            Variant(
                id="treatment",
                name="Treatment",
                traffic_percentage=50.0,
                is_control=False,
                model_version_id="model-v2",
            ),
        ],
        success_metrics=[
            SuccessMetric(
                name="conversion_rate",
                metric_type="conversion",
                minimum_detectable_effect=0.02,
                primary=True,
            ),
        ],
        traffic_allocation="random",
        min_sample_size_per_variant=100,
    )


@pytest.fixture
def running_experiment(
    experiment_manager: ExperimentManager,
    simple_experiment_config: ExperimentConfig,
) -> Experiment:
    """An experiment that has been created and started."""
    experiment = experiment_manager.create_experiment(simple_experiment_config)
    experiment_manager.start_experiment(experiment.id)
    return experiment


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
