"""Experiment lifecycle management with state machine, traffic allocation, and mutual exclusion.

Handles the full lifecycle of A/B experiments from draft through completion,
including gradual rollout support and multi-armed bandit allocation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class ExperimentState(str, Enum):
    """State machine states for experiment lifecycle."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficAllocationType(str, Enum):
    """Strategy for allocating traffic across variants."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


class RolloutStage(BaseModel):
    """A single stage in a gradual rollout plan."""

    percentage: float = Field(ge=0.0, le=100.0)
    min_duration_hours: float = Field(ge=0.0, default=24.0)
    min_sample_size: int = Field(ge=0, default=100)
    auto_advance: bool = False


class Variant(BaseModel):
    """A single variant (treatment or control) in an experiment."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    model_version_id: str | None = None
    description: str = ""
    traffic_percentage: float = Field(ge=0.0, le=100.0)
    is_control: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuccessMetric(BaseModel):
    """Definition of a metric used to evaluate experiment success."""

    name: str
    metric_type: str = Field(description="conversion | continuous | revenue")
    minimum_detectable_effect: float = Field(
        ge=0.0, description="Minimum effect size worth detecting"
    )
    primary: bool = True
    direction: str = Field(default="increase", description="increase | decrease")


class ExperimentConfig(BaseModel):
    """Full configuration for creating a new experiment."""

    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = ""
    description: str = ""
    variants: list[Variant] = Field(min_length=2)
    success_metrics: list[SuccessMetric] = Field(min_length=1)
    traffic_allocation: TrafficAllocationType = TrafficAllocationType.RANDOM
    mutual_exclusion_group: str | None = None
    rollout_stages: list[RolloutStage] | None = None
    max_duration_days: int = Field(default=30, ge=1, le=365)
    min_sample_size_per_variant: int = Field(default=1000, ge=10)
    owner: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("variants")
    @classmethod
    def validate_traffic_split(cls, variants: list[Variant]) -> list[Variant]:
        total = sum(v.traffic_percentage for v in variants)
        if abs(total - 100.0) > 0.01:
            msg = f"Traffic percentages must sum to 100, got {total}"
            raise ValueError(msg)
        controls = [v for v in variants if v.is_control]
        if len(controls) != 1:
            msg = f"Exactly one control variant required, got {len(controls)}"
            raise ValueError(msg)
        return variants


# Valid state transitions
VALID_TRANSITIONS: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.DRAFT: {ExperimentState.RUNNING, ExperimentState.CANCELLED},
    ExperimentState.RUNNING: {
        ExperimentState.PAUSED,
        ExperimentState.ANALYZING,
        ExperimentState.CANCELLED,
    },
    ExperimentState.PAUSED: {ExperimentState.RUNNING, ExperimentState.CANCELLED},
    ExperimentState.ANALYZING: {ExperimentState.COMPLETED, ExperimentState.RUNNING},
    ExperimentState.COMPLETED: set(),
    ExperimentState.CANCELLED: set(),
}


class Experiment(BaseModel):
    """A running experiment with its full state."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: ExperimentConfig
    state: ExperimentState = ExperimentState.DRAFT
    current_rollout_stage: int = 0
    current_rollout_percentage: float = 100.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    variant_sample_counts: dict[str, int] = Field(default_factory=dict)
    bandit_rewards: dict[str, list[float]] = Field(default_factory=dict)


class ExperimentManager:
    """Manages the lifecycle of A/B experiments.

    Supports creating, starting, stopping, and pausing experiments.
    Handles traffic allocation strategies including multi-armed bandit,
    mutual exclusion groups, and gradual rollout plans.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._exclusion_groups: dict[str, set[str]] = {}

    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment in DRAFT state.

        Args:
            config: Full experiment configuration including variants and metrics.

        Returns:
            The newly created Experiment instance.

        Raises:
            ValueError: If a mutual exclusion conflict exists.
        """
        if config.mutual_exclusion_group:
            self._check_exclusion_conflict(config.mutual_exclusion_group)

        experiment = Experiment(config=config)

        if config.rollout_stages:
            experiment.current_rollout_percentage = config.rollout_stages[0].percentage

        # Initialize sample counts and bandit rewards
        for variant in config.variants:
            experiment.variant_sample_counts[variant.id] = 0
            experiment.bandit_rewards[variant.id] = []

        self._experiments[experiment.id] = experiment

        if config.mutual_exclusion_group:
            group = config.mutual_exclusion_group
            if group not in self._exclusion_groups:
                self._exclusion_groups[group] = set()
            self._exclusion_groups[group].add(experiment.id)

        logger.info(
            "experiment_created",
            experiment_id=experiment.id,
            name=config.name,
            variants=len(config.variants),
            allocation=config.traffic_allocation.value,
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by ID.

        Raises:
            KeyError: If experiment not found.
        """
        if experiment_id not in self._experiments:
            msg = f"Experiment {experiment_id} not found"
            raise KeyError(msg)
        return self._experiments[experiment_id]

    def list_experiments(
        self,
        state: ExperimentState | None = None,
        tag: str | None = None,
    ) -> list[Experiment]:
        """List experiments, optionally filtered by state or tag."""
        experiments = list(self._experiments.values())
        if state is not None:
            experiments = [e for e in experiments if e.state == state]
        if tag is not None:
            experiments = [e for e in experiments if tag in e.config.tags]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Transition experiment from DRAFT or PAUSED to RUNNING.

        Raises:
            KeyError: If experiment not found.
            ValueError: If transition is invalid or exclusion conflict exists.
        """
        experiment = self.get_experiment(experiment_id)
        self._transition_state(experiment, ExperimentState.RUNNING)

        if experiment.started_at is None:
            experiment.started_at = datetime.now(UTC)

        # Check mutual exclusion: no two running experiments in same group
        if experiment.config.mutual_exclusion_group:
            group = experiment.config.mutual_exclusion_group
            for other_id in self._exclusion_groups.get(group, set()):
                if other_id != experiment_id:
                    other = self._experiments.get(other_id)
                    if other and other.state == ExperimentState.RUNNING:
                        # Revert state
                        experiment.state = ExperimentState.DRAFT
                        msg = (
                            f"Cannot start: experiment {other_id} in exclusion group "
                            f"'{group}' is already running"
                        )
                        raise ValueError(msg)

        logger.info("experiment_started", experiment_id=experiment_id)
        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        experiment = self.get_experiment(experiment_id)
        self._transition_state(experiment, ExperimentState.PAUSED)
        logger.info("experiment_paused", experiment_id=experiment_id)
        return experiment

    def stop_experiment(self, experiment_id: str) -> Experiment:
        """Move experiment to ANALYZING state for result computation."""
        experiment = self.get_experiment(experiment_id)
        self._transition_state(experiment, ExperimentState.ANALYZING)
        logger.info("experiment_stopped", experiment_id=experiment_id)
        return experiment

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Mark experiment as COMPLETED after analysis."""
        experiment = self.get_experiment(experiment_id)
        self._transition_state(experiment, ExperimentState.COMPLETED)
        experiment.completed_at = datetime.now(UTC)
        logger.info("experiment_completed", experiment_id=experiment_id)
        return experiment

    def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment from any non-terminal state."""
        experiment = self.get_experiment(experiment_id)
        self._transition_state(experiment, ExperimentState.CANCELLED)
        experiment.completed_at = datetime.now(UTC)
        logger.info("experiment_cancelled", experiment_id=experiment_id)
        return experiment

    def advance_rollout(self, experiment_id: str) -> Experiment:
        """Advance to the next rollout stage.

        Raises:
            ValueError: If no more rollout stages or experiment not running.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment.state != ExperimentState.RUNNING:
            msg = "Can only advance rollout for running experiments"
            raise ValueError(msg)

        stages = experiment.config.rollout_stages
        if not stages:
            msg = "Experiment has no rollout stages configured"
            raise ValueError(msg)

        next_stage = experiment.current_rollout_stage + 1
        if next_stage >= len(stages):
            msg = f"Already at final rollout stage ({experiment.current_rollout_stage})"
            raise ValueError(msg)

        experiment.current_rollout_stage = next_stage
        experiment.current_rollout_percentage = stages[next_stage].percentage
        experiment.updated_at = datetime.now(UTC)

        logger.info(
            "rollout_advanced",
            experiment_id=experiment_id,
            stage=next_stage,
            percentage=stages[next_stage].percentage,
        )
        return experiment

    def allocate_variant(self, experiment_id: str, user_id: str) -> Variant:
        """Allocate a user to a variant based on the experiment's traffic allocation strategy.

        Args:
            experiment_id: The experiment to allocate for.
            user_id: Unique identifier for the user being allocated.

        Returns:
            The assigned Variant.

        Raises:
            KeyError: If experiment not found.
            ValueError: If experiment is not running.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment.state != ExperimentState.RUNNING:
            msg = f"Experiment {experiment_id} is not running (state: {experiment.state})"
            raise ValueError(msg)

        strategy = experiment.config.traffic_allocation
        if strategy == TrafficAllocationType.RANDOM:
            variant = self._allocate_random(experiment, user_id)
        elif strategy == TrafficAllocationType.STRATIFIED:
            variant = self._allocate_stratified(experiment, user_id)
        elif strategy == TrafficAllocationType.MULTI_ARMED_BANDIT:
            variant = self._allocate_bandit(experiment)
        else:
            variant = self._allocate_random(experiment, user_id)

        experiment.variant_sample_counts[variant.id] = (
            experiment.variant_sample_counts.get(variant.id, 0) + 1
        )

        return variant

    def record_reward(self, experiment_id: str, variant_id: str, reward: float) -> None:
        """Record a reward observation for bandit allocation.

        Args:
            experiment_id: The experiment.
            variant_id: The variant that produced the reward.
            reward: Reward value (typically 0 or 1 for conversion).
        """
        experiment = self.get_experiment(experiment_id)
        if variant_id in experiment.bandit_rewards:
            experiment.bandit_rewards[variant_id].append(reward)

    def _allocate_random(self, experiment: Experiment, user_id: str) -> Variant:
        """Random allocation based on traffic split percentages.

        Uses a deterministic hash so the same user always gets the same variant.
        """
        # Deterministic hash for consistent assignment
        hash_val = hash(f"{experiment.id}:{user_id}") % 10000
        bucket = hash_val / 100.0  # 0-100

        cumulative = 0.0
        for variant in experiment.config.variants:
            # Scale by rollout percentage
            scaled_pct = variant.traffic_percentage * (
                experiment.current_rollout_percentage / 100.0
            )
            cumulative += scaled_pct
            if bucket < cumulative:
                return variant

        # Fallback to control
        return next(v for v in experiment.config.variants if v.is_control)

    def _allocate_stratified(self, experiment: Experiment, user_id: str) -> Variant:
        """Stratified allocation maintaining exact traffic ratios.

        Assigns the variant with the largest deficit between target and actual ratio.
        Falls back to random allocation with consistent hashing for tie-breaking.
        """
        total_samples = sum(experiment.variant_sample_counts.values())
        if total_samples == 0:
            return self._allocate_random(experiment, user_id)

        best_variant = None
        best_deficit = -float("inf")

        for variant in experiment.config.variants:
            target_ratio = variant.traffic_percentage / 100.0
            actual_count = experiment.variant_sample_counts.get(variant.id, 0)
            actual_ratio = actual_count / total_samples if total_samples > 0 else 0.0
            deficit = target_ratio - actual_ratio

            if deficit > best_deficit:
                best_deficit = deficit
                best_variant = variant

        return best_variant or experiment.config.variants[0]

    def _allocate_bandit(self, experiment: Experiment) -> Variant:
        """Thompson Sampling multi-armed bandit allocation.

        Uses Beta(successes+1, failures+1) posterior for each variant's conversion rate,
        then samples from each posterior and picks the variant with the highest sample.
        """
        best_variant = None
        best_sample = -1.0

        for variant in experiment.config.variants:
            rewards = experiment.bandit_rewards.get(variant.id, [])
            successes = sum(1 for r in rewards if r > 0.5)
            failures = len(rewards) - successes
            # Beta posterior: Beta(alpha=successes+1, beta=failures+1)
            sample = float(np.random.beta(successes + 1, failures + 1))

            if sample > best_sample:
                best_sample = sample
                best_variant = variant

        return best_variant or experiment.config.variants[0]

    def _transition_state(self, experiment: Experiment, new_state: ExperimentState) -> None:
        """Validate and execute a state transition.

        Raises:
            ValueError: If the transition is not valid.
        """
        valid_next = VALID_TRANSITIONS.get(experiment.state, set())
        if new_state not in valid_next:
            msg = (
                f"Invalid state transition: {experiment.state.value} -> {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_next]}"
            )
            raise ValueError(msg)
        experiment.state = new_state
        experiment.updated_at = datetime.now(UTC)

    def _check_exclusion_conflict(self, group: str) -> None:
        """Check if adding to a mutual exclusion group would cause conflicts.

        Note: We allow multiple experiments in a group but only one can be RUNNING.
        This check is informational at creation time; the real enforcement is at start.
        """
        running_in_group = [
            eid
            for eid in self._exclusion_groups.get(group, set())
            if self._experiments.get(
                eid,
                Experiment(
                    config=ExperimentConfig(
                        name="temp",
                        variants=[
                            Variant(name="c", traffic_percentage=50, is_control=True),
                            Variant(name="t", traffic_percentage=50),
                        ],
                        success_metrics=[
                            SuccessMetric(
                                name="temp",
                                metric_type="conversion",
                                minimum_detectable_effect=0.01,
                            )
                        ],
                    )
                ),
            ).state
            == ExperimentState.RUNNING
        ]
        if running_in_group:
            logger.warning(
                "exclusion_group_has_running",
                group=group,
                running_experiments=running_in_group,
            )
