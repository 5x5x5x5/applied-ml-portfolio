"""Tests for experiment lifecycle management.

Validates state machine transitions, experiment creation, mutual exclusion,
gradual rollout, and variant allocation strategies.
"""

from __future__ import annotations

import pytest

from model_lab.experiments.experiment_manager import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentState,
    RolloutStage,
    SuccessMetric,
    Variant,
)


class TestExperimentCreation:
    """Tests for creating experiments."""

    def test_create_valid_experiment(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should create an experiment in DRAFT state."""
        experiment = experiment_manager.create_experiment(simple_experiment_config)

        assert experiment.state == ExperimentState.DRAFT
        assert experiment.config.name == "Test Experiment"
        assert len(experiment.config.variants) == 2
        assert experiment.id is not None

    def test_create_initializes_sample_counts(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Sample counts should be initialized to 0."""
        experiment = experiment_manager.create_experiment(simple_experiment_config)

        for variant in experiment.config.variants:
            assert experiment.variant_sample_counts[variant.id] == 0

    def test_traffic_must_sum_to_100(self) -> None:
        """Traffic percentages must sum to exactly 100."""
        with pytest.raises(ValueError, match="Traffic percentages must sum to 100"):
            ExperimentConfig(
                name="Bad Config",
                variants=[
                    Variant(name="Control", traffic_percentage=40.0, is_control=True),
                    Variant(name="Treatment", traffic_percentage=40.0),
                ],
                success_metrics=[
                    SuccessMetric(
                        name="conversion",
                        metric_type="conversion",
                        minimum_detectable_effect=0.01,
                    ),
                ],
            )

    def test_must_have_exactly_one_control(self) -> None:
        """Exactly one variant must be marked as control."""
        with pytest.raises(ValueError, match="Exactly one control"):
            ExperimentConfig(
                name="No Control",
                variants=[
                    Variant(name="A", traffic_percentage=50.0, is_control=False),
                    Variant(name="B", traffic_percentage=50.0, is_control=False),
                ],
                success_metrics=[
                    SuccessMetric(
                        name="conversion",
                        metric_type="conversion",
                        minimum_detectable_effect=0.01,
                    ),
                ],
            )

    def test_must_have_at_least_two_variants(self) -> None:
        """Must have at least 2 variants."""
        with pytest.raises(ValueError):
            ExperimentConfig(
                name="One Variant",
                variants=[
                    Variant(name="Only", traffic_percentage=100.0, is_control=True),
                ],
                success_metrics=[
                    SuccessMetric(
                        name="conversion",
                        metric_type="conversion",
                        minimum_detectable_effect=0.01,
                    ),
                ],
            )


class TestStateMachine:
    """Tests for experiment state transitions."""

    def test_draft_to_running(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should transition from DRAFT to RUNNING."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        result = experiment_manager.start_experiment(exp.id)

        assert result.state == ExperimentState.RUNNING
        assert result.started_at is not None

    def test_running_to_paused(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should transition from RUNNING to PAUSED."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        result = experiment_manager.pause_experiment(exp.id)

        assert result.state == ExperimentState.PAUSED

    def test_paused_to_running(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should transition from PAUSED back to RUNNING."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        experiment_manager.pause_experiment(exp.id)
        result = experiment_manager.start_experiment(exp.id)

        assert result.state == ExperimentState.RUNNING

    def test_running_to_analyzing(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should transition from RUNNING to ANALYZING."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        result = experiment_manager.stop_experiment(exp.id)

        assert result.state == ExperimentState.ANALYZING

    def test_analyzing_to_completed(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should transition from ANALYZING to COMPLETED."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        experiment_manager.stop_experiment(exp.id)
        result = experiment_manager.complete_experiment(exp.id)

        assert result.state == ExperimentState.COMPLETED
        assert result.completed_at is not None

    def test_invalid_transition_raises(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Invalid state transitions should raise ValueError."""
        exp = experiment_manager.create_experiment(simple_experiment_config)

        # Can't pause a draft
        with pytest.raises(ValueError, match="Invalid state transition"):
            experiment_manager.pause_experiment(exp.id)

    def test_completed_is_terminal(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Completed experiments should not allow further transitions."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        experiment_manager.stop_experiment(exp.id)
        experiment_manager.complete_experiment(exp.id)

        with pytest.raises(ValueError, match="Invalid state transition"):
            experiment_manager.start_experiment(exp.id)

    def test_cancel_from_draft(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should be able to cancel from DRAFT."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        result = experiment_manager.cancel_experiment(exp.id)

        assert result.state == ExperimentState.CANCELLED

    def test_cancel_from_running(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should be able to cancel from RUNNING."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)
        result = experiment_manager.cancel_experiment(exp.id)

        assert result.state == ExperimentState.CANCELLED


class TestMutualExclusion:
    """Tests for mutual exclusion groups."""

    def test_two_experiments_same_group(self, experiment_manager: ExperimentManager) -> None:
        """Only one experiment in an exclusion group can be running."""
        config1 = ExperimentConfig(
            name="Exp 1",
            mutual_exclusion_group="pricing",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        config2 = ExperimentConfig(
            name="Exp 2",
            mutual_exclusion_group="pricing",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        exp1 = experiment_manager.create_experiment(config1)
        exp2 = experiment_manager.create_experiment(config2)

        experiment_manager.start_experiment(exp1.id)

        with pytest.raises(ValueError, match="exclusion group"):
            experiment_manager.start_experiment(exp2.id)

    def test_different_groups_can_run(self, experiment_manager: ExperimentManager) -> None:
        """Experiments in different groups can run simultaneously."""
        config1 = ExperimentConfig(
            name="Exp 1",
            mutual_exclusion_group="pricing",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        config2 = ExperimentConfig(
            name="Exp 2",
            mutual_exclusion_group="recommendations",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        exp1 = experiment_manager.create_experiment(config1)
        exp2 = experiment_manager.create_experiment(config2)

        experiment_manager.start_experiment(exp1.id)
        experiment_manager.start_experiment(exp2.id)

        assert experiment_manager.get_experiment(exp1.id).state == ExperimentState.RUNNING
        assert experiment_manager.get_experiment(exp2.id).state == ExperimentState.RUNNING


class TestGradualRollout:
    """Tests for gradual rollout support."""

    def test_initial_rollout_percentage(self, experiment_manager: ExperimentManager) -> None:
        """Experiment should start at first rollout stage percentage."""
        config = ExperimentConfig(
            name="Rollout Test",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
            rollout_stages=[
                RolloutStage(percentage=1.0, min_duration_hours=24),
                RolloutStage(percentage=5.0, min_duration_hours=24),
                RolloutStage(percentage=25.0, min_duration_hours=48),
                RolloutStage(percentage=100.0, min_duration_hours=0),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        assert exp.current_rollout_percentage == 1.0
        assert exp.current_rollout_stage == 0

    def test_advance_rollout(self, experiment_manager: ExperimentManager) -> None:
        """Should advance through rollout stages."""
        config = ExperimentConfig(
            name="Rollout Test",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
            rollout_stages=[
                RolloutStage(percentage=5.0),
                RolloutStage(percentage=25.0),
                RolloutStage(percentage=100.0),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        experiment_manager.start_experiment(exp.id)

        # Advance to 25%
        result = experiment_manager.advance_rollout(exp.id)
        assert result.current_rollout_percentage == 25.0
        assert result.current_rollout_stage == 1

        # Advance to 100%
        result = experiment_manager.advance_rollout(exp.id)
        assert result.current_rollout_percentage == 100.0
        assert result.current_rollout_stage == 2

    def test_cannot_advance_past_final_stage(self, experiment_manager: ExperimentManager) -> None:
        """Should raise when trying to advance past the last stage."""
        config = ExperimentConfig(
            name="Rollout Test",
            variants=[
                Variant(name="Control", traffic_percentage=50.0, is_control=True),
                Variant(name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
            rollout_stages=[
                RolloutStage(percentage=50.0),
                RolloutStage(percentage=100.0),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        experiment_manager.start_experiment(exp.id)
        experiment_manager.advance_rollout(exp.id)  # 50 -> 100

        with pytest.raises(ValueError, match="final rollout stage"):
            experiment_manager.advance_rollout(exp.id)


class TestVariantAllocation:
    """Tests for different traffic allocation strategies."""

    def test_random_allocation_returns_variant(self, experiment_manager: ExperimentManager) -> None:
        """Random allocation should return a valid variant."""
        config = ExperimentConfig(
            name="Random Test",
            traffic_allocation="random",
            variants=[
                Variant(id="c", name="Control", traffic_percentage=50.0, is_control=True),
                Variant(id="t", name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        experiment_manager.start_experiment(exp.id)

        variant = experiment_manager.allocate_variant(exp.id, "user-1")
        assert variant.id in ["c", "t"]

    def test_stratified_allocation(self, experiment_manager: ExperimentManager) -> None:
        """Stratified allocation should maintain balance."""
        config = ExperimentConfig(
            name="Stratified Test",
            traffic_allocation="stratified",
            variants=[
                Variant(id="c", name="Control", traffic_percentage=50.0, is_control=True),
                Variant(id="t", name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        experiment_manager.start_experiment(exp.id)

        for i in range(100):
            experiment_manager.allocate_variant(exp.id, f"user-{i}")

        counts = exp.variant_sample_counts
        total = sum(counts.values())
        for vid, count in counts.items():
            ratio = count / total
            assert 0.35 <= ratio <= 0.65

    def test_bandit_allocation(self, experiment_manager: ExperimentManager) -> None:
        """Bandit allocation should return valid variants."""
        config = ExperimentConfig(
            name="Bandit Test",
            traffic_allocation="multi_armed_bandit",
            variants=[
                Variant(id="c", name="Control", traffic_percentage=50.0, is_control=True),
                Variant(id="t", name="Treatment", traffic_percentage=50.0),
            ],
            success_metrics=[
                SuccessMetric(
                    name="conversion",
                    metric_type="conversion",
                    minimum_detectable_effect=0.01,
                ),
            ],
        )

        exp = experiment_manager.create_experiment(config)
        experiment_manager.start_experiment(exp.id)

        # Record some rewards to influence bandit
        for _ in range(20):
            experiment_manager.record_reward(exp.id, "t", 1.0)
            experiment_manager.record_reward(exp.id, "c", 0.0)

        # Bandit should favor the winning arm over many allocations
        t_count = 0
        for i in range(100):
            variant = experiment_manager.allocate_variant(exp.id, f"user-{i}")
            if variant.id == "t":
                t_count += 1

        # Treatment should get more traffic after showing better performance
        assert t_count > 50, f"Bandit should favor treatment, got {t_count}/100"

    def test_cannot_allocate_non_running(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should raise when allocating for a non-running experiment."""
        exp = experiment_manager.create_experiment(simple_experiment_config)

        with pytest.raises(ValueError, match="not running"):
            experiment_manager.allocate_variant(exp.id, "user-1")


class TestListAndGet:
    """Tests for listing and retrieving experiments."""

    def test_get_experiment(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should retrieve experiment by ID."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        retrieved = experiment_manager.get_experiment(exp.id)
        assert retrieved.id == exp.id

    def test_get_nonexistent_raises(self, experiment_manager: ExperimentManager) -> None:
        """Should raise KeyError for unknown experiment."""
        with pytest.raises(KeyError):
            experiment_manager.get_experiment("nonexistent")

    def test_list_all(self, experiment_manager: ExperimentManager) -> None:
        """Should list all experiments."""
        for i in range(3):
            config = ExperimentConfig(
                name=f"Exp {i}",
                variants=[
                    Variant(name="Control", traffic_percentage=50.0, is_control=True),
                    Variant(name="Treatment", traffic_percentage=50.0),
                ],
                success_metrics=[
                    SuccessMetric(
                        name="conversion",
                        metric_type="conversion",
                        minimum_detectable_effect=0.01,
                    ),
                ],
            )
            experiment_manager.create_experiment(config)

        experiments = experiment_manager.list_experiments()
        assert len(experiments) == 3

    def test_list_by_state(
        self,
        experiment_manager: ExperimentManager,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should filter experiments by state."""
        exp1 = experiment_manager.create_experiment(simple_experiment_config)
        config2 = simple_experiment_config.model_copy(update={"name": "Exp 2"})
        exp2 = experiment_manager.create_experiment(config2)
        experiment_manager.start_experiment(exp1.id)

        running = experiment_manager.list_experiments(state=ExperimentState.RUNNING)
        drafts = experiment_manager.list_experiments(state=ExperimentState.DRAFT)

        assert len(running) == 1
        assert len(drafts) == 1
        assert running[0].id == exp1.id
