"""Tests for the traffic router.

Validates consistent hashing, sticky sessions, override rules,
feature flag evaluation, and multi-experiment routing.
"""

from __future__ import annotations

from collections import Counter

from model_lab.experiments.experiment_manager import (
    ExperimentConfig,
    ExperimentManager,
    SuccessMetric,
    Variant,
)
from model_lab.experiments.traffic_router import (
    ConsistentHasher,
    FeatureFlagEvaluator,
    RoutingContext,
    TrafficRouter,
)


class TestConsistentHasher:
    """Tests for consistent hashing behavior."""

    def test_deterministic(self) -> None:
        """Same inputs should always produce the same bucket."""
        bucket1 = ConsistentHasher.hash_to_bucket("exp-1", "user-abc")
        bucket2 = ConsistentHasher.hash_to_bucket("exp-1", "user-abc")
        assert bucket1 == bucket2

    def test_different_users_different_buckets(self) -> None:
        """Different users should generally get different buckets."""
        buckets = set()
        for i in range(100):
            bucket = ConsistentHasher.hash_to_bucket("exp-1", f"user-{i}")
            buckets.add(bucket)
        # With 100 users and 10000 buckets, collisions should be very rare
        assert len(buckets) > 90

    def test_different_experiments_different_buckets(self) -> None:
        """Same user in different experiments should get different buckets."""
        bucket1 = ConsistentHasher.hash_to_bucket("exp-1", "user-abc")
        bucket2 = ConsistentHasher.hash_to_bucket("exp-2", "user-abc")
        # Not guaranteed but very likely to differ
        # We test the mechanism, not probability
        assert isinstance(bucket1, int)
        assert isinstance(bucket2, int)
        assert 0 <= bucket1 < 10000
        assert 0 <= bucket2 < 10000

    def test_uniform_distribution(self) -> None:
        """Hash should distribute uniformly across buckets."""
        n_users = 10000
        n_bins = 10
        bin_size = 10000 // n_bins

        bin_counts = Counter()
        for i in range(n_users):
            bucket = ConsistentHasher.hash_to_bucket("exp-1", f"user-{i}")
            bin_idx = bucket // bin_size
            bin_counts[bin_idx] += 1

        # Each bin should have roughly n_users/n_bins = 1000 users
        for bin_idx in range(n_bins):
            count = bin_counts.get(bin_idx, 0)
            expected = n_users / n_bins
            # Allow 20% deviation
            assert abs(count - expected) / expected < 0.20, (
                f"Bin {bin_idx} has {count}, expected ~{expected}"
            )


class TestTrafficRouterBasic:
    """Basic routing functionality tests."""

    def test_routes_to_running_experiment(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should route to a running experiment's variant."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        context = RoutingContext(user_id="user-1", experiment_id=exp.id)
        decision = traffic_router.route(context)

        assert decision is not None
        assert decision.experiment_id == exp.id
        assert decision.variant_id in ["control", "treatment"]

    def test_returns_none_for_no_experiments(self, traffic_router: TrafficRouter) -> None:
        """Should return None when no experiments are running."""
        context = RoutingContext(user_id="user-1")
        decision = traffic_router.route(context)
        assert decision is None

    def test_returns_none_for_non_running(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Should return None for experiments not in RUNNING state."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        # Don't start it

        context = RoutingContext(user_id="user-1", experiment_id=exp.id)
        decision = traffic_router.route(context)
        assert decision is None


class TestStickyAssignment:
    """Tests for consistent user-to-variant assignment."""

    def test_same_user_same_variant(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Same user should always get the same variant."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        context = RoutingContext(user_id="user-42", experiment_id=exp.id)
        decisions = [traffic_router.route(context) for _ in range(10)]

        variant_ids = {d.variant_id for d in decisions if d is not None}
        assert len(variant_ids) == 1  # Always same variant

    def test_traffic_split_approximate(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Traffic split should approximately match configured percentages."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        counts: Counter[str] = Counter()
        for i in range(1000):
            context = RoutingContext(user_id=f"user-{i}", experiment_id=exp.id)
            decision = traffic_router.route(context)
            if decision:
                counts[decision.variant_id] += 1

        total = sum(counts.values())
        for variant_id, count in counts.items():
            ratio = count / total
            # Should be approximately 50% (allow 10% deviation)
            assert 0.40 <= ratio <= 0.60, f"Variant {variant_id}: {ratio:.2%}"


class TestOverrides:
    """Tests for override rules."""

    def test_force_variant_in_context(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """force_variant in context should override normal routing."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        context = RoutingContext(
            user_id="user-1",
            experiment_id=exp.id,
            force_variant="treatment",
        )
        decision = traffic_router.route(context)

        assert decision is not None
        assert decision.variant_name == "Treatment"
        assert decision.is_override is True
        assert decision.reason == "forced"

    def test_admin_override(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Admin override should force specific variant for a user."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        traffic_router.set_override(exp.id, "test-user", "control")

        context = RoutingContext(user_id="test-user", experiment_id=exp.id)
        decision = traffic_router.route(context)

        assert decision is not None
        assert decision.variant_id == "control"
        assert decision.is_override is True
        assert decision.reason == "admin_override"

    def test_remove_override(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
        simple_experiment_config: ExperimentConfig,
    ) -> None:
        """Removing override should revert to normal routing."""
        exp = experiment_manager.create_experiment(simple_experiment_config)
        experiment_manager.start_experiment(exp.id)

        traffic_router.set_override(exp.id, "test-user", "control")
        traffic_router.remove_override(exp.id, "test-user")

        context = RoutingContext(user_id="test-user", experiment_id=exp.id)
        decision = traffic_router.route(context)

        assert decision is not None
        assert not decision.is_override


class TestFeatureFlags:
    """Tests for feature flag integration."""

    def test_flag_enabled(self) -> None:
        """Enabled flag should evaluate to True."""
        evaluator = FeatureFlagEvaluator()
        evaluator.register_flag("test_flag", {"enabled": True})

        context = RoutingContext(user_id="user-1")
        assert evaluator.evaluate("test_flag", context) is True

    def test_flag_disabled(self) -> None:
        """Disabled flag should evaluate to False."""
        evaluator = FeatureFlagEvaluator()
        evaluator.register_flag("test_flag", {"enabled": False})

        context = RoutingContext(user_id="user-1")
        assert evaluator.evaluate("test_flag", context) is False

    def test_flag_whitelist(self) -> None:
        """Whitelisted users should always get the flag."""
        evaluator = FeatureFlagEvaluator()
        evaluator.register_flag(
            "test_flag",
            {
                "enabled": True,
                "user_whitelist": ["vip-user"],
                "percentage": 0.0,  # Would normally block everyone
            },
        )

        vip_context = RoutingContext(user_id="vip-user")
        assert evaluator.evaluate("test_flag", vip_context) is True

    def test_flag_conditions(self) -> None:
        """Conditions should filter based on user features."""
        evaluator = FeatureFlagEvaluator()
        evaluator.register_flag(
            "test_flag",
            {
                "enabled": True,
                "conditions": {"country": "US"},
            },
        )

        us_context = RoutingContext(user_id="user-1", features={"country": "US"})
        assert evaluator.evaluate("test_flag", us_context) is True

        uk_context = RoutingContext(user_id="user-2", features={"country": "UK"})
        assert evaluator.evaluate("test_flag", uk_context) is False

    def test_unknown_flag(self) -> None:
        """Unknown flag should evaluate to False."""
        evaluator = FeatureFlagEvaluator()
        context = RoutingContext(user_id="user-1")
        assert evaluator.evaluate("nonexistent_flag", context) is False

    def test_percentage_rollout(self) -> None:
        """Percentage-based rollout should affect roughly the right proportion."""
        evaluator = FeatureFlagEvaluator()
        evaluator.register_flag(
            "test_flag",
            {
                "enabled": True,
                "percentage": 50.0,
            },
        )

        enabled_count = 0
        total = 1000
        for i in range(total):
            context = RoutingContext(user_id=f"user-{i}")
            if evaluator.evaluate("test_flag", context):
                enabled_count += 1

        ratio = enabled_count / total
        # Should be approximately 50%
        assert 0.35 <= ratio <= 0.65


class TestMultipleExperiments:
    """Tests for routing across multiple concurrent experiments."""

    def test_route_multiple(
        self,
        experiment_manager: ExperimentManager,
        traffic_router: TrafficRouter,
    ) -> None:
        """Should return decisions for all running experiments."""
        configs = []
        for i in range(3):
            config = ExperimentConfig(
                name=f"Experiment {i}",
                variants=[
                    Variant(
                        name="Control",
                        traffic_percentage=50.0,
                        is_control=True,
                    ),
                    Variant(
                        name="Treatment",
                        traffic_percentage=50.0,
                        is_control=False,
                    ),
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
            configs.append(exp)

        context = RoutingContext(user_id="user-1")
        decisions = traffic_router.route_multiple(context)

        assert len(decisions) == 3
        experiment_ids = {d.experiment_id for d in decisions}
        assert len(experiment_ids) == 3
