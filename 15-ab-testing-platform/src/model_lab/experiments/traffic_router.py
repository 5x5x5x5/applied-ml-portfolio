"""Request routing with consistent hashing, sticky sessions, and override support.

Routes incoming prediction requests to the appropriate model variant based on
experiment configuration, user identity, and feature flags.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

import structlog

from model_lab.experiments.experiment_manager import (
    Experiment,
    ExperimentManager,
    ExperimentState,
    Variant,
)

logger = structlog.get_logger(__name__)


@dataclass
class RoutingContext:
    """Context for making a routing decision."""

    user_id: str
    experiment_id: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    force_variant: str | None = None
    session_id: str | None = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    experiment_id: str
    variant_id: str
    variant_name: str
    model_version_id: str | None
    is_override: bool = False
    reason: str = ""


class ConsistentHasher:
    """Consistent hashing with virtual nodes for stable user-to-variant assignment.

    Uses MD5 hash to produce a deterministic bucket for each (experiment, user) pair.
    This ensures users consistently see the same variant across requests.
    """

    @staticmethod
    def hash_to_bucket(experiment_id: str, user_id: str, num_buckets: int = 10000) -> int:
        """Map (experiment_id, user_id) to a deterministic bucket in [0, num_buckets).

        Uses MD5 for uniform distribution and determinism (not security).
        """
        key = f"{experiment_id}:{user_id}".encode()
        digest = hashlib.md5(key, usedforsecurity=False).digest()
        # Use first 4 bytes as unsigned int
        value = struct.unpack("<I", digest[:4])[0]
        return value % num_buckets

    @staticmethod
    def assign_variant(
        experiment: Experiment,
        user_id: str,
        num_buckets: int = 10000,
    ) -> Variant:
        """Assign a variant to a user using consistent hashing.

        Maps the user to a bucket, then maps that bucket to a variant
        based on traffic percentages. The rollout percentage scales down
        all variant allocations proportionally.

        Args:
            experiment: The experiment with variant configurations.
            user_id: The user to assign.
            num_buckets: Resolution of the hash ring.

        Returns:
            The assigned variant.
        """
        bucket = ConsistentHasher.hash_to_bucket(experiment.id, user_id, num_buckets)
        bucket_pct = (bucket / num_buckets) * 100.0

        # Apply rollout scaling
        rollout_scale = experiment.current_rollout_percentage / 100.0

        cumulative = 0.0
        for variant in experiment.config.variants:
            cumulative += variant.traffic_percentage * rollout_scale
            if bucket_pct < cumulative:
                return variant

        # If bucket falls outside the rollout window, assign to control
        return next(
            (v for v in experiment.config.variants if v.is_control),
            experiment.config.variants[0],
        )


class FeatureFlagEvaluator:
    """Evaluates feature flags for experiment targeting and gating."""

    def __init__(self) -> None:
        self._flags: dict[str, dict[str, Any]] = {}

    def register_flag(self, flag_name: str, config: dict[str, Any]) -> None:
        """Register a feature flag configuration.

        Config can include:
            - enabled: bool
            - user_whitelist: list[str]
            - percentage: float (0-100, rollout percentage)
            - conditions: dict of attribute -> value requirements
        """
        self._flags[flag_name] = config

    def evaluate(self, flag_name: str, context: RoutingContext) -> bool:
        """Evaluate whether a feature flag is active for the given context.

        Args:
            flag_name: Name of the feature flag.
            context: Routing context with user/feature information.

        Returns:
            True if the flag is active, False otherwise.
        """
        config = self._flags.get(flag_name)
        if config is None:
            return False

        if not config.get("enabled", True):
            return False

        # Check user whitelist
        whitelist = config.get("user_whitelist", [])
        if whitelist and context.user_id in whitelist:
            return True

        # Check conditions against user features
        conditions = config.get("conditions", {})
        for attr, expected in conditions.items():
            actual = context.features.get(attr)
            if actual != expected:
                return False

        # Percentage-based rollout
        percentage = config.get("percentage", 100.0)
        if percentage < 100.0:
            bucket = ConsistentHasher.hash_to_bucket(flag_name, context.user_id, 10000)
            if (bucket / 100.0) >= percentage:
                return False

        return True


class TrafficRouter:
    """Routes requests to appropriate model variants.

    Provides consistent user assignment, sticky sessions, override rules,
    feature flag integration, and support for multiple concurrent experiments.
    """

    def __init__(self, experiment_manager: ExperimentManager) -> None:
        self._manager = experiment_manager
        self._hasher = ConsistentHasher()
        self._feature_flags = FeatureFlagEvaluator()
        self._overrides: dict[str, dict[str, str]] = {}  # experiment_id -> {user_id -> variant_id}
        self._session_cache: dict[str, RoutingDecision] = {}  # session_key -> decision

    @property
    def feature_flags(self) -> FeatureFlagEvaluator:
        """Access the feature flag evaluator."""
        return self._feature_flags

    def set_override(self, experiment_id: str, user_id: str, variant_id: str) -> None:
        """Set a routing override for internal testing.

        Forces a specific user to always see a specific variant,
        bypassing normal traffic allocation.

        Args:
            experiment_id: The experiment to override.
            user_id: The user to override.
            variant_id: The variant to force.
        """
        if experiment_id not in self._overrides:
            self._overrides[experiment_id] = {}
        self._overrides[experiment_id][user_id] = variant_id
        logger.info(
            "override_set",
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
        )

    def remove_override(self, experiment_id: str, user_id: str) -> None:
        """Remove a routing override."""
        if experiment_id in self._overrides:
            self._overrides[experiment_id].pop(user_id, None)

    def clear_session_cache(self) -> None:
        """Clear all cached session assignments."""
        self._session_cache.clear()

    def route(self, context: RoutingContext) -> RoutingDecision | None:
        """Route a request to the appropriate model variant.

        Routing priority:
        1. Forced variant override (from context.force_variant)
        2. Admin override rules
        3. Sticky session cache
        4. Consistent hashing assignment

        Args:
            context: The routing context with user and experiment info.

        Returns:
            A RoutingDecision, or None if no experiment applies.
        """
        if context.experiment_id:
            return self._route_for_experiment(context, context.experiment_id)

        # Try all running experiments
        running = self._manager.list_experiments(state=ExperimentState.RUNNING)
        if not running:
            return None

        # Route to the first applicable experiment
        for experiment in running:
            decision = self._route_for_experiment(context, experiment.id)
            if decision:
                return decision

        return None

    def route_multiple(self, context: RoutingContext) -> list[RoutingDecision]:
        """Route a request across all applicable running experiments.

        Returns decisions for every running experiment the user qualifies for.
        """
        decisions = []
        running = self._manager.list_experiments(state=ExperimentState.RUNNING)
        for experiment in running:
            decision = self._route_for_experiment(context, experiment.id)
            if decision:
                decisions.append(decision)
        return decisions

    def _route_for_experiment(
        self,
        context: RoutingContext,
        experiment_id: str,
    ) -> RoutingDecision | None:
        """Perform routing for a specific experiment."""
        try:
            experiment = self._manager.get_experiment(experiment_id)
        except KeyError:
            logger.warning("experiment_not_found", experiment_id=experiment_id)
            return None

        if experiment.state != ExperimentState.RUNNING:
            return None

        # 1. Check forced variant from context
        if context.force_variant:
            variant = self._find_variant(experiment, context.force_variant)
            if variant:
                return self._make_decision(
                    experiment_id, variant, is_override=True, reason="forced"
                )

        # 2. Check admin overrides
        override_variant_id = self._overrides.get(experiment_id, {}).get(context.user_id)
        if override_variant_id:
            variant = self._find_variant(experiment, override_variant_id)
            if variant:
                return self._make_decision(
                    experiment_id, variant, is_override=True, reason="admin_override"
                )

        # 3. Check sticky session cache
        session_key = self._session_key(experiment_id, context)
        if session_key in self._session_cache:
            cached = self._session_cache[session_key]
            logger.debug("session_cache_hit", session_key=session_key)
            return cached

        # 4. Consistent hashing assignment
        variant = self._hasher.assign_variant(experiment, context.user_id)

        decision = self._make_decision(
            experiment_id, variant, is_override=False, reason="consistent_hash"
        )

        # Cache for sticky sessions
        if session_key:
            self._session_cache[session_key] = decision

        return decision

    def _find_variant(self, experiment: Experiment, variant_id: str) -> Variant | None:
        """Find a variant by ID or name in an experiment."""
        for v in experiment.config.variants:
            if v.id == variant_id or v.name == variant_id:
                return v
        return None

    def _make_decision(
        self,
        experiment_id: str,
        variant: Variant,
        is_override: bool,
        reason: str,
    ) -> RoutingDecision:
        return RoutingDecision(
            experiment_id=experiment_id,
            variant_id=variant.id,
            variant_name=variant.name,
            model_version_id=variant.model_version_id,
            is_override=is_override,
            reason=reason,
        )

    def _session_key(self, experiment_id: str, context: RoutingContext) -> str:
        """Generate a session cache key."""
        session_part = context.session_id or context.user_id
        return f"{experiment_id}:{session_part}"
