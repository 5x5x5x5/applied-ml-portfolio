"""Tests for Step Functions and SageMaker deployment modules."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from drug_interaction.deployment.sagemaker_deploy import (
    ABTestConfig,
    AutoScalingConfig,
    EndpointConfig,
    SageMakerDeployer,
)
from drug_interaction.deployment.step_functions import (
    StepFunctionsManager,
    build_training_workflow_definition,
)
from drug_interaction.monitoring.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftSeverity,
    compute_psi,
)

# ---------------------------------------------------------------------------
# Step Functions tests
# ---------------------------------------------------------------------------


class TestBuildTrainingWorkflowDefinition:
    """Tests for the Step Functions state machine definition builder."""

    @pytest.fixture
    def workflow_definition(self) -> dict:
        """Build a sample workflow definition."""
        return build_training_workflow_definition(
            extraction_lambda_arn="arn:aws:lambda:us-east-1:123:function:extract",
            feature_eng_lambda_arn="arn:aws:lambda:us-east-1:123:function:feature-eng",
            training_job_lambda_arn="arn:aws:lambda:us-east-1:123:function:train",
            evaluation_lambda_arn="arn:aws:lambda:us-east-1:123:function:evaluate",
            deployment_lambda_arn="arn:aws:lambda:us-east-1:123:function:deploy",
            monitoring_lambda_arn="arn:aws:lambda:us-east-1:123:function:monitor",
            rollback_lambda_arn="arn:aws:lambda:us-east-1:123:function:rollback",
            sns_topic_arn="arn:aws:sns:us-east-1:123:alerts",
            performance_threshold=0.80,
        )

    def test_definition_has_required_fields(self, workflow_definition: dict) -> None:
        """State machine definition has StartAt and States."""
        assert "StartAt" in workflow_definition
        assert "States" in workflow_definition
        assert workflow_definition["StartAt"] == "DataExtraction"

    def test_definition_has_all_states(self, workflow_definition: dict) -> None:
        """All expected states are present."""
        states = workflow_definition["States"]
        expected_states = [
            "DataExtraction",
            "FeatureEngineering",
            "ValidateFeatures",
            "ModelTraining",
            "ModelEvaluation",
            "DeploymentDecision",
            "DeployModel",
            "SetupMonitoring",
            "PipelineComplete",
            "PipelineFailed",
            "DeploymentRollback",
        ]
        for state in expected_states:
            assert state in states, f"Missing state: {state}"

    def test_parallel_extraction(self, workflow_definition: dict) -> None:
        """Data extraction uses parallel branches."""
        extraction = workflow_definition["States"]["DataExtraction"]
        assert extraction["Type"] == "Parallel"
        assert len(extraction["Branches"]) == 2

    def test_deployment_decision_is_choice(self, workflow_definition: dict) -> None:
        """Deployment decision is a Choice state."""
        decision = workflow_definition["States"]["DeploymentDecision"]
        assert decision["Type"] == "Choice"
        assert len(decision["Choices"]) >= 2

    def test_retry_logic_on_training(self, workflow_definition: dict) -> None:
        """Model training has retry configuration."""
        training = workflow_definition["States"]["ModelTraining"]
        assert "Retry" in training
        retry = training["Retry"][0]
        assert retry["MaxAttempts"] >= 1
        assert retry["BackoffRate"] >= 1.0

    def test_error_handling_states(self, workflow_definition: dict) -> None:
        """Error handling states exist with Catch clauses."""
        extraction = workflow_definition["States"]["DataExtraction"]
        assert "Catch" in extraction

        training = workflow_definition["States"]["ModelTraining"]
        assert "Catch" in training

        deploy = workflow_definition["States"]["DeployModel"]
        assert "Catch" in deploy

    def test_rollback_on_deployment_failure(self, workflow_definition: dict) -> None:
        """Deployment failure triggers rollback."""
        deploy = workflow_definition["States"]["DeployModel"]
        catch = deploy["Catch"][0]
        assert catch["Next"] == "DeploymentRollback"

    def test_terminal_states(self, workflow_definition: dict) -> None:
        """Pipeline has proper Succeed and Fail terminal states."""
        assert workflow_definition["States"]["PipelineComplete"]["Type"] == "Succeed"
        assert workflow_definition["States"]["PipelineFailed"]["Type"] == "Fail"

    def test_definition_is_valid_json(self, workflow_definition: dict) -> None:
        """Definition can be serialised to valid JSON."""
        json_str = json.dumps(workflow_definition)
        parsed = json.loads(json_str)
        assert parsed == workflow_definition


class TestStepFunctionsManager:
    """Tests for StepFunctionsManager."""

    @patch("boto3.client")
    def test_create_state_machine(self, mock_boto_client: MagicMock) -> None:
        """State machine creation calls the correct API."""
        mock_sfn = MagicMock()
        mock_sfn.get_paginator.return_value.paginate.return_value = [{"stateMachines": []}]
        mock_sfn.create_state_machine.return_value = {
            "stateMachineArn": "arn:aws:states:us-east-1:123:stateMachine:test"
        }
        mock_boto_client.return_value = mock_sfn

        manager = StepFunctionsManager(region="us-east-1", role_arn="arn:aws:iam::role/test")
        arn = manager.create_state_machine(
            name="test-sm",
            definition={"StartAt": "Init", "States": {"Init": {"Type": "Pass", "End": True}}},
        )

        assert arn == "arn:aws:states:us-east-1:123:stateMachine:test"
        mock_sfn.create_state_machine.assert_called_once()

    @patch("boto3.client")
    def test_start_execution(self, mock_boto_client: MagicMock) -> None:
        """Execution start returns an execution ARN."""
        mock_sfn = MagicMock()
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:test:run1"
        }
        mock_boto_client.return_value = mock_sfn

        manager = StepFunctionsManager(region="us-east-1", role_arn="arn:aws:iam::role/test")
        exec_arn = manager.start_execution(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:test",
            execution_name="run1",
            input_data={"key": "value"},
        )

        assert "execution" in exec_arn


# ---------------------------------------------------------------------------
# SageMaker deployment tests
# ---------------------------------------------------------------------------


class TestSageMakerDeployer:
    """Tests for SageMakerDeployer."""

    @patch("boto3.client")
    def test_create_model(self, mock_boto_client: MagicMock) -> None:
        """Model creation returns an ARN."""
        mock_sm = MagicMock()
        mock_sm.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-east-1:123:model/test-model"
        }
        mock_boto_client.return_value = mock_sm

        deployer = SageMakerDeployer(
            region="us-east-1",
            execution_role_arn="arn:aws:iam::role/sagemaker",
        )
        arn = deployer.create_model(
            model_name="test-model",
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
            model_data_url="s3://bucket/model/model.tar.gz",
        )

        assert "model/test-model" in arn

    @patch("boto3.client")
    def test_create_endpoint(self, mock_boto_client: MagicMock) -> None:
        """Endpoint creation succeeds with data capture."""
        mock_sm = MagicMock()
        mock_sm.create_endpoint_config.return_value = {}
        mock_sm.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-east-1:123:endpoint/test"
        }
        mock_boto_client.return_value = mock_sm

        deployer = SageMakerDeployer(region="us-east-1", execution_role_arn="role")
        config = EndpointConfig(
            endpoint_name="test-endpoint",
            model_name="test-model",
            data_capture_s3_uri="s3://bucket/capture",
        )
        result = deployer.create_endpoint(config)
        assert result  # non-empty

    @patch("boto3.client")
    def test_create_ab_test_endpoint(self, mock_boto_client: MagicMock) -> None:
        """A/B test endpoint creates two variants."""
        mock_sm = MagicMock()
        mock_sm.create_endpoint_config.return_value = {}
        mock_sm.create_endpoint.return_value = {}
        mock_boto_client.return_value = mock_sm

        deployer = SageMakerDeployer(region="us-east-1", execution_role_arn="role")
        config = ABTestConfig(
            endpoint_name="ab-test",
            variant_a_model="model-v1",
            variant_b_model="model-v2",
            variant_a_weight=0.9,
            variant_b_weight=0.1,
        )
        result = deployer.create_ab_test_endpoint(config)

        assert result == "ab-test"
        # Verify two variants in the endpoint config call
        call_kwargs = mock_sm.create_endpoint_config.call_args
        variants = call_kwargs.kwargs.get("ProductionVariants") or call_kwargs[1].get(
            "ProductionVariants"
        )
        assert len(variants) == 2

    @patch("boto3.client")
    def test_configure_autoscaling(self, mock_boto_client: MagicMock) -> None:
        """Auto-scaling configuration calls the correct APIs."""
        mock_autoscaling = MagicMock()
        mock_boto_client.return_value = mock_autoscaling

        deployer = SageMakerDeployer(region="us-east-1", execution_role_arn="role")
        config = AutoScalingConfig(
            endpoint_name="test-endpoint",
            min_capacity=1,
            max_capacity=4,
            target_invocations_per_instance=1000,
        )
        deployer.configure_autoscaling(config)

        mock_autoscaling.register_scalable_target.assert_called_once()
        mock_autoscaling.put_scaling_policy.assert_called_once()


# ---------------------------------------------------------------------------
# Drift detection tests
# ---------------------------------------------------------------------------


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_no_drift_detected(
        self,
        baseline_features: MagicMock,
        current_features_no_drift: MagicMock,
    ) -> None:
        """Similar distributions show minimal drift."""
        detector = DriftDetector(psi_threshold=0.2)
        results = detector.detect_feature_drift(baseline_features, current_features_no_drift)

        drifted = [r for r in results if r.is_drifted]
        assert len(drifted) == 0 or all(r.severity == DriftSeverity.NONE for r in drifted)

    def test_drift_detected(
        self,
        baseline_features: MagicMock,
        current_features_with_drift: MagicMock,
    ) -> None:
        """Shifted distributions are detected as drifted."""
        detector = DriftDetector(psi_threshold=0.2)
        results = detector.detect_feature_drift(baseline_features, current_features_with_drift)

        drifted_names = {r.feature_name for r in results if r.is_drifted}
        # feature_a and feature_b have significant shifts
        assert "feature_a" in drifted_names or "feature_b" in drifted_names

    def test_psi_computation(self) -> None:
        """PSI is 0 for identical distributions and positive for different ones."""
        import numpy as np

        rng = np.random.RandomState(42)
        same = rng.normal(0, 1, 1000)
        shifted = rng.normal(3, 2, 1000)

        psi_same = compute_psi(same, same)
        psi_diff = compute_psi(same, shifted)

        assert psi_same < 0.05  # nearly zero for same distribution
        assert psi_diff > 0.2  # significant for shifted distribution

    def test_generate_full_report(
        self,
        baseline_features: MagicMock,
        current_features_with_drift: MagicMock,
    ) -> None:
        """Full drift report includes all sections."""
        import numpy as np

        detector = DriftDetector(psi_threshold=0.2)

        baseline_pred = np.random.RandomState(42).uniform(0, 1, 500).astype(np.float64)
        current_pred = np.random.RandomState(99).uniform(0.2, 0.8, 500).astype(np.float64)
        baseline_labels = np.random.RandomState(42).choice([0, 1, 2], 500).astype(np.int64)
        current_labels = np.random.RandomState(99).choice([0, 1, 2], 500).astype(np.int64)

        report = detector.generate_drift_report(
            baseline_features=baseline_features,
            current_features=current_features_with_drift,
            baseline_predictions=baseline_pred,
            current_predictions=current_pred,
            baseline_labels=baseline_labels,
            current_labels=current_labels,
        )

        assert isinstance(report, DriftReport)
        assert report.total_features_analyzed > 0
        assert report.prediction_drift is not None
        assert report.label_drift is not None
        assert report.summary  # non-empty string

    def test_prediction_drift_detection(self) -> None:
        """Prediction drift is detected for shifted probabilities."""
        import numpy as np

        detector = DriftDetector(psi_threshold=0.1)
        baseline = np.random.RandomState(42).uniform(0, 1, 1000).astype(np.float64)
        shifted = np.random.RandomState(99).uniform(0.5, 1, 1000).astype(np.float64)

        result = detector.detect_prediction_drift(baseline, shifted)
        assert result.psi > 0
        assert result.prediction_shift != 0.0

    def test_label_drift_detection(self) -> None:
        """Label drift is detected when class distribution changes."""
        import numpy as np

        detector = DriftDetector(ks_alpha=0.05)
        baseline_labels = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int64)
        shifted_labels = np.array([0] * 50 + [1] * 50 + [2] * 200, dtype=np.int64)

        result = detector.detect_label_drift(baseline_labels, shifted_labels)
        assert result.is_drifted
        assert result.chi2_statistic > 0


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestEndpointConfig:
    """Tests for EndpointConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = EndpointConfig(endpoint_name="test", model_name="model")
        assert config.instance_type == "ml.m5.xlarge"
        assert config.initial_instance_count == 1
        assert config.data_capture_percentage == 100


class TestABTestConfig:
    """Tests for ABTestConfig."""

    def test_weight_validation(self) -> None:
        """Variant weights must be between 0 and 1."""
        config = ABTestConfig(
            endpoint_name="test",
            variant_a_model="m1",
            variant_b_model="m2",
            variant_a_weight=0.9,
            variant_b_weight=0.1,
        )
        assert config.variant_a_weight + config.variant_b_weight == pytest.approx(1.0)

    def test_invalid_weight_raises(self) -> None:
        """Weight > 1 is rejected."""
        with pytest.raises(Exception):
            ABTestConfig(
                endpoint_name="test",
                variant_a_model="m1",
                variant_b_model="m2",
                variant_a_weight=1.5,
                variant_b_weight=0.1,
            )
