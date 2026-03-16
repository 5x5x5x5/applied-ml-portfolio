"""Airflow DAG for drug interaction model training pipeline.

Weekly schedule that orchestrates:
1. Extract molecular features from SMILES database
2. Extract patient features from Snowflake
3. Feature engineering and validation
4. Train model with cross-validation
5. Evaluate against production model
6. Register in MLflow
7. Deploy via Step Functions if approved
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.amazon.aws.operators.step_function import (
    StepFunctionStartExecutionOperator,
)
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "drug-interaction-ml",
    "depends_on_past": False,
    "email": ["ml-team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=4),
}

SNOWFLAKE_CONFIG = {
    "account": Variable.get("snowflake_account", default_var=""),
    "user": Variable.get("snowflake_user", default_var=""),
    "password": Variable.get("snowflake_password", default_var=""),
    "warehouse": Variable.get("snowflake_warehouse", default_var="DRUG_INTERACTION_WH"),
    "database": Variable.get("snowflake_database", default_var="PHARMA_DB"),
    "schema": Variable.get("snowflake_schema", default_var="DRUG_INTERACTION"),
}

MLFLOW_CONFIG = {
    "tracking_uri": Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000"),
    "experiment_name": "drug-interaction-prediction",
    "registered_model_name": "drug-interaction-xgboost",
}

S3_BUCKET = Variable.get("ml_artifacts_bucket", default_var="drug-interaction-ml-artifacts")
STEP_FUNCTIONS_ARN = Variable.get("training_step_functions_arn", default_var="")


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def extract_molecular_features(**context: Any) -> dict[str, Any]:
    """Extract molecular features from the drug SMILES database."""
    import pandas as pd

    from drug_interaction.features.molecular_features import MolecularFeatureExtractor

    extractor = MolecularFeatureExtractor(fingerprint_radius=2, fingerprint_nbits=1024)

    # Load drug pairs from S3 or a database table
    execution_date = context["ds"]
    drug_pairs_path = f"s3://{S3_BUCKET}/input/drug_pairs/{execution_date}/pairs.csv"

    logger.info("Loading drug pairs from %s", drug_pairs_path)
    # In production, this would read from S3; here we show the logic
    drug_pairs_df = pd.read_csv(drug_pairs_path)
    pairs = list(zip(drug_pairs_df["smiles_a"], drug_pairs_df["smiles_b"]))

    pairwise_df = extractor.extract_pairwise_batch(pairs, include_fingerprint=True)

    output_path = f"s3://{S3_BUCKET}/features/molecular/{execution_date}/features.parquet"
    pairwise_df.to_parquet(output_path, index=False)
    logger.info("Saved %d molecular feature rows to %s", len(pairwise_df), output_path)

    return {
        "features_path": output_path,
        "n_pairs": len(pairwise_df),
        "n_features": pairwise_df.shape[1],
    }


def extract_snowflake_features(**context: Any) -> dict[str, Any]:
    """Extract patient-level features from Snowflake."""
    from datetime import date

    from drug_interaction.features.snowflake_features import (
        SnowflakeConfig,
        SnowflakeFeatureExtractor,
    )

    execution_date = context["ds"]
    end_date = date.fromisoformat(execution_date)
    start_date = end_date - timedelta(days=365)

    config = SnowflakeConfig(**SNOWFLAKE_CONFIG)
    extractor = SnowflakeFeatureExtractor(config=config)

    all_features = extractor.extract_all_features(start_date, end_date)

    output_paths: dict[str, str] = {}
    total_rows = 0
    for name, df in all_features.items():
        path = f"s3://{S3_BUCKET}/features/snowflake/{execution_date}/{name}.parquet"
        df.to_parquet(path, index=False)
        output_paths[name] = path
        total_rows += len(df)
        logger.info("Saved %s: %d rows to %s", name, len(df), path)

    return {
        "feature_paths": output_paths,
        "total_rows": total_rows,
    }


def engineer_and_validate_features(**context: Any) -> dict[str, Any]:
    """Combine features, engineer new ones, and validate quality."""
    import numpy as np
    import pandas as pd

    ti = context["ti"]
    molecular_result = ti.xcom_pull(task_ids="extract_molecular_features")
    snowflake_result = ti.xcom_pull(task_ids="extract_snowflake_features")

    # Load feature DataFrames
    molecular_df = pd.read_parquet(molecular_result["features_path"])

    sf_paths = snowflake_result["feature_paths"]
    coprescription_df = pd.read_parquet(sf_paths["co_prescription"])
    demographics_df = pd.read_parquet(sf_paths["demographics"])
    adverse_df = pd.read_parquet(sf_paths["adverse_events"])

    # Merge features on drug pair identifiers
    combined_df = molecular_df.copy()
    if "drug_a_ndc" in coprescription_df.columns:
        combined_df = combined_df.merge(
            coprescription_df,
            left_on=["drug_a_smiles", "drug_b_smiles"],
            right_on=["drug_a_ndc", "drug_b_ndc"],
            how="left",
        )

    # Fill missing values
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)

    # Validation checks
    validation = {
        "sample_count": len(combined_df),
        "feature_count": combined_df.shape[1],
        "null_pct": float(combined_df.isnull().sum().sum() / combined_df.size),
        "has_duplicates": bool(combined_df.duplicated().any()),
        "validation_passed": len(combined_df) >= 100 and combined_df.shape[1] >= 10,
    }

    output_path = f"s3://{S3_BUCKET}/features/combined/{context['ds']}/features.parquet"
    combined_df.to_parquet(output_path, index=False)

    logger.info("Feature engineering complete: %s", validation)
    return {**validation, "features_path": output_path}


def train_model_with_cv(**context: Any) -> dict[str, Any]:
    """Train XGBoost model with cross-validation."""
    import pandas as pd

    from drug_interaction.models.interaction_predictor import (
        DrugInteractionPredictor,
        HyperparameterConfig,
    )

    ti = context["ti"]
    feature_result = ti.xcom_pull(task_ids="engineer_and_validate_features")
    df = pd.read_parquet(feature_result["features_path"])

    # Separate features and targets
    target_cols = ["severity_label", "interaction_type_label"]
    feature_cols = [
        c for c in df.columns if c not in target_cols + ["drug_a_smiles", "drug_b_smiles"]
    ]

    X = df[feature_cols]
    y_severity = df["severity_label"]
    y_type = df["interaction_type_label"]

    # Train-validation split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_sev_train, y_sev_val, y_type_train, y_type_val = train_test_split(
        X, y_severity, y_type, test_size=0.2, random_state=42, stratify=y_severity
    )

    predictor = DrugInteractionPredictor(
        hyperparams=HyperparameterConfig(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
        ),
        n_cv_folds=5,
    )

    # Cross-validation
    cv_metrics = predictor.cross_validate(X_train, y_sev_train)
    logger.info("CV metrics: f1=%.4f (+/- %.4f)", cv_metrics.cv_f1_mean, cv_metrics.cv_f1_std)

    # Full training
    train_metrics = predictor.train(
        X_train,
        y_sev_train,
        y_type_train,
        eval_set=(X_val, y_sev_val, y_type_val),
    )

    # Save model
    model_path = f"/tmp/drug_interaction_model/{context['ds']}"
    predictor.save(model_path)

    return {
        "model_path": model_path,
        "train_metrics": train_metrics.model_dump(),
        "cv_metrics": cv_metrics.model_dump(),
        "feature_names": feature_cols,
    }


def evaluate_against_production(**context: Any) -> dict[str, Any]:
    """Compare candidate model against production model."""
    from drug_interaction.models.model_registry import DrugInteractionModelRegistry

    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model_with_cv")

    registry = DrugInteractionModelRegistry(**MLFLOW_CONFIG)

    # Start MLflow run and log metrics
    run_id = registry.start_run(
        run_name=f"training-{context['ds']}",
        tags={"dag_run": context["run_id"], "execution_date": context["ds"]},
    )
    registry.log_params({"execution_date": context["ds"]})

    train_metrics = train_result["train_metrics"]
    registry.log_metrics(
        {
            "accuracy": train_metrics["accuracy"],
            "precision_macro": train_metrics["precision_macro"],
            "recall_macro": train_metrics["recall_macro"],
            "f1_macro": train_metrics["f1_macro"],
        }
    )

    if train_metrics.get("roc_auc_ovr") is not None:
        registry.log_metrics({"roc_auc_ovr": train_metrics["roc_auc_ovr"]})

    cv_metrics = train_result["cv_metrics"]
    if cv_metrics.get("cv_f1_mean") is not None:
        registry.log_metrics(
            {
                "cv_f1_mean": cv_metrics["cv_f1_mean"],
                "cv_f1_std": cv_metrics["cv_f1_std"],
            }
        )

    # Log feature importance
    if train_metrics.get("feature_importances"):
        registry.log_feature_importance(train_metrics["feature_importances"])

    # Compare with production
    comparison = registry.compare_with_production(run_id, primary_metric="f1_macro")
    registry.end_run()

    return {
        "run_id": run_id,
        "is_improvement": comparison.is_improvement,
        "recommendation": comparison.recommendation,
        "candidate_f1": train_metrics["f1_macro"],
        "production_f1": comparison.production_metrics.get("f1_macro"),
    }


def decide_deployment(**context: Any) -> str:
    """Branch: deploy if the model is an improvement."""
    ti = context["ti"]
    eval_result = ti.xcom_pull(task_ids="evaluate_against_production")

    if eval_result["is_improvement"]:
        logger.info("Model is an improvement. Proceeding to registration and deployment.")
        return "register_model"
    logger.info("Model is not an improvement. Skipping deployment.")
    return "skip_deployment"


def register_model_in_mlflow(**context: Any) -> dict[str, Any]:
    """Register the model in MLflow and promote to staging."""
    from drug_interaction.models.model_registry import DrugInteractionModelRegistry

    ti = context["ti"]
    eval_result = ti.xcom_pull(task_ids="evaluate_against_production")
    train_result = ti.xcom_pull(task_ids="train_model_with_cv")

    registry = DrugInteractionModelRegistry(**MLFLOW_CONFIG)

    model_uri = f"runs:/{eval_result['run_id']}/model"
    model_info = registry.register_model(
        model_uri,
        description=f"Training run {context['ds']}. F1={eval_result['candidate_f1']:.4f}",
    )
    registry.promote_to_staging(model_info.version)

    return {
        "model_version": model_info.version,
        "model_name": model_info.name,
        "stage": "Staging",
    }


def skip_deployment_task(**context: Any) -> None:
    """No-op task when deployment is skipped."""
    logger.info("Deployment skipped for run %s", context["ds"])


def notify_completion(**context: Any) -> None:
    """Send completion notification."""
    ti = context["ti"]
    eval_result = ti.xcom_pull(task_ids="evaluate_against_production")
    logger.info("Pipeline complete. Result: %s", eval_result.get("recommendation", "N/A"))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="drug_interaction_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Weekly drug interaction model training and deployment pipeline",
    schedule="0 2 * * 0",  # Every Sunday at 02:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "drug-interaction", "training"],
) as dag:
    # -- Extraction (parallel) --
    extract_molecular = PythonOperator(
        task_id="extract_molecular_features",
        python_callable=extract_molecular_features,
    )

    extract_snowflake = PythonOperator(
        task_id="extract_snowflake_features",
        python_callable=extract_snowflake_features,
    )

    # -- Feature engineering --
    feature_eng = PythonOperator(
        task_id="engineer_and_validate_features",
        python_callable=engineer_and_validate_features,
    )

    # -- Training --
    train = PythonOperator(
        task_id="train_model_with_cv",
        python_callable=train_model_with_cv,
        execution_timeout=timedelta(hours=2),
    )

    # -- Evaluation --
    evaluate = PythonOperator(
        task_id="evaluate_against_production",
        python_callable=evaluate_against_production,
    )

    # -- Deployment decision --
    deploy_decision = BranchPythonOperator(
        task_id="deployment_decision",
        python_callable=decide_deployment,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model_in_mlflow,
    )

    deploy_via_sfn = StepFunctionStartExecutionOperator(
        task_id="deploy_via_step_functions",
        state_machine_arn=STEP_FUNCTIONS_ARN,
        name="airflow-deploy-{{ ds_nodash }}",
        input=json.dumps(
            {
                "trigger": "airflow_training_pipeline",
                "execution_date": "{{ ds }}",
            }
        ),
        aws_conn_id="aws_default",
    )

    skip_deploy = PythonOperator(
        task_id="skip_deployment",
        python_callable=skip_deployment_task,
    )

    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # -- Task dependencies --
    [extract_molecular, extract_snowflake] >> feature_eng >> train >> evaluate >> deploy_decision
    deploy_decision >> register >> deploy_via_sfn >> notify
    deploy_decision >> skip_deploy >> notify
