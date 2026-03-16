"""XGBoost-based drug-drug interaction predictor.

Provides model training with cross-validation, hyperparameter tuning,
SHAP-based explanations, and feature importance analysis. Predicts
interaction severity and type (pharmacokinetic vs pharmacodynamic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class InteractionType(str, Enum):
    """Category of drug-drug interaction mechanism."""

    PHARMACOKINETIC = "pharmacokinetic"
    PHARMACODYNAMIC = "pharmacodynamic"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """Severity classification for predicted interactions."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CONTRAINDICATED = "contraindicated"


class InteractionPrediction(BaseModel):
    """Structured prediction for a single drug pair."""

    drug_a_id: str
    drug_b_id: str
    interaction_probability: float = Field(..., ge=0.0, le=1.0)
    severity: SeverityLevel
    interaction_type: InteractionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_contributing_features: list[dict[str, float]] = Field(default_factory=list)


class ModelMetrics(BaseModel):
    """Aggregated model evaluation metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: float | None = None
    cv_f1_mean: float | None = None
    cv_f1_std: float | None = None
    feature_importances: dict[str, float] = Field(default_factory=dict)


class HyperparameterConfig(BaseModel):
    """Hyperparameter search space and defaults."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int = 50


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@dataclass
class DrugInteractionPredictor:
    """XGBoost classifier for drug-drug interaction prediction.

    Supports multi-class severity prediction and binary interaction
    type classification.

    Parameters
    ----------
    hyperparams : HyperparameterConfig
        Model hyperparameters.
    n_cv_folds : int
        Number of folds for cross-validation.
    random_state : int
        Seed for reproducibility.
    """

    hyperparams: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    n_cv_folds: int = 5
    random_state: int = 42

    _severity_model: xgb.XGBClassifier | None = field(default=None, init=False, repr=False)
    _type_model: xgb.XGBClassifier | None = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    _severity_classes: list[str] = field(default_factory=list, init=False, repr=False)

    def _build_classifier(self, num_classes: int) -> xgb.XGBClassifier:
        """Create an XGBClassifier with the configured hyperparameters."""
        hp = self.hyperparams
        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
        params: dict[str, Any] = {
            "n_estimators": hp.n_estimators,
            "max_depth": hp.max_depth,
            "learning_rate": hp.learning_rate,
            "subsample": hp.subsample,
            "colsample_bytree": hp.colsample_bytree,
            "min_child_weight": hp.min_child_weight,
            "gamma": hp.gamma,
            "reg_alpha": hp.reg_alpha,
            "reg_lambda": hp.reg_lambda,
            "scale_pos_weight": hp.scale_pos_weight,
            "objective": objective,
            "eval_metric": hp.eval_metric,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if num_classes > 2:
            params["num_class"] = num_classes
        return xgb.XGBClassifier(**params)

    # -- Training -----------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y_severity: pd.Series,
        y_type: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series, pd.Series] | None = None,
    ) -> ModelMetrics:
        """Train severity and interaction-type models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y_severity : pd.Series
            Target labels for severity (multi-class).
        y_type : pd.Series
            Target labels for interaction type (binary/multi).
        eval_set : tuple, optional
            (X_val, y_severity_val, y_type_val) for early stopping.

        Returns
        -------
        ModelMetrics
            Training metrics for the severity model.
        """
        self._feature_names = list(X.columns)
        self._severity_classes = sorted(y_severity.unique().tolist())
        n_severity = len(self._severity_classes)
        n_types = len(y_type.unique())

        logger.info(
            "Training severity model: %d samples, %d features, %d classes",
            len(X),
            X.shape[1],
            n_severity,
        )

        # Build models
        self._severity_model = self._build_classifier(n_severity)
        self._type_model = self._build_classifier(n_types)

        # Fit severity model
        fit_params: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            X_val, y_sev_val, _ = eval_set
            fit_params["eval_set"] = [(X_val, y_sev_val)]
        self._severity_model.fit(X, y_severity, **fit_params)

        # Fit type model
        type_fit_params: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            X_val, _, y_type_val = eval_set
            type_fit_params["eval_set"] = [(X_val, y_type_val)]
        self._type_model.fit(X, y_type, **type_fit_params)

        # Compute metrics
        y_pred = self._severity_model.predict(X)
        y_proba = self._severity_model.predict_proba(X)

        roc_auc = None
        if n_severity > 2:
            try:
                roc_auc = roc_auc_score(y_severity, y_proba, multi_class="ovr", average="macro")
            except ValueError:
                logger.warning("Could not compute ROC AUC (likely single-class fold)")
        elif n_severity == 2:
            roc_auc = roc_auc_score(y_severity, y_proba[:, 1])

        importances = dict(
            zip(self._feature_names, self._severity_model.feature_importances_.tolist())
        )
        sorted_imp = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:20])

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_severity, y_pred),
            precision_macro=precision_score(y_severity, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_severity, y_pred, average="macro", zero_division=0),
            f1_macro=f1_score(y_severity, y_pred, average="macro", zero_division=0),
            roc_auc_ovr=roc_auc,
            feature_importances=sorted_imp,
        )
        logger.info(
            "Training metrics: accuracy=%.4f, f1_macro=%.4f", metrics.accuracy, metrics.f1_macro
        )
        return metrics

    # -- Cross-validation ---------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y_severity: pd.Series,
    ) -> ModelMetrics:
        """Run stratified k-fold cross-validation on the severity model.

        Returns
        -------
        ModelMetrics
            Includes ``cv_f1_mean`` and ``cv_f1_std``.
        """
        n_classes = len(y_severity.unique())
        model = self._build_classifier(n_classes)
        skf = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state
        )
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        }
        logger.info("Running %d-fold cross-validation", self.n_cv_folds)
        cv_results = cross_validate(model, X, y_severity, cv=skf, scoring=scoring, n_jobs=-1)

        metrics = ModelMetrics(
            accuracy=float(np.mean(cv_results["test_accuracy"])),
            precision_macro=float(np.mean(cv_results["test_precision_macro"])),
            recall_macro=float(np.mean(cv_results["test_recall_macro"])),
            f1_macro=float(np.mean(cv_results["test_f1_macro"])),
            cv_f1_mean=float(np.mean(cv_results["test_f1_macro"])),
            cv_f1_std=float(np.std(cv_results["test_f1_macro"])),
        )
        logger.info(
            "CV results: f1_macro=%.4f (+/- %.4f)",
            metrics.cv_f1_mean,
            metrics.cv_f1_std,
        )
        return metrics

    # -- Hyperparameter tuning ----------------------------------------------

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y_severity: pd.Series,
        param_grid: dict[str, list[Any]] | None = None,
        n_iter: int = 20,
    ) -> HyperparameterConfig:
        """Randomised hyperparameter search.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y_severity : pd.Series
            Target labels.
        param_grid : dict, optional
            Parameter distributions. If None, uses sensible defaults.
        n_iter : int
            Number of random combinations to try.

        Returns
        -------
        HyperparameterConfig
            Best hyperparameters found.
        """
        from sklearn.model_selection import RandomizedSearchCV

        if param_grid is None:
            param_grid = {
                "max_depth": [3, 4, 5, 6, 7, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
                "n_estimators": [100, 200, 300, 500, 700],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0.0, 0.05, 0.1, 0.2, 0.5],
                "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            }

        n_classes = len(y_severity.unique())
        base_model = self._build_classifier(n_classes)
        skf = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state
        )

        logger.info("Starting hyperparameter search with %d iterations", n_iter)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=skf,
            scoring="f1_macro",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y_severity)

        best = search.best_params_
        logger.info("Best hyperparameters: %s (score=%.4f)", best, search.best_score_)

        self.hyperparams = HyperparameterConfig(
            **{k: v for k, v in best.items() if k in HyperparameterConfig.model_fields}
        )
        return self.hyperparams

    # -- Prediction ---------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        drug_a_ids: list[str],
        drug_b_ids: list[str],
    ) -> list[InteractionPrediction]:
        """Generate interaction predictions for drug pairs.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same schema as training data).
        drug_a_ids, drug_b_ids : list[str]
            Identifiers for each drug in the pairs.

        Returns
        -------
        list[InteractionPrediction]
        """
        if self._severity_model is None or self._type_model is None:
            raise RuntimeError("Models not trained. Call train() first.")

        sev_proba = self._severity_model.predict_proba(X)
        sev_pred = self._severity_model.predict(X)
        type_pred = self._type_model.predict(X)
        type_proba = self._type_model.predict_proba(X)

        predictions: list[InteractionPrediction] = []
        for i in range(len(X)):
            max_prob = float(np.max(sev_proba[i]))
            severity = SeverityLevel(self._severity_classes[int(sev_pred[i])])
            interaction_type = InteractionType(type_pred[i])
            # Confidence: max probability minus entropy-based uncertainty
            entropy = -float(np.sum(sev_proba[i] * np.log(sev_proba[i] + 1e-10)))
            max_entropy = np.log(len(self._severity_classes))
            confidence = max_prob * (1.0 - entropy / max_entropy) if max_entropy > 0 else max_prob

            predictions.append(
                InteractionPrediction(
                    drug_a_id=drug_a_ids[i],
                    drug_b_id=drug_b_ids[i],
                    interaction_probability=max_prob,
                    severity=severity,
                    interaction_type=interaction_type,
                    confidence=round(confidence, 4),
                )
            )
        return predictions

    # -- SHAP explanations --------------------------------------------------

    def explain_predictions(
        self,
        X: pd.DataFrame,
        max_display: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate SHAP-based feature explanations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for the instances to explain.
        max_display : int
            Maximum number of top features to include per instance.

        Returns
        -------
        list[dict]
            One dict per instance with feature-level SHAP values.
        """
        import shap

        if self._severity_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info("Computing SHAP explanations for %d instances", len(X))
        explainer = shap.TreeExplainer(self._severity_model)
        shap_values = explainer.shap_values(X)

        explanations: list[dict[str, Any]] = []
        for i in range(len(X)):
            if isinstance(shap_values, list):
                # Multi-class: shap_values is a list of arrays, one per class
                instance_shap = np.abs(np.array([sv[i] for sv in shap_values])).mean(axis=0)
            else:
                instance_shap = np.abs(shap_values[i])

            feature_importance = sorted(
                zip(self._feature_names, instance_shap.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:max_display]

            explanations.append(
                {
                    "instance_index": i,
                    "top_features": [
                        {"feature": name, "shap_value": round(val, 6)}
                        for name, val in feature_importance
                    ],
                    "base_value": float(explainer.expected_value[0])
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else float(explainer.expected_value),
                }
            )

        return explanations

    # -- Feature importance -------------------------------------------------

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame.

        Parameters
        ----------
        importance_type : str
            One of ``"gain"``, ``"weight"``, ``"cover"``.
        """
        if self._severity_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        booster = self._severity_model.get_booster()
        raw_importance = booster.get_score(importance_type=importance_type)

        df = (
            pd.DataFrame([{"feature": k, "importance": v} for k, v in raw_importance.items()])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return df

    # -- Serialisation ------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save both models to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._severity_model is not None:
            self._severity_model.save_model(str(directory / "severity_model.json"))
        if self._type_model is not None:
            self._type_model.save_model(str(directory / "type_model.json"))
        logger.info("Models saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        """Load both models from disk."""
        directory = Path(directory)
        self._severity_model = xgb.XGBClassifier()
        self._severity_model.load_model(str(directory / "severity_model.json"))
        self._type_model = xgb.XGBClassifier()
        self._type_model.load_model(str(directory / "type_model.json"))
        logger.info("Models loaded from %s", directory)
