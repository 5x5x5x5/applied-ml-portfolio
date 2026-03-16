"""Adverse Event Classifier using scikit-learn pipelines.

Classifies FDA adverse event reports into severity categories using a
TF-IDF + Logistic Regression pipeline with NLP preprocessing.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Adverse event severity classification levels."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


SEVERITY_LABELS: list[str] = [s.value for s in SeverityLevel]

# Keywords associated with severity levels for feature enrichment
SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "critical": [
        "death",
        "fatal",
        "life-threatening",
        "cardiac arrest",
        "anaphylaxis",
        "anaphylactic",
        "respiratory failure",
        "organ failure",
        "coma",
        "sepsis",
        "hemorrhage",
        "stroke",
        "myocardial infarction",
        "pulmonary embolism",
    ],
    "severe": [
        "hospitalization",
        "hospitalized",
        "disability",
        "incapacity",
        "surgery",
        "transplant",
        "seizure",
        "convulsion",
        "liver failure",
        "renal failure",
        "pancreatitis",
        "stevens-johnson",
        "toxic epidermal",
        "rhabdomyolysis",
        "agranulocytosis",
        "thrombocytopenia",
    ],
    "moderate": [
        "vomiting",
        "diarrhea",
        "fever",
        "infection",
        "rash",
        "edema",
        "hypertension",
        "hypotension",
        "tachycardia",
        "bradycardia",
        "pneumonia",
        "bleeding",
        "fracture",
        "dehydration",
        "syncope",
    ],
    "mild": [
        "nausea",
        "headache",
        "dizziness",
        "fatigue",
        "insomnia",
        "constipation",
        "dry mouth",
        "itching",
        "pruritus",
        "mild rash",
        "drowsiness",
        "appetite loss",
        "weight gain",
        "muscle ache",
    ],
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom sklearn transformer for clinical text preprocessing.

    Cleans and normalizes adverse event report text, extracts severity
    indicator features, and prepares text for TF-IDF vectorization.
    """

    def __init__(self, lowercase: bool = True, remove_numbers: bool = False) -> None:
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers

    def fit(self, X: list[str], y: Any = None) -> TextPreprocessor:
        """Fit is a no-op; preprocessing is stateless.

        Args:
            X: Input text documents.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: list[str], y: Any = None) -> list[str]:
        """Transform input texts with clinical NLP preprocessing.

        Args:
            X: Raw adverse event report texts.
            y: Ignored.

        Returns:
            Preprocessed text documents.
        """
        return [self._preprocess_single(text) for text in X]

    def _preprocess_single(self, text: str) -> str:
        """Preprocess a single adverse event report text.

        Args:
            text: Raw report text.

        Returns:
            Cleaned and normalized text with severity indicator tokens.
        """
        if not text or not text.strip():
            return ""

        processed = text.strip()

        if self.lowercase:
            processed = processed.lower()

        # Normalize whitespace and remove control characters
        processed = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", processed)
        processed = re.sub(r"\s+", " ", processed)

        # Remove PHI-like patterns (dates, IDs) for safety
        processed = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", " PHI_SSN ", processed)
        processed = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", " PHI_DATE ", processed)

        if self.remove_numbers:
            processed = re.sub(r"\b\d+\.?\d*\b", " NUM ", processed)

        # Append severity indicator tokens based on keyword presence
        severity_tokens = self._extract_severity_tokens(processed)
        if severity_tokens:
            processed = f"{processed} {severity_tokens}"

        return processed.strip()

    @staticmethod
    def _extract_severity_tokens(text: str) -> str:
        """Extract severity indicator tokens from text.

        Scans for known clinical severity keywords and appends
        structured indicator tokens for the classifier.

        Args:
            text: Preprocessed text.

        Returns:
            Space-separated severity indicator tokens.
        """
        tokens: list[str] = []
        text_lower = text.lower()

        for severity, keywords in SEVERITY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                # Add weighted tokens based on keyword match count
                indicator = f"SEV_{severity.upper()}"
                tokens.extend([indicator] * min(count, 3))

        return " ".join(tokens)


class AdverseEventClassifier:
    """ML classifier for FDA adverse event report severity.

    Uses a scikit-learn pipeline with TF-IDF vectorization and Logistic
    Regression to classify adverse event reports into severity levels:
    mild, moderate, severe, or critical.

    Attributes:
        pipeline: The fitted sklearn Pipeline, or None if not trained.
        is_fitted: Whether the model has been trained.
        metadata: Model metadata including training metrics.
    """

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: tuple[int, int] = (1, 3),
        c_param: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the classifier with hyperparameters.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: Range of n-grams for TF-IDF.
            c_param: Regularization strength for Logistic Regression.
            max_iter: Maximum iterations for solver convergence.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.c_param = c_param
        self.max_iter = max_iter

        self.pipeline: Pipeline | None = None
        self.is_fitted: bool = False
        self.metadata: dict[str, Any] = {}

        self._build_pipeline()
        logger.info(
            "AdverseEventClassifier initialized: max_features=%d, ngram_range=%s, C=%.2f",
            max_features,
            ngram_range,
            c_param,
        )

    def _build_pipeline(self) -> None:
        """Construct the sklearn classification pipeline."""
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", TextPreprocessor(lowercase=True, remove_numbers=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"(?u)\b\w\w+\b",
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c_param,
                        max_iter=self.max_iter,
                        class_weight="balanced",
                        solver="lbfgs",
                        multi_class="multinomial",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def train(
        self,
        texts: list[str],
        labels: list[str],
        validate: bool = True,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Train the classifier on labeled adverse event reports.

        Args:
            texts: List of adverse event report texts.
            labels: List of severity labels corresponding to texts.
            validate: Whether to perform cross-validation during training.
            cv_folds: Number of cross-validation folds.

        Returns:
            Dictionary containing training metrics (accuracy, f1, cv_scores).

        Raises:
            ValueError: If texts and labels have mismatched lengths or
                        if labels contain invalid severity levels.
        """
        if len(texts) != len(labels):
            raise ValueError(f"Mismatched lengths: {len(texts)} texts vs {len(labels)} labels")

        if not texts:
            raise ValueError("Cannot train on empty dataset")

        # Validate labels
        invalid_labels = set(labels) - set(SEVERITY_LABELS)
        if invalid_labels:
            raise ValueError(
                f"Invalid severity labels: {invalid_labels}. Valid labels: {SEVERITY_LABELS}"
            )

        logger.info(
            "Training classifier on %d samples with label distribution: %s",
            len(texts),
            {label: labels.count(label) for label in set(labels)},
        )

        if self.pipeline is None:
            self._build_pipeline()
        assert self.pipeline is not None

        # Cross-validation
        metrics: dict[str, Any] = {}
        if validate and len(texts) >= cv_folds * 2:
            logger.info("Running %d-fold stratified cross-validation", cv_folds)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.pipeline, texts, labels, cv=skf, scoring="f1_weighted", n_jobs=-1
            )
            metrics["cv_f1_mean"] = float(np.mean(cv_scores))
            metrics["cv_f1_std"] = float(np.std(cv_scores))
            metrics["cv_scores"] = [float(s) for s in cv_scores]
            logger.info(
                "Cross-validation F1: %.4f (+/- %.4f)",
                metrics["cv_f1_mean"],
                metrics["cv_f1_std"],
            )

        # Train on full dataset
        self.pipeline.fit(texts, labels)
        self.is_fitted = True

        # Compute training metrics
        train_predictions = self.pipeline.predict(texts)
        metrics["train_f1_weighted"] = float(
            f1_score(labels, train_predictions, average="weighted")
        )
        metrics["classification_report"] = classification_report(
            labels, train_predictions, output_dict=True
        )
        metrics["n_samples"] = len(texts)
        metrics["n_features"] = self.pipeline.named_steps["tfidf"].vocabulary_.__len__()

        self.metadata = {
            "training_metrics": metrics,
            "hyperparameters": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "c_param": self.c_param,
                "max_iter": self.max_iter,
            },
        }

        logger.info(
            "Training complete: train_f1=%.4f, n_features=%d",
            metrics["train_f1_weighted"],
            metrics["n_features"],
        )

        return metrics

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify adverse event reports by severity.

        Args:
            texts: List of adverse event report texts to classify.

        Returns:
            List of prediction dictionaries, each containing:
                - severity: Predicted severity label
                - confidence: Prediction confidence score
                - probabilities: Dict of class probabilities

        Raises:
            RuntimeError: If the model has not been trained or loaded.
            ValueError: If texts list is empty.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() or load_model() first.")

        if not texts:
            raise ValueError("Cannot predict on empty input")

        logger.debug("Predicting severity for %d texts", len(texts))

        predictions = self.pipeline.predict(texts)
        probabilities: NDArray[np.floating[Any]] = self.pipeline.predict_proba(texts)
        class_labels: list[str] = list(self.pipeline.classes_)

        results: list[dict[str, Any]] = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = float(np.max(probs))
            prob_dict = {label: float(prob) for label, prob in zip(class_labels, probs)}

            result = {
                "severity": str(pred),
                "confidence": confidence,
                "probabilities": prob_dict,
            }

            if confidence < 0.5:
                logger.warning(
                    "Low confidence prediction (%.3f) for text %d: %s",
                    confidence,
                    i,
                    pred,
                )

            results.append(result)

        return results

    def predict_single(self, text: str) -> dict[str, Any]:
        """Classify a single adverse event report.

        Args:
            text: A single adverse event report text.

        Returns:
            Prediction dictionary with severity, confidence, and probabilities.
        """
        results = self.predict([text])
        return results[0]

    def save_model(self, path: str | Path) -> None:
        """Save the trained model and metadata to disk.

        Args:
            path: File path to save the model artifact.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "pipeline": self.pipeline,
            "metadata": self.metadata,
            "version": "1.0.0",
            "hyperparameters": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "c_param": self.c_param,
                "max_iter": self.max_iter,
            },
        }

        joblib.dump(artifact, path, compress=3)
        logger.info("Model saved to %s (%.2f MB)", path, path.stat().st_size / 1e6)

    def load_model(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to the saved model artifact.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the artifact is corrupted or incompatible.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        logger.info("Loading model from %s", path)

        artifact = joblib.load(path)

        if not isinstance(artifact, dict) or "pipeline" not in artifact:
            raise ValueError(
                f"Invalid model artifact format at {path}. Expected dict with 'pipeline' key."
            )

        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})
        self.is_fitted = True

        # Restore hyperparameters if available
        hp = artifact.get("hyperparameters", {})
        self.max_features = hp.get("max_features", self.max_features)
        self.ngram_range = hp.get("ngram_range", self.ngram_range)
        self.c_param = hp.get("c_param", self.c_param)
        self.max_iter = hp.get("max_iter", self.max_iter)

        logger.info(
            "Model loaded successfully. Metadata: %s",
            {k: v for k, v in self.metadata.items() if k != "training_metrics"},
        )

    def get_feature_importance(self, top_n: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Get top features by importance for each severity class.

        Args:
            top_n: Number of top features to return per class.

        Returns:
            Dictionary mapping class labels to lists of (feature, weight) dicts.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be fitted to extract feature importance")

        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]

        feature_names = tfidf.get_feature_names_out()
        importance: dict[str, list[dict[str, Any]]] = {}

        for i, class_label in enumerate(classifier.classes_):
            coefficients = classifier.coef_[i]
            top_indices = np.argsort(coefficients)[-top_n:][::-1]

            importance[str(class_label)] = [
                {
                    "feature": str(feature_names[idx]),
                    "weight": float(coefficients[idx]),
                }
                for idx in top_indices
            ]

        return importance
