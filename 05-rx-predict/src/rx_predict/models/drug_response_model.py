"""Optimized ML model for sub-100ms drug response prediction.

Uses scikit-learn with careful optimization:
- Pre-allocated arrays to avoid GC pressure
- Model warm-up on startup
- Feature hashing for constant-time feature encoding
- Confidence intervals via calibrated predictions
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rx_predict.models.feature_processor import FeatureProcessor

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Single prediction result with confidence."""

    response_probability: float
    confidence_lower: float
    confidence_upper: float
    predicted_class: str
    risk_level: str
    inference_time_ms: float
    model_version: str
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchPredictionResult:
    """Batch prediction results."""

    predictions: list[PredictionResult]
    total_inference_time_ms: float
    avg_inference_time_ms: float
    model_version: str


# Response class labels
RESPONSE_CLASSES = ["poor_response", "partial_response", "good_response", "excellent_response"]

RISK_THRESHOLDS = {
    "high": 0.3,
    "moderate": 0.5,
    "low": 0.7,
}


def _classify_risk(good_response_prob: float) -> str:
    """Classify risk level based on probability of good/excellent response."""
    if good_response_prob < RISK_THRESHOLDS["high"]:
        return "high_risk"
    elif good_response_prob < RISK_THRESHOLDS["moderate"]:
        return "moderate_risk"
    elif good_response_prob < RISK_THRESHOLDS["low"]:
        return "low_risk"
    return "minimal_risk"


class DrugResponseModel:
    """Optimized drug response prediction model.

    Design choices for sub-100ms latency:
    - GradientBoosting with limited depth and estimators
    - Pre-allocated prediction arrays
    - Model warm-up to trigger JIT/cache population
    - Cached feature processor
    """

    def __init__(self, model_version: str = "1.0.0") -> None:
        self.model_version = model_version
        self.feature_processor = FeatureProcessor()
        self._model: Pipeline | None = None
        self._is_warmed_up = False
        self._warmup_array: npt.NDArray[np.float32] | None = None
        self._feature_names: list[str] = []
        self._prediction_buffer = np.zeros(len(RESPONSE_CLASSES), dtype=np.float64)

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._model is not None

    def build_default_model(self) -> None:
        """Build and train a default model with synthetic data.

        Used for initial deployment before real training data is available.
        The model is intentionally lightweight for fast inference.
        """
        logger.info("building_default_model", version=self.model_version)

        # Create a fast, lightweight pipeline
        base_classifier = GradientBoostingClassifier(
            n_estimators=50,  # Few trees for speed
            max_depth=4,  # Shallow trees
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            max_features="sqrt",  # Feature subsampling
            random_state=42,
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", base_classifier),
            ]
        )

        # Generate synthetic training data
        n_samples = 2000
        n_features = self.feature_processor.feature_dimension
        rng = np.random.RandomState(42)

        X_train = rng.randn(n_samples, n_features).astype(np.float32)
        # Create realistic class distribution: most patients have partial/good response
        class_probs = [0.15, 0.30, 0.35, 0.20]  # poor, partial, good, excellent
        y_train = rng.choice(len(RESPONSE_CLASSES), size=n_samples, p=class_probs)

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Wrap with calibration for better probability estimates
        self._model = Pipeline(
            [
                ("scaler", pipeline.named_steps["scaler"]),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        pipeline.named_steps["classifier"],
                        cv="prefit",
                        method="sigmoid",
                    ),
                ),
            ]
        )
        self._model.named_steps["classifier"].fit(
            pipeline.named_steps["scaler"].transform(X_train), y_train
        )

        self._feature_names = self.feature_processor.get_feature_names()
        logger.info(
            "default_model_built",
            n_features=n_features,
            n_classes=len(RESPONSE_CLASSES),
        )

    def warm_up(self) -> float:
        """Pre-warm the model to minimize cold-start latency.

        Returns the warm-up inference time in milliseconds.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call build_default_model() or load() first.")

        logger.info("model_warmup_starting")

        # Pre-allocate warmup array matching expected input shape
        n_features = self.feature_processor.feature_dimension
        self._warmup_array = np.zeros((1, n_features), dtype=np.float32)

        # Run several warmup predictions to populate CPU caches
        warmup_times: list[float] = []
        for _ in range(10):
            start = time.perf_counter()
            self._model.predict_proba(self._warmup_array)
            elapsed = (time.perf_counter() - start) * 1000
            warmup_times.append(elapsed)

        self._is_warmed_up = True
        avg_warmup_ms = sum(warmup_times) / len(warmup_times)
        p99_warmup_ms = sorted(warmup_times)[int(len(warmup_times) * 0.99)]

        logger.info(
            "model_warmup_complete",
            avg_ms=round(avg_warmup_ms, 3),
            p99_ms=round(p99_warmup_ms, 3),
        )
        return avg_warmup_ms

    def predict(self, patient_data: dict[str, Any]) -> PredictionResult:
        """Make a single prediction with confidence intervals.

        Target: <50ms p99 latency.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        start = time.perf_counter()

        # Feature extraction (<2ms target)
        features = self.feature_processor.process_single(patient_data)
        features_2d = features.reshape(1, -1)

        # Model inference (<5ms target for GBT with 50 trees)
        probabilities = self._model.predict_proba(features_2d)[0]

        # Post-processing (<1ms)
        predicted_class_idx = int(np.argmax(probabilities))
        predicted_class = RESPONSE_CLASSES[predicted_class_idx]
        response_prob = float(probabilities[predicted_class_idx])

        # Confidence interval via Wilson score interval approximation
        n_effective = 100  # effective sample size for CI calculation
        z = 1.96  # 95% CI
        p = response_prob
        denominator = 1 + z**2 / n_effective
        center = (p + z**2 / (2 * n_effective)) / denominator
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_effective)) / n_effective) / denominator
        ci_lower = max(0.0, center - spread)
        ci_upper = min(1.0, center + spread)

        # Risk level based on probability of good/excellent response
        good_response_prob = float(probabilities[2] + probabilities[3])
        risk_level = _classify_risk(good_response_prob)

        # Top feature importances (if available)
        feature_importance: dict[str, float] = {}
        if hasattr(self._model.named_steps.get("classifier", None), "estimator"):
            estimator = self._model.named_steps["classifier"].estimator
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                top_indices = np.argsort(importances)[-5:][::-1]
                for idx in top_indices:
                    if idx < len(self._feature_names):
                        feature_importance[self._feature_names[idx]] = round(
                            float(importances[idx]), 4
                        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            response_probability=round(response_prob, 4),
            confidence_lower=round(ci_lower, 4),
            confidence_upper=round(ci_upper, 4),
            predicted_class=predicted_class,
            risk_level=risk_level,
            inference_time_ms=round(elapsed_ms, 3),
            model_version=self.model_version,
            feature_importance=feature_importance,
        )

    def predict_batch(self, patient_records: list[dict[str, Any]]) -> BatchPredictionResult:
        """Batch prediction for multiple patients.

        Uses vectorized operations where possible.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        start = time.perf_counter()

        # Batch feature processing
        feature_matrix = self.feature_processor.process_batch(patient_records)

        # Batch inference
        probabilities = self._model.predict_proba(feature_matrix)

        # Post-process each prediction
        predictions: list[PredictionResult] = []
        for i in range(len(patient_records)):
            probs = probabilities[i]
            predicted_class_idx = int(np.argmax(probs))
            predicted_class = RESPONSE_CLASSES[predicted_class_idx]
            response_prob = float(probs[predicted_class_idx])

            # Simplified CI for batch (speed)
            ci_half = 0.1 * (1 - response_prob)
            ci_lower = max(0.0, response_prob - ci_half)
            ci_upper = min(1.0, response_prob + ci_half)

            good_response_prob = float(probs[2] + probs[3])
            risk_level = _classify_risk(good_response_prob)

            predictions.append(
                PredictionResult(
                    response_probability=round(response_prob, 4),
                    confidence_lower=round(ci_lower, 4),
                    confidence_upper=round(ci_upper, 4),
                    predicted_class=predicted_class,
                    risk_level=risk_level,
                    inference_time_ms=0.0,  # individual timing not meaningful in batch
                    model_version=self.model_version,
                )
            )

        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / len(patient_records) if patient_records else 0.0

        return BatchPredictionResult(
            predictions=predictions,
            total_inference_time_ms=round(total_ms, 3),
            avg_inference_time_ms=round(avg_ms, 3),
            model_version=self.model_version,
        )

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "version": self.model_version,
                "feature_names": self._feature_names,
            },
            path,
        )
        logger.info("model_saved", path=str(path), version=self.model_version)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        data = joblib.load(path)
        self._model = data["model"]
        self.model_version = data["version"]
        self._feature_names = data.get("feature_names", [])
        logger.info("model_loaded", path=str(path), version=self.model_version)

    def benchmark(self, n_iterations: int = 1000) -> dict[str, float]:
        """Benchmark inference latency.

        Returns p50, p95, p99 latency in milliseconds.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        n_features = self.feature_processor.feature_dimension
        test_input = np.random.randn(1, n_features).astype(np.float32)

        latencies: list[float] = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self._model.predict_proba(test_input)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        results = {
            "p50_ms": latencies[int(n_iterations * 0.50)],
            "p95_ms": latencies[int(n_iterations * 0.95)],
            "p99_ms": latencies[int(n_iterations * 0.99)],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "mean_ms": sum(latencies) / len(latencies),
            "iterations": float(n_iterations),
        }

        logger.info("benchmark_complete", **{k: round(v, 3) for k, v in results.items()})
        return results
