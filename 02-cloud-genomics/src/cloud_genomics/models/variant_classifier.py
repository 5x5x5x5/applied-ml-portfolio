"""ML model for genomic variant classification using Random Forest.

Classifies genetic variants into ACMG/AMP categories:
benign, likely_benign, VUS, likely_pathogenic, pathogenic.

Features include conservation scores, allele frequencies, functional
impact predictions, and protein domain information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class VariantClass(str, Enum):
    """ACMG/AMP variant classification categories."""

    BENIGN = "benign"
    LIKELY_BENIGN = "likely_benign"
    VUS = "VUS"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    PATHOGENIC = "pathogenic"


@dataclass
class VariantFeatures:
    """Input features for variant classification."""

    # Conservation scores
    phylop_score: float = 0.0  # PhyloP conservation score (-14 to 6.4)
    phastcons_score: float = 0.0  # PhastCons probability (0 to 1)
    gerp_score: float = 0.0  # GERP++ rejected substitution score

    # Allele frequency data
    gnomad_af: float = 0.0  # gnomAD global allele frequency
    gnomad_af_afr: float = 0.0  # gnomAD African population AF
    gnomad_af_eas: float = 0.0  # gnomAD East Asian population AF
    gnomad_af_nfe: float = 0.0  # gnomAD Non-Finnish European AF
    gnomad_homozygote_count: int = 0  # Number of homozygotes in gnomAD

    # Functional impact predictions
    sift_score: float = 1.0  # SIFT score (0=damaging, 1=tolerated)
    polyphen2_score: float = 0.0  # PolyPhen-2 score (0=benign, 1=damaging)
    cadd_phred: float = 0.0  # CADD Phred-scaled score
    revel_score: float = 0.0  # REVEL ensemble score (0 to 1)
    mutation_taster_score: float = 0.0  # MutationTaster score

    # Protein domain information
    in_protein_domain: bool = False  # Variant falls within known protein domain
    domain_conservation: float = 0.0  # Conservation of the protein domain
    distance_to_active_site: float = -1.0  # Distance to nearest active site (-1 = unknown)
    pfam_domain_count: int = 0  # Number of overlapping Pfam domains

    # Variant characteristics
    variant_type: str = "SNV"  # SNV, insertion, deletion, MNV
    consequence: str = "missense"  # missense, nonsense, frameshift, synonymous, etc.
    exon_number: int = 0  # Exon number (0 = intron/intergenic)
    total_exons: int = 0  # Total exons in transcript
    amino_acid_change_blosum62: float = 0.0  # BLOSUM62 score for AA substitution
    grantham_distance: float = 0.0  # Grantham distance for AA change

    # Splice predictions
    splice_ai_score: float = 0.0  # SpliceAI delta score
    max_splice_distance: int = 0  # Distance to nearest splice site

    def to_array(self) -> NDArray[np.float64]:
        """Convert features to numpy array for model input."""
        return np.array(self._to_numeric_list(), dtype=np.float64)

    def _to_numeric_list(self) -> list[float]:
        """Convert all features to numeric values."""
        consequence_map = {
            "missense": 0,
            "nonsense": 1,
            "frameshift": 2,
            "synonymous": 3,
            "splice_site": 4,
            "splice_region": 5,
            "intron": 6,
            "utr_5": 7,
            "utr_3": 8,
            "intergenic": 9,
            "inframe_insertion": 10,
            "inframe_deletion": 11,
            "start_lost": 12,
            "stop_lost": 13,
        }
        variant_type_map = {"SNV": 0, "insertion": 1, "deletion": 2, "MNV": 3}

        return [
            self.phylop_score,
            self.phastcons_score,
            self.gerp_score,
            self.gnomad_af,
            self.gnomad_af_afr,
            self.gnomad_af_eas,
            self.gnomad_af_nfe,
            float(self.gnomad_homozygote_count),
            self.sift_score,
            self.polyphen2_score,
            self.cadd_phred,
            self.revel_score,
            self.mutation_taster_score,
            float(self.in_protein_domain),
            self.domain_conservation,
            self.distance_to_active_site,
            float(self.pfam_domain_count),
            float(variant_type_map.get(self.variant_type, 0)),
            float(consequence_map.get(self.consequence, 0)),
            float(self.exon_number),
            float(self.total_exons),
            self.amino_acid_change_blosum62,
            self.grantham_distance,
            self.splice_ai_score,
            float(self.max_splice_distance),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return ordered list of feature names matching to_array() output."""
        return [
            "phylop_score",
            "phastcons_score",
            "gerp_score",
            "gnomad_af",
            "gnomad_af_afr",
            "gnomad_af_eas",
            "gnomad_af_nfe",
            "gnomad_homozygote_count",
            "sift_score",
            "polyphen2_score",
            "cadd_phred",
            "revel_score",
            "mutation_taster_score",
            "in_protein_domain",
            "domain_conservation",
            "distance_to_active_site",
            "pfam_domain_count",
            "variant_type",
            "consequence",
            "exon_number",
            "total_exons",
            "amino_acid_change_blosum62",
            "grantham_distance",
            "splice_ai_score",
            "max_splice_distance",
        ]


@dataclass
class PredictionResult:
    """Result of a variant classification prediction."""

    variant_class: VariantClass
    confidence: float
    class_probabilities: dict[str, float]
    feature_importances: dict[str, float]
    explanation: list[str]


@dataclass
class ModelMetrics:
    """Training metrics for the classifier."""

    accuracy: float
    cross_val_mean: float
    cross_val_std: float
    classification_report: str
    confusion_matrix: NDArray[np.int64]
    feature_importances: dict[str, float]


class VariantClassifier:
    """Random Forest classifier for genomic variant classification.

    Uses an ensemble of decision trees trained on annotated variant data
    with features spanning conservation, population frequency, functional
    impact, and protein domain information.
    """

    FEATURE_NAMES: list[str] = VariantFeatures.feature_names()
    N_FEATURES: int = len(FEATURE_NAMES)

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = 30,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        self._model: RandomForestClassifier | None = None
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._is_trained: bool = False
        self._model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        self._training_metrics: ModelMetrics | None = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_metrics(self) -> ModelMetrics | None:
        return self._training_metrics

    def _engineer_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply feature engineering transformations.

        Creates interaction features and non-linear transforms to improve
        model discriminative power.
        """
        df = pd.DataFrame(X, columns=self.FEATURE_NAMES)

        # Log-transform allele frequencies (add small epsilon to avoid log(0))
        epsilon = 1e-10
        for af_col in ["gnomad_af", "gnomad_af_afr", "gnomad_af_eas", "gnomad_af_nfe"]:
            df[f"{af_col}_log"] = np.log10(df[af_col] + epsilon)

        # Combined pathogenicity score: weighted ensemble of predictors
        df["combined_pathogenicity"] = (
            (1.0 - df["sift_score"]) * 0.15
            + df["polyphen2_score"] * 0.15
            + (df["cadd_phred"] / 40.0).clip(0, 1) * 0.25
            + df["revel_score"] * 0.30
            + df["mutation_taster_score"] * 0.15
        )

        # Conservation composite score
        df["conservation_composite"] = (
            (df["phylop_score"] / 6.4).clip(-1, 1) * 0.35
            + df["phastcons_score"] * 0.35
            + (df["gerp_score"] / 6.0).clip(-1, 1) * 0.30
        )

        # Interaction: high conservation + damaging prediction = more likely pathogenic
        df["conservation_x_pathogenicity"] = (
            df["conservation_composite"] * df["combined_pathogenicity"]
        )

        # Rarity score: inverse of allele frequency (rarer = potentially more pathogenic)
        df["rarity_score"] = 1.0 / (1.0 + df["gnomad_af"] * 1e6)

        # Exon position (relative position within gene)
        df["exon_position_ratio"] = np.where(
            df["total_exons"] > 0, df["exon_number"] / df["total_exons"], 0.0
        )

        # Domain impact score
        df["domain_impact"] = (
            df["in_protein_domain"].astype(float)
            * df["domain_conservation"]
            * (1.0 / (1.0 + np.abs(df["distance_to_active_site"])))
        )

        # Splice impact: high SpliceAI near splice site
        df["splice_impact"] = df["splice_ai_score"] * (
            1.0 / (1.0 + df["max_splice_distance"] / 10.0)
        )

        return df.values.astype(np.float64)

    def train(
        self,
        features: list[VariantFeatures],
        labels: list[VariantClass],
        calibrate: bool = True,
    ) -> ModelMetrics:
        """Train the variant classifier on labeled data.

        Args:
            features: List of VariantFeatures for each training sample.
            labels: Corresponding ACMG classification labels.
            calibrate: Whether to apply probability calibration.

        Returns:
            ModelMetrics with training performance statistics.

        Raises:
            ValueError: If features and labels have different lengths or
                        if fewer than 10 samples are provided.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"Features ({len(features)}) and labels ({len(labels)}) must have same length"
            )
        if len(features) < 10:
            raise ValueError(f"Need at least 10 training samples, got {len(features)}")

        logger.info("Starting model training with %d samples", len(features))

        # Convert to arrays
        X_raw = np.array([f.to_array() for f in features])
        y_str = np.array([label.value for label in labels])

        # Encode labels
        self._label_encoder.fit([vc.value for vc in VariantClass])
        y = self._label_encoder.transform(y_str)

        # Feature engineering
        X_engineered = self._engineer_features(X_raw)

        # Scale features
        X_scaled = self._scaler.fit_transform(X_engineered)

        # Train Random Forest
        self._model = RandomForestClassifier(**self._model_params)
        self._model.fit(X_scaled, y)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self._model, X_scaled, y, cv=cv, scoring="accuracy")

        # Probability calibration for better-calibrated confidence scores
        if calibrate and len(features) >= 20:
            cal_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            self._calibrated_model = CalibratedClassifierCV(
                self._model, cv=cal_cv, method="isotonic"
            )
            self._calibrated_model.fit(X_scaled, y)
            logger.info("Applied isotonic probability calibration")

        # Compute feature importances
        raw_importances = self._model.feature_importances_
        engineered_feature_names = self.FEATURE_NAMES + [
            "gnomad_af_log",
            "gnomad_af_afr_log",
            "gnomad_af_eas_log",
            "gnomad_af_nfe_log",
            "combined_pathogenicity",
            "conservation_composite",
            "conservation_x_pathogenicity",
            "rarity_score",
            "exon_position_ratio",
            "domain_impact",
            "splice_impact",
        ]
        # Pad names if mismatch (engineered features may vary)
        while len(engineered_feature_names) < len(raw_importances):
            engineered_feature_names.append(f"engineered_{len(engineered_feature_names)}")
        importances_dict = {
            name: float(imp)
            for name, imp in zip(
                engineered_feature_names[: len(raw_importances)],
                raw_importances,
                strict=False,
            )
        }

        # Generate classification report
        y_pred = self._model.predict(X_scaled)
        class_names = list(self._label_encoder.classes_)
        report = classification_report(y, y_pred, target_names=class_names, zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)

        self._is_trained = True

        self._training_metrics = ModelMetrics(
            accuracy=float(self._model.score(X_scaled, y)),
            cross_val_mean=float(cv_scores.mean()),
            cross_val_std=float(cv_scores.std()),
            classification_report=report,
            confusion_matrix=conf_matrix,
            feature_importances=dict(
                sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
            ),
        )

        logger.info(
            "Model trained: accuracy=%.4f, cv_mean=%.4f (+/- %.4f)",
            self._training_metrics.accuracy,
            self._training_metrics.cross_val_mean,
            self._training_metrics.cross_val_std,
        )

        return self._training_metrics

    def predict(self, features: VariantFeatures) -> PredictionResult:
        """Classify a single variant.

        Args:
            features: VariantFeatures for the variant to classify.

        Returns:
            PredictionResult with classification, confidence, and explanation.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")

        X_raw = features.to_array().reshape(1, -1)
        X_engineered = self._engineer_features(X_raw)
        X_scaled = self._scaler.transform(X_engineered)

        # Use calibrated model if available, otherwise raw model
        predictor = self._calibrated_model if self._calibrated_model is not None else self._model
        probabilities = predictor.predict_proba(X_scaled)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = self._label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])

        # Build class probability map
        class_probs = {
            str(cls): float(prob)
            for cls, prob in zip(self._label_encoder.classes_, probabilities, strict=False)
        }

        # Feature importances for this prediction
        feat_importances = self._get_prediction_feature_importances(X_scaled[0])

        # Generate human-readable explanation
        explanation = self.explain_prediction(features, class_probs, feat_importances)

        return PredictionResult(
            variant_class=VariantClass(predicted_label),
            confidence=confidence,
            class_probabilities=class_probs,
            feature_importances=feat_importances,
            explanation=explanation,
        )

    def predict_batch(self, features_list: list[VariantFeatures]) -> list[PredictionResult]:
        """Classify multiple variants in batch.

        Args:
            features_list: List of VariantFeatures for variants to classify.

        Returns:
            List of PredictionResult objects.
        """
        return [self.predict(f) for f in features_list]

    def _get_prediction_feature_importances(
        self, x_scaled: NDArray[np.float64]
    ) -> dict[str, float]:
        """Compute per-prediction feature importances using tree path analysis."""
        if self._model is None:
            return {}

        global_importances = self._model.feature_importances_
        engineered_names = self.FEATURE_NAMES + [
            "gnomad_af_log",
            "gnomad_af_afr_log",
            "gnomad_af_eas_log",
            "gnomad_af_nfe_log",
            "combined_pathogenicity",
            "conservation_composite",
            "conservation_x_pathogenicity",
            "rarity_score",
            "exon_position_ratio",
            "domain_impact",
            "splice_impact",
        ]
        # Weight global importance by feature deviation from mean
        deviations = np.abs(x_scaled - 0.0)  # Scaled data centered at ~0
        weighted = global_importances[: len(deviations)] * deviations[: len(global_importances)]
        total = float(weighted.sum()) or 1.0

        result: dict[str, float] = {}
        for i, val in enumerate(weighted):
            name = engineered_names[i] if i < len(engineered_names) else f"feature_{i}"
            result[name] = float(val) / total

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def explain_prediction(
        self,
        features: VariantFeatures,
        class_probs: dict[str, float],
        feat_importances: dict[str, float],
    ) -> list[str]:
        """Generate human-readable explanation for a prediction.

        Produces a list of plain-English statements explaining why the
        model made its classification decision, suitable for clinical
        review.
        """
        explanations: list[str] = []

        # Top predicted class
        top_class = max(class_probs, key=class_probs.get)  # type: ignore[arg-type]
        top_prob = class_probs[top_class]
        explanations.append(f"Predicted class: {top_class} (confidence: {top_prob:.1%})")

        # Allele frequency interpretation
        if features.gnomad_af > 0.01:
            explanations.append(
                f"Common variant in population (gnomAD AF={features.gnomad_af:.4f}). "
                "Allele frequency >1% strongly supports benign classification (BS1/BA1)."
            )
        elif features.gnomad_af > 0.001:
            explanations.append(
                f"Low-frequency variant (gnomAD AF={features.gnomad_af:.4f}). "
                "Present in population but uncommon."
            )
        elif features.gnomad_af < 1e-5 and features.gnomad_af > 0:
            explanations.append(
                f"Extremely rare variant (gnomAD AF={features.gnomad_af:.2e}). "
                "Rarity is consistent with pathogenic classification (PM2)."
            )
        elif features.gnomad_af == 0:
            explanations.append(
                "Absent from gnomAD population database. "
                "Novel variant; rarity supports pathogenic hypothesis (PM2)."
            )

        # In-silico predictor concordance
        damaging_predictors: list[str] = []
        benign_predictors: list[str] = []

        if features.sift_score < 0.05:
            damaging_predictors.append(f"SIFT={features.sift_score:.3f}")
        else:
            benign_predictors.append(f"SIFT={features.sift_score:.3f}")

        if features.polyphen2_score > 0.85:
            damaging_predictors.append(f"PolyPhen-2={features.polyphen2_score:.3f}")
        elif features.polyphen2_score < 0.15:
            benign_predictors.append(f"PolyPhen-2={features.polyphen2_score:.3f}")

        if features.cadd_phred > 25:
            damaging_predictors.append(f"CADD={features.cadd_phred:.1f}")
        elif features.cadd_phred < 10:
            benign_predictors.append(f"CADD={features.cadd_phred:.1f}")

        if features.revel_score > 0.7:
            damaging_predictors.append(f"REVEL={features.revel_score:.3f}")
        elif features.revel_score < 0.3:
            benign_predictors.append(f"REVEL={features.revel_score:.3f}")

        if damaging_predictors:
            explanations.append(
                f"In-silico predictors suggesting damaging: {', '.join(damaging_predictors)} "
                f"(PP3 evidence)."
            )
        if benign_predictors:
            explanations.append(
                f"In-silico predictors suggesting benign: {', '.join(benign_predictors)} "
                f"(BP4 evidence)."
            )

        # Conservation
        if features.phylop_score > 4.0:
            explanations.append(
                f"Highly conserved position (PhyloP={features.phylop_score:.2f}). "
                "Strong evolutionary constraint at this site."
            )
        elif features.phylop_score < -2.0:
            explanations.append(
                f"Fast-evolving position (PhyloP={features.phylop_score:.2f}). "
                "Low conservation supports benign classification."
            )

        # Protein domain
        if features.in_protein_domain:
            explanations.append(
                "Variant located within a known protein domain "
                f"(domain conservation={features.domain_conservation:.2f}). "
                "Functional domain disruption may affect protein activity (PM1)."
            )

        # Splice impact
        if features.splice_ai_score > 0.5:
            explanations.append(
                f"High predicted splice impact (SpliceAI={features.splice_ai_score:.2f}). "
                "Variant may disrupt normal mRNA splicing."
            )

        # Consequence
        if features.consequence in ("nonsense", "frameshift"):
            explanations.append(
                f"Loss-of-function variant type ({features.consequence}). "
                "Null variants in genes where LOF is a mechanism of disease support "
                "pathogenic classification (PVS1)."
            )

        # Top contributing features
        top_features = list(feat_importances.items())[:3]
        feat_strs = [f"{name} ({importance:.1%})" for name, importance in top_features]
        explanations.append(f"Most influential features: {', '.join(feat_strs)}.")

        return explanations

    def save(self, path: str | Path) -> None:
        """Save trained model to disk.

        Args:
            path: File path for the saved model artifact.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        artifact = {
            "model": self._model,
            "calibrated_model": self._calibrated_model,
            "scaler": self._scaler,
            "label_encoder": self._label_encoder,
            "params": self._model_params,
            "metrics": self._training_metrics,
            "version": "1.0.0",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: File path of the saved model artifact.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._calibrated_model = artifact.get("calibrated_model")
        self._scaler = artifact["scaler"]
        self._label_encoder = artifact["label_encoder"]
        self._model_params = artifact["params"]
        self._training_metrics = artifact.get("metrics")
        self._is_trained = True
        logger.info("Model loaded from %s (version %s)", path, artifact.get("version"))


def generate_synthetic_training_data(
    n_samples: int = 1000, random_state: int = 42
) -> tuple[list[VariantFeatures], list[VariantClass]]:
    """Generate synthetic training data for development and testing.

    Produces realistic-looking variant features and labels using known
    relationships between features and pathogenicity.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (features_list, labels_list).
    """
    rng = np.random.default_rng(random_state)
    features_list: list[VariantFeatures] = []
    labels: list[VariantClass] = []

    class_distribution = [
        VariantClass.BENIGN,
        VariantClass.LIKELY_BENIGN,
        VariantClass.VUS,
        VariantClass.LIKELY_PATHOGENIC,
        VariantClass.PATHOGENIC,
    ]
    class_weights = [0.25, 0.20, 0.25, 0.15, 0.15]

    consequences = [
        "missense",
        "nonsense",
        "frameshift",
        "synonymous",
        "splice_site",
        "inframe_insertion",
        "inframe_deletion",
    ]

    for _ in range(n_samples):
        label = rng.choice(class_distribution, p=class_weights)

        # Generate features correlated with the label
        pathogenicity_bias = {
            VariantClass.BENIGN: 0.0,
            VariantClass.LIKELY_BENIGN: 0.25,
            VariantClass.VUS: 0.5,
            VariantClass.LIKELY_PATHOGENIC: 0.75,
            VariantClass.PATHOGENIC: 1.0,
        }[label]

        noise = rng.normal(0, 0.15)

        # Conservation: higher for pathogenic
        phylop = float(np.clip(pathogenicity_bias * 8 - 2 + rng.normal(0, 1.5), -14, 6.4))
        phastcons = float(np.clip(pathogenicity_bias * 0.8 + rng.normal(0, 0.2), 0, 1))
        gerp = float(np.clip(pathogenicity_bias * 6 - 1 + rng.normal(0, 1), -12, 6))

        # Allele frequency: lower for pathogenic
        if label in (VariantClass.BENIGN, VariantClass.LIKELY_BENIGN):
            gnomad_af = float(10 ** rng.uniform(-4, -1))
        elif label == VariantClass.VUS:
            gnomad_af = float(10 ** rng.uniform(-6, -3))
        else:
            gnomad_af = float(10 ** rng.uniform(-8, -4))

        gnomad_af_afr = gnomad_af * float(rng.uniform(0.5, 2.0))
        gnomad_af_eas = gnomad_af * float(rng.uniform(0.3, 3.0))
        gnomad_af_nfe = gnomad_af * float(rng.uniform(0.5, 1.5))
        homozygotes = int(max(0, gnomad_af * 150000 * rng.uniform(0, 2)))

        # Functional predictions: higher pathogenicity scores for pathogenic
        sift = float(np.clip(1.0 - pathogenicity_bias + rng.normal(0, 0.2), 0, 1))
        polyphen = float(np.clip(pathogenicity_bias + rng.normal(0, 0.15), 0, 1))
        cadd = float(np.clip(pathogenicity_bias * 35 + rng.normal(0, 5), 0, 50))
        revel = float(np.clip(pathogenicity_bias * 0.8 + 0.1 + rng.normal(0, 0.1), 0, 1))
        mt_score = float(np.clip(pathogenicity_bias + rng.normal(0, 0.15), 0, 1))

        # Protein domain
        in_domain = bool(rng.random() < 0.3 + pathogenicity_bias * 0.4)
        domain_cons = float(rng.uniform(0.5, 1.0)) if in_domain else 0.0
        dist_active = float(rng.uniform(0, 50)) if in_domain else -1.0
        pfam_count = int(rng.integers(0, 4)) if in_domain else 0

        # Variant type and consequence
        if label in (VariantClass.PATHOGENIC, VariantClass.LIKELY_PATHOGENIC):
            consequence_weights = [0.35, 0.15, 0.15, 0.05, 0.15, 0.08, 0.07]
        else:
            consequence_weights = [0.30, 0.02, 0.02, 0.40, 0.05, 0.10, 0.11]

        consequence = str(rng.choice(consequences, p=consequence_weights))
        variant_type = (
            "SNV"
            if consequence in ("missense", "nonsense", "synonymous")
            else str(rng.choice(["insertion", "deletion", "SNV"]))
        )

        total_exons = int(rng.integers(2, 40))
        exon_num = int(rng.integers(1, total_exons + 1))

        blosum62 = float(rng.normal(-pathogenicity_bias * 3, 1.5))
        grantham = float(np.clip(pathogenicity_bias * 150 + rng.normal(0, 30), 0, 215))

        splice_ai = float(
            np.clip(rng.exponential(0.1) + (0.3 if consequence == "splice_site" else 0.0), 0, 1)
        )
        splice_dist = int(rng.integers(0, 200))

        features_list.append(
            VariantFeatures(
                phylop_score=phylop,
                phastcons_score=phastcons,
                gerp_score=gerp,
                gnomad_af=gnomad_af,
                gnomad_af_afr=gnomad_af_afr,
                gnomad_af_eas=gnomad_af_eas,
                gnomad_af_nfe=gnomad_af_nfe,
                gnomad_homozygote_count=homozygotes,
                sift_score=sift,
                polyphen2_score=polyphen,
                cadd_phred=cadd,
                revel_score=revel,
                mutation_taster_score=mt_score,
                in_protein_domain=in_domain,
                domain_conservation=domain_cons,
                distance_to_active_site=dist_active,
                pfam_domain_count=pfam_count,
                variant_type=variant_type,
                consequence=consequence,
                exon_number=exon_num,
                total_exons=total_exons,
                amino_acid_change_blosum62=blosum62,
                grantham_distance=grantham,
                splice_ai_score=splice_ai,
                max_splice_distance=splice_dist,
            )
        )
        labels.append(label)

    logger.info(
        "Generated %d synthetic samples: %s",
        n_samples,
        {vc.value: labels.count(vc) for vc in VariantClass},
    )
    return features_list, labels
