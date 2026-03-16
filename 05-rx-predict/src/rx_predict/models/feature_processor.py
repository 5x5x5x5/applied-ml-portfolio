"""Fast feature processing for drug response prediction.

All operations are vectorized with numpy for sub-millisecond processing.
Feature transformations are cached to avoid redundant computation.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

logger = structlog.get_logger(__name__)

# Pre-defined genetic variant catalogs for fast lookup
PHARMACOGENOMIC_GENES: dict[str, list[str]] = {
    "CYP2D6": ["*1", "*2", "*3", "*4", "*5", "*6", "*10", "*17", "*41"],
    "CYP2C19": ["*1", "*2", "*3", "*17"],
    "CYP3A4": ["*1", "*1B", "*22"],
    "CYP2C9": ["*1", "*2", "*3"],
    "VKORC1": ["-1639G>A_GG", "-1639G>A_GA", "-1639G>A_AA"],
    "DPYD": ["*1", "*2A", "*13"],
    "TPMT": ["*1", "*2", "*3A", "*3C"],
    "UGT1A1": ["*1", "*28", "*6"],
    "SLCO1B1": ["*1A", "*5", "*15"],
    "HLA-B": ["*57:01_pos", "*57:01_neg", "*58:01_pos", "*58:01_neg"],
}

# Pre-computed gene-to-index mapping for O(1) lookups
_GENE_INDEX_MAP: dict[str, dict[str, int]] = {}
_TOTAL_GENETIC_FEATURES = 0
_offset = 0
for gene, variants in PHARMACOGENOMIC_GENES.items():
    _GENE_INDEX_MAP[gene] = {v: _offset + i for i, v in enumerate(variants)}
    _offset += len(variants)
_TOTAL_GENETIC_FEATURES = _offset

# Metabolizer phenotype encoding
METABOLIZER_PHENOTYPES: dict[str, float] = {
    "poor": 0.0,
    "intermediate": 0.33,
    "normal": 0.66,
    "rapid": 0.85,
    "ultrarapid": 1.0,
}

# Drug class encoding for feature hashing
DRUG_CLASSES: dict[str, int] = {
    "ssri": 0,
    "snri": 1,
    "tca": 2,
    "maoi": 3,
    "statin": 4,
    "anticoagulant": 5,
    "antiplatelet": 6,
    "opioid": 7,
    "nsaid": 8,
    "ppi": 9,
    "beta_blocker": 10,
    "ace_inhibitor": 11,
    "arb": 12,
    "immunosuppressant": 13,
    "antiepileptic": 14,
    "other": 15,
}

# Pre-allocated arrays for speed
_EMPTY_GENETIC_VEC = np.zeros(_TOTAL_GENETIC_FEATURES, dtype=np.float32)
_DEMOGRAPHIC_MEANS = np.array(
    [45.0, 75.0, 170.0, 25.9], dtype=np.float32
)  # age, weight, height, bmi
_DEMOGRAPHIC_STDS = np.array([18.0, 20.0, 15.0, 5.5], dtype=np.float32)


class FeatureProcessor:
    """Vectorized feature processor optimized for low-latency inference.

    Features are organized into blocks:
    - Genetic variants: one-hot encoded (variable length per gene)
    - Metabolizer phenotype: ordinal encoded
    - Demographics: z-score normalized
    - Drug info: feature-hashed
    - Medical history: summarized into counts and flags
    """

    def __init__(self, cache_size: int = 2048) -> None:
        self._cache_size = cache_size
        self._feature_dim: int | None = None
        # Pre-warm the cached function with appropriate size
        self._encode_genetic_cached = lru_cache(maxsize=cache_size)(
            self._encode_genetic_variants_impl
        )
        logger.info(
            "feature_processor_initialized",
            total_genetic_features=_TOTAL_GENETIC_FEATURES,
            cache_size=cache_size,
        )

    @property
    def feature_dimension(self) -> int:
        """Total feature vector dimension."""
        # genetic + metabolizer + demographics(4) + drug(16) + medical_history(10)
        return _TOTAL_GENETIC_FEATURES + 1 + 4 + len(DRUG_CLASSES) + 10

    def process_single(self, patient_data: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Process a single patient record into a feature vector.

        Target: <2ms processing time.
        """
        features = np.zeros(self.feature_dimension, dtype=np.float32)
        offset = 0

        # Block 1: Genetic variants (one-hot)
        genetic_data = patient_data.get("genetic_profile", {})
        genetic_key = self._make_genetic_key(genetic_data)
        genetic_vec = self._encode_genetic_cached(genetic_key)
        features[offset : offset + _TOTAL_GENETIC_FEATURES] = genetic_vec
        offset += _TOTAL_GENETIC_FEATURES

        # Block 2: Metabolizer phenotype
        phenotype = patient_data.get("metabolizer_phenotype", "normal")
        features[offset] = METABOLIZER_PHENOTYPES.get(phenotype.lower(), 0.66)
        offset += 1

        # Block 3: Demographics (z-score normalized)
        demo_vec = self._encode_demographics(patient_data.get("demographics", {}))
        features[offset : offset + 4] = demo_vec
        offset += 4

        # Block 4: Drug info (feature hashed)
        drug_vec = self._encode_drug(patient_data.get("drug", {}))
        features[offset : offset + len(DRUG_CLASSES)] = drug_vec
        offset += len(DRUG_CLASSES)

        # Block 5: Medical history summary
        history_vec = self._encode_medical_history(patient_data.get("medical_history", {}))
        features[offset : offset + 10] = history_vec

        return features

    def process_batch(self, patient_records: list[dict[str, Any]]) -> npt.NDArray[np.float32]:
        """Process a batch of patient records into a feature matrix.

        Uses pre-allocated matrix for zero-copy efficiency.
        """
        n = len(patient_records)
        feature_matrix = np.zeros((n, self.feature_dimension), dtype=np.float32)
        for i, record in enumerate(patient_records):
            feature_matrix[i] = self.process_single(record)
        return feature_matrix

    @staticmethod
    def _make_genetic_key(genetic_data: dict[str, Any]) -> str:
        """Create a hashable key from genetic data for caching."""
        parts: list[str] = []
        for gene in sorted(PHARMACOGENOMIC_GENES.keys()):
            variants = genetic_data.get(gene, [])
            if isinstance(variants, str):
                variants = [variants]
            parts.append(f"{gene}:{','.join(sorted(variants))}")
        return "|".join(parts)

    @staticmethod
    def _encode_genetic_variants_impl(genetic_key: str) -> tuple[float, ...]:
        """One-hot encode genetic variants. Returns tuple for LRU cache compatibility."""
        vec = np.zeros(_TOTAL_GENETIC_FEATURES, dtype=np.float32)
        if not genetic_key:
            return tuple(vec.tolist())

        for part in genetic_key.split("|"):
            if ":" not in part:
                continue
            gene, variants_str = part.split(":", 1)
            if not variants_str or gene not in _GENE_INDEX_MAP:
                continue
            index_map = _GENE_INDEX_MAP[gene]
            for variant in variants_str.split(","):
                variant = variant.strip()
                if variant in index_map:
                    vec[index_map[variant]] = 1.0

        return tuple(vec.tolist())

    @staticmethod
    def _encode_demographics(demographics: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Z-score normalize demographic features."""
        raw = np.array(
            [
                float(demographics.get("age", 45)),
                float(demographics.get("weight_kg", 75)),
                float(demographics.get("height_cm", 170)),
                float(demographics.get("bmi", 25.9)),
            ],
            dtype=np.float32,
        )
        # Vectorized z-score normalization
        return (raw - _DEMOGRAPHIC_MEANS) / _DEMOGRAPHIC_STDS

    @staticmethod
    def _encode_drug(drug_info: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Feature-hash drug information."""
        vec = np.zeros(len(DRUG_CLASSES), dtype=np.float32)

        drug_class = drug_info.get("drug_class", "other").lower()
        if drug_class in DRUG_CLASSES:
            vec[DRUG_CLASSES[drug_class]] = 1.0
        else:
            # Feature hashing fallback for unknown drug classes
            idx = int(hashlib.md5(drug_class.encode()).hexdigest(), 16) % len(DRUG_CLASSES)
            vec[idx] = 1.0

        # Encode dosage as normalized value (0-1 range, assuming max 1000mg)
        dosage = float(drug_info.get("dosage_mg", 0))
        max_dosage = float(drug_info.get("max_dosage_mg", 1000))
        if max_dosage > 0:
            normalized_dosage = min(dosage / max_dosage, 1.0)
            # Add dosage signal to the drug class feature
            class_idx = DRUG_CLASSES.get(drug_class, len(DRUG_CLASSES) - 1)
            vec[class_idx] *= 0.5 + 0.5 * normalized_dosage

        return vec

    @staticmethod
    def _encode_medical_history(history: dict[str, Any]) -> npt.NDArray[np.float32]:
        """Summarize medical history into a fixed-length vector.

        Features:
        [0] number of current medications (normalized)
        [1] number of known allergies (normalized)
        [2] number of prior adverse reactions (normalized)
        [3] liver_function_flag (0 or 1)
        [4] kidney_function_flag (0 or 1)
        [5] cardiac_condition_flag (0 or 1)
        [6] diabetes_flag (0 or 1)
        [7] pregnancy_flag (0 or 1)
        [8] age_group_encoded (0-1)
        [9] polypharmacy_risk (computed)
        """
        vec = np.zeros(10, dtype=np.float32)

        # Counts normalized to 0-1 range
        vec[0] = min(float(history.get("num_current_medications", 0)) / 20.0, 1.0)
        vec[1] = min(float(history.get("num_allergies", 0)) / 10.0, 1.0)
        vec[2] = min(float(history.get("num_adverse_reactions", 0)) / 5.0, 1.0)

        # Binary flags
        conditions = history.get("conditions", [])
        if isinstance(conditions, str):
            conditions = [conditions]
        condition_set = {c.lower() for c in conditions}

        vec[3] = (
            1.0
            if any(c in condition_set for c in ["liver_disease", "hepatitis", "cirrhosis"])
            else 0.0
        )
        vec[4] = (
            1.0
            if any(c in condition_set for c in ["kidney_disease", "ckd", "renal_failure"])
            else 0.0
        )
        vec[5] = (
            1.0
            if any(c in condition_set for c in ["heart_disease", "arrhythmia", "heart_failure"])
            else 0.0
        )
        vec[6] = (
            1.0
            if any(c in condition_set for c in ["diabetes", "type_1_diabetes", "type_2_diabetes"])
            else 0.0
        )
        vec[7] = 1.0 if history.get("pregnant", False) else 0.0

        # Age group encoding
        age = float(history.get("age", 45))
        vec[8] = min(age / 100.0, 1.0)

        # Polypharmacy risk score
        num_meds = float(history.get("num_current_medications", 0))
        num_conditions = len(condition_set)
        vec[9] = min((num_meds * 0.1 + num_conditions * 0.15), 1.0)

        return vec

    def get_feature_names(self) -> list[str]:
        """Return ordered feature names for interpretability."""
        names: list[str] = []

        # Genetic features
        for gene, variants in PHARMACOGENOMIC_GENES.items():
            for variant in variants:
                names.append(f"gene_{gene}_{variant}")

        names.append("metabolizer_phenotype")

        # Demographics
        names.extend(["demo_age", "demo_weight", "demo_height", "demo_bmi"])

        # Drug features
        for drug_class in DRUG_CLASSES:
            names.append(f"drug_{drug_class}")

        # Medical history
        names.extend(
            [
                "hist_num_medications",
                "hist_num_allergies",
                "hist_num_adverse_reactions",
                "hist_liver_flag",
                "hist_kidney_flag",
                "hist_cardiac_flag",
                "hist_diabetes_flag",
                "hist_pregnancy_flag",
                "hist_age_group",
                "hist_polypharmacy_risk",
            ]
        )

        return names

    def clear_cache(self) -> None:
        """Clear the genetic encoding cache."""
        self._encode_genetic_cached.cache_clear()
        logger.info("feature_cache_cleared")
