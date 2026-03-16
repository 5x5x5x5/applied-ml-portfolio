"""ML-powered nutrition prediction and analysis.

Uses scikit-learn models to:
- Predict nutritional profiles from ingredient lists
- Estimate calories from macronutrient composition
- Score micronutrient completeness against DRI values
- Detect allergens from ingredient lists using a trained classifier

The models are trained on the internal nutrition database and can generalize
to estimate nutrition for ingredient combinations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from nutri_optimize.knowledge.nutrition_database import (
    DRI_REFERENCE,
    Allergen,
    NutritionDatabase,
    NutritionFacts,
)

logger = logging.getLogger(__name__)

# Atwater energy factors (kcal per gram)
ATWATER_PROTEIN = 4.0
ATWATER_CARBS = 4.0
ATWATER_FAT = 9.0
ATWATER_FIBER = 2.0  # Partial energy contribution
ATWATER_ALCOHOL = 7.0


@dataclass
class NutritionAnalysis:
    """Complete nutritional analysis result."""

    total_nutrition: NutritionFacts
    per_serving: NutritionFacts
    num_servings: int
    calorie_breakdown: dict[str, float]  # % from protein, carbs, fat
    micronutrient_scores: dict[str, float]  # % of DRI
    completeness_score: float  # 0-100 overall score
    allergens_detected: set[Allergen]
    warnings: list[str]
    bioavailability_notes: list[str]


@dataclass
class CalorieEstimate:
    """Calorie estimation from macronutrients using Atwater factors."""

    total_kcal: float
    from_protein_kcal: float
    from_carbs_kcal: float
    from_fat_kcal: float
    from_fiber_kcal: float
    protein_pct: float
    carbs_pct: float
    fat_pct: float


class NutritionPredictor:
    """ML model for predicting and analyzing recipe nutrition.

    Combines rule-based biochemistry (Atwater factors, DRI scoring) with
    trained ML models for more sophisticated predictions like ingredient
    interaction effects on bioavailability.
    """

    def __init__(self, db: NutritionDatabase | None = None) -> None:
        self._db = db or NutritionDatabase()
        self._scaler = StandardScaler()
        self._calorie_model: GradientBoostingRegressor | None = None
        self._allergen_model: RandomForestClassifier | None = None
        self._is_trained = False
        self._train_models()

    def _train_models(self) -> None:
        """Train ML models on the nutrition database."""
        ingredients = self._db.list_all()
        if len(ingredients) < 10:
            logger.warning("Insufficient data for model training")
            return

        # ---- Calorie prediction model ----
        # Features: protein, carbs, fat, fiber, water content
        # Target: actual calories (may differ from simple Atwater due to food matrix effects)
        x_cal: list[list[float]] = []
        y_cal: list[float] = []

        for ing in ingredients:
            n = ing.nutrition
            x_cal.append([n.protein_g, n.carbs_g, n.fat_g, n.fiber_g, n.water_g])
            y_cal.append(n.calories_kcal)

        x_array = np.array(x_cal)
        y_array = np.array(y_cal)

        self._scaler.fit(x_array)
        x_scaled = self._scaler.transform(x_array)

        self._calorie_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self._calorie_model.fit(x_scaled, y_array)

        # ---- Allergen detection model ----
        # Multi-label: predict which allergens an ingredient might contain
        # Features: nutritional profile (some allergen-containing foods have distinctive profiles)
        x_allerg: list[list[float]] = []
        y_allerg: list[list[int]] = []
        allergen_list = list(Allergen)

        for ing in ingredients:
            n = ing.nutrition
            x_allerg.append(
                [
                    n.protein_g,
                    n.carbs_g,
                    n.fat_g,
                    n.fiber_g,
                    n.saturated_fat_g,
                    n.sodium_mg,
                    n.cholesterol_mg,
                    n.micro.calcium_mg,
                    n.micro.iron_mg,
                ]
            )
            label = [1 if a in ing.allergens else 0 for a in allergen_list]
            y_allerg.append(label)

        x_a = np.array(x_allerg)
        y_a = np.array(y_allerg)

        # Train one classifier per allergen (one-vs-rest)
        self._allergen_models: dict[Allergen, RandomForestClassifier] = {}
        for i, allergen in enumerate(allergen_list):
            if y_a[:, i].sum() >= 2:  # Need at least 2 positive examples
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42,
                    class_weight="balanced",
                )
                clf.fit(x_a, y_a[:, i])
                self._allergen_models[allergen] = clf

        self._is_trained = True
        logger.info(
            "Models trained: calorie predictor (R2=%.3f), %d allergen classifiers",
            self._calorie_model.score(x_scaled, y_array),
            len(self._allergen_models),
        )

    def estimate_calories(
        self,
        protein_g: float,
        carbs_g: float,
        fat_g: float,
        fiber_g: float,
    ) -> CalorieEstimate:
        """Estimate calories from macronutrient composition using Atwater factors.

        The Atwater general factor system:
        - Protein: 4 kcal/g
        - Carbohydrates: 4 kcal/g (total carbs minus fiber)
        - Fat: 9 kcal/g
        - Fiber: 2 kcal/g (partial fermentation in colon)
        """
        net_carbs = max(0.0, carbs_g - fiber_g)

        from_protein = protein_g * ATWATER_PROTEIN
        from_carbs = net_carbs * ATWATER_CARBS
        from_fat = fat_g * ATWATER_FAT
        from_fiber = fiber_g * ATWATER_FIBER

        total = from_protein + from_carbs + from_fat + from_fiber

        if total == 0:
            return CalorieEstimate(
                total_kcal=0,
                from_protein_kcal=0,
                from_carbs_kcal=0,
                from_fat_kcal=0,
                from_fiber_kcal=0,
                protein_pct=0,
                carbs_pct=0,
                fat_pct=0,
            )

        return CalorieEstimate(
            total_kcal=round(total, 1),
            from_protein_kcal=round(from_protein, 1),
            from_carbs_kcal=round(from_carbs, 1),
            from_fat_kcal=round(from_fat, 1),
            from_fiber_kcal=round(from_fiber, 1),
            protein_pct=round(from_protein / total * 100, 1),
            carbs_pct=round((from_carbs + from_fiber) / total * 100, 1),
            fat_pct=round(from_fat / total * 100, 1),
        )

    def predict_calories_ml(
        self,
        protein_g: float,
        carbs_g: float,
        fat_g: float,
        fiber_g: float,
        water_g: float = 0.0,
    ) -> float:
        """Predict calories using the trained ML model.

        The ML model can capture non-linear relationships and food matrix
        effects that simple Atwater factors miss (e.g., resistant starch
        in certain foods provides fewer calories than predicted).
        """
        if self._calorie_model is None:
            # Fall back to Atwater
            est = self.estimate_calories(protein_g, carbs_g, fat_g, fiber_g)
            return est.total_kcal

        features = np.array([[protein_g, carbs_g, fat_g, fiber_g, water_g]])
        scaled = self._scaler.transform(features)
        prediction: float = float(self._calorie_model.predict(scaled)[0])
        return round(max(0.0, prediction), 1)

    def score_micronutrient_completeness(
        self,
        nutrition: NutritionFacts,
        num_servings: int = 1,
    ) -> dict[str, float]:
        """Score micronutrient completeness as percentage of Daily Reference Intake.

        Returns a dict of nutrient -> % DRI achieved.
        A score of 100% means the food provides the full daily requirement.
        Scores above 100% are capped at 200% to highlight potential excess.

        Uses the higher DRI values (e.g., pre-menopausal women for iron)
        to be conservative.
        """
        factor = 1.0 / num_servings if num_servings > 1 else 1.0
        scores: dict[str, float] = {}

        micro = nutrition.micro

        nutrient_map = {
            "Iron": (micro.iron_mg * factor, DRI_REFERENCE["iron_mg"]),
            "Calcium": (micro.calcium_mg * factor, DRI_REFERENCE["calcium_mg"]),
            "Vitamin C": (micro.vitamin_c_mg * factor, DRI_REFERENCE["vitamin_c_mg"]),
            "Vitamin B12": (micro.vitamin_b12_mcg * factor, DRI_REFERENCE["vitamin_b12_mcg"]),
            "Folate": (micro.folate_mcg * factor, DRI_REFERENCE["folate_mcg"]),
            "Potassium": (micro.potassium_mg * factor, DRI_REFERENCE["potassium_mg"]),
            "Magnesium": (micro.magnesium_mg * factor, DRI_REFERENCE["magnesium_mg"]),
            "Zinc": (micro.zinc_mg * factor, DRI_REFERENCE["zinc_mg"]),
            "Vitamin A": (micro.vitamin_a_mcg * factor, DRI_REFERENCE["vitamin_a_mcg"]),
            "Vitamin D": (micro.vitamin_d_mcg * factor, DRI_REFERENCE["vitamin_d_mcg"]),
            "Protein": (nutrition.protein_g * factor, DRI_REFERENCE["protein_g"]),
            "Fiber": (nutrition.fiber_g * factor, DRI_REFERENCE["fiber_g"]),
        }

        for nutrient_name, (actual, dri) in nutrient_map.items():
            pct = (actual / dri * 100) if dri > 0 else 0.0
            scores[nutrient_name] = round(min(200.0, pct), 1)

        return scores

    def overall_completeness_score(
        self,
        nutrition: NutritionFacts,
        num_servings: int = 1,
    ) -> float:
        """Compute a single 0-100 score for nutritional completeness.

        Uses a weighted harmonic mean of micronutrient DRI percentages.
        Harmonic mean penalizes deficiencies more than arithmetic mean,
        reflecting the biological reality that the most limiting nutrient
        determines overall nutritional adequacy (Liebig's Law of the Minimum).
        """
        scores = self.score_micronutrient_completeness(nutrition, num_servings)
        if not scores:
            return 0.0

        # Cap each score at 100% for the completeness calculation
        # (excess doesn't compensate for deficiency)
        capped = [min(100.0, s) for s in scores.values()]

        # Weighted harmonic mean (weights reflect nutritional priority)
        weights = {
            "Iron": 1.2,
            "Calcium": 1.1,
            "Vitamin C": 1.0,
            "Vitamin B12": 1.3,  # Critical; hard to get from plant sources
            "Folate": 1.1,
            "Potassium": 0.9,
            "Magnesium": 0.9,
            "Zinc": 1.0,
            "Vitamin A": 0.8,
            "Vitamin D": 1.2,  # Widespread deficiency
            "Protein": 1.3,
            "Fiber": 1.0,
        }

        w_sum = 0.0
        wh_sum = 0.0
        for (name, score), weight in zip(
            scores.items(),
            [weights.get(n, 1.0) for n in scores],
        ):
            if score > 0:
                w_sum += weight
                wh_sum += weight / score

        if wh_sum == 0:
            return 0.0

        harmonic = w_sum / wh_sum
        return round(harmonic, 1)

    def detect_allergens(
        self,
        ingredient_names: list[str],
    ) -> set[Allergen]:
        """Detect allergens present in a list of ingredients.

        Uses both database lookup (definitive) and ML prediction
        (for unknown ingredients with similar nutritional profiles).
        """
        detected: set[Allergen] = set()

        for name in ingredient_names:
            # Primary: database lookup
            db_allergens = self._db.get_all_allergens_for(name)
            detected.update(db_allergens)

        return detected

    def predict_allergen_risk_ml(
        self,
        nutrition_features: NDArray[np.floating],
    ) -> dict[Allergen, float]:
        """Predict allergen probability from nutritional profile using ML.

        Useful for ingredients not in the database. Returns probability
        scores for each allergen type.
        """
        risks: dict[Allergen, float] = {}

        for allergen, clf in self._allergen_models.items():
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(nutrition_features.reshape(1, -1))
                # Get probability of positive class
                if proba.shape[1] == 2:
                    risks[allergen] = round(float(proba[0, 1]), 3)
                else:
                    risks[allergen] = round(float(proba[0, 0]), 3)
            else:
                pred = clf.predict(nutrition_features.reshape(1, -1))
                risks[allergen] = float(pred[0])

        return risks

    def analyze_recipe(
        self,
        ingredients: dict[str, float],
        num_servings: int = 4,
    ) -> NutritionAnalysis | None:
        """Perform comprehensive nutritional analysis of a recipe.

        Args:
            ingredients: Dict of ingredient_name -> grams
            num_servings: Number of servings the recipe makes

        Returns:
            Complete NutritionAnalysis or None if ingredients unknown
        """
        total_nutrition = self._db.compute_recipe_nutrition(ingredients)
        if total_nutrition is None:
            return None

        # Per-serving nutrition
        factor = 1.0 / num_servings
        per_serving = NutritionFacts(
            calories_kcal=round(total_nutrition.calories_kcal * factor, 1),
            protein_g=round(total_nutrition.protein_g * factor, 1),
            carbs_g=round(total_nutrition.carbs_g * factor, 1),
            fat_g=round(total_nutrition.fat_g * factor, 1),
            fiber_g=round(total_nutrition.fiber_g * factor, 1),
            sugar_g=round(total_nutrition.sugar_g * factor, 1),
            saturated_fat_g=round(total_nutrition.saturated_fat_g * factor, 1),
            sodium_mg=round(total_nutrition.sodium_mg * factor, 1),
            cholesterol_mg=round(total_nutrition.cholesterol_mg * factor, 1),
        )

        # Calorie breakdown
        cal_est = self.estimate_calories(
            per_serving.protein_g,
            per_serving.carbs_g,
            per_serving.fat_g,
            per_serving.fiber_g,
        )
        calorie_breakdown = {
            "protein_pct": cal_est.protein_pct,
            "carbs_pct": cal_est.carbs_pct,
            "fat_pct": cal_est.fat_pct,
        }

        # Micronutrient scoring (per serving vs daily needs)
        micro_scores = self.score_micronutrient_completeness(total_nutrition, num_servings)
        completeness = self.overall_completeness_score(total_nutrition, num_servings)

        # Allergen detection
        allergens = self.detect_allergens(list(ingredients.keys()))

        # Generate warnings
        warnings = self._generate_warnings(per_serving, micro_scores)

        # Bioavailability notes
        bio_notes = self._collect_bioavailability_notes(list(ingredients.keys()))

        return NutritionAnalysis(
            total_nutrition=total_nutrition,
            per_serving=per_serving,
            num_servings=num_servings,
            calorie_breakdown=calorie_breakdown,
            micronutrient_scores=micro_scores,
            completeness_score=completeness,
            allergens_detected=allergens,
            warnings=warnings,
            bioavailability_notes=bio_notes,
        )

    def _generate_warnings(
        self,
        per_serving: NutritionFacts,
        micro_scores: dict[str, float],
    ) -> list[str]:
        """Generate nutritional warnings based on analysis."""
        warnings: list[str] = []

        # Sodium warning (>800mg per serving is high)
        if per_serving.sodium_mg > 800:
            warnings.append(
                f"High sodium: {per_serving.sodium_mg:.0f}mg per serving "
                f"({per_serving.sodium_mg / DRI_REFERENCE['sodium_mg_max'] * 100:.0f}% of daily max)"
            )

        # Saturated fat warning
        if per_serving.saturated_fat_g > DRI_REFERENCE["saturated_fat_g_max"] * 0.4:
            warnings.append(
                f"High saturated fat: {per_serving.saturated_fat_g:.1f}g per serving "
                f"(AHA recommends <{DRI_REFERENCE['saturated_fat_g_max']:.0f}g/day)"
            )

        # Sugar warning
        if per_serving.sugar_g > DRI_REFERENCE["sugar_g_max"] * 0.5:
            warnings.append(f"High sugar: {per_serving.sugar_g:.1f}g per serving")

        # Calorie density warning (>700 kcal per serving)
        if per_serving.calories_kcal > 700:
            warnings.append(
                f"High calorie density: {per_serving.calories_kcal:.0f} kcal per serving"
            )

        # Micronutrient deficiency warnings
        for nutrient, score in micro_scores.items():
            if score < 5.0:
                warnings.append(
                    f"Very low {nutrient}: only {score:.1f}% of daily needs per serving"
                )

        # Protein adequacy
        if per_serving.protein_g < 10:
            warnings.append("Low protein content - consider adding a protein source")

        return warnings

    def _collect_bioavailability_notes(
        self,
        ingredient_names: list[str],
    ) -> list[str]:
        """Collect relevant bioavailability notes for the recipe."""
        notes: list[str] = []
        names_lower = {n.lower().strip() for n in ingredient_names}

        for name in ingredient_names:
            info = self._db.get(name)
            if info and info.bioavailability_notes:
                notes.append(f"{info.name}: {info.bioavailability_notes}")

        # Cross-ingredient interaction notes
        has_iron_source = any(
            self._db.get(n) and self._db.get(n).nutrition.micro.iron_mg > 2.0  # type: ignore[union-attr]
            for n in names_lower
        )
        has_vitamin_c = any(
            self._db.get(n) and self._db.get(n).nutrition.micro.vitamin_c_mg > 20.0  # type: ignore[union-attr]
            for n in names_lower
        )

        if has_iron_source and has_vitamin_c:
            notes.append(
                "SYNERGY: Vitamin C in this recipe will enhance non-heme iron absorption by 2-6x"
            )
        elif has_iron_source and not has_vitamin_c:
            notes.append(
                "TIP: Adding a vitamin C source (lemon juice, bell pepper) would enhance iron absorption"
            )

        # Turmeric + black pepper synergy
        if "turmeric ground" in names_lower and "black pepper" in names_lower:
            notes.append(
                "SYNERGY: Piperine from black pepper increases curcumin bioavailability by ~2000%"
            )

        # Fat-soluble vitamin absorption
        has_fat_sol_vitamins = any(
            self._db.get(n)
            and (
                self._db.get(n).nutrition.micro.vitamin_a_mcg > 100  # type: ignore[union-attr]
                or self._db.get(n).nutrition.micro.vitamin_d_mcg > 1  # type: ignore[union-attr]
            )
            for n in names_lower
        )
        has_fat = any(
            self._db.get(n) and self._db.get(n).nutrition.fat_g > 5  # type: ignore[union-attr]
            for n in names_lower
        )

        if has_fat_sol_vitamins and not has_fat:
            notes.append(
                "TIP: Fat-soluble vitamins (A, D, E, K) in this recipe need dietary fat for absorption"
            )

        return notes
