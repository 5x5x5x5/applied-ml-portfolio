"""Multi-objective recipe optimization using scipy.optimize.

Optimizes recipes across three competing objectives:
1. Maximize nutritional quality (micronutrient completeness)
2. Minimize calorie density (or match target calories)
3. Preserve taste quality (flavor compatibility scores)

Uses Sequential Least Squares Programming (SLSQP) from scipy for
constrained nonlinear optimization with dietary restrictions as constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import minimize

from nutri_optimize.knowledge.nutrition_database import (
    Allergen,
    NutritionDatabase,
    NutritionFacts,
)
from nutri_optimize.knowledge.taste_model import TasteModel
from nutri_optimize.models.nutrition_predictor import NutritionPredictor

logger = logging.getLogger(__name__)


class DietaryGoal(str, Enum):
    """Pre-defined nutritional optimization goals."""

    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    HEART_HEALTH = "heart_health"
    BALANCED = "balanced"
    HIGH_FIBER = "high_fiber"
    LOW_SODIUM = "low_sodium"
    IRON_RICH = "iron_rich"


class DietaryRestriction(str, Enum):
    """Dietary restriction categories."""

    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    LOW_SODIUM = "low_sodium"
    LOW_FAT = "low_fat"
    LOW_CARB = "low_carb"
    HIGH_PROTEIN = "high_protein"


# Goal-specific optimization weights
GOAL_WEIGHTS: dict[DietaryGoal, dict[str, float]] = {
    DietaryGoal.WEIGHT_LOSS: {
        "calorie_weight": 2.0,  # Strongly minimize calories
        "protein_weight": 1.5,  # Maintain protein for satiety
        "fiber_weight": 1.5,  # High fiber for fullness
        "nutrition_weight": 1.0,
        "taste_weight": 0.8,
        "target_cal_per_serving": 400,
        "min_protein_pct": 25,  # At least 25% calories from protein
    },
    DietaryGoal.MUSCLE_GAIN: {
        "calorie_weight": 0.5,
        "protein_weight": 3.0,  # Maximize protein
        "fiber_weight": 0.5,
        "nutrition_weight": 1.2,
        "taste_weight": 0.8,
        "target_cal_per_serving": 600,
        "min_protein_pct": 30,
    },
    DietaryGoal.HEART_HEALTH: {
        "calorie_weight": 1.0,
        "protein_weight": 1.0,
        "fiber_weight": 2.0,  # Soluble fiber reduces cholesterol
        "nutrition_weight": 1.5,
        "taste_weight": 0.8,
        "sat_fat_penalty": 3.0,  # Heavily penalize saturated fat
        "sodium_penalty": 2.5,  # Penalize sodium
        "target_cal_per_serving": 500,
    },
    DietaryGoal.BALANCED: {
        "calorie_weight": 1.0,
        "protein_weight": 1.0,
        "fiber_weight": 1.0,
        "nutrition_weight": 1.5,
        "taste_weight": 1.0,
        "target_cal_per_serving": 500,
    },
    DietaryGoal.HIGH_FIBER: {
        "calorie_weight": 0.8,
        "protein_weight": 0.8,
        "fiber_weight": 3.0,
        "nutrition_weight": 1.0,
        "taste_weight": 0.8,
        "target_cal_per_serving": 450,
    },
    DietaryGoal.LOW_SODIUM: {
        "calorie_weight": 1.0,
        "protein_weight": 1.0,
        "fiber_weight": 1.0,
        "nutrition_weight": 1.0,
        "taste_weight": 0.8,
        "sodium_penalty": 4.0,
        "target_cal_per_serving": 500,
    },
    DietaryGoal.IRON_RICH: {
        "calorie_weight": 0.8,
        "protein_weight": 1.0,
        "fiber_weight": 1.0,
        "nutrition_weight": 1.5,
        "taste_weight": 0.8,
        "iron_bonus": 3.0,
        "target_cal_per_serving": 500,
    },
}


@dataclass
class OptimizationResult:
    """Result of recipe optimization."""

    original_ingredients: dict[str, float]
    optimized_ingredients: dict[str, float]
    original_nutrition: NutritionFacts
    optimized_nutrition: NutritionFacts
    original_taste_score: float
    optimized_taste_score: float
    original_completeness: float
    optimized_completeness: float
    calories_change_pct: float
    substitutions: list[dict[str, str]]
    optimization_notes: list[str]
    success: bool


@dataclass
class SubstitutionRecommendation:
    """A recommended ingredient substitution."""

    original: str
    substitute: str
    reason: str
    nutrition_impact: str
    taste_impact: str
    confidence: float


@dataclass
class MealPlanDay:
    """A single day in a meal plan."""

    breakfast: dict[str, float]
    lunch: dict[str, float]
    dinner: dict[str, float]
    snacks: list[dict[str, float]] = field(default_factory=list)
    daily_nutrition: NutritionFacts | None = None
    daily_completeness: float = 0.0


@dataclass
class WeeklyMealPlan:
    """A week-long optimized meal plan."""

    days: list[MealPlanDay]
    weekly_avg_calories: float
    weekly_avg_completeness: float
    dietary_goal: DietaryGoal
    restrictions: list[DietaryRestriction]
    notes: list[str]


class RecipeOptimizer:
    """Multi-objective recipe optimizer using scipy constrained optimization."""

    def __init__(
        self,
        db: NutritionDatabase | None = None,
        taste_model: TasteModel | None = None,
        predictor: NutritionPredictor | None = None,
    ) -> None:
        self._db = db or NutritionDatabase()
        self._taste = taste_model or TasteModel()
        self._predictor = predictor or NutritionPredictor(self._db)

    def optimize_recipe(
        self,
        ingredients: dict[str, float],
        goal: DietaryGoal = DietaryGoal.BALANCED,
        restrictions: list[DietaryRestriction] | None = None,
        num_servings: int = 4,
        max_iterations: int = 200,
    ) -> OptimizationResult:
        """Optimize ingredient quantities for nutritional goals.

        Uses SLSQP (Sequential Least Squares Quadratic Programming) to
        find optimal ingredient quantities subject to dietary constraints.

        Args:
            ingredients: Current recipe {ingredient_name: grams}
            goal: Nutritional optimization goal
            restrictions: Dietary restrictions to enforce
            num_servings: Number of servings
            max_iterations: Max optimizer iterations

        Returns:
            OptimizationResult with before/after comparison
        """
        restrictions = restrictions or []
        weights = GOAL_WEIGHTS[goal]

        # Validate all ingredients exist in database
        ingredient_names = list(ingredients.keys())
        for name in ingredient_names:
            if self._db.get(name) is None:
                logger.warning("Unknown ingredient: %s", name)

        # Original analysis
        orig_nutrition = self._db.compute_recipe_nutrition(ingredients)
        if orig_nutrition is None:
            return OptimizationResult(
                original_ingredients=ingredients,
                optimized_ingredients=ingredients,
                original_nutrition=NutritionFacts(0, 0, 0, 0, 0),
                optimized_nutrition=NutritionFacts(0, 0, 0, 0, 0),
                original_taste_score=0,
                optimized_taste_score=0,
                original_completeness=0,
                optimized_completeness=0,
                calories_change_pct=0,
                substitutions=[],
                optimization_notes=["Error: Unknown ingredients in recipe"],
                success=False,
            )

        orig_taste = self._taste.recipe_taste_score(ingredient_names)
        orig_completeness = self._predictor.overall_completeness_score(orig_nutrition, num_servings)

        # Set up optimization variables (ingredient quantities)
        x0 = np.array([ingredients[name] for name in ingredient_names])
        original_total = x0.sum()

        # Bounds: each ingredient between 5% and 300% of original quantity
        bounds = [(max(1.0, xi * 0.05), xi * 3.0) for xi in x0]

        # Constraint: total recipe weight within 50-150% of original
        constraints: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda x: x.sum() - original_total * 0.5,
            },
            {
                "type": "ineq",
                "fun": lambda x: original_total * 1.5 - x.sum(),
            },
        ]

        # Add dietary restriction constraints
        constraints.extend(
            self._build_restriction_constraints(ingredient_names, restrictions, num_servings)
        )

        # Objective function (to minimize)
        def objective(x: np.ndarray) -> float:
            recipe = dict(zip(ingredient_names, x.tolist()))
            return self._compute_objective(
                recipe, ingredient_names, weights, num_servings, orig_taste
            )

        # Run SLSQP optimization
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-8},
        )

        # Build optimized recipe
        optimized = dict(zip(ingredient_names, [round(v, 1) for v in result.x]))

        opt_nutrition = self._db.compute_recipe_nutrition(optimized)
        if opt_nutrition is None:
            opt_nutrition = orig_nutrition

        opt_taste = self._taste.recipe_taste_score(ingredient_names)
        opt_completeness = self._predictor.overall_completeness_score(opt_nutrition, num_servings)

        cal_change = 0.0
        if orig_nutrition.calories_kcal > 0:
            cal_change = (
                (opt_nutrition.calories_kcal - orig_nutrition.calories_kcal)
                / orig_nutrition.calories_kcal
                * 100
            )

        # Generate substitution suggestions
        substitutions = self._suggest_substitutions(ingredient_names, goal, restrictions)

        # Optimization notes
        notes = self._generate_optimization_notes(
            orig_nutrition, opt_nutrition, goal, result.success, num_servings
        )

        return OptimizationResult(
            original_ingredients=ingredients,
            optimized_ingredients=optimized,
            original_nutrition=orig_nutrition,
            optimized_nutrition=opt_nutrition,
            original_taste_score=round(orig_taste, 3),
            optimized_taste_score=round(opt_taste, 3),
            original_completeness=round(orig_completeness, 1),
            optimized_completeness=round(opt_completeness, 1),
            calories_change_pct=round(cal_change, 1),
            substitutions=substitutions,
            optimization_notes=notes,
            success=result.success,
        )

    def _compute_objective(
        self,
        recipe: dict[str, float],
        ingredient_names: list[str],
        weights: dict[str, float],
        num_servings: int,
        original_taste: float,
    ) -> float:
        """Compute the multi-objective cost function.

        Lower is better. Combines:
        - Calorie deviation from target
        - Negative nutrition score (we want to maximize)
        - Taste preservation penalty
        - Specific nutrient bonuses/penalties based on goal
        """
        nutrition = self._db.compute_recipe_nutrition(recipe)
        if nutrition is None:
            return 1e6  # Penalty for invalid recipe

        per_serving_cal = nutrition.calories_kcal / num_servings
        target_cal = weights.get("target_cal_per_serving", 500)

        # Calorie objective: squared deviation from target
        cal_cost = (
            weights.get("calorie_weight", 1.0) * ((per_serving_cal - target_cal) / target_cal) ** 2
        )

        # Nutrition objective: maximize completeness (negate for minimization)
        completeness = self._predictor.overall_completeness_score(nutrition, num_servings)
        nutrition_cost = -weights.get("nutrition_weight", 1.0) * completeness / 100.0

        # Protein objective
        per_serving_protein = nutrition.protein_g / num_servings
        protein_target = target_cal * weights.get("min_protein_pct", 20) / 100 / 4
        protein_cost = (
            weights.get("protein_weight", 1.0)
            * max(0, (protein_target - per_serving_protein) / protein_target) ** 2
        )

        # Fiber objective
        per_serving_fiber = nutrition.fiber_g / num_servings
        fiber_target = 7.0  # ~25% of daily 28g per meal
        fiber_cost = -weights.get("fiber_weight", 1.0) * min(1.0, per_serving_fiber / fiber_target)

        # Taste preservation: penalize deviation from original taste score
        taste_score = self._taste.recipe_taste_score(ingredient_names)
        taste_cost = weights.get("taste_weight", 1.0) * max(0, (original_taste - taste_score)) ** 2

        # Saturated fat penalty
        per_serving_sat_fat = nutrition.saturated_fat_g / num_servings
        sat_fat_penalty = (
            weights.get("sat_fat_penalty", 0.0) * max(0, (per_serving_sat_fat - 5.0) / 5.0) ** 2
        )

        # Sodium penalty
        per_serving_sodium = nutrition.sodium_mg / num_servings
        sodium_penalty = (
            weights.get("sodium_penalty", 0.0) * max(0, (per_serving_sodium - 500) / 500) ** 2
        )

        # Iron bonus (negative cost = reward)
        iron_bonus = 0.0
        if weights.get("iron_bonus", 0) > 0:
            per_serving_iron = nutrition.micro.iron_mg / num_servings
            iron_bonus = -weights["iron_bonus"] * min(1.0, per_serving_iron / 4.5)

        total = (
            cal_cost
            + nutrition_cost
            + protein_cost
            + fiber_cost
            + taste_cost
            + sat_fat_penalty
            + sodium_penalty
            + iron_bonus
        )
        return float(total)

    def _build_restriction_constraints(
        self,
        ingredient_names: list[str],
        restrictions: list[DietaryRestriction],
        num_servings: int,
    ) -> list[dict[str, Any]]:
        """Build scipy constraint dicts for dietary restrictions."""
        constraints: list[dict[str, Any]] = []

        for restriction in restrictions:
            if restriction == DietaryRestriction.LOW_SODIUM:

                def sodium_constraint(x: np.ndarray) -> float:
                    recipe = dict(zip(ingredient_names, x.tolist()))
                    n = self._db.compute_recipe_nutrition(recipe)
                    if n is None:
                        return -1.0
                    return 575.0 * num_servings - n.sodium_mg  # <575mg per serving

                constraints.append({"type": "ineq", "fun": sodium_constraint})

            elif restriction == DietaryRestriction.LOW_FAT:

                def fat_constraint(x: np.ndarray) -> float:
                    recipe = dict(zip(ingredient_names, x.tolist()))
                    n = self._db.compute_recipe_nutrition(recipe)
                    if n is None:
                        return -1.0
                    per_serving_fat = n.fat_g / num_servings
                    return 15.0 - per_serving_fat  # <15g fat per serving

                constraints.append({"type": "ineq", "fun": fat_constraint})

            elif restriction == DietaryRestriction.LOW_CARB:

                def carb_constraint(x: np.ndarray) -> float:
                    recipe = dict(zip(ingredient_names, x.tolist()))
                    n = self._db.compute_recipe_nutrition(recipe)
                    if n is None:
                        return -1.0
                    per_serving_carbs = n.carbs_g / num_servings
                    return 30.0 - per_serving_carbs  # <30g carbs per serving (keto-adjacent)

                constraints.append({"type": "ineq", "fun": carb_constraint})

            elif restriction == DietaryRestriction.HIGH_PROTEIN:

                def protein_constraint(x: np.ndarray) -> float:
                    recipe = dict(zip(ingredient_names, x.tolist()))
                    n = self._db.compute_recipe_nutrition(recipe)
                    if n is None:
                        return -1.0
                    per_serving_protein = n.protein_g / num_servings
                    return per_serving_protein - 20.0  # >20g protein per serving

                constraints.append({"type": "ineq", "fun": protein_constraint})

        return constraints

    def _suggest_substitutions(
        self,
        ingredient_names: list[str],
        goal: DietaryGoal,
        restrictions: list[DietaryRestriction],
    ) -> list[dict[str, str]]:
        """Suggest ingredient substitutions based on goal and restrictions."""
        subs: list[dict[str, str]] = []

        vegan = DietaryRestriction.VEGAN in restrictions
        gf = DietaryRestriction.GLUTEN_FREE in restrictions
        dairy_free = DietaryRestriction.DAIRY_FREE in restrictions

        exclude_allergens: tuple[Allergen, ...] = ()
        if DietaryRestriction.NUT_FREE in restrictions:
            exclude_allergens = (Allergen.TREE_NUTS, Allergen.PEANUTS)

        for name in ingredient_names:
            info = self._db.get(name)
            if info is None:
                continue

            # Check if current ingredient violates restrictions
            needs_sub = False
            reason = ""

            if vegan and not info.is_vegan:
                needs_sub = True
                reason = "Not vegan"
            elif gf and not info.is_gluten_free:
                needs_sub = True
                reason = "Contains gluten"
            elif dairy_free and Allergen.MILK in info.allergens:
                needs_sub = True
                reason = "Contains dairy"
            elif exclude_allergens and any(a in info.allergens for a in exclude_allergens):
                needs_sub = True
                reason = "Contains restricted allergen"

            # Goal-based substitutions
            if not needs_sub and goal == DietaryGoal.HEART_HEALTH:
                if info.nutrition.saturated_fat_g > 15:
                    needs_sub = True
                    reason = "High saturated fat (heart health goal)"
                elif info.nutrition.sodium_mg > 800:
                    needs_sub = True
                    reason = "High sodium (heart health goal)"

            if not needs_sub and goal == DietaryGoal.WEIGHT_LOSS:
                if info.nutrition.calories_kcal > 500 and info.nutrition.fiber_g < 3:
                    needs_sub = True
                    reason = "High calorie, low fiber (weight loss goal)"

            if needs_sub:
                candidates = self._db.find_substitutes(
                    name,
                    vegan_only=vegan,
                    gluten_free_only=gf,
                    exclude_allergens=exclude_allergens,
                )
                if candidates:
                    best = candidates[0]
                    subs.append(
                        {
                            "original": name,
                            "substitute": best.name,
                            "reason": reason,
                        }
                    )

        return subs

    def _generate_optimization_notes(
        self,
        orig: NutritionFacts,
        opt: NutritionFacts,
        goal: DietaryGoal,
        converged: bool,
        num_servings: int,
    ) -> list[str]:
        """Generate human-readable notes about the optimization."""
        notes: list[str] = []

        if not converged:
            notes.append("Optimization did not fully converge; results are approximate")

        # Calorie change
        if orig.calories_kcal > 0:
            cal_pct = (opt.calories_kcal - orig.calories_kcal) / orig.calories_kcal * 100
            if abs(cal_pct) > 5:
                direction = "reduced" if cal_pct < 0 else "increased"
                notes.append(
                    f"Calories {direction} by {abs(cal_pct):.1f}% "
                    f"({orig.calories_kcal / num_servings:.0f} -> "
                    f"{opt.calories_kcal / num_servings:.0f} kcal/serving)"
                )

        # Protein change
        if orig.protein_g > 0:
            prot_pct = (opt.protein_g - orig.protein_g) / orig.protein_g * 100
            if abs(prot_pct) > 10:
                direction = "increased" if prot_pct > 0 else "decreased"
                notes.append(
                    f"Protein {direction} by {abs(prot_pct):.1f}% "
                    f"({opt.protein_g / num_servings:.1f}g/serving)"
                )

        # Fiber change
        if orig.fiber_g > 0:
            fiber_pct = (opt.fiber_g - orig.fiber_g) / orig.fiber_g * 100
            if fiber_pct > 15:
                notes.append(
                    f"Fiber increased by {fiber_pct:.1f}% "
                    f"({opt.fiber_g / num_servings:.1f}g/serving)"
                )

        # Goal-specific notes
        if goal == DietaryGoal.HEART_HEALTH:
            notes.append(
                f"Saturated fat: {opt.saturated_fat_g / num_servings:.1f}g/serving "
                f"(AHA limit: <13g/day)"
            )
            notes.append(
                f"Sodium: {opt.sodium_mg / num_servings:.0f}mg/serving (AHA limit: <2300mg/day)"
            )

        if goal == DietaryGoal.MUSCLE_GAIN:
            cal_from_protein = opt.protein_g * 4
            total_cal = opt.calories_kcal if opt.calories_kcal > 0 else 1
            prot_pct_cal = cal_from_protein / total_cal * 100
            notes.append(
                f"Protein provides {prot_pct_cal:.1f}% of calories "
                f"({opt.protein_g / num_servings:.1f}g/serving)"
            )

        return notes

    def optimize_portion_size(
        self,
        ingredients: dict[str, float],
        target_calories: float,
        num_servings: int = 1,
    ) -> dict[str, float]:
        """Scale all ingredients proportionally to hit a calorie target.

        Simple linear scaling - maintains recipe proportions while
        adjusting total calories per serving.
        """
        nutrition = self._db.compute_recipe_nutrition(ingredients)
        if nutrition is None or nutrition.calories_kcal == 0:
            return ingredients

        total_target = target_calories * num_servings
        scale = total_target / nutrition.calories_kcal

        return {name: round(qty * scale, 1) for name, qty in ingredients.items()}

    def get_substitution_recommendations(
        self,
        ingredient_name: str,
        goal: DietaryGoal = DietaryGoal.BALANCED,
        restrictions: list[DietaryRestriction] | None = None,
    ) -> list[SubstitutionRecommendation]:
        """Get detailed substitution recommendations for an ingredient."""
        restrictions = restrictions or []
        original = self._db.get(ingredient_name)
        if original is None:
            return []

        vegan = DietaryRestriction.VEGAN in restrictions
        gf = DietaryRestriction.GLUTEN_FREE in restrictions
        exclude_allergens: tuple[Allergen, ...] = ()
        if DietaryRestriction.NUT_FREE in restrictions:
            exclude_allergens = (Allergen.TREE_NUTS, Allergen.PEANUTS)

        candidates = self._db.find_substitutes(
            ingredient_name,
            vegan_only=vegan,
            gluten_free_only=gf,
            exclude_allergens=exclude_allergens,
        )

        recommendations: list[SubstitutionRecommendation] = []
        for candidate in candidates[:5]:  # Top 5
            orig_n = original.nutrition
            cand_n = candidate.nutrition

            # Nutrition impact
            cal_diff = cand_n.calories_kcal - orig_n.calories_kcal
            prot_diff = cand_n.protein_g - orig_n.protein_g
            impact_parts = []
            if abs(cal_diff) > 20:
                impact_parts.append(f"{'+' if cal_diff > 0 else ''}{cal_diff:.0f} kcal")
            if abs(prot_diff) > 2:
                impact_parts.append(f"{'+' if prot_diff > 0 else ''}{prot_diff:.1f}g protein")
            nutrition_impact = ", ".join(impact_parts) if impact_parts else "Similar nutrition"

            # Taste impact
            taste_pair = self._taste.pairing_score(ingredient_name, candidate.name)
            if taste_pair > 0.7:
                taste_impact = "Excellent flavor compatibility"
            elif taste_pair > 0.5:
                taste_impact = "Good flavor match"
            else:
                taste_impact = "Different flavor profile - may change dish character"

            # Confidence based on category match + nutrition similarity
            cat_match = 1.0 if original.category == candidate.category else 0.5
            cal_sim = 1.0 - min(1.0, abs(cal_diff) / 300)
            confidence = round(0.6 * cat_match + 0.4 * cal_sim, 2)

            # Reason
            reason_parts = []
            if candidate.substitution_group == original.substitution_group:
                reason_parts.append("Same food group")
            if cand_n.calories_kcal < orig_n.calories_kcal:
                reason_parts.append("Lower calorie")
            if cand_n.protein_g > orig_n.protein_g:
                reason_parts.append("Higher protein")
            if cand_n.fiber_g > orig_n.fiber_g:
                reason_parts.append("More fiber")
            reason = "; ".join(reason_parts) if reason_parts else "Alternative option"

            recommendations.append(
                SubstitutionRecommendation(
                    original=ingredient_name,
                    substitute=candidate.name,
                    reason=reason,
                    nutrition_impact=nutrition_impact,
                    taste_impact=taste_impact,
                    confidence=confidence,
                )
            )

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations

    def generate_meal_plan(
        self,
        goal: DietaryGoal = DietaryGoal.BALANCED,
        restrictions: list[DietaryRestriction] | None = None,
        days: int = 7,
    ) -> WeeklyMealPlan:
        """Generate an optimized weekly meal plan.

        Creates balanced meals across the day and week, ensuring
        micronutrient variety and avoiding repetition.
        """
        restrictions = restrictions or []
        weights = GOAL_WEIGHTS[goal]
        target_daily_cal = weights.get("target_cal_per_serving", 500) * 3

        vegan = DietaryRestriction.VEGAN in restrictions
        gf = DietaryRestriction.GLUTEN_FREE in restrictions

        # Build ingredient pools by category
        proteins = self._get_valid_ingredients("protein", vegan, gf)
        grains = self._get_valid_ingredients("grain", vegan, gf)
        vegetables = self._get_valid_ingredients("vegetable", vegan, gf)
        fruits = self._get_valid_ingredients("fruit", vegan, gf)

        plan_days: list[MealPlanDay] = []
        rng = np.random.RandomState(42)  # Reproducible

        for day_idx in range(days):
            # Rotate through protein sources to ensure variety
            day_proteins = rng.choice(proteins, size=min(3, len(proteins)), replace=False).tolist()
            day_vegs = rng.choice(vegetables, size=min(4, len(vegetables)), replace=False).tolist()
            day_grain = rng.choice(grains) if grains else "brown rice cooked"
            day_fruit = rng.choice(fruits) if fruits else "banana"

            # Breakfast
            breakfast: dict[str, float] = {
                "oats rolled" if gf else "oats rolled": 60.0,
                day_fruit: 100.0,
            }
            if not vegan:
                breakfast["greek yogurt"] = 150.0
            else:
                breakfast["almond milk"] = 200.0
                breakfast["chia seeds"] = 15.0

            # Lunch
            lunch: dict[str, float] = {
                day_proteins[0]: 120.0,
                day_grain: 150.0,
                day_vegs[0]: 100.0,
                day_vegs[1] if len(day_vegs) > 1 else "tomatoes": 80.0,
                "olive oil": 10.0,
            }

            # Dinner
            dinner: dict[str, float] = {
                day_proteins[1] if len(day_proteins) > 1 else day_proteins[0]: 150.0,
                day_vegs[2] if len(day_vegs) > 2 else "broccoli": 120.0,
                day_vegs[3] if len(day_vegs) > 3 else "carrots": 80.0,
                "olive oil": 10.0,
                "garlic": 5.0,
            }
            if not gf:
                dinner["whole wheat bread"] = 50.0

            # Snacks
            snack: dict[str, float] = {"almonds": 30.0}
            if not vegan:
                snack["cottage cheese"] = 100.0
            else:
                snack["peanut butter"] = 20.0

            # Compute daily nutrition
            all_ingredients: dict[str, float] = {}
            for meal in [breakfast, lunch, dinner, snack]:
                for ing, qty in meal.items():
                    all_ingredients[ing] = all_ingredients.get(ing, 0) + qty

            daily_nutrition = self._db.compute_recipe_nutrition(all_ingredients)
            daily_completeness = 0.0
            if daily_nutrition:
                daily_completeness = self._predictor.overall_completeness_score(daily_nutrition, 1)

            plan_days.append(
                MealPlanDay(
                    breakfast=breakfast,
                    lunch=lunch,
                    dinner=dinner,
                    snacks=[snack],
                    daily_nutrition=daily_nutrition,
                    daily_completeness=daily_completeness,
                )
            )

        # Compute weekly averages
        valid_days = [d for d in plan_days if d.daily_nutrition is not None]
        avg_cal = (
            sum(d.daily_nutrition.calories_kcal for d in valid_days) / len(valid_days)
            if valid_days
            else 0
        )
        avg_comp = (
            sum(d.daily_completeness for d in valid_days) / len(valid_days) if valid_days else 0
        )

        notes = [
            f"Target: ~{target_daily_cal:.0f} kcal/day for {goal.value}",
            f"Average daily calories: {avg_cal:.0f} kcal",
            f"Average nutritional completeness: {avg_comp:.1f}/100",
        ]

        if vegan:
            notes.append(
                "Note: B12 supplementation recommended for vegan diets "
                "(or include fortified nutritional yeast)"
            )

        return WeeklyMealPlan(
            days=plan_days,
            weekly_avg_calories=round(avg_cal, 1),
            weekly_avg_completeness=round(avg_comp, 1),
            dietary_goal=goal,
            restrictions=restrictions,
            notes=notes,
        )

    def _get_valid_ingredients(
        self,
        category_hint: str,
        vegan: bool,
        gluten_free: bool,
    ) -> list[str]:
        """Get ingredients valid for the given restrictions."""
        from nutri_optimize.knowledge.nutrition_database import IngredientCategory

        category_map = {
            "protein": [
                IngredientCategory.PROTEIN_ANIMAL,
                IngredientCategory.PROTEIN_PLANT,
                IngredientCategory.LEGUME,
            ],
            "grain": [IngredientCategory.GRAIN],
            "vegetable": [IngredientCategory.VEGETABLE],
            "fruit": [IngredientCategory.FRUIT],
        }

        categories = category_map.get(category_hint, [])
        results: list[str] = []

        for cat in categories:
            for info in self._db.get_by_category(cat):
                if vegan and not info.is_vegan:
                    continue
                if gluten_free and not info.is_gluten_free:
                    continue
                results.append(info.name)

        return results if results else ["broccoli"]  # Fallback
