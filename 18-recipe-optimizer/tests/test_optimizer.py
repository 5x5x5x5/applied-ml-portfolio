"""Tests for the recipe optimizer."""

from __future__ import annotations

import pytest

from nutri_optimize.optimizer.recipe_optimizer import (
    DietaryGoal,
    DietaryRestriction,
    RecipeOptimizer,
)


class TestRecipeOptimizer:
    """Tests for multi-objective recipe optimization."""

    def test_optimize_returns_result(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Optimization should return a valid result."""
        result = optimizer.optimize_recipe(sample_recipe, goal=DietaryGoal.BALANCED)
        assert result.success or len(result.optimization_notes) > 0
        assert result.optimized_ingredients is not None
        assert len(result.optimized_ingredients) == len(sample_recipe)

    def test_optimize_preserves_ingredients(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Optimization should keep the same ingredient set."""
        result = optimizer.optimize_recipe(sample_recipe)
        assert set(result.optimized_ingredients.keys()) == set(sample_recipe.keys())

    def test_optimize_positive_quantities(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """All optimized quantities should be positive."""
        result = optimizer.optimize_recipe(sample_recipe)
        assert all(v > 0 for v in result.optimized_ingredients.values())

    def test_weight_loss_reduces_calories(
        self,
        optimizer: RecipeOptimizer,
        high_calorie_recipe: dict[str, float],
    ) -> None:
        """Weight loss goal should tend to reduce calories."""
        result = optimizer.optimize_recipe(high_calorie_recipe, goal=DietaryGoal.WEIGHT_LOSS)
        # The optimized version should have fewer or similar total calories
        assert result.optimized_nutrition.calories_kcal <= (
            result.original_nutrition.calories_kcal * 1.1  # Allow 10% tolerance
        )

    def test_muscle_gain_maintains_protein(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Muscle gain goal should maintain or increase protein."""
        result = optimizer.optimize_recipe(sample_recipe, goal=DietaryGoal.MUSCLE_GAIN)
        # Protein should not decrease significantly
        assert result.optimized_nutrition.protein_g >= (
            result.original_nutrition.protein_g * 0.8  # Allow 20% tolerance
        )

    def test_heart_health_notes(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Heart health goal should generate relevant notes."""
        result = optimizer.optimize_recipe(sample_recipe, goal=DietaryGoal.HEART_HEALTH)
        sat_fat_notes = [n for n in result.optimization_notes if "saturated" in n.lower()]
        sodium_notes = [n for n in result.optimization_notes if "sodium" in n.lower()]
        # At least one heart-health relevant note should be present
        assert len(sat_fat_notes) + len(sodium_notes) > 0

    def test_vegan_substitutions(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Vegan restriction should suggest substitutions for animal products."""
        result = optimizer.optimize_recipe(
            sample_recipe,
            restrictions=[DietaryRestriction.VEGAN],
        )
        # Should suggest replacing chicken breast
        sub_originals = [s["original"] for s in result.substitutions]
        assert "chicken breast" in sub_originals

    def test_gluten_free_substitutions(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Gluten-free restriction should flag wheat-containing ingredients."""
        recipe = {
            "pasta cooked": 200.0,
            "tomato sauce": 150.0,
            "parmesan": 30.0,
        }
        result = optimizer.optimize_recipe(
            recipe,
            restrictions=[DietaryRestriction.GLUTEN_FREE],
        )
        sub_originals = [s["original"] for s in result.substitutions]
        assert "pasta cooked" in sub_originals

    def test_taste_score_computed(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Both original and optimized taste scores should be computed."""
        result = optimizer.optimize_recipe(sample_recipe)
        assert 0.0 <= result.original_taste_score <= 1.0
        assert 0.0 <= result.optimized_taste_score <= 1.0

    def test_completeness_computed(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Completeness scores should be computed."""
        result = optimizer.optimize_recipe(sample_recipe)
        assert 0 <= result.original_completeness <= 100
        assert 0 <= result.optimized_completeness <= 100

    def test_optimize_unknown_ingredients_fails_gracefully(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Unknown ingredients should be handled without crashing."""
        recipe = {"unicorn steak": 200.0}
        result = optimizer.optimize_recipe(recipe)
        assert not result.success

    def test_portion_size_optimization(
        self,
        optimizer: RecipeOptimizer,
        sample_recipe: dict[str, float],
    ) -> None:
        """Portion optimization should scale to hit calorie target."""
        scaled = optimizer.optimize_portion_size(sample_recipe, target_calories=400, num_servings=1)
        assert all(v > 0 for v in scaled.values())
        # All ingredients should scale by the same factor
        factors = [scaled[k] / sample_recipe[k] for k in sample_recipe if sample_recipe[k] > 0]
        assert max(factors) == pytest.approx(min(factors), rel=0.01)


class TestSubstitutionRecommendations:
    """Tests for ingredient substitution recommendations."""

    def test_substitution_returns_results(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Should return substitution recommendations."""
        recs = optimizer.get_substitution_recommendations("chicken breast")
        assert len(recs) > 0

    def test_substitution_has_required_fields(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Each recommendation should have all required fields."""
        recs = optimizer.get_substitution_recommendations("butter")
        for rec in recs:
            assert rec.original == "butter"
            assert rec.substitute != "butter"
            assert rec.reason
            assert rec.nutrition_impact
            assert rec.taste_impact
            assert 0.0 <= rec.confidence <= 1.0

    def test_substitution_respects_vegan(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Vegan restriction should only return plant-based substitutes."""
        recs = optimizer.get_substitution_recommendations(
            "whole milk",
            restrictions=[DietaryRestriction.VEGAN],
        )
        for rec in recs:
            info = optimizer._db.get(rec.substitute)
            if info is not None:
                assert info.is_vegan, f"{rec.substitute} is not vegan"

    def test_substitution_unknown_ingredient(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Unknown ingredients should return empty list."""
        recs = optimizer.get_substitution_recommendations("imaginary food")
        assert recs == []


class TestMealPlan:
    """Tests for meal plan generation."""

    def test_meal_plan_has_correct_days(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Meal plan should have the requested number of days."""
        plan = optimizer.generate_meal_plan(days=3)
        assert len(plan.days) == 3

    def test_meal_plan_seven_days(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Default 7-day plan should have 7 days."""
        plan = optimizer.generate_meal_plan()
        assert len(plan.days) == 7

    def test_meal_plan_has_all_meals(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Each day should have breakfast, lunch, and dinner."""
        plan = optimizer.generate_meal_plan(days=1)
        day = plan.days[0]
        assert len(day.breakfast) > 0
        assert len(day.lunch) > 0
        assert len(day.dinner) > 0

    def test_meal_plan_vegan(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Vegan meal plan should not contain animal products."""
        plan = optimizer.generate_meal_plan(restrictions=[DietaryRestriction.VEGAN], days=3)
        for day in plan.days:
            all_ingredients = (
                list(day.breakfast.keys()) + list(day.lunch.keys()) + list(day.dinner.keys())
            )
            for ing_name in all_ingredients:
                info = optimizer._db.get(ing_name)
                if info is not None:
                    assert info.is_vegan, f"Non-vegan ingredient {ing_name} in vegan meal plan"

    def test_meal_plan_has_completeness(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Meal plan should compute daily completeness scores."""
        plan = optimizer.generate_meal_plan(days=1)
        assert plan.weekly_avg_completeness >= 0
        assert plan.days[0].daily_completeness >= 0

    def test_meal_plan_has_notes(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Meal plan should include informational notes."""
        plan = optimizer.generate_meal_plan()
        assert len(plan.notes) > 0

    def test_meal_plan_calorie_average(
        self,
        optimizer: RecipeOptimizer,
    ) -> None:
        """Weekly average calories should be positive."""
        plan = optimizer.generate_meal_plan()
        assert plan.weekly_avg_calories > 0
