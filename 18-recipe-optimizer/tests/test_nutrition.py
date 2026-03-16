"""Tests for the nutrition database and predictor models."""

from __future__ import annotations

import pytest

from nutri_optimize.knowledge.nutrition_database import (
    Allergen,
    IngredientCategory,
    NutritionDatabase,
    NutritionFacts,
)
from nutri_optimize.knowledge.taste_model import TasteModel
from nutri_optimize.models.nutrition_predictor import NutritionPredictor


class TestNutritionDatabase:
    """Tests for the nutrition database."""

    def test_database_has_minimum_ingredients(self, nutrition_db: NutritionDatabase) -> None:
        """Database should contain 100+ ingredients."""
        assert nutrition_db.ingredient_count >= 100

    def test_lookup_known_ingredient(self, nutrition_db: NutritionDatabase) -> None:
        """Should find a known ingredient by name."""
        chicken = nutrition_db.get("chicken breast")
        assert chicken is not None
        assert chicken.name == "chicken breast"
        assert chicken.category == IngredientCategory.PROTEIN_ANIMAL

    def test_lookup_case_insensitive(self, nutrition_db: NutritionDatabase) -> None:
        """Lookup should be case-insensitive."""
        result = nutrition_db.get("Chicken Breast")
        assert result is not None
        assert result.name == "chicken breast"

    def test_lookup_unknown_returns_none(self, nutrition_db: NutritionDatabase) -> None:
        """Unknown ingredients should return None."""
        assert nutrition_db.get("unicorn meat") is None

    def test_chicken_breast_nutrition_values(self, nutrition_db: NutritionDatabase) -> None:
        """Chicken breast should have realistic USDA values."""
        chicken = nutrition_db.get("chicken breast")
        assert chicken is not None
        n = chicken.nutrition
        # USDA: chicken breast ~165 kcal, 31g protein, 3.6g fat per 100g
        assert 155 <= n.calories_kcal <= 175
        assert 28 <= n.protein_g <= 34
        assert 2 <= n.fat_g <= 6
        assert n.carbs_g < 2  # Chicken has negligible carbs

    def test_salmon_is_not_vegan(self, nutrition_db: NutritionDatabase) -> None:
        """Animal products should not be marked vegan."""
        salmon = nutrition_db.get("salmon")
        assert salmon is not None
        assert not salmon.is_vegan

    def test_broccoli_is_vegan_and_gluten_free(self, nutrition_db: NutritionDatabase) -> None:
        """Vegetables should be vegan and gluten-free."""
        broccoli = nutrition_db.get("broccoli")
        assert broccoli is not None
        assert broccoli.is_vegan
        assert broccoli.is_gluten_free

    def test_wheat_bread_has_allergens(self, nutrition_db: NutritionDatabase) -> None:
        """Wheat bread should flag wheat allergen."""
        bread = nutrition_db.get("whole wheat bread")
        assert bread is not None
        assert Allergen.WHEAT in bread.allergens
        assert not bread.is_gluten_free

    def test_eggs_allergen(self, nutrition_db: NutritionDatabase) -> None:
        """Eggs should flag egg allergen."""
        eggs = nutrition_db.get("eggs")
        assert eggs is not None
        assert Allergen.EGGS in eggs.allergens

    def test_search_partial_match(self, nutrition_db: NutritionDatabase) -> None:
        """Search should find ingredients by partial name."""
        results = nutrition_db.search("chick")
        names = [r.name for r in results]
        assert any("chicken" in n for n in names)

    def test_search_returns_limited_results(self, nutrition_db: NutritionDatabase) -> None:
        """Search should respect the limit parameter."""
        results = nutrition_db.search("a", limit=5)
        assert len(results) <= 5

    def test_get_by_category(self, nutrition_db: NutritionDatabase) -> None:
        """Should retrieve ingredients by category."""
        veggies = nutrition_db.get_by_category(IngredientCategory.VEGETABLE)
        assert len(veggies) >= 10
        assert all(v.category == IngredientCategory.VEGETABLE for v in veggies)

    def test_find_substitutes_basic(self, nutrition_db: NutritionDatabase) -> None:
        """Should find substitutes for an ingredient."""
        subs = nutrition_db.find_substitutes("chicken breast")
        assert len(subs) > 0
        # Substitutes should not include the original
        assert all(s.name != "chicken breast" for s in subs)

    def test_find_substitutes_vegan_filter(self, nutrition_db: NutritionDatabase) -> None:
        """Vegan filter should exclude animal products."""
        subs = nutrition_db.find_substitutes("chicken breast", vegan_only=True)
        assert all(s.is_vegan for s in subs)

    def test_find_substitutes_gluten_free_filter(self, nutrition_db: NutritionDatabase) -> None:
        """Gluten-free filter should exclude gluten-containing items."""
        subs = nutrition_db.find_substitutes("pasta cooked", gluten_free_only=True)
        assert all(s.is_gluten_free for s in subs)

    def test_compute_recipe_nutrition(self, nutrition_db: NutritionDatabase) -> None:
        """Should compute aggregate nutrition for a recipe."""
        recipe = {"chicken breast": 200.0, "brown rice cooked": 300.0}
        result = nutrition_db.compute_recipe_nutrition(recipe)
        assert result is not None
        # 200g chicken (~330 kcal) + 300g brown rice (~369 kcal) ~ 700 kcal
        assert 650 <= result.calories_kcal <= 750
        assert result.protein_g > 50  # Chicken alone has ~62g protein

    def test_compute_recipe_unknown_ingredient(self, nutrition_db: NutritionDatabase) -> None:
        """Should return None when an ingredient is unknown."""
        recipe = {"chicken breast": 200.0, "mystery ingredient": 100.0}
        result = nutrition_db.compute_recipe_nutrition(recipe)
        assert result is None

    def test_allergen_detection(self, nutrition_db: NutritionDatabase) -> None:
        """Should detect all allergens for an ingredient."""
        allergens = nutrition_db.get_all_allergens_for("soy sauce")
        assert Allergen.SOY in allergens
        assert Allergen.WHEAT in allergens

    def test_calories_follow_atwater(self, nutrition_db: NutritionDatabase) -> None:
        """Calorie values should roughly follow Atwater factors."""
        olive_oil = nutrition_db.get("olive oil")
        assert olive_oil is not None
        # Pure fat: 100g * 9 kcal/g = 900 kcal (USDA says 884)
        assert 850 <= olive_oil.nutrition.calories_kcal <= 900

    def test_micronutrients_present(self, nutrition_db: NutritionDatabase) -> None:
        """Key ingredients should have micronutrient data."""
        spinach = nutrition_db.get("spinach raw")
        assert spinach is not None
        micro = spinach.nutrition.micro
        assert micro.iron_mg > 2.0  # Spinach is iron-rich
        assert micro.folate_mcg > 100  # Spinach is folate-rich
        assert micro.vitamin_a_mcg > 400  # Spinach is vitamin A-rich


class TestNutritionPredictor:
    """Tests for the ML nutrition predictor."""

    def test_calorie_estimation_atwater(self, predictor: NutritionPredictor) -> None:
        """Atwater estimation should be consistent with biochemistry."""
        est = predictor.estimate_calories(protein_g=31.0, carbs_g=0.0, fat_g=3.6, fiber_g=0.0)
        # 31*4 + 3.6*9 = 124 + 32.4 = 156.4
        assert 155 <= est.total_kcal <= 158
        assert est.protein_pct > 75  # Mostly protein

    def test_calorie_estimation_balanced(self, predictor: NutritionPredictor) -> None:
        """Balanced macros should give reasonable calorie split."""
        est = predictor.estimate_calories(protein_g=20.0, carbs_g=50.0, fat_g=10.0, fiber_g=5.0)
        assert est.total_kcal > 0
        assert est.protein_pct + est.carbs_pct + est.fat_pct == pytest.approx(100.0, abs=0.5)

    def test_ml_calorie_prediction(self, predictor: NutritionPredictor) -> None:
        """ML model should produce reasonable calorie predictions."""
        predicted = predictor.predict_calories_ml(
            protein_g=31.0, carbs_g=0.0, fat_g=3.6, fiber_g=0.0, water_g=65.0
        )
        # Should be in the ballpark of chicken breast (165 kcal)
        assert 100 <= predicted <= 250

    def test_micronutrient_scoring(self, predictor: NutritionPredictor) -> None:
        """Micronutrient scoring should return % DRI values."""
        nutrition = NutritionFacts(
            calories_kcal=500,
            protein_g=30.0,
            carbs_g=50.0,
            fat_g=15.0,
            fiber_g=10.0,
        )
        scores = predictor.score_micronutrient_completeness(nutrition, num_servings=1)
        assert "Iron" in scores
        assert "Calcium" in scores
        assert "Protein" in scores
        assert all(0 <= v <= 200 for v in scores.values())

    def test_completeness_score_range(self, predictor: NutritionPredictor) -> None:
        """Completeness score should be between 0 and 100."""
        nutrition = NutritionFacts(
            calories_kcal=500,
            protein_g=30.0,
            carbs_g=50.0,
            fat_g=15.0,
            fiber_g=10.0,
        )
        score = predictor.overall_completeness_score(nutrition)
        assert 0 <= score <= 100

    def test_allergen_detection(self, predictor: NutritionPredictor) -> None:
        """Should detect allergens from ingredient list."""
        allergens = predictor.detect_allergens(["salmon", "whole wheat bread", "eggs"])
        assert Allergen.FISH in allergens
        assert Allergen.WHEAT in allergens
        assert Allergen.EGGS in allergens

    def test_recipe_analysis(
        self,
        predictor: NutritionPredictor,
        sample_recipe: dict[str, float],
    ) -> None:
        """Full recipe analysis should return complete results."""
        analysis = predictor.analyze_recipe(sample_recipe, num_servings=4)
        assert analysis is not None
        assert analysis.per_serving.calories_kcal > 0
        assert analysis.per_serving.protein_g > 0
        assert analysis.num_servings == 4
        assert 0 <= analysis.completeness_score <= 100
        assert len(analysis.calorie_breakdown) == 3

    def test_recipe_analysis_unknown_ingredient(self, predictor: NutritionPredictor) -> None:
        """Analysis should fail gracefully for unknown ingredients."""
        analysis = predictor.analyze_recipe({"mystery food": 100.0})
        assert analysis is None

    def test_bioavailability_notes(
        self,
        predictor: NutritionPredictor,
    ) -> None:
        """Should generate bioavailability interaction notes."""
        recipe = {
            "spinach raw": 100.0,
            "lemon juice": 30.0,  # Vitamin C source
        }
        analysis = predictor.analyze_recipe(recipe, num_servings=1)
        assert analysis is not None
        # Should note vitamin C + iron synergy
        synergy_notes = [n for n in analysis.bioavailability_notes if "iron" in n.lower()]
        assert len(synergy_notes) > 0


class TestTasteModel:
    """Tests for the taste compatibility model."""

    def test_great_pairing_scores_high(self, taste_model: TasteModel) -> None:
        """Known great pairings should score highly."""
        score = taste_model.pairing_score("tomatoes", "basil dried")
        assert score >= 0.8

    def test_bad_pairing_scores_low(self, taste_model: TasteModel) -> None:
        """Known bad pairings should score low."""
        score = taste_model.pairing_score("salmon", "cinnamon ground")
        assert score < 0.3

    def test_unknown_pairing_neutral(self, taste_model: TasteModel) -> None:
        """Unknown ingredients should get neutral score."""
        score = taste_model.pairing_score("unknown_food_1", "unknown_food_2")
        assert score == pytest.approx(0.5)

    def test_recipe_taste_score_range(self, taste_model: TasteModel) -> None:
        """Recipe taste score should be 0.0 to 1.0."""
        ingredients = ["chicken breast", "garlic", "olive oil", "lemon juice"]
        score = taste_model.recipe_taste_score(ingredients)
        assert 0.0 <= score <= 1.0

    def test_good_recipe_scores_well(self, taste_model: TasteModel) -> None:
        """A classic flavor combination should score well."""
        ingredients = ["tomatoes", "mozzarella", "basil dried", "olive oil"]
        score = taste_model.recipe_taste_score(ingredients)
        assert score > 0.5

    def test_flavor_enhancer_suggestions(self, taste_model: TasteModel) -> None:
        """Should suggest enhancers for bland recipes."""
        bland_recipe = ["white rice cooked", "chicken breast"]
        suggestions = taste_model.suggest_flavor_enhancers(bland_recipe)
        assert len(suggestions) > 0

    def test_flavor_profile_similarity(self, taste_model: TasteModel) -> None:
        """Similar ingredients should have similar flavor profiles."""
        profile_a = taste_model.get_profile("chicken breast")
        profile_b = taste_model.get_profile("turkey breast")
        assert profile_a is not None and profile_b is not None
        sim = profile_a.flavor.similarity(profile_b.flavor)
        assert sim > 0.5  # Both are mild umami proteins
