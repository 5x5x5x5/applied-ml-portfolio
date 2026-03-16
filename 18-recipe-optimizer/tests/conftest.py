"""Shared test fixtures for NutriOptimize tests."""

from __future__ import annotations

import pytest

from nutri_optimize.knowledge.nutrition_database import NutritionDatabase
from nutri_optimize.knowledge.taste_model import TasteModel
from nutri_optimize.models.nutrition_predictor import NutritionPredictor
from nutri_optimize.optimizer.recipe_optimizer import RecipeOptimizer


@pytest.fixture
def nutrition_db() -> NutritionDatabase:
    """Shared NutritionDatabase instance."""
    return NutritionDatabase()


@pytest.fixture
def taste_model() -> TasteModel:
    """Shared TasteModel instance."""
    return TasteModel()


@pytest.fixture
def predictor(nutrition_db: NutritionDatabase) -> NutritionPredictor:
    """Shared NutritionPredictor instance."""
    return NutritionPredictor(nutrition_db)


@pytest.fixture
def optimizer(
    nutrition_db: NutritionDatabase,
    taste_model: TasteModel,
    predictor: NutritionPredictor,
) -> RecipeOptimizer:
    """Shared RecipeOptimizer instance."""
    return RecipeOptimizer(nutrition_db, taste_model, predictor)


@pytest.fixture
def sample_recipe() -> dict[str, float]:
    """A sample chicken + rice + broccoli recipe in grams."""
    return {
        "chicken breast": 200.0,
        "brown rice cooked": 300.0,
        "broccoli": 150.0,
        "olive oil": 15.0,
        "garlic": 5.0,
    }


@pytest.fixture
def vegan_recipe() -> dict[str, float]:
    """A sample vegan recipe in grams."""
    return {
        "tofu firm": 200.0,
        "quinoa cooked": 250.0,
        "spinach raw": 100.0,
        "bell pepper red": 100.0,
        "olive oil": 10.0,
        "lemon juice": 15.0,
    }


@pytest.fixture
def high_calorie_recipe() -> dict[str, float]:
    """A calorie-dense recipe for testing weight loss optimization."""
    return {
        "ground beef 85%": 300.0,
        "cheddar cheese": 60.0,
        "white bread": 100.0,
        "butter": 20.0,
        "bacon": 50.0,
    }
