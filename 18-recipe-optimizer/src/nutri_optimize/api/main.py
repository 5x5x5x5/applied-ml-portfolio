"""FastAPI application for the NutriOptimize recipe optimizer.

Provides REST endpoints for recipe analysis, optimization,
ingredient substitution, and meal plan generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nutri_optimize.knowledge.nutrition_database import NutritionDatabase
from nutri_optimize.knowledge.taste_model import TasteModel
from nutri_optimize.models.nutrition_predictor import NutritionPredictor
from nutri_optimize.optimizer.recipe_optimizer import (
    DietaryGoal,
    DietaryRestriction,
    RecipeOptimizer,
)

logger = logging.getLogger(__name__)

# Initialize shared instances
db = NutritionDatabase()
taste_model = TasteModel()
predictor = NutritionPredictor(db)
optimizer = RecipeOptimizer(db, taste_model, predictor)

app = FastAPI(
    title="NutriOptimize",
    description=(
        "AI-powered Recipe Optimizer that modifies recipes for optimal nutrition "
        "while preserving taste. Combines nutritional biochemistry with ML."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---- Request/Response Models ----


class IngredientInput(BaseModel):
    """Single ingredient with quantity."""

    name: str = Field(..., description="Ingredient name", examples=["chicken breast"])
    grams: float = Field(..., gt=0, description="Quantity in grams", examples=[200.0])


class RecipeInput(BaseModel):
    """Recipe to optimize or analyze."""

    ingredients: list[IngredientInput] = Field(
        ..., min_length=1, description="List of ingredients with quantities"
    )
    num_servings: int = Field(default=4, ge=1, le=20, description="Number of servings")


class OptimizeRequest(RecipeInput):
    """Request to optimize a recipe."""

    goal: str = Field(
        default="balanced",
        description="Optimization goal",
        examples=["balanced", "weight_loss", "muscle_gain", "heart_health"],
    )
    restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions",
        examples=[["vegan", "gluten_free"]],
    )


class SubstituteRequest(BaseModel):
    """Request for ingredient substitutions."""

    ingredient: str = Field(..., description="Ingredient to find substitutes for")
    goal: str = Field(default="balanced", description="Optimization goal")
    restrictions: list[str] = Field(default_factory=list)


class MealPlanRequest(BaseModel):
    """Request for meal plan generation."""

    goal: str = Field(default="balanced", description="Dietary goal")
    restrictions: list[str] = Field(default_factory=list)
    days: int = Field(default=7, ge=1, le=14, description="Number of days")


class NutritionResponse(BaseModel):
    """Nutritional information response."""

    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    saturated_fat_g: float
    sodium_mg: float
    micronutrients: dict[str, float]


class AnalysisResponse(BaseModel):
    """Full nutritional analysis response."""

    per_serving: NutritionResponse
    total: NutritionResponse
    num_servings: int
    calorie_breakdown: dict[str, float]
    micronutrient_scores: dict[str, float]
    completeness_score: float
    allergens: list[str]
    warnings: list[str]
    bioavailability_notes: list[str]


class OptimizeResponse(BaseModel):
    """Recipe optimization response."""

    original: NutritionResponse
    optimized: NutritionResponse
    original_ingredients: dict[str, float]
    optimized_ingredients: dict[str, float]
    original_taste_score: float
    optimized_taste_score: float
    original_completeness: float
    optimized_completeness: float
    calories_change_pct: float
    substitutions: list[dict[str, str]]
    notes: list[str]
    success: bool


class SubstitutionResponse(BaseModel):
    """Ingredient substitution response."""

    original: str
    substitutes: list[dict[str, Any]]


class MealResponse(BaseModel):
    """Single meal in a plan."""

    ingredients: dict[str, float]


class DayPlanResponse(BaseModel):
    """Single day in a meal plan."""

    breakfast: dict[str, float]
    lunch: dict[str, float]
    dinner: dict[str, float]
    snacks: list[dict[str, float]]
    daily_calories: float
    daily_completeness: float


class MealPlanResponse(BaseModel):
    """Weekly meal plan response."""

    days: list[DayPlanResponse]
    weekly_avg_calories: float
    weekly_avg_completeness: float
    goal: str
    restrictions: list[str]
    notes: list[str]


class IngredientSearchResult(BaseModel):
    """Ingredient search result."""

    name: str
    category: str
    calories_per_100g: float
    protein_per_100g: float
    is_vegan: bool
    is_gluten_free: bool
    allergens: list[str]


# ---- Helper functions ----


def _nutrition_to_response(n: Any) -> NutritionResponse:
    """Convert NutritionFacts to API response."""
    return NutritionResponse(
        calories=n.calories_kcal,
        protein_g=n.protein_g,
        carbs_g=n.carbs_g,
        fat_g=n.fat_g,
        fiber_g=n.fiber_g,
        sugar_g=n.sugar_g,
        saturated_fat_g=n.saturated_fat_g,
        sodium_mg=n.sodium_mg,
        micronutrients={
            "iron_mg": n.micro.iron_mg,
            "calcium_mg": n.micro.calcium_mg,
            "vitamin_c_mg": n.micro.vitamin_c_mg,
            "vitamin_b12_mcg": n.micro.vitamin_b12_mcg,
            "folate_mcg": n.micro.folate_mcg,
            "potassium_mg": n.micro.potassium_mg,
            "magnesium_mg": n.micro.magnesium_mg,
            "zinc_mg": n.micro.zinc_mg,
            "vitamin_a_mcg": n.micro.vitamin_a_mcg,
            "vitamin_d_mcg": n.micro.vitamin_d_mcg,
        },
    )


def _parse_goal(goal_str: str) -> DietaryGoal:
    """Parse goal string to enum."""
    try:
        return DietaryGoal(goal_str.lower())
    except ValueError:
        return DietaryGoal.BALANCED


def _parse_restrictions(restriction_strs: list[str]) -> list[DietaryRestriction]:
    """Parse restriction strings to enums."""
    result: list[DietaryRestriction] = []
    for r in restriction_strs:
        try:
            result.append(DietaryRestriction(r.lower()))
        except ValueError:
            logger.warning("Unknown restriction: %s", r)
    return result


# ---- Endpoints ----


@app.get("/")
async def root() -> FileResponse:
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_recipe(request: RecipeInput) -> AnalysisResponse:
    """Analyze nutritional content of a recipe.

    Returns detailed nutritional breakdown including macros, micros,
    allergen detection, DRI completeness scoring, and bioavailability notes.
    """
    ingredients = {ing.name: ing.grams for ing in request.ingredients}

    analysis = predictor.analyze_recipe(ingredients, request.num_servings)
    if analysis is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown ingredient(s). Use /ingredients/search to find valid names.",
        )

    return AnalysisResponse(
        per_serving=_nutrition_to_response(analysis.per_serving),
        total=_nutrition_to_response(analysis.total_nutrition),
        num_servings=analysis.num_servings,
        calorie_breakdown=analysis.calorie_breakdown,
        micronutrient_scores=analysis.micronutrient_scores,
        completeness_score=analysis.completeness_score,
        allergens=[a.value for a in analysis.allergens_detected],
        warnings=analysis.warnings,
        bioavailability_notes=analysis.bioavailability_notes,
    )


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_recipe(request: OptimizeRequest) -> OptimizeResponse:
    """Optimize a recipe for nutritional goals while preserving taste.

    Uses multi-objective optimization (SLSQP) to adjust ingredient quantities.
    Supports goals: weight_loss, muscle_gain, heart_health, balanced, high_fiber, low_sodium.
    """
    ingredients = {ing.name: ing.grams for ing in request.ingredients}
    goal = _parse_goal(request.goal)
    restrictions = _parse_restrictions(request.restrictions)

    result = optimizer.optimize_recipe(
        ingredients=ingredients,
        goal=goal,
        restrictions=restrictions,
        num_servings=request.num_servings,
    )

    return OptimizeResponse(
        original=_nutrition_to_response(result.original_nutrition),
        optimized=_nutrition_to_response(result.optimized_nutrition),
        original_ingredients=result.original_ingredients,
        optimized_ingredients=result.optimized_ingredients,
        original_taste_score=result.original_taste_score,
        optimized_taste_score=result.optimized_taste_score,
        original_completeness=result.original_completeness,
        optimized_completeness=result.optimized_completeness,
        calories_change_pct=result.calories_change_pct,
        substitutions=result.substitutions,
        notes=result.optimization_notes,
        success=result.success,
    )


@app.post("/substitute", response_model=SubstitutionResponse)
async def get_substitutions(request: SubstituteRequest) -> SubstitutionResponse:
    """Get ingredient substitution recommendations.

    Returns ranked alternatives with nutrition impact, taste compatibility,
    and confidence scores.
    """
    goal = _parse_goal(request.goal)
    restrictions = _parse_restrictions(request.restrictions)

    recommendations = optimizer.get_substitution_recommendations(
        request.ingredient, goal, restrictions
    )

    subs = [
        {
            "name": r.substitute,
            "reason": r.reason,
            "nutrition_impact": r.nutrition_impact,
            "taste_impact": r.taste_impact,
            "confidence": r.confidence,
        }
        for r in recommendations
    ]

    return SubstitutionResponse(
        original=request.ingredient,
        substitutes=subs,
    )


@app.post("/meal-plan", response_model=MealPlanResponse)
async def generate_meal_plan(request: MealPlanRequest) -> MealPlanResponse:
    """Generate an optimized weekly meal plan.

    Creates balanced meals across the day and week, ensuring
    micronutrient variety and calorie targets based on the dietary goal.
    """
    goal = _parse_goal(request.goal)
    restrictions = _parse_restrictions(request.restrictions)

    plan = optimizer.generate_meal_plan(
        goal=goal,
        restrictions=restrictions,
        days=request.days,
    )

    day_responses: list[DayPlanResponse] = []
    for day in plan.days:
        daily_cal = day.daily_nutrition.calories_kcal if day.daily_nutrition else 0
        day_responses.append(
            DayPlanResponse(
                breakfast=day.breakfast,
                lunch=day.lunch,
                dinner=day.dinner,
                snacks=day.snacks,
                daily_calories=daily_cal,
                daily_completeness=day.daily_completeness,
            )
        )

    return MealPlanResponse(
        days=day_responses,
        weekly_avg_calories=plan.weekly_avg_calories,
        weekly_avg_completeness=plan.weekly_avg_completeness,
        goal=plan.dietary_goal.value,
        restrictions=[r.value for r in plan.restrictions],
        notes=plan.notes,
    )


@app.get("/ingredients/search", response_model=list[IngredientSearchResult])
async def search_ingredients(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[IngredientSearchResult]:
    """Search the ingredient database.

    Returns matching ingredients with basic nutritional info.
    Use these names in recipe analysis and optimization requests.
    """
    results = db.search(q, limit=limit)

    return [
        IngredientSearchResult(
            name=r.name,
            category=r.category.value,
            calories_per_100g=r.nutrition.calories_kcal,
            protein_per_100g=r.nutrition.protein_g,
            is_vegan=r.is_vegan,
            is_gluten_free=r.is_gluten_free,
            allergens=[a.value for a in r.allergens],
        )
        for r in results
    ]


@app.get("/ingredients/count")
async def ingredient_count() -> dict[str, int]:
    """Get the number of ingredients in the database."""
    return {"count": db.ingredient_count}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}
