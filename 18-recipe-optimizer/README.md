# NutriOptimize - AI-Powered Recipe Optimizer

Modify recipes for optimal nutrition while preserving taste. Combines pharmaceutical sciences knowledge (nutrition, biochemistry, bioavailability) with machine learning optimization.

## What It Does

- **Nutritional Analysis**: Complete macro/micronutrient breakdown with DRI (Daily Reference Intake) scoring
- **Recipe Optimization**: Multi-objective optimization using scipy SLSQP to balance nutrition, calories, and taste
- **Ingredient Substitution**: Smart alternatives respecting dietary restrictions (vegan, gluten-free, allergen-free)
- **Meal Planning**: Weekly meal plans optimized for nutritional completeness and variety
- **Allergen Detection**: FDA Big 9 allergen identification from ingredient lists
- **Bioavailability Insights**: Notes on nutrient interactions (e.g., vitamin C + iron synergy, curcumin + piperine)

## Nutritional Science Highlights

- **110+ ingredients** with USDA FoodData Central-calibrated nutritional values
- **Atwater factor** calorie estimation (protein=4, carbs=4, fat=9, fiber=2 kcal/g)
- **Bioavailability modeling**: Accounts for oxalate-calcium interactions, heme vs non-heme iron absorption, fat-soluble vitamin requirements
- **Flavor pairing theory**: Taste compatibility based on shared volatile organic compounds
- **DRI scoring**: Weighted harmonic mean reflecting Liebig's Law of the Minimum

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Run the API server
uv run uvicorn nutri_optimize.api.main:app --reload

# Open http://localhost:8000 in your browser

# Run tests
uv run pytest
```

## Docker

```bash
docker build -t nutri-optimize .
docker run -p 8000:8000 nutri-optimize
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Analyze nutritional content of a recipe |
| POST | `/optimize` | Optimize a recipe for nutrition goals |
| POST | `/substitute` | Get ingredient substitution recommendations |
| POST | `/meal-plan` | Generate an optimized weekly meal plan |
| GET | `/ingredients/search?q=` | Search the ingredient database |

### Example: Optimize a Recipe

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": [
      {"name": "chicken breast", "grams": 200},
      {"name": "brown rice cooked", "grams": 300},
      {"name": "broccoli", "grams": 150}
    ],
    "num_servings": 4,
    "goal": "weight_loss",
    "restrictions": ["low_sodium"]
  }'
```

## Architecture

```
src/nutri_optimize/
  knowledge/
    nutrition_database.py   # 110+ ingredients with USDA-calibrated values
    taste_model.py          # Flavor profiles, pairing scores, texture compatibility
  models/
    nutrition_predictor.py  # ML calorie prediction, allergen detection, DRI scoring
  optimizer/
    recipe_optimizer.py     # scipy SLSQP multi-objective optimization
  api/
    main.py                 # FastAPI REST endpoints
frontend/
  index.html               # Recipe optimizer web interface
  styles.css               # Food-themed responsive CSS
  app.js                   # Charts and API integration
```

## Optimization Goals

| Goal | Strategy |
|------|----------|
| **Balanced** | Equal weight on nutrition, calories, and taste |
| **Weight Loss** | Minimize calories, maximize protein and fiber for satiety |
| **Muscle Gain** | Maximize protein (30%+ of calories), adequate total calories |
| **Heart Health** | Minimize saturated fat and sodium, maximize fiber |
| **High Fiber** | Maximize dietary fiber intake |
| **Iron Rich** | Optimize for iron content with absorption enhancers |

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Pydantic
- **ML**: scikit-learn (GradientBoosting, RandomForest), scipy.optimize (SLSQP)
- **Data**: NumPy, Pandas
- **Frontend**: Vanilla HTML/CSS/JS with Canvas API charts
- **Testing**: pytest
