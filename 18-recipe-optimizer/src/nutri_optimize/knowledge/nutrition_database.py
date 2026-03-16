"""USDA-style nutrition database for common ingredients.

Nutritional values are per 100g of edible portion, sourced from USDA FoodData Central
reference values and peer-reviewed nutritional science literature.

Macronutrients in grams, micronutrients in milligrams unless noted.
Energy computed via Atwater factors: protein=4 kcal/g, carbs=4 kcal/g, fat=9 kcal/g, fiber=2 kcal/g.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Allergen(str, Enum):
    """Major food allergens (FDA Big 9)."""

    MILK = "milk"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT = "wheat"
    SOY = "soy"
    SESAME = "sesame"


class IngredientCategory(str, Enum):
    """Broad ingredient classification for substitution logic."""

    PROTEIN_ANIMAL = "protein_animal"
    PROTEIN_PLANT = "protein_plant"
    GRAIN = "grain"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    DAIRY = "dairy"
    FAT_OIL = "fat_oil"
    NUT_SEED = "nut_seed"
    LEGUME = "legume"
    SWEETENER = "sweetener"
    HERB_SPICE = "herb_spice"
    CONDIMENT = "condiment"


@dataclass(frozen=True)
class Micronutrients:
    """Key micronutrients per 100g (mg unless noted)."""

    iron_mg: float = 0.0
    calcium_mg: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_b12_mcg: float = 0.0  # micrograms
    folate_mcg: float = 0.0  # micrograms
    potassium_mg: float = 0.0
    magnesium_mg: float = 0.0
    zinc_mg: float = 0.0
    vitamin_a_mcg: float = 0.0  # micrograms RAE
    vitamin_d_mcg: float = 0.0  # micrograms


@dataclass(frozen=True)
class NutritionFacts:
    """Complete nutrition profile per 100g edible portion."""

    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float = 0.0
    saturated_fat_g: float = 0.0
    sodium_mg: float = 0.0
    cholesterol_mg: float = 0.0
    water_g: float = 0.0
    micro: Micronutrients = field(default_factory=Micronutrients)


@dataclass(frozen=True)
class IngredientInfo:
    """Full ingredient entry in the database."""

    name: str
    category: IngredientCategory
    nutrition: NutritionFacts
    allergens: tuple[Allergen, ...] = ()
    is_vegan: bool = True
    is_gluten_free: bool = True
    substitution_group: str = ""
    bioavailability_notes: str = ""


# ---------------------------------------------------------------------------
# The database: 110+ common ingredients with realistic USDA-like values
# ---------------------------------------------------------------------------

_INGREDIENTS: dict[str, IngredientInfo] = {}


def _add(
    name: str,
    category: IngredientCategory,
    cal: float,
    protein: float,
    carbs: float,
    fat: float,
    fiber: float,
    *,
    sugar: float = 0.0,
    sat_fat: float = 0.0,
    sodium: float = 0.0,
    cholesterol: float = 0.0,
    water: float = 0.0,
    iron: float = 0.0,
    calcium: float = 0.0,
    vit_c: float = 0.0,
    b12: float = 0.0,
    folate: float = 0.0,
    potassium: float = 0.0,
    magnesium: float = 0.0,
    zinc: float = 0.0,
    vit_a: float = 0.0,
    vit_d: float = 0.0,
    allergens: tuple[Allergen, ...] = (),
    vegan: bool = True,
    gf: bool = True,
    sub_group: str = "",
    bio_notes: str = "",
) -> None:
    key = name.lower().strip()
    _INGREDIENTS[key] = IngredientInfo(
        name=name,
        category=category,
        nutrition=NutritionFacts(
            calories_kcal=cal,
            protein_g=protein,
            carbs_g=carbs,
            fat_g=fat,
            fiber_g=fiber,
            sugar_g=sugar,
            saturated_fat_g=sat_fat,
            sodium_mg=sodium,
            cholesterol_mg=cholesterol,
            water_g=water,
            micro=Micronutrients(
                iron_mg=iron,
                calcium_mg=calcium,
                vitamin_c_mg=vit_c,
                vitamin_b12_mcg=b12,
                folate_mcg=folate,
                potassium_mg=potassium,
                magnesium_mg=magnesium,
                zinc_mg=zinc,
                vitamin_a_mcg=vit_a,
                vitamin_d_mcg=vit_d,
            ),
        ),
        allergens=allergens,
        is_vegan=vegan,
        is_gluten_free=gf,
        substitution_group=sub_group,
        bioavailability_notes=bio_notes,
    )


# ---- PROTEINS (Animal) ----
_add(
    "chicken breast",
    IngredientCategory.PROTEIN_ANIMAL,
    165,
    31.0,
    0.0,
    3.6,
    0.0,
    sodium=74,
    cholesterol=85,
    water=65,
    iron=1.0,
    calcium=15,
    b12=0.3,
    zinc=1.0,
    potassium=256,
    magnesium=29,
    vegan=False,
    sub_group="lean_protein",
    bio_notes="Complete protein; high bioavailability of heme iron and B12",
)
_add(
    "salmon",
    IngredientCategory.PROTEIN_ANIMAL,
    208,
    20.4,
    0.0,
    13.4,
    0.0,
    sat_fat=3.0,
    sodium=59,
    cholesterol=55,
    water=64,
    iron=0.8,
    calcium=12,
    b12=3.2,
    vit_d=11.0,
    zinc=0.6,
    potassium=363,
    magnesium=29,
    allergens=(Allergen.FISH,),
    vegan=False,
    sub_group="fatty_fish",
    bio_notes="Rich in EPA/DHA omega-3 fatty acids; excellent vitamin D source",
)
_add(
    "ground beef 85%",
    IngredientCategory.PROTEIN_ANIMAL,
    215,
    18.6,
    0.0,
    15.0,
    0.0,
    sat_fat=5.9,
    sodium=66,
    cholesterol=73,
    water=64,
    iron=2.3,
    calcium=18,
    b12=2.6,
    zinc=4.8,
    potassium=305,
    magnesium=20,
    vegan=False,
    sub_group="red_meat",
    bio_notes="High heme iron bioavailability (~25%); complete essential amino acids",
)
_add(
    "tuna canned",
    IngredientCategory.PROTEIN_ANIMAL,
    116,
    25.5,
    0.0,
    0.8,
    0.0,
    sodium=338,
    cholesterol=42,
    water=72,
    iron=1.3,
    calcium=11,
    b12=2.9,
    vit_d=1.7,
    zinc=0.8,
    potassium=237,
    allergens=(Allergen.FISH,),
    vegan=False,
    sub_group="lean_protein",
)
_add(
    "shrimp",
    IngredientCategory.PROTEIN_ANIMAL,
    99,
    24.0,
    0.2,
    0.3,
    0.0,
    sodium=111,
    cholesterol=189,
    water=76,
    iron=0.5,
    calcium=52,
    b12=1.1,
    zinc=1.6,
    potassium=259,
    allergens=(Allergen.SHELLFISH,),
    vegan=False,
    sub_group="lean_protein",
)
_add(
    "pork tenderloin",
    IngredientCategory.PROTEIN_ANIMAL,
    143,
    26.2,
    0.0,
    3.5,
    0.0,
    sat_fat=1.2,
    sodium=57,
    cholesterol=73,
    water=69,
    iron=1.1,
    b12=0.7,
    zinc=2.4,
    potassium=421,
    magnesium=28,
    vegan=False,
    sub_group="lean_protein",
)
_add(
    "turkey breast",
    IngredientCategory.PROTEIN_ANIMAL,
    135,
    30.1,
    0.0,
    0.7,
    0.0,
    sodium=46,
    cholesterol=83,
    water=68,
    iron=0.7,
    b12=0.4,
    zinc=1.6,
    potassium=293,
    magnesium=27,
    vegan=False,
    sub_group="lean_protein",
)
_add(
    "eggs",
    IngredientCategory.PROTEIN_ANIMAL,
    155,
    12.6,
    1.1,
    10.6,
    0.0,
    sat_fat=3.3,
    sodium=124,
    cholesterol=373,
    water=75,
    iron=1.8,
    calcium=56,
    b12=0.9,
    vit_a=160,
    vit_d=2.0,
    zinc=1.3,
    folate=47,
    allergens=(Allergen.EGGS,),
    vegan=False,
    sub_group="eggs",
    bio_notes="Highest PDCAAS (1.0); choline-rich; fat-soluble vitamin carrier",
)
_add(
    "cod",
    IngredientCategory.PROTEIN_ANIMAL,
    82,
    17.8,
    0.0,
    0.7,
    0.0,
    sodium=54,
    cholesterol=43,
    water=81,
    iron=0.4,
    calcium=16,
    b12=0.9,
    potassium=413,
    allergens=(Allergen.FISH,),
    vegan=False,
    sub_group="lean_protein",
)

# ---- DAIRY ----
_add(
    "whole milk",
    IngredientCategory.DAIRY,
    61,
    3.2,
    4.8,
    3.3,
    0.0,
    sugar=5.0,
    sat_fat=1.9,
    sodium=43,
    cholesterol=10,
    water=88,
    iron=0.03,
    calcium=113,
    b12=0.4,
    vit_a=46,
    vit_d=1.3,
    potassium=132,
    magnesium=10,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="milk",
    bio_notes="Excellent calcium bioavailability (~30%); vitamin D fortified",
)
_add(
    "greek yogurt",
    IngredientCategory.DAIRY,
    97,
    9.0,
    3.6,
    5.0,
    0.0,
    sugar=3.2,
    sat_fat=3.1,
    sodium=36,
    cholesterol=13,
    water=81,
    calcium=110,
    b12=0.7,
    potassium=141,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="yogurt",
    bio_notes="Probiotic cultures enhance mineral absorption",
)
_add(
    "cheddar cheese",
    IngredientCategory.DAIRY,
    403,
    24.9,
    1.3,
    33.1,
    0.0,
    sat_fat=21.1,
    sodium=621,
    cholesterol=105,
    water=37,
    iron=0.7,
    calcium=721,
    b12=0.8,
    vit_a=265,
    zinc=3.1,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="hard_cheese",
)
_add(
    "mozzarella",
    IngredientCategory.DAIRY,
    280,
    27.5,
    3.1,
    17.1,
    0.0,
    sat_fat=10.9,
    sodium=627,
    cholesterol=54,
    water=50,
    calcium=505,
    b12=0.7,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="soft_cheese",
)
_add(
    "cottage cheese",
    IngredientCategory.DAIRY,
    98,
    11.1,
    3.4,
    4.3,
    0.0,
    sodium=364,
    cholesterol=17,
    water=80,
    calcium=83,
    b12=0.4,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="yogurt",
)
_add(
    "butter",
    IngredientCategory.FAT_OIL,
    717,
    0.9,
    0.1,
    81.1,
    0.0,
    sat_fat=51.4,
    sodium=11,
    cholesterol=215,
    water=18,
    vit_a=684,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="solid_fat",
)
_add(
    "parmesan",
    IngredientCategory.DAIRY,
    431,
    38.5,
    4.1,
    28.6,
    0.0,
    sat_fat=18.5,
    sodium=1529,
    cholesterol=68,
    calcium=1184,
    b12=1.2,
    zinc=2.8,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="hard_cheese",
)

# ---- GRAINS ----
_add(
    "white rice cooked",
    IngredientCategory.GRAIN,
    130,
    2.7,
    28.2,
    0.3,
    0.4,
    sodium=1,
    water=69,
    iron=0.2,
    folate=58,
    magnesium=12,
    potassium=35,
    gf=True,
    sub_group="rice",
    bio_notes="High glycemic index (~73); low in micronutrients unless fortified",
)
_add(
    "brown rice cooked",
    IngredientCategory.GRAIN,
    123,
    2.7,
    25.6,
    1.0,
    1.8,
    sodium=1,
    water=70,
    iron=0.5,
    magnesium=39,
    potassium=79,
    folate=4,
    zinc=0.6,
    gf=True,
    sub_group="rice",
    bio_notes="Phytic acid reduces mineral absorption; soaking improves bioavailability",
)
_add(
    "quinoa cooked",
    IngredientCategory.GRAIN,
    120,
    4.4,
    21.3,
    1.9,
    2.8,
    sodium=7,
    water=72,
    iron=1.5,
    calcium=17,
    magnesium=64,
    folate=42,
    potassium=172,
    zinc=1.1,
    gf=True,
    sub_group="pseudograin",
    bio_notes="Complete plant protein (all 9 EAAs); good iron source but non-heme",
)
_add(
    "oats rolled",
    IngredientCategory.GRAIN,
    379,
    13.2,
    67.7,
    6.5,
    10.1,
    sodium=6,
    iron=4.3,
    calcium=52,
    magnesium=138,
    potassium=362,
    folate=32,
    zinc=3.6,
    gf=False,
    sub_group="whole_grain",
    bio_notes="Beta-glucan fiber lowers LDL cholesterol; may contain gluten from cross-contamination",
)
_add(
    "whole wheat bread",
    IngredientCategory.GRAIN,
    247,
    12.9,
    41.3,
    3.4,
    6.8,
    sugar=5.6,
    sodium=400,
    iron=2.5,
    calcium=107,
    magnesium=75,
    folate=42,
    potassium=254,
    zinc=1.8,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="bread",
)
_add(
    "white bread",
    IngredientCategory.GRAIN,
    265,
    9.4,
    49.2,
    3.3,
    2.7,
    sugar=5.3,
    sodium=491,
    iron=3.6,
    calcium=151,
    folate=111,
    potassium=100,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="bread",
)
_add(
    "pasta cooked",
    IngredientCategory.GRAIN,
    158,
    5.8,
    30.9,
    0.9,
    1.8,
    sodium=1,
    iron=1.3,
    folate=7,
    magnesium=18,
    potassium=44,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="pasta",
)
_add(
    "whole wheat pasta cooked",
    IngredientCategory.GRAIN,
    124,
    5.3,
    26.5,
    0.5,
    3.9,
    sodium=3,
    iron=1.1,
    magnesium=30,
    potassium=44,
    folate=7,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="pasta",
)
_add(
    "corn tortilla",
    IngredientCategory.GRAIN,
    218,
    5.7,
    44.6,
    2.8,
    5.3,
    sodium=46,
    calcium=46,
    iron=1.5,
    magnesium=48,
    gf=True,
    sub_group="flatbread",
)
_add(
    "flour tortilla",
    IngredientCategory.GRAIN,
    312,
    8.2,
    52.4,
    7.6,
    2.2,
    sodium=558,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="flatbread",
)
_add(
    "couscous cooked",
    IngredientCategory.GRAIN,
    112,
    3.8,
    23.2,
    0.2,
    1.4,
    sodium=5,
    iron=0.4,
    folate=15,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="pasta",
)
_add(
    "barley cooked",
    IngredientCategory.GRAIN,
    123,
    2.3,
    28.2,
    0.4,
    3.8,
    iron=1.3,
    magnesium=22,
    potassium=93,
    gf=False,
    sub_group="whole_grain",
)

# ---- VEGETABLES ----
_add(
    "broccoli",
    IngredientCategory.VEGETABLE,
    34,
    2.8,
    6.6,
    0.4,
    2.6,
    sugar=1.7,
    water=89,
    iron=0.7,
    calcium=47,
    vit_c=89.2,
    folate=63,
    potassium=316,
    magnesium=21,
    vit_a=31,
    sub_group="cruciferous",
    bio_notes="Sulforaphane content; vitamin C enhances non-heme iron absorption",
)
_add(
    "spinach raw",
    IngredientCategory.VEGETABLE,
    23,
    2.9,
    3.6,
    0.4,
    2.2,
    water=91,
    iron=2.7,
    calcium=99,
    vit_c=28.1,
    folate=194,
    potassium=558,
    magnesium=79,
    vit_a=469,
    zinc=0.5,
    sub_group="leafy_green",
    bio_notes="High oxalate reduces calcium absorption to ~5%; non-heme iron ~2-5% bioavailable",
)
_add(
    "kale",
    IngredientCategory.VEGETABLE,
    49,
    4.3,
    8.8,
    0.9,
    3.6,
    water=84,
    iron=1.5,
    calcium=150,
    vit_c=120,
    folate=141,
    potassium=491,
    magnesium=47,
    vit_a=500,
    sub_group="leafy_green",
    bio_notes="Lower oxalate than spinach; ~49% calcium bioavailability",
)
_add(
    "sweet potato",
    IngredientCategory.VEGETABLE,
    86,
    1.6,
    20.1,
    0.1,
    3.0,
    sugar=4.2,
    water=77,
    iron=0.6,
    calcium=30,
    vit_c=2.4,
    vit_a=709,
    potassium=337,
    magnesium=25,
    folate=11,
    sub_group="starchy_veg",
    bio_notes="Beta-carotene (pro-vitamin A) enhanced by fat co-consumption",
)
_add(
    "carrots",
    IngredientCategory.VEGETABLE,
    41,
    0.9,
    9.6,
    0.2,
    2.8,
    sugar=4.7,
    water=88,
    iron=0.3,
    calcium=33,
    vit_c=5.9,
    vit_a=835,
    potassium=320,
    magnesium=12,
    folate=19,
    sub_group="root_veg",
)
_add(
    "tomatoes",
    IngredientCategory.VEGETABLE,
    18,
    0.9,
    3.9,
    0.2,
    1.2,
    sugar=2.6,
    water=95,
    iron=0.3,
    calcium=10,
    vit_c=13.7,
    potassium=237,
    folate=15,
    vit_a=42,
    sub_group="nightshade",
    bio_notes="Lycopene bioavailability increases with cooking and fat",
)
_add(
    "bell pepper red",
    IngredientCategory.VEGETABLE,
    31,
    1.0,
    6.0,
    0.3,
    2.1,
    sugar=4.2,
    water=92,
    iron=0.4,
    calcium=7,
    vit_c=127.7,
    vit_a=157,
    potassium=211,
    folate=46,
    sub_group="nightshade",
)
_add(
    "onion",
    IngredientCategory.VEGETABLE,
    40,
    1.1,
    9.3,
    0.1,
    1.7,
    sugar=4.2,
    water=89,
    iron=0.2,
    calcium=23,
    vit_c=7.4,
    potassium=146,
    folate=19,
    sub_group="allium",
)
_add(
    "garlic",
    IngredientCategory.HERB_SPICE,
    149,
    6.4,
    33.1,
    0.5,
    2.1,
    sodium=17,
    iron=1.7,
    calcium=181,
    vit_c=31.2,
    potassium=401,
    magnesium=25,
    folate=3,
    zinc=1.2,
    sub_group="allium",
    bio_notes="Allicin has antimicrobial properties; enhances iron absorption",
)
_add(
    "mushrooms white",
    IngredientCategory.VEGETABLE,
    22,
    3.1,
    3.3,
    0.3,
    1.0,
    water=92,
    iron=0.5,
    potassium=318,
    folate=17,
    vit_d=0.2,
    zinc=0.5,
    sub_group="mushroom",
    bio_notes="UV-exposed mushrooms can provide vitamin D2",
)
_add(
    "zucchini",
    IngredientCategory.VEGETABLE,
    17,
    1.2,
    3.1,
    0.3,
    1.0,
    water=95,
    iron=0.4,
    calcium=16,
    vit_c=17.9,
    potassium=261,
    magnesium=18,
    folate=24,
    sub_group="squash",
)
_add(
    "cauliflower",
    IngredientCategory.VEGETABLE,
    25,
    1.9,
    5.0,
    0.3,
    2.0,
    water=92,
    iron=0.4,
    calcium=22,
    vit_c=48.2,
    folate=57,
    potassium=299,
    magnesium=15,
    sub_group="cruciferous",
)
_add(
    "green beans",
    IngredientCategory.VEGETABLE,
    31,
    1.8,
    7.0,
    0.2,
    2.7,
    water=90,
    iron=1.0,
    calcium=37,
    vit_c=12.2,
    folate=33,
    potassium=211,
    magnesium=25,
    vit_a=35,
    sub_group="pod_veg",
)
_add(
    "asparagus",
    IngredientCategory.VEGETABLE,
    20,
    2.2,
    3.9,
    0.1,
    2.1,
    water=93,
    iron=2.1,
    calcium=24,
    vit_c=5.6,
    folate=52,
    potassium=202,
    vit_a=38,
    sub_group="stem_veg",
)
_add(
    "celery",
    IngredientCategory.VEGETABLE,
    14,
    0.7,
    3.0,
    0.2,
    1.6,
    sodium=80,
    water=95,
    calcium=40,
    vit_c=3.1,
    potassium=260,
    folate=36,
    sub_group="stem_veg",
)
_add(
    "cucumber",
    IngredientCategory.VEGETABLE,
    15,
    0.7,
    3.6,
    0.1,
    0.5,
    water=96,
    calcium=16,
    vit_c=2.8,
    potassium=147,
    magnesium=13,
    sub_group="gourd",
)
_add(
    "peas green",
    IngredientCategory.VEGETABLE,
    81,
    5.4,
    14.5,
    0.4,
    5.7,
    sugar=5.7,
    water=79,
    iron=1.5,
    calcium=25,
    vit_c=40,
    folate=65,
    potassium=244,
    magnesium=33,
    vit_a=38,
    zinc=1.2,
    sub_group="pod_veg",
)
_add(
    "corn kernels",
    IngredientCategory.VEGETABLE,
    86,
    3.3,
    19.0,
    1.4,
    2.7,
    sugar=3.2,
    water=76,
    iron=0.5,
    magnesium=37,
    potassium=270,
    folate=46,
    sub_group="starchy_veg",
)
_add(
    "eggplant",
    IngredientCategory.VEGETABLE,
    25,
    1.0,
    5.9,
    0.2,
    3.0,
    water=92,
    iron=0.2,
    calcium=9,
    potassium=229,
    folate=22,
    sub_group="nightshade",
)
_add(
    "cabbage",
    IngredientCategory.VEGETABLE,
    25,
    1.3,
    5.8,
    0.1,
    2.5,
    water=92,
    iron=0.5,
    calcium=40,
    vit_c=36.6,
    folate=43,
    potassium=170,
    sub_group="cruciferous",
)
_add(
    "lettuce romaine",
    IngredientCategory.VEGETABLE,
    17,
    1.2,
    3.3,
    0.3,
    2.1,
    water=95,
    iron=1.0,
    calcium=33,
    vit_c=4.0,
    folate=136,
    potassium=247,
    vit_a=436,
    sub_group="leafy_green",
)

# ---- FRUITS ----
_add(
    "banana",
    IngredientCategory.FRUIT,
    89,
    1.1,
    22.8,
    0.3,
    2.6,
    sugar=12.2,
    water=75,
    iron=0.3,
    calcium=5,
    vit_c=8.7,
    potassium=358,
    magnesium=27,
    folate=20,
    sub_group="tropical_fruit",
    bio_notes="Good potassium source; resistant starch in unripe form",
)
_add(
    "apple",
    IngredientCategory.FRUIT,
    52,
    0.3,
    13.8,
    0.2,
    2.4,
    sugar=10.4,
    water=86,
    vit_c=4.6,
    potassium=107,
    sub_group="pome_fruit",
)
_add(
    "blueberries",
    IngredientCategory.FRUIT,
    57,
    0.7,
    14.5,
    0.3,
    2.4,
    sugar=10.0,
    water=84,
    iron=0.3,
    vit_c=9.7,
    potassium=77,
    sub_group="berry",
    bio_notes="Anthocyanins with antioxidant properties; polyphenol-rich",
)
_add(
    "strawberries",
    IngredientCategory.FRUIT,
    32,
    0.7,
    7.7,
    0.3,
    2.0,
    sugar=4.9,
    water=91,
    iron=0.4,
    calcium=16,
    vit_c=58.8,
    folate=24,
    potassium=153,
    sub_group="berry",
)
_add(
    "avocado",
    IngredientCategory.FRUIT,
    160,
    2.0,
    8.5,
    14.7,
    6.7,
    sugar=0.7,
    water=73,
    iron=0.6,
    calcium=12,
    vit_c=10,
    potassium=485,
    magnesium=29,
    folate=81,
    sub_group="fatty_fruit",
    bio_notes="Monounsaturated fats enhance carotenoid absorption from other foods",
)
_add(
    "orange",
    IngredientCategory.FRUIT,
    47,
    0.9,
    11.8,
    0.1,
    2.4,
    sugar=9.4,
    water=87,
    calcium=40,
    vit_c=53.2,
    folate=30,
    potassium=181,
    sub_group="citrus",
)
_add(
    "lemon juice",
    IngredientCategory.FRUIT,
    22,
    0.4,
    6.9,
    0.2,
    0.3,
    water=92,
    vit_c=38.7,
    potassium=103,
    calcium=6,
    folate=20,
    sub_group="citrus",
    bio_notes="Citric acid enhances non-heme iron absorption significantly",
)
_add(
    "grapes",
    IngredientCategory.FRUIT,
    69,
    0.7,
    18.1,
    0.2,
    0.9,
    sugar=15.5,
    water=81,
    potassium=191,
    vit_c=3.2,
    sub_group="vine_fruit",
)
_add(
    "mango",
    IngredientCategory.FRUIT,
    60,
    0.8,
    15.0,
    0.4,
    1.6,
    sugar=13.7,
    water=84,
    vit_c=36.4,
    vit_a=54,
    folate=43,
    potassium=168,
    sub_group="tropical_fruit",
)
_add(
    "pineapple",
    IngredientCategory.FRUIT,
    50,
    0.5,
    13.1,
    0.1,
    1.4,
    sugar=9.9,
    water=86,
    vit_c=47.8,
    potassium=109,
    magnesium=12,
    sub_group="tropical_fruit",
)

# ---- LEGUMES ----
_add(
    "black beans cooked",
    IngredientCategory.LEGUME,
    132,
    8.9,
    23.7,
    0.5,
    8.7,
    water=66,
    iron=2.1,
    calcium=27,
    folate=149,
    potassium=355,
    magnesium=70,
    zinc=1.1,
    sub_group="beans",
    bio_notes="Non-heme iron; phytates reduce absorption but soaking/cooking helps",
)
_add(
    "chickpeas cooked",
    IngredientCategory.LEGUME,
    164,
    8.9,
    27.4,
    2.6,
    7.6,
    water=60,
    iron=2.9,
    calcium=49,
    folate=172,
    potassium=291,
    magnesium=48,
    zinc=1.5,
    sub_group="beans",
    bio_notes="Good lysine source; pair with grains for complete amino acid profile",
)
_add(
    "lentils cooked",
    IngredientCategory.LEGUME,
    116,
    9.0,
    20.1,
    0.4,
    7.9,
    water=70,
    iron=3.3,
    calcium=19,
    folate=181,
    potassium=369,
    magnesium=36,
    zinc=1.3,
    sub_group="lentils",
    bio_notes="Highest folate among legumes; pair with vitamin C for iron absorption",
)
_add(
    "kidney beans cooked",
    IngredientCategory.LEGUME,
    127,
    8.7,
    22.8,
    0.5,
    6.4,
    water=67,
    iron=2.9,
    calcium=28,
    folate=130,
    potassium=403,
    magnesium=45,
    zinc=1.0,
    sub_group="beans",
)
_add(
    "edamame",
    IngredientCategory.LEGUME,
    121,
    11.9,
    8.9,
    5.2,
    5.2,
    water=73,
    iron=2.3,
    calcium=63,
    folate=311,
    potassium=436,
    magnesium=64,
    zinc=1.4,
    allergens=(Allergen.SOY,),
    sub_group="soy",
    bio_notes="Complete plant protein; isoflavones with estrogenic activity",
)
_add(
    "tofu firm",
    IngredientCategory.PROTEIN_PLANT,
    144,
    17.3,
    2.8,
    8.7,
    2.3,
    sodium=14,
    calcium=683,
    iron=2.7,
    magnesium=58,
    potassium=237,
    zinc=1.6,
    folate=29,
    allergens=(Allergen.SOY,),
    sub_group="soy",
    bio_notes="Calcium-set tofu excellent calcium source; complete amino acids",
)
_add(
    "tempeh",
    IngredientCategory.PROTEIN_PLANT,
    192,
    20.3,
    7.6,
    10.8,
    0.0,
    iron=2.7,
    calcium=111,
    magnesium=81,
    potassium=412,
    zinc=1.1,
    folate=24,
    allergens=(Allergen.SOY,),
    sub_group="soy",
    bio_notes="Fermentation increases B12 and mineral bioavailability",
)
_add(
    "peanut butter",
    IngredientCategory.LEGUME,
    588,
    25.1,
    19.6,
    50.4,
    6.0,
    sugar=9.2,
    sodium=426,
    iron=1.7,
    calcium=43,
    magnesium=154,
    potassium=649,
    zinc=2.5,
    folate=87,
    allergens=(Allergen.PEANUTS,),
    sub_group="nut_butter",
)

# ---- NUTS & SEEDS ----
_add(
    "almonds",
    IngredientCategory.NUT_SEED,
    579,
    21.2,
    21.6,
    49.9,
    12.5,
    sugar=4.4,
    iron=3.7,
    calcium=269,
    magnesium=270,
    potassium=733,
    zinc=3.1,
    folate=44,
    allergens=(Allergen.TREE_NUTS,),
    sub_group="tree_nut",
    bio_notes="Vitamin E rich; calcium absorption ~21% (moderate)",
)
_add(
    "walnuts",
    IngredientCategory.NUT_SEED,
    654,
    15.2,
    13.7,
    65.2,
    6.7,
    iron=2.9,
    calcium=98,
    magnesium=158,
    potassium=441,
    zinc=3.1,
    folate=98,
    allergens=(Allergen.TREE_NUTS,),
    sub_group="tree_nut",
    bio_notes="ALA omega-3 source; polyphenols support gut microbiome",
)
_add(
    "cashews",
    IngredientCategory.NUT_SEED,
    553,
    18.2,
    30.2,
    43.9,
    3.3,
    iron=6.7,
    calcium=37,
    magnesium=292,
    potassium=660,
    zinc=5.8,
    folate=25,
    allergens=(Allergen.TREE_NUTS,),
    sub_group="tree_nut",
)
_add(
    "chia seeds",
    IngredientCategory.NUT_SEED,
    486,
    16.5,
    42.1,
    30.7,
    34.4,
    iron=7.7,
    calcium=631,
    magnesium=335,
    potassium=407,
    zinc=4.6,
    folate=49,
    sub_group="seed",
    bio_notes="Highest plant omega-3 (ALA); exceptional fiber; forms gel (soluble fiber)",
)
_add(
    "flax seeds",
    IngredientCategory.NUT_SEED,
    534,
    18.3,
    28.9,
    42.2,
    27.3,
    iron=5.7,
    calcium=255,
    magnesium=392,
    potassium=813,
    zinc=4.3,
    folate=87,
    sub_group="seed",
    bio_notes="Lignans with antioxidant properties; must grind to access nutrients",
)
_add(
    "sunflower seeds",
    IngredientCategory.NUT_SEED,
    584,
    20.8,
    20.0,
    51.5,
    8.6,
    iron=5.3,
    calcium=78,
    magnesium=325,
    potassium=645,
    zinc=5.0,
    folate=227,
    sub_group="seed",
)
_add(
    "pumpkin seeds",
    IngredientCategory.NUT_SEED,
    559,
    30.2,
    10.7,
    49.1,
    6.0,
    iron=8.8,
    calcium=46,
    magnesium=550,
    potassium=809,
    zinc=7.8,
    folate=58,
    sub_group="seed",
    bio_notes="Exceptional magnesium and zinc source",
)
_add(
    "sesame seeds",
    IngredientCategory.NUT_SEED,
    573,
    17.7,
    23.5,
    49.7,
    11.8,
    iron=14.6,
    calcium=975,
    magnesium=351,
    potassium=468,
    zinc=7.8,
    folate=97,
    allergens=(Allergen.SESAME,),
    sub_group="seed",
    bio_notes="Highest iron among seeds; calcium in unhulled form; sesamin antioxidant",
)

# ---- FATS & OILS ----
_add(
    "olive oil",
    IngredientCategory.FAT_OIL,
    884,
    0.0,
    0.0,
    100.0,
    0.0,
    sat_fat=13.8,
    sub_group="liquid_oil",
    bio_notes="73% oleic acid (MUFA); polyphenols; anti-inflammatory properties",
)
_add(
    "coconut oil",
    IngredientCategory.FAT_OIL,
    862,
    0.0,
    0.0,
    100.0,
    0.0,
    sat_fat=82.5,
    sub_group="solid_fat",
    bio_notes="~50% lauric acid (MCT); high saturated fat raises LDL and HDL",
)
_add(
    "canola oil",
    IngredientCategory.FAT_OIL,
    884,
    0.0,
    0.0,
    100.0,
    0.0,
    sat_fat=7.4,
    sub_group="liquid_oil",
    bio_notes="Favorable omega-6:omega-3 ratio (~2:1); low saturated fat",
)
_add(
    "sesame oil",
    IngredientCategory.FAT_OIL,
    884,
    0.0,
    0.0,
    100.0,
    0.0,
    sat_fat=14.2,
    allergens=(Allergen.SESAME,),
    sub_group="liquid_oil",
)

# ---- SWEETENERS ----
_add(
    "honey",
    IngredientCategory.SWEETENER,
    304,
    0.3,
    82.4,
    0.0,
    0.2,
    sugar=82.1,
    iron=0.4,
    calcium=6,
    potassium=52,
    vegan=False,
    sub_group="liquid_sweetener",
)
_add(
    "maple syrup",
    IngredientCategory.SWEETENER,
    260,
    0.0,
    67.0,
    0.1,
    0.0,
    sugar=60.5,
    calcium=102,
    potassium=212,
    magnesium=21,
    zinc=1.5,
    manganese=2.9,
    sub_group="liquid_sweetener",
)
_add(
    "white sugar",
    IngredientCategory.SWEETENER,
    387,
    0.0,
    100.0,
    0.0,
    0.0,
    sugar=100.0,
    sub_group="dry_sweetener",
)
_add(
    "brown sugar",
    IngredientCategory.SWEETENER,
    380,
    0.1,
    98.1,
    0.0,
    0.0,
    sugar=97.0,
    calcium=83,
    iron=0.7,
    potassium=133,
    magnesium=9,
    sub_group="dry_sweetener",
)

# ---- HERBS & SPICES (per 100g dry, used in small quantities) ----
_add(
    "turmeric ground",
    IngredientCategory.HERB_SPICE,
    312,
    9.7,
    67.1,
    3.3,
    22.7,
    iron=55.0,
    calcium=168,
    potassium=2080,
    magnesium=208,
    vit_c=0.7,
    sub_group="spice",
    bio_notes="Curcumin bioavailability enhanced 2000% with piperine (black pepper)",
)
_add(
    "black pepper",
    IngredientCategory.HERB_SPICE,
    251,
    10.4,
    63.9,
    3.3,
    25.3,
    iron=9.7,
    calcium=443,
    magnesium=171,
    potassium=1329,
    sub_group="spice",
    bio_notes="Piperine enhances bioavailability of curcumin, beta-carotene, and other nutrients",
)
_add(
    "cinnamon ground",
    IngredientCategory.HERB_SPICE,
    247,
    4.0,
    80.6,
    1.2,
    53.1,
    iron=8.3,
    calcium=1002,
    magnesium=60,
    potassium=431,
    sub_group="spice",
    bio_notes="May improve insulin sensitivity; cinnamaldehyde is the active compound",
)
_add(
    "cumin ground",
    IngredientCategory.HERB_SPICE,
    375,
    17.8,
    44.2,
    22.3,
    10.5,
    iron=66.4,
    calcium=931,
    magnesium=366,
    potassium=1788,
    zinc=4.8,
    sub_group="spice",
)
_add(
    "ginger ground",
    IngredientCategory.HERB_SPICE,
    335,
    8.9,
    71.6,
    4.2,
    14.1,
    iron=19.8,
    calcium=114,
    magnesium=214,
    potassium=1320,
    zinc=3.3,
    vit_c=0.7,
    sub_group="spice",
    bio_notes="Gingerols and shogaols have anti-nausea and anti-inflammatory effects",
)
_add(
    "basil dried",
    IngredientCategory.HERB_SPICE,
    233,
    22.98,
    47.75,
    4.07,
    37.7,
    iron=89.8,
    calcium=2240,
    magnesium=711,
    potassium=2630,
    zinc=5.8,
    vit_a=744,
    sub_group="herb",
)
_add(
    "oregano dried",
    IngredientCategory.HERB_SPICE,
    265,
    9.0,
    68.9,
    4.3,
    42.5,
    iron=36.8,
    calcium=1597,
    magnesium=270,
    potassium=1260,
    zinc=2.7,
    vit_a=85,
    sub_group="herb",
)
_add(
    "paprika",
    IngredientCategory.HERB_SPICE,
    282,
    14.1,
    53.9,
    13.0,
    34.9,
    iron=21.1,
    calcium=229,
    vit_c=0.9,
    vit_a=2463,
    potassium=2280,
    magnesium=178,
    sub_group="spice",
)
_add(
    "chili powder",
    IngredientCategory.HERB_SPICE,
    282,
    12.3,
    49.7,
    14.3,
    34.8,
    iron=14.3,
    calcium=278,
    vit_c=0.0,
    vit_a=1483,
    potassium=1916,
    magnesium=170,
    sub_group="spice",
)

# ---- CONDIMENTS ----
_add(
    "soy sauce",
    IngredientCategory.CONDIMENT,
    53,
    8.1,
    4.9,
    0.0,
    0.8,
    sodium=5493,
    iron=2.4,
    potassium=212,
    allergens=(Allergen.SOY, Allergen.WHEAT),
    vegan=True,
    gf=False,
    sub_group="fermented_sauce",
)
_add(
    "tomato sauce",
    IngredientCategory.CONDIMENT,
    29,
    1.3,
    5.4,
    0.5,
    1.5,
    sodium=427,
    iron=0.9,
    calcium=13,
    vit_c=7.3,
    potassium=331,
    vit_a=25,
    sub_group="tomato_product",
)
_add(
    "balsamic vinegar",
    IngredientCategory.CONDIMENT,
    88,
    0.5,
    17.0,
    0.0,
    0.0,
    sugar=14.9,
    sodium=23,
    calcium=27,
    iron=0.7,
    potassium=112,
    magnesium=12,
    sub_group="vinegar",
)
_add(
    "tahini",
    IngredientCategory.NUT_SEED,
    595,
    17.0,
    21.2,
    53.8,
    9.3,
    iron=8.9,
    calcium=426,
    magnesium=95,
    potassium=414,
    zinc=4.6,
    allergens=(Allergen.SESAME,),
    sub_group="nut_butter",
)
_add(
    "dijon mustard",
    IngredientCategory.CONDIMENT,
    66,
    4.4,
    5.8,
    3.3,
    3.3,
    sodium=1135,
    iron=1.5,
    calcium=58,
    potassium=138,
    sub_group="condiment",
)

# ---- MISC PLANT PROTEINS ----
_add(
    "seitan",
    IngredientCategory.PROTEIN_PLANT,
    370,
    75.2,
    13.8,
    1.9,
    0.6,
    sodium=29,
    iron=5.2,
    calcium=53,
    allergens=(Allergen.WHEAT,),
    gf=False,
    sub_group="wheat_protein",
    bio_notes="Pure wheat gluten; very high protein but low in lysine",
)
_add(
    "nutritional yeast",
    IngredientCategory.PROTEIN_PLANT,
    325,
    50.0,
    36.0,
    4.0,
    21.0,
    iron=5.0,
    calcium=20,
    b12=17.6,
    folate=3120,
    zinc=9.9,
    sub_group="supplement",
    bio_notes="Fortified B12 source for vegans; complete protein",
)

# ---- ADDITIONAL COMMON ITEMS ----
_add(
    "potato",
    IngredientCategory.VEGETABLE,
    77,
    2.0,
    17.5,
    0.1,
    2.2,
    sugar=0.8,
    water=79,
    iron=0.8,
    calcium=12,
    vit_c=19.7,
    potassium=421,
    magnesium=23,
    folate=15,
    sub_group="starchy_veg",
    bio_notes="Vitamin C content decreases significantly with cooking",
)
_add(
    "rice milk",
    IngredientCategory.DAIRY,
    47,
    0.3,
    9.2,
    1.0,
    0.3,
    calcium=118,
    vit_d=1.0,
    sub_group="plant_milk",
    gf=True,
)
_add(
    "almond milk",
    IngredientCategory.DAIRY,
    15,
    0.6,
    0.6,
    1.1,
    0.2,
    calcium=184,
    vit_d=1.0,
    allergens=(Allergen.TREE_NUTS,),
    sub_group="plant_milk",
)
_add(
    "oat milk",
    IngredientCategory.DAIRY,
    43,
    1.0,
    7.0,
    1.5,
    0.8,
    calcium=120,
    vit_d=1.0,
    gf=False,
    sub_group="plant_milk",
)
_add(
    "soy milk",
    IngredientCategory.DAIRY,
    33,
    2.8,
    1.6,
    1.6,
    0.4,
    calcium=123,
    vit_d=1.1,
    b12=0.5,
    potassium=141,
    allergens=(Allergen.SOY,),
    sub_group="plant_milk",
)
_add(
    "coconut milk canned",
    IngredientCategory.FAT_OIL,
    197,
    2.3,
    2.8,
    21.3,
    0.0,
    sat_fat=18.9,
    iron=1.6,
    magnesium=37,
    potassium=263,
    sub_group="coconut_product",
)
_add(
    "dark chocolate 70%",
    IngredientCategory.SWEETENER,
    598,
    7.8,
    45.9,
    42.6,
    10.9,
    sugar=24.0,
    sat_fat=24.5,
    iron=11.9,
    calcium=73,
    magnesium=228,
    potassium=715,
    zinc=3.3,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="chocolate",
    bio_notes="Flavanols support vascular health; theobromine is a mild stimulant",
)
_add(
    "cocoa powder",
    IngredientCategory.SWEETENER,
    228,
    19.6,
    57.9,
    13.7,
    33.2,
    iron=13.9,
    calcium=128,
    magnesium=499,
    potassium=1524,
    zinc=6.8,
    sub_group="chocolate",
)
_add(
    "coconut flakes",
    IngredientCategory.NUT_SEED,
    660,
    6.9,
    23.6,
    64.5,
    16.3,
    sat_fat=57.2,
    iron=2.4,
    calcium=14,
    magnesium=32,
    potassium=543,
    zinc=2.0,
    sub_group="coconut_product",
)

# Additional to reach 110+
_add(
    "ham",
    IngredientCategory.PROTEIN_ANIMAL,
    145,
    20.9,
    1.5,
    5.5,
    0.0,
    sat_fat=1.8,
    sodium=1203,
    cholesterol=53,
    iron=0.9,
    b12=0.6,
    zinc=2.2,
    potassium=287,
    vegan=False,
    sub_group="cured_meat",
)
_add(
    "bacon",
    IngredientCategory.PROTEIN_ANIMAL,
    541,
    37.0,
    1.4,
    42.0,
    0.0,
    sat_fat=14.0,
    sodium=1717,
    cholesterol=110,
    iron=1.4,
    b12=1.0,
    zinc=3.2,
    potassium=565,
    vegan=False,
    sub_group="cured_meat",
)
_add(
    "cream cheese",
    IngredientCategory.DAIRY,
    342,
    5.9,
    4.1,
    34.2,
    0.0,
    sat_fat=19.2,
    sodium=321,
    cholesterol=110,
    calcium=98,
    vit_a=362,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="soft_cheese",
)
_add(
    "ricotta",
    IngredientCategory.DAIRY,
    174,
    11.3,
    3.0,
    13.0,
    0.0,
    sat_fat=8.3,
    sodium=84,
    cholesterol=51,
    calcium=207,
    b12=0.3,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="soft_cheese",
)
_add(
    "heavy cream",
    IngredientCategory.DAIRY,
    340,
    2.1,
    2.8,
    36.1,
    0.0,
    sat_fat=23.0,
    cholesterol=137,
    calcium=65,
    vit_a=411,
    allergens=(Allergen.MILK,),
    vegan=False,
    sub_group="cream",
)


# ========================================================================
# Public API
# ========================================================================


class NutritionDatabase:
    """Queryable nutrition database with search, lookup, and substitution support."""

    def __init__(self) -> None:
        self._db = dict(_INGREDIENTS)
        logger.info("NutritionDatabase loaded with %d ingredients", len(self._db))

    @property
    def ingredient_count(self) -> int:
        return len(self._db)

    def get(self, name: str) -> IngredientInfo | None:
        """Look up an ingredient by exact name (case-insensitive)."""
        return self._db.get(name.lower().strip())

    def search(self, query: str, limit: int = 10) -> list[IngredientInfo]:
        """Fuzzy search for ingredients matching a query string."""
        query_lower = query.lower().strip()
        exact = self._db.get(query_lower)
        if exact:
            return [exact]

        # Score by substring match quality
        scored: list[tuple[float, IngredientInfo]] = []
        for key, info in self._db.items():
            if query_lower in key:
                # Prefer shorter keys (more specific matches)
                score = len(query_lower) / len(key)
                scored.append((score, info))
            elif any(word.startswith(query_lower) for word in key.split()):
                scored.append((0.5, info))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [info for _, info in scored[:limit]]

    def get_by_category(self, category: IngredientCategory) -> list[IngredientInfo]:
        """Get all ingredients in a category."""
        return [info for info in self._db.values() if info.category == category]

    def get_substitution_group(self, group_name: str) -> list[IngredientInfo]:
        """Get all ingredients in a substitution group."""
        return [info for info in self._db.values() if info.substitution_group == group_name]

    def find_substitutes(
        self,
        ingredient_name: str,
        *,
        vegan_only: bool = False,
        gluten_free_only: bool = False,
        exclude_allergens: tuple[Allergen, ...] = (),
    ) -> list[IngredientInfo]:
        """Find suitable substitutes for an ingredient, respecting dietary constraints."""
        original = self.get(ingredient_name)
        if original is None:
            return []

        # First try same substitution group
        candidates = self.get_substitution_group(original.substitution_group)

        # Fall back to same category
        if len(candidates) <= 1:
            candidates = self.get_by_category(original.category)

        results = []
        for candidate in candidates:
            if candidate.name.lower() == ingredient_name.lower():
                continue
            if vegan_only and not candidate.is_vegan:
                continue
            if gluten_free_only and not candidate.is_gluten_free:
                continue
            if exclude_allergens and any(a in candidate.allergens for a in exclude_allergens):
                continue
            results.append(candidate)

        return results

    def get_all_allergens_for(self, ingredient_name: str) -> set[Allergen]:
        """Detect allergens present in an ingredient."""
        info = self.get(ingredient_name)
        if info is None:
            return set()
        return set(info.allergens)

    def list_all(self) -> list[IngredientInfo]:
        """Return all ingredients in the database."""
        return list(self._db.values())

    def compute_recipe_nutrition(self, ingredients: dict[str, float]) -> NutritionFacts | None:
        """Compute aggregate nutrition for a recipe.

        Args:
            ingredients: Mapping of ingredient name -> quantity in grams.

        Returns:
            Aggregated NutritionFacts or None if any ingredient is unknown.
        """
        total_cal = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        total_fiber = 0.0
        total_sugar = 0.0
        total_sat_fat = 0.0
        total_sodium = 0.0
        total_cholesterol = 0.0
        total_iron = 0.0
        total_calcium = 0.0
        total_vit_c = 0.0
        total_b12 = 0.0
        total_folate = 0.0
        total_potassium = 0.0
        total_magnesium = 0.0
        total_zinc = 0.0
        total_vit_a = 0.0
        total_vit_d = 0.0

        for name, grams in ingredients.items():
            info = self.get(name)
            if info is None:
                logger.warning("Unknown ingredient: %s", name)
                return None
            factor = grams / 100.0
            n = info.nutrition
            total_cal += n.calories_kcal * factor
            total_protein += n.protein_g * factor
            total_carbs += n.carbs_g * factor
            total_fat += n.fat_g * factor
            total_fiber += n.fiber_g * factor
            total_sugar += n.sugar_g * factor
            total_sat_fat += n.saturated_fat_g * factor
            total_sodium += n.sodium_mg * factor
            total_cholesterol += n.cholesterol_mg * factor
            total_iron += n.micro.iron_mg * factor
            total_calcium += n.micro.calcium_mg * factor
            total_vit_c += n.micro.vitamin_c_mg * factor
            total_b12 += n.micro.vitamin_b12_mcg * factor
            total_folate += n.micro.folate_mcg * factor
            total_potassium += n.micro.potassium_mg * factor
            total_magnesium += n.micro.magnesium_mg * factor
            total_zinc += n.micro.zinc_mg * factor
            total_vit_a += n.micro.vitamin_a_mcg * factor
            total_vit_d += n.micro.vitamin_d_mcg * factor

        return NutritionFacts(
            calories_kcal=round(total_cal, 1),
            protein_g=round(total_protein, 1),
            carbs_g=round(total_carbs, 1),
            fat_g=round(total_fat, 1),
            fiber_g=round(total_fiber, 1),
            sugar_g=round(total_sugar, 1),
            saturated_fat_g=round(total_sat_fat, 1),
            sodium_mg=round(total_sodium, 1),
            cholesterol_mg=round(total_cholesterol, 1),
            micro=Micronutrients(
                iron_mg=round(total_iron, 2),
                calcium_mg=round(total_calcium, 1),
                vitamin_c_mg=round(total_vit_c, 1),
                vitamin_b12_mcg=round(total_b12, 2),
                folate_mcg=round(total_folate, 1),
                potassium_mg=round(total_potassium, 1),
                magnesium_mg=round(total_magnesium, 1),
                zinc_mg=round(total_zinc, 2),
                vitamin_a_mcg=round(total_vit_a, 1),
                vitamin_d_mcg=round(total_vit_d, 2),
            ),
        )


# Daily Reference Intakes (DRI) for adults, used for completeness scoring
DRI_REFERENCE: dict[str, float] = {
    "calories_kcal": 2000.0,
    "protein_g": 50.0,
    "carbs_g": 275.0,
    "fat_g": 78.0,
    "fiber_g": 28.0,
    "iron_mg": 18.0,  # Higher value (pre-menopausal women)
    "calcium_mg": 1000.0,
    "vitamin_c_mg": 90.0,
    "vitamin_b12_mcg": 2.4,
    "folate_mcg": 400.0,
    "potassium_mg": 2600.0,
    "magnesium_mg": 420.0,
    "zinc_mg": 11.0,
    "vitamin_a_mcg": 900.0,
    "vitamin_d_mcg": 15.0,
    "sodium_mg_max": 2300.0,  # Upper limit
    "saturated_fat_g_max": 22.0,  # ~10% of 2000 kcal
    "sugar_g_max": 50.0,  # WHO recommendation
}
