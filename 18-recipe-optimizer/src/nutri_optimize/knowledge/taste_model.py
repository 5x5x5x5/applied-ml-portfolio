"""Taste compatibility model for recipe optimization.

Models the five basic tastes (sweet, salty, sour, bitter, umami) plus
texture profiles and ingredient pairing compatibility. Used to ensure
that nutritional optimizations don't destroy the flavor of a dish.

Based on food science research on flavor pairing theory and the
concept of shared volatile organic compounds between compatible ingredients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TasteDimension(str, Enum):
    """The five basic tastes recognized by gustatory science."""

    SWEET = "sweet"
    SALTY = "salty"
    SOUR = "sour"
    BITTER = "bitter"
    UMAMI = "umami"


class TextureType(str, Enum):
    """Primary texture categories relevant to mouthfeel."""

    CRISPY = "crispy"
    CRUNCHY = "crunchy"
    CHEWY = "chewy"
    TENDER = "tender"
    CREAMY = "creamy"
    SILKY = "silky"
    FIRM = "firm"
    SOFT = "soft"
    JUICY = "juicy"
    DRY = "dry"
    GELATINOUS = "gelatinous"
    FLAKY = "flaky"


class CookingMethod(str, Enum):
    """Common cooking methods that affect flavor development."""

    RAW = "raw"
    BOILED = "boiled"
    STEAMED = "steamed"
    ROASTED = "roasted"
    GRILLED = "grilled"
    SAUTEED = "sauteed"
    FRIED = "fried"
    BAKED = "baked"
    BRAISED = "braised"
    SMOKED = "smoked"
    FERMENTED = "fermented"
    PICKLED = "pickled"


@dataclass(frozen=True)
class FlavorProfile:
    """Normalized flavor intensity profile (0.0 to 1.0 per dimension)."""

    sweet: float = 0.0
    salty: float = 0.0
    sour: float = 0.0
    bitter: float = 0.0
    umami: float = 0.0

    def dominant_taste(self) -> TasteDimension:
        """Return the most prominent taste dimension."""
        tastes = {
            TasteDimension.SWEET: self.sweet,
            TasteDimension.SALTY: self.salty,
            TasteDimension.SOUR: self.sour,
            TasteDimension.BITTER: self.bitter,
            TasteDimension.UMAMI: self.umami,
        }
        return max(tastes, key=tastes.get)  # type: ignore[arg-type]

    def intensity(self) -> float:
        """Overall flavor intensity (euclidean magnitude, normalized)."""
        return (
            (self.sweet**2 + self.salty**2 + self.sour**2 + self.bitter**2 + self.umami**2) ** 0.5
        ) / (5**0.5)

    def similarity(self, other: FlavorProfile) -> float:
        """Cosine similarity between two flavor profiles (0.0 to 1.0)."""
        dot = (
            self.sweet * other.sweet
            + self.salty * other.salty
            + self.sour * other.sour
            + self.bitter * other.bitter
            + self.umami * other.umami
        )
        mag_a = (
            self.sweet**2 + self.salty**2 + self.sour**2 + self.bitter**2 + self.umami**2
        ) ** 0.5
        mag_b = (
            other.sweet**2 + other.salty**2 + other.sour**2 + other.bitter**2 + other.umami**2
        ) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


@dataclass(frozen=True)
class IngredientTasteProfile:
    """Complete taste and texture profile for an ingredient."""

    name: str
    flavor: FlavorProfile
    textures: tuple[TextureType, ...]
    aroma_intensity: float = 0.5  # 0.0 = bland, 1.0 = very aromatic
    heat_level: float = 0.0  # 0.0 = none, 1.0 = extremely spicy


# ---------------------------------------------------------------------------
# Flavor profiles for common ingredients
# Values are subjective but informed by food science literature
# Scale: 0.0 (absent) to 1.0 (dominant)
# ---------------------------------------------------------------------------

_TASTE_PROFILES: dict[str, IngredientTasteProfile] = {}


def _t(
    name: str,
    sweet: float = 0.0,
    salty: float = 0.0,
    sour: float = 0.0,
    bitter: float = 0.0,
    umami: float = 0.0,
    textures: tuple[TextureType, ...] = (TextureType.SOFT,),
    aroma: float = 0.5,
    heat: float = 0.0,
) -> None:
    _TASTE_PROFILES[name.lower().strip()] = IngredientTasteProfile(
        name=name,
        flavor=FlavorProfile(sweet=sweet, salty=salty, sour=sour, bitter=bitter, umami=umami),
        textures=textures,
        aroma_intensity=aroma,
        heat_level=heat,
    )


# Proteins
_t(
    "chicken breast",
    umami=0.5,
    salty=0.1,
    textures=(TextureType.TENDER, TextureType.FIRM),
    aroma=0.3,
)
_t("salmon", umami=0.7, salty=0.2, textures=(TextureType.FLAKY, TextureType.TENDER), aroma=0.6)
_t("ground beef 85%", umami=0.8, salty=0.2, textures=(TextureType.CHEWY,), aroma=0.6)
_t("tuna canned", umami=0.7, salty=0.4, textures=(TextureType.FLAKY,), aroma=0.5)
_t(
    "shrimp",
    umami=0.6,
    sweet=0.2,
    salty=0.2,
    textures=(TextureType.FIRM, TextureType.CHEWY),
    aroma=0.5,
)
_t("pork tenderloin", umami=0.5, sweet=0.1, textures=(TextureType.TENDER,), aroma=0.4)
_t("turkey breast", umami=0.4, textures=(TextureType.TENDER,), aroma=0.3)
_t("eggs", umami=0.5, salty=0.1, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.4)
_t("cod", umami=0.4, textures=(TextureType.FLAKY, TextureType.TENDER), aroma=0.3)
_t("tofu firm", umami=0.2, textures=(TextureType.FIRM, TextureType.SILKY), aroma=0.1)
_t("tempeh", umami=0.5, bitter=0.1, textures=(TextureType.FIRM, TextureType.CHEWY), aroma=0.5)
_t("seitan", umami=0.3, textures=(TextureType.CHEWY, TextureType.FIRM), aroma=0.2)
_t("bacon", umami=0.8, salty=0.8, sweet=0.1, textures=(TextureType.CRISPY,), aroma=0.9)
_t("ham", umami=0.6, salty=0.7, textures=(TextureType.TENDER,), aroma=0.5)

# Dairy
_t("whole milk", sweet=0.3, textures=(TextureType.CREAMY, TextureType.SILKY), aroma=0.2)
_t("greek yogurt", sour=0.4, sweet=0.1, textures=(TextureType.CREAMY,), aroma=0.3)
_t("cheddar cheese", umami=0.6, salty=0.5, textures=(TextureType.FIRM,), aroma=0.7)
_t("mozzarella", umami=0.3, salty=0.3, textures=(TextureType.CHEWY, TextureType.SOFT), aroma=0.3)
_t("parmesan", umami=0.9, salty=0.7, textures=(TextureType.CRUNCHY, TextureType.DRY), aroma=0.8)
_t("cream cheese", sweet=0.1, sour=0.1, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.2)
_t("ricotta", sweet=0.2, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.2)
_t(
    "cottage cheese",
    sour=0.2,
    salty=0.2,
    textures=(TextureType.SOFT, TextureType.CREAMY),
    aroma=0.2,
)
_t("butter", sweet=0.2, salty=0.2, textures=(TextureType.CREAMY,), aroma=0.5)
_t("heavy cream", sweet=0.2, textures=(TextureType.CREAMY, TextureType.SILKY), aroma=0.3)

# Grains
_t("white rice cooked", sweet=0.1, textures=(TextureType.SOFT,), aroma=0.1)
_t("brown rice cooked", sweet=0.1, textures=(TextureType.CHEWY,), aroma=0.2)
_t("quinoa cooked", bitter=0.1, textures=(TextureType.CRUNCHY,), aroma=0.2)
_t("oats rolled", sweet=0.2, textures=(TextureType.CHEWY, TextureType.SOFT), aroma=0.3)
_t("whole wheat bread", sweet=0.1, textures=(TextureType.CHEWY, TextureType.SOFT), aroma=0.4)
_t("white bread", sweet=0.2, textures=(TextureType.SOFT,), aroma=0.3)
_t("pasta cooked", sweet=0.1, textures=(TextureType.CHEWY, TextureType.SOFT), aroma=0.2)
_t("corn tortilla", sweet=0.2, textures=(TextureType.CHEWY,), aroma=0.4)

# Vegetables
_t("broccoli", bitter=0.3, sweet=0.1, textures=(TextureType.CRUNCHY, TextureType.FIRM), aroma=0.4)
_t("spinach raw", bitter=0.2, textures=(TextureType.SOFT,), aroma=0.3)
_t("kale", bitter=0.4, textures=(TextureType.CHEWY,), aroma=0.3)
_t("sweet potato", sweet=0.6, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.4)
_t("carrots", sweet=0.5, textures=(TextureType.CRUNCHY, TextureType.FIRM), aroma=0.3)
_t(
    "tomatoes",
    sweet=0.3,
    sour=0.3,
    umami=0.4,
    textures=(TextureType.JUICY, TextureType.SOFT),
    aroma=0.5,
)
_t("bell pepper red", sweet=0.5, textures=(TextureType.CRUNCHY, TextureType.JUICY), aroma=0.4)
_t("onion", sweet=0.3, textures=(TextureType.CRUNCHY,), aroma=0.7)
_t("garlic", umami=0.3, textures=(TextureType.FIRM,), aroma=0.9, heat=0.1)
_t("mushrooms white", umami=0.7, textures=(TextureType.CHEWY, TextureType.SOFT), aroma=0.5)
_t("zucchini", sweet=0.1, textures=(TextureType.SOFT, TextureType.JUICY), aroma=0.2)
_t(
    "cauliflower",
    bitter=0.2,
    sweet=0.1,
    textures=(TextureType.CRUNCHY, TextureType.FIRM),
    aroma=0.3,
)
_t("potato", sweet=0.1, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.2)
_t("asparagus", bitter=0.2, sweet=0.1, textures=(TextureType.FIRM, TextureType.CRUNCHY), aroma=0.4)
_t("celery", bitter=0.1, salty=0.1, textures=(TextureType.CRUNCHY, TextureType.JUICY), aroma=0.3)
_t("eggplant", bitter=0.2, textures=(TextureType.SOFT, TextureType.CREAMY), aroma=0.2)
_t("cabbage", bitter=0.2, sweet=0.1, textures=(TextureType.CRUNCHY,), aroma=0.3)
_t("peas green", sweet=0.4, textures=(TextureType.SOFT,), aroma=0.3)
_t("corn kernels", sweet=0.5, textures=(TextureType.CRUNCHY, TextureType.JUICY), aroma=0.3)
_t("lettuce romaine", bitter=0.1, textures=(TextureType.CRUNCHY, TextureType.JUICY), aroma=0.1)

# Fruits
_t("banana", sweet=0.7, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.6)
_t("apple", sweet=0.6, sour=0.2, textures=(TextureType.CRUNCHY, TextureType.JUICY), aroma=0.5)
_t("blueberries", sweet=0.5, sour=0.2, textures=(TextureType.JUICY, TextureType.SOFT), aroma=0.5)
_t("strawberries", sweet=0.6, sour=0.2, textures=(TextureType.JUICY, TextureType.SOFT), aroma=0.7)
_t("avocado", sweet=0.1, textures=(TextureType.CREAMY,), aroma=0.2)
_t("orange", sweet=0.5, sour=0.4, textures=(TextureType.JUICY,), aroma=0.7)
_t("lemon juice", sour=0.9, textures=(TextureType.SILKY,), aroma=0.8)
_t("mango", sweet=0.8, sour=0.1, textures=(TextureType.JUICY, TextureType.SOFT), aroma=0.7)
_t("pineapple", sweet=0.6, sour=0.4, textures=(TextureType.JUICY, TextureType.FIRM), aroma=0.7)

# Legumes
_t("black beans cooked", umami=0.2, textures=(TextureType.CREAMY, TextureType.SOFT), aroma=0.2)
_t("chickpeas cooked", sweet=0.1, textures=(TextureType.FIRM, TextureType.CREAMY), aroma=0.2)
_t("lentils cooked", umami=0.3, textures=(TextureType.SOFT,), aroma=0.3)
_t("kidney beans cooked", sweet=0.1, textures=(TextureType.CREAMY,), aroma=0.2)
_t("edamame", sweet=0.2, umami=0.3, textures=(TextureType.FIRM,), aroma=0.2)

# Nuts & Seeds
_t("almonds", sweet=0.2, textures=(TextureType.CRUNCHY,), aroma=0.5)
_t("walnuts", bitter=0.2, textures=(TextureType.CRUNCHY,), aroma=0.5)
_t("cashews", sweet=0.3, textures=(TextureType.CRUNCHY, TextureType.CREAMY), aroma=0.4)
_t("peanut butter", sweet=0.2, salty=0.3, textures=(TextureType.CREAMY,), aroma=0.6)
_t("chia seeds", textures=(TextureType.GELATINOUS,), aroma=0.1)
_t("flax seeds", textures=(TextureType.CRUNCHY,), aroma=0.2)
_t("sunflower seeds", sweet=0.1, textures=(TextureType.CRUNCHY,), aroma=0.3)
_t("pumpkin seeds", sweet=0.1, textures=(TextureType.CRUNCHY,), aroma=0.3)
_t("sesame seeds", sweet=0.1, textures=(TextureType.CRUNCHY,), aroma=0.5)
_t("tahini", bitter=0.2, textures=(TextureType.CREAMY,), aroma=0.5)

# Fats & Oils
_t("olive oil", bitter=0.2, textures=(TextureType.SILKY,), aroma=0.6)
_t("coconut oil", sweet=0.2, textures=(TextureType.SILKY,), aroma=0.5)
_t("butter", sweet=0.2, salty=0.2, textures=(TextureType.CREAMY,), aroma=0.5)
_t("sesame oil", umami=0.2, textures=(TextureType.SILKY,), aroma=0.8)

# Sweeteners
_t("honey", sweet=0.9, textures=(TextureType.SILKY,), aroma=0.6)
_t("maple syrup", sweet=0.8, textures=(TextureType.SILKY,), aroma=0.6)
_t("white sugar", sweet=1.0, textures=(TextureType.DRY,), aroma=0.0)
_t("brown sugar", sweet=0.9, textures=(TextureType.SOFT,), aroma=0.3)
_t(
    "dark chocolate 70%",
    bitter=0.5,
    sweet=0.4,
    textures=(TextureType.FIRM, TextureType.CREAMY),
    aroma=0.7,
)

# Herbs & Spices
_t("turmeric ground", bitter=0.3, textures=(TextureType.DRY,), aroma=0.6)
_t("black pepper", heat=0.4, textures=(TextureType.DRY,), aroma=0.7)
_t("cinnamon ground", sweet=0.3, textures=(TextureType.DRY,), aroma=0.8)
_t("cumin ground", bitter=0.2, umami=0.2, textures=(TextureType.DRY,), aroma=0.8)
_t("ginger ground", sweet=0.1, textures=(TextureType.DRY,), aroma=0.8, heat=0.3)
_t("basil dried", sweet=0.1, textures=(TextureType.DRY,), aroma=0.8)
_t("oregano dried", bitter=0.2, textures=(TextureType.DRY,), aroma=0.8)
_t("paprika", sweet=0.2, textures=(TextureType.DRY,), aroma=0.6, heat=0.1)
_t("chili powder", bitter=0.1, textures=(TextureType.DRY,), aroma=0.7, heat=0.6)

# Condiments
_t("soy sauce", umami=0.9, salty=0.9, textures=(TextureType.SILKY,), aroma=0.7)
_t("tomato sauce", sweet=0.2, sour=0.3, umami=0.4, textures=(TextureType.SILKY,), aroma=0.5)
_t("balsamic vinegar", sweet=0.3, sour=0.7, textures=(TextureType.SILKY,), aroma=0.7)
_t("dijon mustard", sour=0.3, bitter=0.2, textures=(TextureType.CREAMY,), aroma=0.6, heat=0.2)

# Plant milks
_t("almond milk", sweet=0.2, textures=(TextureType.SILKY,), aroma=0.3)
_t("oat milk", sweet=0.3, textures=(TextureType.CREAMY,), aroma=0.3)
_t("soy milk", sweet=0.1, textures=(TextureType.SILKY,), aroma=0.2)
_t("coconut milk canned", sweet=0.3, textures=(TextureType.CREAMY, TextureType.SILKY), aroma=0.5)
_t("rice milk", sweet=0.3, textures=(TextureType.SILKY,), aroma=0.2)


# ---------------------------------------------------------------------------
# Ingredient pairing compatibility matrix
# Based on flavor pairing theory: ingredients sharing volatile compounds pair well.
# Scores: 1.0 = perfect match, 0.5 = neutral, 0.0 = clash
# Only store notable pairings (>0.7 or <0.3); assume 0.5 for unlisted pairs.
# ---------------------------------------------------------------------------

_GREAT_PAIRINGS: set[frozenset[str]] = {
    # Classic complementary pairings from culinary science
    frozenset({"tomatoes", "basil dried"}),
    frozenset({"tomatoes", "mozzarella"}),
    frozenset({"tomatoes", "olive oil"}),
    frozenset({"garlic", "olive oil"}),
    frozenset({"garlic", "ginger ground"}),
    frozenset({"lemon juice", "salmon"}),
    frozenset({"lemon juice", "chicken breast"}),
    frozenset({"avocado", "lemon juice"}),
    frozenset({"black pepper", "turmeric ground"}),  # Also biochemistry: piperine + curcumin
    frozenset({"cinnamon ground", "banana"}),
    frozenset({"cinnamon ground", "apple"}),
    frozenset({"cinnamon ground", "oats rolled"}),
    frozenset({"honey", "lemon juice"}),
    frozenset({"honey", "ginger ground"}),
    frozenset({"dark chocolate 70%", "strawberries"}),
    frozenset({"dark chocolate 70%", "banana"}),
    frozenset({"peanut butter", "banana"}),
    frozenset({"peanut butter", "dark chocolate 70%"}),
    frozenset({"soy sauce", "ginger ground"}),
    frozenset({"soy sauce", "sesame oil"}),
    frozenset({"soy sauce", "garlic"}),
    frozenset({"salmon", "soy sauce"}),
    frozenset({"chicken breast", "garlic"}),
    frozenset({"ground beef 85%", "onion"}),
    frozenset({"eggs", "bacon"}),
    frozenset({"eggs", "cheddar cheese"}),
    frozenset({"pasta cooked", "parmesan"}),
    frozenset({"pasta cooked", "tomato sauce"}),
    frozenset({"black beans cooked", "cumin ground"}),
    frozenset({"chickpeas cooked", "tahini"}),
    frozenset({"chickpeas cooked", "lemon juice"}),
    frozenset({"spinach raw", "lemon juice"}),
    frozenset({"spinach raw", "garlic"}),
    frozenset({"sweet potato", "cinnamon ground"}),
    frozenset({"sweet potato", "black pepper"}),
    frozenset({"carrots", "ginger ground"}),
    frozenset({"mushrooms white", "garlic"}),
    frozenset({"mushrooms white", "soy sauce"}),
    frozenset({"broccoli", "garlic"}),
    frozenset({"broccoli", "lemon juice"}),
    frozenset({"quinoa cooked", "lemon juice"}),
    frozenset({"blueberries", "oats rolled"}),
    frozenset({"maple syrup", "oats rolled"}),
    frozenset({"greek yogurt", "honey"}),
    frozenset({"greek yogurt", "blueberries"}),
    frozenset({"coconut milk canned", "turmeric ground"}),
    frozenset({"coconut milk canned", "ginger ground"}),
    frozenset({"walnuts", "honey"}),
    frozenset({"almonds", "honey"}),
    frozenset({"bell pepper red", "onion"}),
    frozenset({"corn tortilla", "black beans cooked"}),
    frozenset({"avocado", "tomatoes"}),
    frozenset({"rice milk", "cinnamon ground"}),
    frozenset({"tofu firm", "soy sauce"}),
    frozenset({"tofu firm", "sesame oil"}),
}

_BAD_PAIRINGS: set[frozenset[str]] = {
    # Flavor clashes based on competing strong flavors or textural dissonance
    frozenset({"salmon", "cheddar cheese"}),
    frozenset({"salmon", "cinnamon ground"}),
    frozenset({"tuna canned", "dark chocolate 70%"}),
    frozenset({"shrimp", "parmesan"}),
    frozenset({"cod", "maple syrup"}),
    frozenset({"eggs", "mango"}),
    frozenset({"soy sauce", "honey"}),  # Debatable, but very different culinary traditions
    frozenset({"balsamic vinegar", "soy sauce"}),
    frozenset({"ground beef 85%", "pineapple"}),  # Contested pairing
    frozenset({"banana", "garlic"}),
    frozenset({"strawberries", "soy sauce"}),
    frozenset({"blueberries", "garlic"}),
    frozenset({"orange", "ground beef 85%"}),
    frozenset({"dark chocolate 70%", "garlic"}),
    frozenset({"cinnamon ground", "soy sauce"}),
}


# ---------------------------------------------------------------------------
# Texture compatibility matrix
# Some textures are enhanced by contrast; others conflict.
# ---------------------------------------------------------------------------

_TEXTURE_AFFINITY: dict[tuple[TextureType, TextureType], float] = {
    # Contrast pairings (complementary textures score high)
    (TextureType.CRISPY, TextureType.CREAMY): 0.9,
    (TextureType.CRUNCHY, TextureType.SOFT): 0.85,
    (TextureType.CRUNCHY, TextureType.CREAMY): 0.85,
    (TextureType.FIRM, TextureType.SOFT): 0.7,
    (TextureType.CHEWY, TextureType.CRISPY): 0.75,
    (TextureType.JUICY, TextureType.DRY): 0.7,
    (TextureType.SILKY, TextureType.CRUNCHY): 0.8,
    # Similar textures (too much of the same can be monotonous)
    (TextureType.SOFT, TextureType.SOFT): 0.4,
    (TextureType.CREAMY, TextureType.CREAMY): 0.5,
    (TextureType.CHEWY, TextureType.CHEWY): 0.35,
    (TextureType.DRY, TextureType.DRY): 0.3,
    (TextureType.GELATINOUS, TextureType.GELATINOUS): 0.3,
    # Good similar pairings
    (TextureType.CRISPY, TextureType.CRUNCHY): 0.6,
    (TextureType.TENDER, TextureType.JUICY): 0.8,
    (TextureType.SILKY, TextureType.CREAMY): 0.7,
    (TextureType.FLAKY, TextureType.TENDER): 0.7,
}

# ---------------------------------------------------------------------------
# Cooking method flavor impact
# How cooking methods modify the base flavor profile
# ---------------------------------------------------------------------------

COOKING_METHOD_MODIFIERS: dict[CookingMethod, dict[str, float]] = {
    CookingMethod.RAW: {
        "sweet_mult": 1.0,
        "umami_mult": 1.0,
        "bitter_mult": 1.0,
        "aroma_mult": 0.8,
        "nutrient_retention": 1.0,
    },
    CookingMethod.BOILED: {
        "sweet_mult": 0.9,
        "umami_mult": 1.1,
        "bitter_mult": 0.8,
        "aroma_mult": 0.7,
        "nutrient_retention": 0.6,  # Water-soluble vitamins lost
    },
    CookingMethod.STEAMED: {
        "sweet_mult": 1.0,
        "umami_mult": 1.05,
        "bitter_mult": 0.9,
        "aroma_mult": 0.85,
        "nutrient_retention": 0.85,  # Best for preserving nutrients
    },
    CookingMethod.ROASTED: {
        "sweet_mult": 1.3,
        "umami_mult": 1.4,
        "bitter_mult": 1.1,
        "aroma_mult": 1.4,
        "nutrient_retention": 0.75,  # Maillard reaction enhances flavor
    },
    CookingMethod.GRILLED: {
        "sweet_mult": 1.2,
        "umami_mult": 1.5,
        "bitter_mult": 1.2,
        "aroma_mult": 1.5,
        "nutrient_retention": 0.7,
    },
    CookingMethod.SAUTEED: {
        "sweet_mult": 1.2,
        "umami_mult": 1.3,
        "bitter_mult": 0.9,
        "aroma_mult": 1.3,
        "nutrient_retention": 0.8,
    },
    CookingMethod.FRIED: {
        "sweet_mult": 1.1,
        "umami_mult": 1.3,
        "bitter_mult": 0.8,
        "aroma_mult": 1.4,
        "nutrient_retention": 0.65,
    },
    CookingMethod.BAKED: {
        "sweet_mult": 1.2,
        "umami_mult": 1.2,
        "bitter_mult": 1.0,
        "aroma_mult": 1.3,
        "nutrient_retention": 0.75,
    },
    CookingMethod.BRAISED: {
        "sweet_mult": 1.1,
        "umami_mult": 1.6,
        "bitter_mult": 0.7,
        "aroma_mult": 1.2,
        "nutrient_retention": 0.8,  # Liquid retains nutrients
    },
    CookingMethod.SMOKED: {
        "sweet_mult": 1.0,
        "umami_mult": 1.5,
        "bitter_mult": 1.3,
        "aroma_mult": 1.8,
        "nutrient_retention": 0.7,
    },
    CookingMethod.FERMENTED: {
        "sweet_mult": 0.7,
        "umami_mult": 1.8,
        "bitter_mult": 1.0,
        "aroma_mult": 1.5,
        "nutrient_retention": 0.9,  # Can increase bioavailability
    },
    CookingMethod.PICKLED: {
        "sweet_mult": 0.8,
        "umami_mult": 1.1,
        "bitter_mult": 0.9,
        "aroma_mult": 1.2,
        "nutrient_retention": 0.7,
    },
}


class TasteModel:
    """Evaluates taste compatibility and flavor balance in recipes."""

    def __init__(self) -> None:
        self._profiles = dict(_TASTE_PROFILES)
        logger.info("TasteModel loaded with %d ingredient profiles", len(self._profiles))

    def get_profile(self, ingredient_name: str) -> IngredientTasteProfile | None:
        """Get the taste profile for an ingredient."""
        return self._profiles.get(ingredient_name.lower().strip())

    def pairing_score(self, ingredient_a: str, ingredient_b: str) -> float:
        """Score how well two ingredients pair together (0.0 to 1.0).

        Uses explicit pairing data first, then falls back to flavor profile similarity.
        """
        key = frozenset({ingredient_a.lower().strip(), ingredient_b.lower().strip()})

        if key in _GREAT_PAIRINGS:
            return 0.9
        if key in _BAD_PAIRINGS:
            return 0.15

        # Fall back to flavor profile similarity
        profile_a = self.get_profile(ingredient_a)
        profile_b = self.get_profile(ingredient_b)

        if profile_a is None or profile_b is None:
            return 0.5  # Unknown = neutral

        # Complementary flavors can be good (sweet+sour, umami+salty)
        flavor_sim = profile_a.flavor.similarity(profile_b.flavor)

        # Some contrast is good, too much similarity can be boring
        # Sweet spot is moderate similarity (0.3-0.7)
        if 0.3 <= flavor_sim <= 0.7:
            score = 0.7
        elif flavor_sim > 0.7:
            score = 0.6  # Very similar - acceptable but not exciting
        else:
            score = 0.5  # Very different - risky

        # Texture compatibility bonus
        texture_bonus = self._texture_compatibility(profile_a.textures, profile_b.textures)
        score = 0.7 * score + 0.3 * texture_bonus

        return round(min(1.0, max(0.0, score)), 2)

    def recipe_taste_score(self, ingredient_names: list[str]) -> float:
        """Score the overall taste harmony of a recipe (0.0 to 1.0).

        Considers:
        - Pairwise ingredient compatibility
        - Flavor balance across taste dimensions
        - Texture variety
        """
        if len(ingredient_names) < 2:
            return 0.7  # Single ingredient is neutral

        # Pairwise compatibility (average)
        pair_scores: list[float] = []
        for i, name_a in enumerate(ingredient_names):
            for name_b in ingredient_names[i + 1 :]:
                pair_scores.append(self.pairing_score(name_a, name_b))

        avg_pairing = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5

        # Flavor balance: a good recipe has multiple taste dimensions active
        profiles = [self.get_profile(n) for n in ingredient_names]
        valid_profiles = [p for p in profiles if p is not None]

        if not valid_profiles:
            return avg_pairing

        avg_flavor = FlavorProfile(
            sweet=sum(p.flavor.sweet for p in valid_profiles) / len(valid_profiles),
            salty=sum(p.flavor.salty for p in valid_profiles) / len(valid_profiles),
            sour=sum(p.flavor.sour for p in valid_profiles) / len(valid_profiles),
            bitter=sum(p.flavor.bitter for p in valid_profiles) / len(valid_profiles),
            umami=sum(p.flavor.umami for p in valid_profiles) / len(valid_profiles),
        )

        # Count active dimensions (>0.1 intensity)
        active_dims = sum(
            1
            for val in [
                avg_flavor.sweet,
                avg_flavor.salty,
                avg_flavor.sour,
                avg_flavor.bitter,
                avg_flavor.umami,
            ]
            if val > 0.1
        )

        # 2-3 active dimensions is ideal; 1 is boring; 5 is chaotic
        balance_score = {0: 0.3, 1: 0.5, 2: 0.8, 3: 0.9, 4: 0.7, 5: 0.5}.get(active_dims, 0.5)

        # Texture variety bonus
        all_textures = set()
        for p in valid_profiles:
            all_textures.update(p.textures)
        texture_variety = min(1.0, len(all_textures) / 4.0)  # 4+ textures = max variety

        # Weighted combination
        return round(
            0.5 * avg_pairing + 0.3 * balance_score + 0.2 * texture_variety,
            3,
        )

    def suggest_flavor_enhancers(self, ingredient_names: list[str]) -> list[str]:
        """Suggest ingredients that would improve flavor balance.

        Identifies missing taste dimensions and recommends ingredients
        that fill the gap.
        """
        profiles = [self.get_profile(n) for n in ingredient_names]
        valid = [p for p in profiles if p is not None]
        if not valid:
            return []

        # Find weak dimensions
        avg_sweet = sum(p.flavor.sweet for p in valid) / len(valid)
        avg_salty = sum(p.flavor.salty for p in valid) / len(valid)
        avg_sour = sum(p.flavor.sour for p in valid) / len(valid)
        avg_umami = sum(p.flavor.umami for p in valid) / len(valid)

        suggestions: list[str] = []

        if avg_umami < 0.15:
            suggestions.extend(["mushrooms white", "soy sauce", "tomatoes", "parmesan"])
        if avg_sour < 0.1:
            suggestions.extend(["lemon juice", "balsamic vinegar"])
        if avg_sweet < 0.1 and avg_salty > 0.3:
            suggestions.extend(["honey", "carrots", "sweet potato"])
        if avg_salty < 0.1 and avg_sweet < 0.5:
            suggestions.append("soy sauce")

        # Remove ingredients already in the recipe
        existing = {n.lower().strip() for n in ingredient_names}
        return [s for s in suggestions if s not in existing][:5]

    def _texture_compatibility(
        self,
        textures_a: tuple[TextureType, ...],
        textures_b: tuple[TextureType, ...],
    ) -> float:
        """Compute texture compatibility between two ingredients."""
        if not textures_a or not textures_b:
            return 0.5

        scores: list[float] = []
        for ta in textures_a:
            for tb in textures_b:
                key = (ta, tb)
                rev_key = (tb, ta)
                if key in _TEXTURE_AFFINITY:
                    scores.append(_TEXTURE_AFFINITY[key])
                elif rev_key in _TEXTURE_AFFINITY:
                    scores.append(_TEXTURE_AFFINITY[rev_key])
                else:
                    scores.append(0.55)  # Default neutral-positive

        return sum(scores) / len(scores) if scores else 0.5

    def apply_cooking_method(
        self,
        ingredient_name: str,
        method: CookingMethod,
    ) -> FlavorProfile | None:
        """Estimate how a cooking method modifies an ingredient's flavor.

        Maillard reaction (roasting, grilling) enhances sweetness and umami.
        Boiling can leach flavor. Fermentation dramatically increases umami.
        """
        profile = self.get_profile(ingredient_name)
        if profile is None:
            return None

        mods = COOKING_METHOD_MODIFIERS.get(method)
        if mods is None:
            return profile.flavor

        return FlavorProfile(
            sweet=min(1.0, profile.flavor.sweet * mods["sweet_mult"]),
            salty=profile.flavor.salty,  # Cooking doesn't change inherent saltiness
            sour=profile.flavor.sour,
            bitter=min(1.0, profile.flavor.bitter * mods["bitter_mult"]),
            umami=min(1.0, profile.flavor.umami * mods["umami_mult"]),
        )
