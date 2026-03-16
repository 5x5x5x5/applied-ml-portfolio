"""WildEye - Wildlife Species Classifier from camera trap images.

A conservation biology toolkit that combines deep learning with ecological
analytics to identify wildlife species from camera trap imagery and compute
biodiversity metrics for monitoring ecosystem health.
"""

__version__ = "0.1.0"
__app_name__ = "WildEye"

# Taxonomic classes supported by the classifier.
# Follows ITIS (Integrated Taxonomic Information System) common-name conventions.
SPECIES_LABELS: list[str] = [
    "white_tailed_deer",  # Odocoileus virginianus
    "mule_deer",  # Odocoileus hemionus
    "elk",  # Cervus canadensis
    "moose",  # Alces alces
    "black_bear",  # Ursus americanus
    "grizzly_bear",  # Ursus arctos horribilis
    "gray_wolf",  # Canis lupus
    "coyote",  # Canis latrans
    "red_fox",  # Vulpes vulpes
    "bobcat",  # Lynx rufus
    "mountain_lion",  # Puma concolor
    "raccoon",  # Procyon lotor
    "striped_skunk",  # Mephitis mephitis
    "wild_turkey",  # Meleagris gallopavo
    "bald_eagle",  # Haliaeetus leucocephalus
    "great_horned_owl",  # Bubo virginianus
    "pronghorn",  # Antilocapra americana
    "american_beaver",  # Castor canadensis
    "river_otter",  # Lontra canadensis
    "snowshoe_hare",  # Lepus americanus
    "bighorn_sheep",  # Ovis canadensis
    "wolverine",  # Gulo gulo
    "empty",  # No animal present (vegetation trigger)
    "human",  # Human presence (filter from wildlife data)
]

NUM_SPECIES: int = len(SPECIES_LABELS)
