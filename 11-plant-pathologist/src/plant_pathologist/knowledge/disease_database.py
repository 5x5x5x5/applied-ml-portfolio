"""Plant disease knowledge base with pathology data and treatment recommendations.

Contains curated information for 23 disease/healthy classes across 5 crop species.
Disease descriptions use standard phytopathological terminology. Treatment
recommendations include both organic and conventional chemical options with
appropriate safety caveats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Disease severity levels based on visual symptom assessment."""

    HEALTHY = "healthy"
    MILD = "mild"  # <10% leaf area affected
    MODERATE = "moderate"  # 10-25% leaf area affected
    SEVERE = "severe"  # 25-50% leaf area affected
    CRITICAL = "critical"  # >50% leaf area affected


class PathogenType(str, Enum):
    """Classification of the causative agent."""

    NONE = "none"
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    OOMYCETE = "oomycete"


@dataclass
class TreatmentInfo:
    """Treatment recommendations for a plant disease."""

    organic: list[str]
    chemical: list[str]
    cultural: list[str]
    biological: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DiseaseInfo:
    """Complete information record for a single disease class."""

    disease_id: str
    common_name: str
    scientific_name: str
    pathogen: str
    pathogen_type: PathogenType
    plant_species: str
    description: str
    symptoms: list[str]
    conditions: str
    severity_criteria: dict[str, str]
    treatment: TreatmentInfo
    prevention: list[str]
    spread_mechanism: str
    is_healthy: bool = False


# ---------------------------------------------------------------------------
# Complete disease knowledge base
# ---------------------------------------------------------------------------

DISEASE_DATABASE: dict[str, DiseaseInfo] = {
    # ===================== TOMATO =====================
    "tomato_healthy": DiseaseInfo(
        disease_id="tomato_healthy",
        common_name="Healthy Tomato",
        scientific_name="Solanum lycopersicum",
        pathogen="N/A",
        pathogen_type=PathogenType.NONE,
        plant_species="tomato",
        description="Healthy tomato foliage with no visible disease symptoms. Leaves are uniformly green with normal morphology.",
        symptoms=[
            "Uniform green coloration",
            "No spots or lesions",
            "Normal leaf shape and turgor",
        ],
        conditions="N/A",
        severity_criteria={
            "healthy": "No symptoms present. Plant tissue appears vigorous and normal.",
        },
        treatment=TreatmentInfo(
            organic=["Maintain current cultural practices"],
            chemical=["No treatment needed"],
            cultural=["Continue proper watering and fertilization"],
        ),
        prevention=[
            "Crop rotation",
            "Proper spacing for air circulation",
            "Avoid overhead irrigation",
        ],
        spread_mechanism="N/A",
        is_healthy=True,
    ),
    "tomato_early_blight": DiseaseInfo(
        disease_id="tomato_early_blight",
        common_name="Tomato Early Blight",
        scientific_name="Alternaria solani",
        pathogen="Alternaria solani",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="tomato",
        description=(
            "Fungal foliar disease caused by Alternaria solani. Characterized by dark brown to black "
            "lesions with distinctive concentric rings forming a 'target spot' or 'bull's eye' pattern. "
            "Typically begins on older, lower leaves and progresses upward. Can cause significant "
            "defoliation and yield loss if left untreated."
        ),
        symptoms=[
            "Dark brown to black circular lesions (1-2 cm diameter)",
            "Concentric rings within lesions ('target spot' pattern)",
            "Yellow chlorotic halo surrounding lesions",
            "Lesions first appear on oldest lower leaves",
            "Progressive defoliation from bottom upward",
            "Stem lesions may develop as dark, sunken cankers",
        ],
        conditions="Warm temperatures (24-29 C), high humidity, alternating wet and dry periods. Favored by heavy dew and frequent rainfall.",
        severity_criteria={
            "mild": "1-5 small lesions on lower leaves only, <10% leaf area affected",
            "moderate": "Multiple lesions spreading to middle canopy, 10-25% defoliation",
            "severe": "Widespread lesions with coalescence, 25-50% defoliation",
            "critical": "Extensive defoliation >50%, stem cankers present, fruit lesions",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides (copper hydroxide, Bordeaux mixture)",
                "Bacillus subtilis-based biofungicides (e.g., Serenade)",
                "Neem oil foliar spray (70% neem oil concentrate)",
            ],
            chemical=[
                "Chlorothalonil (Bravo, Daconil) - protectant",
                "Mancozeb (Dithane) - protectant",
                "Azoxystrobin (Quadris) - systemic strobilurin",
                "Difenoconazole + cyprodinil (Inspire Super) - systemic",
            ],
            cultural=[
                "Remove and destroy infected lower leaves promptly",
                "Mulch around base to prevent soil splash",
                "Improve air circulation via proper spacing and pruning",
                "Avoid overhead irrigation; use drip irrigation",
                "Water in early morning so foliage dries quickly",
            ],
            biological=[
                "Trichoderma harzianum soil application",
                "Bacillus amyloliquefaciens foliar spray",
            ],
            notes="Rotate fungicide modes of action to prevent resistance development. Begin preventive applications before symptom onset in high-risk conditions.",
        ),
        prevention=[
            "3-year crop rotation away from Solanaceae",
            "Use certified disease-free seed and transplants",
            "Select resistant or tolerant cultivars (e.g., 'Mountain Magic', 'Defiant PhR')",
            "Remove and destroy crop debris after harvest",
            "Maintain adequate plant nutrition (especially calcium and potassium)",
            "Stake or cage plants to improve air circulation",
        ],
        spread_mechanism="Airborne conidia, rain splash, contaminated seed, infected crop debris",
    ),
    "tomato_late_blight": DiseaseInfo(
        disease_id="tomato_late_blight",
        common_name="Tomato Late Blight",
        scientific_name="Phytophthora infestans",
        pathogen="Phytophthora infestans",
        pathogen_type=PathogenType.OOMYCETE,
        plant_species="tomato",
        description=(
            "Devastating oomycete disease caused by Phytophthora infestans - the same pathogen "
            "responsible for the Irish Potato Famine. Rapidly spreading under cool, wet conditions. "
            "Can destroy an entire field within days if conditions are favorable. Requires immediate "
            "action upon detection."
        ),
        symptoms=[
            "Large (>2 cm) water-soaked, pale green to dark brown lesions",
            "White cottony sporulation on leaf undersides in humid conditions",
            "Lesions rapidly expand and turn dark brown/black",
            "Affected tissue has a distinctly 'greasy' or water-soaked appearance",
            "Entire leaflets and stems can collapse within days",
            "Brown firm rot on fruit, often starting at stem end",
        ],
        conditions="Cool temperatures (15-22 C), prolonged leaf wetness (>10 hours), high relative humidity (>90%). Most severe in cool, rainy weather.",
        severity_criteria={
            "mild": "Few scattered lesions, <10% leaf area, no sporulation visible",
            "moderate": "Multiple lesions with active sporulation, 10-25% canopy affected",
            "severe": "Rapid lesion expansion, 25-50% canopy affected, stem lesions",
            "critical": "Widespread canopy collapse >50%, fruit infection, potential total loss",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides applied preventively (copper hydroxide or copper octanoate)",
                "Bacillus subtilis strain QST 713 (Serenade Optimum)",
            ],
            chemical=[
                "Mefenoxam/metalaxyl + mancozeb (Ridomil Gold MZ) - systemic",
                "Cymoxanil + famoxadone (Tanos) - locally systemic",
                "Mandipropamid (Revus) - translaminar",
                "Fluopicolide + propamocarb (Infinito) - systemic",
                "Dimethomorph (Forum) - locally systemic",
            ],
            cultural=[
                "Remove and destroy ALL infected plant material immediately",
                "Do not compost infected tissue - burn or bag and discard",
                "Improve drainage and air circulation",
                "Avoid overhead irrigation entirely during outbreaks",
            ],
            biological=[
                "Trichoderma virens (SoilGard) for soil applications",
            ],
            notes="CRITICAL: Late blight spreads extremely rapidly. Report outbreaks to local extension service. Sporangia can travel 10+ km on wind. Timing is critical - begin applications before or at very first symptoms.",
        ),
        prevention=[
            "Plant only certified disease-free seed potatoes and transplants",
            "Destroy all volunteer potato and tomato plants",
            "Monitor USABlight.org for regional late blight alerts",
            "Use resistant varieties where available (e.g., 'Iron Lady', 'Defiant PhR')",
            "Ensure excellent air circulation and drainage",
            "Apply protectant fungicides preventively during high-risk periods",
        ],
        spread_mechanism="Airborne sporangia (can travel long distances on wind), infected seed tubers, contaminated transplants, volunteer plants",
    ),
    "tomato_bacterial_spot": DiseaseInfo(
        disease_id="tomato_bacterial_spot",
        common_name="Tomato Bacterial Spot",
        scientific_name="Xanthomonas campestris pv. vesicatoria",
        pathogen="Xanthomonas spp. (X. euvesicatoria, X. vesicatoria, X. perforans, X. gardneri)",
        pathogen_type=PathogenType.BACTERIAL,
        plant_species="tomato",
        description=(
            "Bacterial foliar disease caused by several Xanthomonas species. Produces small, "
            "dark, water-soaked lesions on leaves, stems, and fruit. Particularly problematic "
            "in warm, humid climates with frequent rainfall. No curative treatments exist once "
            "established."
        ),
        symptoms=[
            "Small (1-3 mm) dark, water-soaked spots on leaves",
            "Spots become raised, dark brown to black, and scab-like",
            "Yellow halo may surround lesions",
            "Spots coalesce causing irregular necrotic patches",
            "Severe infections cause defoliation exposing fruit to sunscald",
            "Fruit spots are raised, scab-like, brown to black",
        ],
        conditions="Warm temperatures (24-30 C), high humidity, rain and wind-driven rain for splash dispersal. Enters through stomata and wounds.",
        severity_criteria={
            "mild": "Scattered small spots on a few leaves, <10% leaf area",
            "moderate": "Spots spreading across canopy, some defoliation, 10-25%",
            "severe": "Significant defoliation, fruit spotting, 25-50%",
            "critical": "Severe defoliation >50%, heavy fruit spotting, major yield loss",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper hydroxide or copper sulfate (preventive only)",
                "Copper + mancozeb tank mix for improved efficacy",
                "Acibenzolar-S-methyl (Actigard) - plant defense activator",
            ],
            chemical=[
                "Copper-based bactericides + mancozeb",
                "Streptomycin (limited efficacy due to widespread resistance)",
                "Fixed copper formulations (Kocide, Badge)",
            ],
            cultural=[
                "Remove severely infected plants to reduce inoculum",
                "Avoid working in fields when foliage is wet",
                "Disinfect tools between plants with 10% bleach solution",
                "Use drip irrigation instead of overhead sprinklers",
            ],
            biological=[
                "Bacteriophage-based products (e.g., AgriPhage)",
            ],
            notes="No curative treatments exist for bacterial diseases. Management focuses on prevention and reducing spread. Copper resistance is increasingly common in Xanthomonas populations.",
        ),
        prevention=[
            "Use certified pathogen-free seed (hot water treated at 50 C for 25 min)",
            "Plant resistant varieties where available",
            "Crop rotation (minimum 1 year away from solanaceous crops)",
            "Avoid overhead irrigation",
            "Control weeds that may serve as alternate hosts",
            "Remove and destroy crop residue after harvest",
        ],
        spread_mechanism="Rain splash, wind-driven rain, contaminated seed, transplants, worker hands and tools",
    ),
    "tomato_septoria_leaf_spot": DiseaseInfo(
        disease_id="tomato_septoria_leaf_spot",
        common_name="Septoria Leaf Spot",
        scientific_name="Septoria lycopersici",
        pathogen="Septoria lycopersici",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="tomato",
        description=(
            "One of the most common foliar diseases of tomato worldwide. Causes numerous small, "
            "circular spots with dark borders and gray-white centers containing tiny black pycnidia "
            "(fruiting bodies). Primarily a leaf disease that can cause severe defoliation."
        ),
        symptoms=[
            "Numerous small (2-3 mm) circular spots with dark brown margins",
            "Gray to white centers with tiny black dots (pycnidia)",
            "Begins on lowest leaves and progresses upward",
            "Severe infection causes extensive yellowing and defoliation",
            "Rarely affects fruit directly",
            "Pycnidia visible with hand lens as black specks in lesion centers",
        ],
        conditions="Moderate temperatures (20-25 C), prolonged leaf wetness, frequent rain. Can occur anytime during growing season but most common after fruit set.",
        severity_criteria={
            "mild": "Few spots on lower leaves, pycnidia beginning to form",
            "moderate": "Spreading to mid-canopy, lower leaf yellowing and drop",
            "severe": "Extensive spotting through canopy, 25-50% defoliation",
            "critical": "Near-total defoliation, fruit exposed to sunscald",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides (preventive application)",
                "Bacillus subtilis biofungicides",
                "Potassium bicarbonate sprays",
            ],
            chemical=[
                "Chlorothalonil (Bravo/Daconil) - standard protectant",
                "Mancozeb (Dithane/Penncozeb) - protectant",
                "Azoxystrobin (Quadris) - systemic",
                "Pyraclostrobin + boscalid (Pristine) - systemic premix",
            ],
            cultural=[
                "Remove infected lower leaves as soon as spots appear",
                "Mulch to prevent rain splash from soil",
                "Improve air circulation through wider spacing",
                "Stake plants and prune suckers to promote airflow",
            ],
        ),
        prevention=[
            "Use pathogen-free seed and transplants",
            "3-year rotation away from tomato and other solanaceous crops",
            "Remove and destroy all crop debris at end of season",
            "Avoid overhead watering",
            "Space plants adequately for good air flow",
        ],
        spread_mechanism="Rain splash dispersal of pycnidiospores, contaminated crop debris, seed-borne",
    ),
    "tomato_leaf_mold": DiseaseInfo(
        disease_id="tomato_leaf_mold",
        common_name="Tomato Leaf Mold",
        scientific_name="Passalora fulva (syn. Cladosporium fulvum)",
        pathogen="Passalora fulva",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="tomato",
        description=(
            "Primarily a greenhouse disease caused by Passalora fulva. Produces distinctive olive-green "
            "to brown velvety sporulation on leaf undersides. Favored by high humidity and poor "
            "ventilation. Can be severe in high tunnels and greenhouses."
        ),
        symptoms=[
            "Pale green to yellowish diffuse spots on upper leaf surface",
            "Olive-green to brown velvety mold on corresponding lower leaf surface",
            "Older leaves affected first, progressing upward",
            "Leaves curl, wither, and drop prematurely",
            "Rarely affects stems or fruit in field conditions",
        ],
        conditions="High humidity (>85% RH), moderate temperatures (22-24 C), poor air circulation. Most common in greenhouses and high tunnels.",
        severity_criteria={
            "mild": "Few spots on lower leaves, light sporulation visible",
            "moderate": "Spreading across lower and middle canopy, visible mold",
            "severe": "Heavy sporulation, significant leaf curling and drop",
            "critical": "Widespread defoliation, yield reduction from poor photosynthesis",
        },
        treatment=TreatmentInfo(
            organic=[
                "Improve ventilation and reduce humidity below 85%",
                "Potassium bicarbonate foliar sprays",
                "Neem oil applications",
            ],
            chemical=[
                "Chlorothalonil (preventive)",
                "Mancozeb (preventive)",
                "Difenoconazole (systemic triazole)",
            ],
            cultural=[
                "Increase greenhouse ventilation significantly",
                "Reduce humidity with fans and proper venting",
                "Remove and destroy infected leaves",
                "Avoid wetting foliage during irrigation",
            ],
        ),
        prevention=[
            "Select resistant varieties (many Cf gene-carrying cultivars available)",
            "Maintain greenhouse humidity below 85%",
            "Ensure adequate plant spacing and ventilation",
            "Sanitize greenhouse structures between crops",
        ],
        spread_mechanism="Airborne conidia, can survive on greenhouse structures and debris for extended periods",
    ),
    "tomato_yellow_leaf_curl_virus": DiseaseInfo(
        disease_id="tomato_yellow_leaf_curl_virus",
        common_name="Tomato Yellow Leaf Curl Virus (TYLCV)",
        scientific_name="Tomato yellow leaf curl virus (Begomovirus)",
        pathogen="Tomato yellow leaf curl virus (TYLCV)",
        pathogen_type=PathogenType.VIRAL,
        plant_species="tomato",
        description=(
            "Devastating viral disease transmitted by the sweetpotato whitefly (Bemisia tabaci). "
            "Causes severe stunting, leaf curling, and yellowing. Once infected, plants cannot be "
            "cured. Major threat in tropical and subtropical regions, increasingly problematic in "
            "temperate zones with whitefly migration."
        ),
        symptoms=[
            "Severe upward curling and cupping of leaves",
            "Interveinal yellowing (chlorosis) of young leaves",
            "Stunted growth with shortened internodes",
            "Plants become bushy with abnormally small leaves",
            "Flower drop and drastically reduced fruit set",
            "No necrosis or spotting (distinguishes from other diseases)",
        ],
        conditions="Warm weather favoring whitefly populations (>25 C). Transmitted by Bemisia tabaci biotype B with a persistent, circulative transmission mechanism. Virus acquired in 15-30 min feeding.",
        severity_criteria={
            "mild": "Slight leaf curling on newest growth, virus may not yet be systemic",
            "moderate": "Pronounced curling and yellowing, stunting evident",
            "severe": "Severe stunting, most leaves affected, flower drop",
            "critical": "Plant severely dwarfed, minimal fruit production, complete crop loss likely",
        },
        treatment=TreatmentInfo(
            organic=[
                "Remove and destroy infected plants immediately",
                "Control whitefly vectors with insecticidal soap or neem oil",
                "Use yellow sticky traps to monitor and reduce whitefly populations",
                "Reflective mulch to repel whiteflies",
            ],
            chemical=[
                "Imidacloprid or thiamethoxam (neonicotinoids) for whitefly control",
                "Cyantraniliprole (Cyazypyr/Verimark) - newer chemistry for whiteflies",
                "Spiromesifen (Oberon) - whitefly control",
                "NOTE: No antiviral treatments exist for plants",
            ],
            cultural=[
                "Remove and destroy infected plants as soon as symptoms appear",
                "Control whitefly populations aggressively",
                "Use UV-reflective mulch to deter whitefly landing",
                "Install fine-mesh insect exclusion screens on greenhouse openings",
                "Plant whitefly-free transplants",
            ],
            notes="There is NO cure for viral infections in plants. Management focuses entirely on vector (whitefly) control and use of resistant varieties. Remove infected plants to reduce virus reservoir.",
        ),
        prevention=[
            "Plant TYLCV-resistant varieties (Ty-1, Ty-2, Ty-3 gene carriers)",
            "Use whitefly-free transplants from certified nurseries",
            "Implement whitefly IPM program before planting",
            "Install 50-mesh insect screens on greenhouse vents",
            "Maintain a crop-free period to break the virus cycle",
            "Remove all solanaceous weeds that can harbor the virus",
        ],
        spread_mechanism="Vectored by Bemisia tabaci (sweetpotato whitefly) in a persistent, circulative manner. Not mechanically transmitted. Not seed-borne.",
    ),
    "tomato_target_spot": DiseaseInfo(
        disease_id="tomato_target_spot",
        common_name="Tomato Target Spot",
        scientific_name="Corynespora cassiicola",
        pathogen="Corynespora cassiicola",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="tomato",
        description=(
            "Foliar disease caused by Corynespora cassiicola, producing target-like lesions "
            "that can be confused with early blight. Distinguished by more uniform concentric "
            "rings and lighter tan centers. Can affect leaves, stems, and fruit."
        ),
        symptoms=[
            "Circular lesions with concentric rings (target-like pattern)",
            "Lesion centers are tan to light brown (lighter than early blight)",
            "Lesions range from 1 mm to 1 cm in diameter",
            "Dark brown borders with occasional yellow halos",
            "Can cause premature defoliation starting from lower canopy",
            "Fruit lesions appear as small, sunken, dark spots",
        ],
        conditions="Warm (25-30 C), humid conditions with prolonged leaf wetness. Common in tropical and subtropical regions.",
        severity_criteria={
            "mild": "Few lesions on lower leaves, <10% area affected",
            "moderate": "Lesions on multiple leaf levels, 10-25% area",
            "severe": "Widespread lesions, significant defoliation, 25-50%",
            "critical": "Extensive defoliation and fruit infection, >50% damage",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides",
                "Bacillus subtilis biofungicides",
            ],
            chemical=[
                "Azoxystrobin (Quadris)",
                "Chlorothalonil (Daconil)",
                "Difenoconazole (Score/Inspire)",
                "Pyraclostrobin (Cabrio)",
            ],
            cultural=[
                "Remove and destroy infected plant debris",
                "Improve air circulation",
                "Avoid overhead irrigation",
                "Prune lower branches to reduce humidity near soil",
            ],
        ),
        prevention=[
            "Crop rotation (2-3 years away from susceptible hosts)",
            "Remove crop residue promptly after harvest",
            "Use resistant varieties when available",
            "Maintain proper plant spacing",
        ],
        spread_mechanism="Airborne conidia, rain splash, infected crop debris",
    ),
    # ===================== POTATO =====================
    "potato_healthy": DiseaseInfo(
        disease_id="potato_healthy",
        common_name="Healthy Potato",
        scientific_name="Solanum tuberosum",
        pathogen="N/A",
        pathogen_type=PathogenType.NONE,
        plant_species="potato",
        description="Healthy potato foliage showing normal growth and coloration with no disease symptoms.",
        symptoms=["Uniform dark green leaves", "No spots or lesions", "Normal growth habit"],
        conditions="N/A",
        severity_criteria={"healthy": "No symptoms present."},
        treatment=TreatmentInfo(
            organic=["Maintain current practices"],
            chemical=["No treatment needed"],
            cultural=["Continue balanced fertilization and irrigation"],
        ),
        prevention=["Use certified seed potatoes", "Proper hill cultivation", "Crop rotation"],
        spread_mechanism="N/A",
        is_healthy=True,
    ),
    "potato_early_blight": DiseaseInfo(
        disease_id="potato_early_blight",
        common_name="Potato Early Blight",
        scientific_name="Alternaria solani",
        pathogen="Alternaria solani",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="potato",
        description=(
            "Fungal disease of potato caused by Alternaria solani (same pathogen as tomato early blight). "
            "Produces characteristic target-spot lesions on leaves and can cause tuber lesions. "
            "Typically a disease of plant stress and senescence."
        ),
        symptoms=[
            "Dark brown circular lesions with concentric rings on leaves",
            "Lesions first on older lower leaves",
            "Yellowing around lesions",
            "Progressive defoliation reducing tuber bulking",
            "Dark, dry, sunken lesions on tubers with raised borders",
        ],
        conditions="Warm temperatures (20-30 C), alternating wet and dry conditions, plant stress (nutrient deficiency, drought).",
        severity_criteria={
            "mild": "Scattered lesions on lower leaves, <10% area",
            "moderate": "Lesions on mid-canopy, 10-25% defoliation",
            "severe": "Widespread lesions and defoliation, 25-50%",
            "critical": "Severe defoliation, premature vine death, tuber lesions",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper hydroxide sprays",
                "Bacillus subtilis (Serenade)",
                "Potassium bicarbonate",
            ],
            chemical=[
                "Chlorothalonil (Bravo) - protectant",
                "Azoxystrobin (Quadris) - systemic",
                "Difenoconazole (Inspire/Score) - systemic",
                "Boscalid + pyraclostrobin (Pristine) - premix",
            ],
            cultural=[
                "Maintain adequate plant nutrition (especially nitrogen)",
                "Avoid drought stress through consistent irrigation",
                "Remove infected foliage",
                "Hill potatoes adequately to protect tubers",
            ],
        ),
        prevention=[
            "Use certified disease-free seed potatoes",
            "Crop rotation (3+ years away from Solanaceae)",
            "Maintain adequate fertility and irrigation",
            "Destroy crop debris and volunteer plants",
        ],
        spread_mechanism="Airborne conidia, rain splash, infected tubers",
    ),
    "potato_late_blight": DiseaseInfo(
        disease_id="potato_late_blight",
        common_name="Potato Late Blight",
        scientific_name="Phytophthora infestans",
        pathogen="Phytophthora infestans",
        pathogen_type=PathogenType.OOMYCETE,
        plant_species="potato",
        description=(
            "The most historically significant plant disease, responsible for the Irish Potato "
            "Famine of the 1840s. Caused by the oomycete Phytophthora infestans. Can destroy "
            "a potato field within a week under favorable conditions. Remains the most expensive "
            "disease to manage in potato production worldwide."
        ),
        symptoms=[
            "Water-soaked, pale green lesions on leaf tips and margins",
            "Lesions rapidly turn dark brown to black",
            "White cottony sporangia on leaf undersides (in humid conditions)",
            "Characteristic 'musty' odor from infected tissue",
            "Reddish-brown granular rot in tubers, often starting from eyes",
            "Entire plants can collapse within days",
        ],
        conditions="Cool (10-20 C), wet conditions. Requires >10 hours leaf wetness. Sporangia produced at >90% RH and 15-20 C.",
        severity_criteria={
            "mild": "Small water-soaked lesions on a few leaves, <5% area",
            "moderate": "Active sporulation visible, 10-25% canopy affected",
            "severe": "Rapid spread, stem lesions, 25-50% canopy affected",
            "critical": "Vine collapse, tuber rot starting, potential total loss",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper sulfate (Bordeaux mixture) - preventive only",
                "Copper hydroxide (Kocide) - preventive",
            ],
            chemical=[
                "Mefenoxam + mancozeb (Ridomil Gold MZ) - systemic",
                "Cymoxanil + famoxadone (Tanos) - locally systemic",
                "Fluopicolide (Presidio) - systemic",
                "Mandipropamid (Revus) - translaminar",
                "Cyazofamid (Ranman) - preventive",
            ],
            cultural=[
                "Destroy ALL infected plants and tubers immediately",
                "Kill vines 2-3 weeks before harvest to prevent tuber infection",
                "Avoid harvesting in wet conditions",
                "Ensure tuber curing before storage",
                "Do NOT compost infected material",
            ],
            notes="Late blight is a community disease - one infected field threatens the entire region. Report to local extension service immediately.",
        ),
        prevention=[
            "Plant only certified seed potatoes",
            "Eliminate all cull piles and volunteer plants",
            "Monitor DSV (Disease Severity Values) or BLITECAST forecasts",
            "Select resistant cultivars (e.g., 'Defender', 'Elba')",
            "Ensure proper hilling to protect tubers",
            "Destroy all crop debris after harvest",
        ],
        spread_mechanism="Airborne sporangia (wind-dispersed 10+ km), infected seed tubers, cull piles, volunteer plants",
    ),
    # ===================== CORN =====================
    "corn_healthy": DiseaseInfo(
        disease_id="corn_healthy",
        common_name="Healthy Corn",
        scientific_name="Zea mays",
        pathogen="N/A",
        pathogen_type=PathogenType.NONE,
        plant_species="corn",
        description="Healthy corn foliage with uniform green coloration and no disease symptoms.",
        symptoms=["Uniform green leaves", "No lesions or discoloration", "Normal growth"],
        conditions="N/A",
        severity_criteria={"healthy": "No symptoms present."},
        treatment=TreatmentInfo(
            organic=["Maintain current practices"],
            chemical=["No treatment needed"],
            cultural=["Continue proper cultural management"],
        ),
        prevention=[
            "Crop rotation",
            "Hybrid selection for local disease pressure",
            "Balanced nutrition",
        ],
        spread_mechanism="N/A",
        is_healthy=True,
    ),
    "corn_northern_leaf_blight": DiseaseInfo(
        disease_id="corn_northern_leaf_blight",
        common_name="Northern Corn Leaf Blight (NCLB)",
        scientific_name="Exserohilum turcicum (syn. Setosphaeria turcica)",
        pathogen="Exserohilum turcicum",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="corn",
        description=(
            "Major foliar disease of corn caused by Exserohilum turcicum. Characterized by long, "
            "elliptical, grayish-green to tan lesions. Can cause significant yield reduction if "
            "infection occurs before or during silking. One of the most important corn diseases in "
            "temperate regions."
        ),
        symptoms=[
            "Large (2-15 cm) elliptical, cigar-shaped lesions",
            "Lesions are grayish-green when young, turning tan/buff at maturity",
            "Dark gray-green sporulation may be visible on lesion surface in humid conditions",
            "Lesions may coalesce, blighting large portions of leaf",
            "Lower leaves affected first, progressing upward",
            "Severe infections reduce photosynthetic area and grain fill",
        ],
        conditions="Moderate temperatures (18-27 C), heavy dew, frequent rainfall, 6+ hours of leaf wetness required for infection.",
        severity_criteria={
            "mild": "Few lesions on lower leaves, <5% leaf area affected",
            "moderate": "Lesions on middle leaves, 10-25% leaf area blighted",
            "severe": "Lesions on upper leaves including ear leaf, 25-50% blighted",
            "critical": "Extensive blighting of ear leaf and above, >50%, stalk rot risk",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides (limited efficacy)",
                "Crop rotation to reduce inoculum",
            ],
            chemical=[
                "Azoxystrobin + propiconazole (Quilt Xcel) - foliar",
                "Pyraclostrobin + metconazole (Headline AMP) - foliar",
                "Trifloxystrobin + prothioconazole (Stratego YLD)",
                "Apply at VT-R1 (tasseling-silking) for maximum ROI",
            ],
            cultural=[
                "Tillage to bury infected crop residue",
                "Crop rotation (corn-soybean or corn-small grain)",
                "Plant hybrids with Ht resistance genes (Ht1, Ht2, Ht3, HtN)",
            ],
        ),
        prevention=[
            "Select hybrids with polygenic or Ht gene resistance",
            "Rotate away from continuous corn",
            "Manage crop residue through tillage or decomposition",
            "Scout fields regularly from V8 through grain fill",
        ],
        spread_mechanism="Airborne conidia produced on overwintered crop residue, rain splash",
    ),
    "corn_common_rust": DiseaseInfo(
        disease_id="corn_common_rust",
        common_name="Common Corn Rust",
        scientific_name="Puccinia sorghi",
        pathogen="Puccinia sorghi",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="corn",
        description=(
            "Obligate biotrophic rust fungus that produces characteristic cinnamon-brown pustules "
            "(uredinia) on both leaf surfaces. Spores (urediniospores) are wind-dispersed and cannot "
            "overwinter in northern regions - inoculum blows northward from tropical/subtropical areas "
            "annually."
        ),
        symptoms=[
            "Small (1-3 mm), circular to elongate, cinnamon-brown pustules (uredinia)",
            "Pustules on both upper and lower leaf surfaces",
            "Pustules rupture epidermis, releasing powdery brown-red urediniospores",
            "Late-season pustules turn dark brown to black (telia)",
            "Heavily infected leaves may yellow and senesce prematurely",
            "Pustules can also appear on husks and leaf sheaths",
        ],
        conditions="Cool to moderate temperatures (16-23 C), high humidity, heavy dew. Urediniospores require free water for germination (6+ hours).",
        severity_criteria={
            "mild": "Scattered pustules, <1% leaf area, mostly lower leaves",
            "moderate": "Numerous pustules across canopy, 1-10% leaf area",
            "severe": "Dense pustules on most leaves, 10-25% area, early senescence",
            "critical": "Extreme pustule density, premature death, >25% leaf area covered",
        },
        treatment=TreatmentInfo(
            organic=[
                "Sulfur-based fungicides (preventive, limited efficacy)",
                "Resistant hybrid selection is the primary organic strategy",
            ],
            chemical=[
                "Azoxystrobin (Quadris) - strobilurin",
                "Propiconazole (Tilt) - triazole",
                "Trifloxystrobin + prothioconazole (Stratego YLD) - premix",
                "Pyraclostrobin + metconazole (Headline AMP) - premix",
            ],
            cultural=[
                "Plant resistant hybrids (Rp genes provide race-specific resistance)",
                "Avoid late planting which increases exposure to late-season spore loads",
                "Scout regularly, especially in cool, humid weather",
            ],
        ),
        prevention=[
            "Plant hybrids with Rp resistance genes or partial resistance",
            "Early planting to avoid peak spore dispersal periods",
            "Monitor for southern rust (Puccinia polysora) which is more aggressive",
        ],
        spread_mechanism="Wind-dispersed urediniospores over long distances. Cannot overwinter as uredinia in temperate climates; re-established annually from southern sources.",
    ),
    "corn_gray_leaf_spot": DiseaseInfo(
        disease_id="corn_gray_leaf_spot",
        common_name="Gray Leaf Spot",
        scientific_name="Cercospora zeae-maydis",
        pathogen="Cercospora zeae-maydis",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="corn",
        description=(
            "One of the most yield-limiting diseases of corn globally, caused by Cercospora zeae-maydis. "
            "Produces distinctive rectangular, gray to tan lesions bounded by leaf veins. Most severe "
            "in continuous corn under reduced tillage with extended periods of high humidity."
        ),
        symptoms=[
            "Rectangular, gray to pale brown lesions bounded by leaf veins",
            "Lesions 2-7 cm long, 2-4 mm wide, distinctly parallel-sided",
            "Gray sporulation visible on lesion surface in humid conditions",
            "Lesions may coalesce, killing large leaf sections",
            "Starts on lower leaves and progresses upward",
            "Severe blighting of upper canopy reduces grain fill",
        ],
        conditions="Prolonged high humidity (>95% RH for >12 hours), warm temperatures (22-30 C), heavy dew, minimum tillage with corn residue on surface.",
        severity_criteria={
            "mild": "Few rectangular lesions on lower leaves",
            "moderate": "Lesions on mid-canopy, 10-25% of total leaf area",
            "severe": "Upper leaf blighting including ear leaf, 25-50%",
            "critical": "Extensive canopy loss >50%, stalk integrity compromised",
        },
        treatment=TreatmentInfo(
            organic=[
                "Tillage to bury infested crop residue",
                "Crop rotation away from continuous corn",
            ],
            chemical=[
                "Pyraclostrobin + fluxapyroxad (Priaxor) - premix",
                "Azoxystrobin + propiconazole (Quilt Xcel) - premix",
                "Prothioconazole + trifloxystrobin (Stratego YLD)",
                "Timing: apply at VT-R1 for best results",
            ],
            cultural=[
                "Rotate to non-host crops (soybeans, small grains)",
                "Tillage to accelerate residue decomposition",
                "Plant resistant hybrids",
                "Maintain balanced nutrition",
            ],
        ),
        prevention=[
            "Select hybrids with high gray leaf spot resistance ratings",
            "Crop rotation (the pathogen survives only on corn residue)",
            "Tillage to reduce surface residue",
            "Avoid fields with history of gray leaf spot for continuous corn",
        ],
        spread_mechanism="Airborne conidia produced on overwintered corn residue, favored by no-till continuous corn",
    ),
    # ===================== APPLE =====================
    "apple_healthy": DiseaseInfo(
        disease_id="apple_healthy",
        common_name="Healthy Apple",
        scientific_name="Malus domestica",
        pathogen="N/A",
        pathogen_type=PathogenType.NONE,
        plant_species="apple",
        description="Healthy apple foliage with uniform green coloration and normal leaf morphology.",
        symptoms=["Uniform green leaves", "No spots or discoloration", "Normal leaf shape"],
        conditions="N/A",
        severity_criteria={"healthy": "No symptoms present."},
        treatment=TreatmentInfo(
            organic=["Maintain current practices"],
            chemical=["No treatment needed"],
            cultural=["Continue balanced orchard management"],
        ),
        prevention=[
            "Regular pruning for air circulation",
            "Balanced nutrition",
            "Integrated pest management",
        ],
        spread_mechanism="N/A",
        is_healthy=True,
    ),
    "apple_scab": DiseaseInfo(
        disease_id="apple_scab",
        common_name="Apple Scab",
        scientific_name="Venturia inaequalis",
        pathogen="Venturia inaequalis",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="apple",
        description=(
            "The most economically important disease of apple worldwide. Caused by the ascomycete "
            "Venturia inaequalis. Produces olive-green to dark brown, velvety scab lesions on "
            "leaves, fruit, and petioles. Unmanaged orchards can lose 70%+ of marketable fruit."
        ),
        symptoms=[
            "Olive-green to dark brown velvety lesions on upper leaf surface",
            "Lesions follow veins and may distort leaf shape",
            "Severe infections cause leaf curling, yellowing, and premature drop",
            "Fruit lesions: dark, corky, scab-like spots",
            "Fruit cracking may occur at scab lesion sites",
            "Late-season 'pinpoint scab' appears as tiny spots near harvest",
        ],
        conditions="Cool, wet spring weather. Ascospore release from overwintered pseudothecia during rain events. 'Mills table' relates temperature and wetting duration to infection risk.",
        severity_criteria={
            "mild": "Few lesions on leaves, no fruit infection visible",
            "moderate": "Moderate leaf infection, early fruit lesions appearing",
            "severe": "Heavy leaf scab, significant fruit infection, defoliation starting",
            "critical": "Severe defoliation, heavy fruit scab, crop unmarketable",
        },
        treatment=TreatmentInfo(
            organic=[
                "Sulfur (wettable or lime sulfur) - protectant",
                "Copper hydroxide (early season only, before bloom)",
                "Potassium bicarbonate (Kaligreen)",
                "Bacillus subtilis (Serenade)",
            ],
            chemical=[
                "Myclobutanil (Rally/Nova) - SI fungicide",
                "Difenoconazole + cyprodinil (Inspire Super)",
                "Captan (Captan 80 WDG) - protectant",
                "Mancozeb (Dithane/Penncozeb) - protectant",
                "Trifloxystrobin (Flint) - strobilurin",
                "Fenarimol (Rubigan) - SI fungicide",
            ],
            cultural=[
                "Shred or remove fallen leaves in autumn (reduces primary inoculum 50-80%)",
                "Apply 5% urea to fallen leaves to accelerate decomposition",
                "Prune to open canopy and improve air drying",
                "Remove water sprouts and suckers",
            ],
            biological=[
                "Cladosporium cladosporioides (leaf litter decomposition)",
            ],
            notes="Follow Mills table for infection period tracking. Protectants must be applied before rain; kickback systemics can be applied up to 72 hours post-infection depending on product.",
        ),
        prevention=[
            "Plant scab-resistant cultivars (e.g., 'Liberty', 'Enterprise', 'GoldRush', Vf gene)",
            "Reduce overwintering inoculum by leaf litter management",
            "Monitor weather and apply fungicides based on infection period predictions",
            "Prune for open canopy to speed leaf drying",
            "Maintain balanced nutrition (avoid excess nitrogen)",
        ],
        spread_mechanism="Ascospores from overwintered pseudothecia in leaf litter (primary), rain-splashed conidia (secondary cycles)",
    ),
    "apple_black_rot": DiseaseInfo(
        disease_id="apple_black_rot",
        common_name="Apple Black Rot",
        scientific_name="Botryosphaeria obtusa",
        pathogen="Botryosphaeria obtusa (anamorph: Diplodia seriata)",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="apple",
        description=(
            "Fungal disease complex affecting leaves, fruit, and limbs. Leaf symptoms include "
            "'frogeye leaf spot' - circular brown lesions with purple borders. Fruit rot begins "
            "at calyx end producing firm, black, mummified fruit. Cankers on branches serve as "
            "perennial inoculum sources."
        ),
        symptoms=[
            "'Frogeye leaf spot': circular lesions with purple margins and tan/brown centers",
            "Lesions 3-6 mm diameter, sometimes with concentric zonation",
            "Fruit rot: firm, brown expanding from calyx end, turning black",
            "Mummified fruit remain on tree as inoculum source",
            "Limb cankers: sunken, reddish-brown bark, often at pruning wounds",
            "Pycnidia (black dots) visible in old lesions and mummies",
        ],
        conditions="Warm (20-28 C), humid conditions. Conidia dispersed by rain splash. Infects through wounds, lenticels, and natural openings.",
        severity_criteria={
            "mild": "Few frogeye spots on leaves, no fruit symptoms",
            "moderate": "Moderate leaf spots, occasional fruit rot",
            "severe": "Heavy leaf infection, multiple fruit affected, cankers visible",
            "critical": "Extensive frogeye spots, significant fruit loss, branch dieback",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper sprays during dormant season",
                "Remove and destroy mummified fruit and cankered wood",
                "Lime sulfur during dormant season",
            ],
            chemical=[
                "Captan (protectant) - also controls scab",
                "Thiophanate-methyl (Topsin-M) - systemic",
                "Myclobutanil (Rally) - systemic",
                "Strobilurin fungicides (Flint, Sovran)",
            ],
            cultural=[
                "Remove and burn mummified fruit from trees and ground",
                "Prune out all cankers and dead wood during dormant season",
                "Cut at least 15 cm below visible canker margin",
                "Maintain tree vigor with proper nutrition and irrigation",
            ],
        ),
        prevention=[
            "Sanitation: remove all mummies, cankers, and dead wood annually",
            "Maintain fungicide program (most scab programs also control black rot)",
            "Avoid wounding bark during cultivation",
            "Fire blight management reduces wounds that black rot exploits",
        ],
        spread_mechanism="Rain-splashed conidia from pycnidia in cankers, mummified fruit, and dead wood",
    ),
    "apple_cedar_apple_rust": DiseaseInfo(
        disease_id="apple_cedar_apple_rust",
        common_name="Cedar Apple Rust",
        scientific_name="Gymnosporangium juniperi-virginianae",
        pathogen="Gymnosporangium juniperi-virginianae",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="apple",
        description=(
            "Heteroecious rust requiring two hosts to complete its life cycle: apple (Malus) and "
            "eastern red cedar (Juniperus virginiana). On apple, produces bright orange-yellow "
            "spots on leaves and fruit. The distinctive gelatinous orange spore horns on cedar "
            "galls release basidiospores that infect apple in spring."
        ),
        symptoms=[
            "Bright yellow-orange spots on upper leaf surface (1-3 mm initial)",
            "Spots enlarge and develop red-orange border",
            "Black dots (spermogonia) appear in center of spots on upper surface",
            "Tube-like projections (aecia) form on lower leaf surface",
            "Aecia are fringed, cup-shaped, and exude orange aeciospores",
            "Fruit lesions: yellowish spots that may distort fruit shape",
        ],
        conditions="Warm spring rains trigger basidiospore release from cedar galls. Requires both hosts within ~2 miles. Wetting period of 4-6 hours at 10-27 C for infection.",
        severity_criteria={
            "mild": "Few scattered yellow-orange spots on leaves",
            "moderate": "Moderate leaf spotting, beginning to see aecia formation",
            "severe": "Heavy leaf spotting, fruit infection, some defoliation",
            "critical": "Extensive leaf and fruit infection, severe defoliation",
        },
        treatment=TreatmentInfo(
            organic=[
                "Sulfur sprays during spring infection periods",
                "Remove nearby cedar trees if feasible",
                "Remove cedar galls before they mature in spring",
            ],
            chemical=[
                "Myclobutanil (Rally/Nova) - most effective single product",
                "Fenarimol (Rubigan) - systemic",
                "Triadimefon (Bayleton) - systemic",
                "Apply from pink bud through petal fall during wet spring weather",
            ],
            cultural=[
                "Remove eastern red cedar (Juniperus virginiana) within 1-2 miles if possible",
                "Physically remove galls from cedar trees before spring",
                "Choose rust-resistant apple cultivars",
            ],
        ),
        prevention=[
            "Plant rust-resistant apple varieties (e.g., 'Liberty', 'Freedom', 'Redfree')",
            "Eliminate eastern red cedar and other Juniperus spp. within 2 miles",
            "Remove galls from cedars in late winter before they produce spore horns",
            "Apply targeted fungicides during spring infection periods",
        ],
        spread_mechanism="Basidiospores from cedar galls infect apple (spring). Aeciospores from apple infect cedar (late summer). Two-year cycle on cedar.",
    ),
    # ===================== GRAPE =====================
    "grape_healthy": DiseaseInfo(
        disease_id="grape_healthy",
        common_name="Healthy Grape",
        scientific_name="Vitis vinifera",
        pathogen="N/A",
        pathogen_type=PathogenType.NONE,
        plant_species="grape",
        description="Healthy grapevine foliage with uniform green coloration and normal leaf morphology.",
        symptoms=["Uniform green leaves", "No spots or discoloration", "Normal vine growth"],
        conditions="N/A",
        severity_criteria={"healthy": "No symptoms present."},
        treatment=TreatmentInfo(
            organic=["Maintain current practices"],
            chemical=["No treatment needed"],
            cultural=["Continue integrated vineyard management"],
        ),
        prevention=["Proper canopy management", "Balanced nutrition", "Good drainage"],
        spread_mechanism="N/A",
        is_healthy=True,
    ),
    "grape_black_rot": DiseaseInfo(
        disease_id="grape_black_rot",
        common_name="Grape Black Rot",
        scientific_name="Guignardia bidwellii (anamorph: Phyllosticta ampelicida)",
        pathogen="Guignardia bidwellii",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="grape",
        description=(
            "Major disease of grapes east of the Rocky Mountains, caused by Guignardia bidwellii. "
            "Can destroy 80%+ of a crop in susceptible varieties during wet seasons. Affects leaves, "
            "shoots, tendrils, and berries. Most critical infections occur on developing fruit."
        ),
        symptoms=[
            "Circular, tan to brown leaf spots (2-10 mm) with dark brown to black margins",
            "Minute black pycnidia visible within leaf spots",
            "Shoot lesions: dark, elongated cankers",
            "Berry symptoms: light brown soft rot progressing to hard, black, shriveled mummy",
            "Mummified berries covered with black pycnidia",
            "Berries most susceptible from bloom to veraison (4-5 weeks post-bloom)",
        ],
        conditions="Warm (21-32 C) and wet conditions. Requires 6+ hours of leaf wetness for infection. Ascospores from overwintered mummies initiate primary infections.",
        severity_criteria={
            "mild": "Few leaf spots, no berry symptoms, <5% leaves affected",
            "moderate": "Moderate leaf spots, first berry infections appearing",
            "severe": "Heavy leaf spotting, multiple berry mummies, 25-50% crop loss",
            "critical": ">50% crop loss, heavy mummy formation, vine stress",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides (Bordeaux mixture, copper hydroxide)",
                "Sulfur (wettable sulfur) - protectant",
                "Careful mummy and debris removal is critical",
            ],
            chemical=[
                "Myclobutanil (Rally/Nova) - systemic SI",
                "Mancozeb (Dithane/Penncozeb) - protectant",
                "Azoxystrobin (Abound) - strobilurin",
                "Tebuconazole (Elite) - systemic",
                "Critical spray window: immediate pre-bloom through 4 weeks post-bloom",
            ],
            cultural=[
                "Remove and destroy ALL mummified berries from vines and ground",
                "Maintain open canopy for air circulation and spray penetration",
                "Leaf pulling and shoot positioning in fruit zone",
                "Cultivate under vines to bury fallen mummies",
            ],
        ),
        prevention=[
            "Sanitation: remove every mummy from vineyard (primary inoculum source)",
            "Plant resistant or tolerant varieties when possible",
            "Open canopy through proper training and summer pruning",
            "Prophylactic fungicide program from pre-bloom through veraison",
        ],
        spread_mechanism="Ascospores from overwintered mummies (primary), conidia from pycnidia for secondary spread during growing season",
    ),
    "grape_esca": DiseaseInfo(
        disease_id="grape_esca",
        common_name="Grape Esca (Black Measles)",
        scientific_name="Phaeomoniella chlamydospora / Phaeoacremonium minimum complex",
        pathogen="Phaeomoniella chlamydospora, Phaeoacremonium minimum, Fomitiporia mediterranea",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="grape",
        description=(
            "Complex trunk disease caused by multiple fungal species that colonize the woody "
            "vascular tissue. One of the most destructive grapevine trunk diseases worldwide. "
            "Symptoms can appear suddenly ('apoplexy') or develop gradually over years. Increasingly "
            "prevalent since the ban on sodium arsenite in most countries."
        ),
        symptoms=[
            "Interveinal 'tiger stripe' chlorosis and necrosis on leaves",
            "Irregular dry, scorched leaf margins",
            "Berry symptoms: dark spots ('black measles') on berry skin",
            "Cross-section of trunk shows dark streaking and necrotic wood",
            "White rot (soft spongy wood) in trunk interior",
            "Acute form: sudden vine collapse ('apoplexy') during hot weather",
        ],
        conditions="Chronic: symptoms develop over multiple growing seasons. Apoplexy triggered by sudden heat stress. Infection through pruning wounds. Fungi are slow-growing wood colonizers.",
        severity_criteria={
            "mild": "Mild tiger striping on a few leaves, vine otherwise vigorous",
            "moderate": "Pronounced foliar symptoms, some berry spotting",
            "severe": "Extensive leaf scorch, significant crop impact, wood symptoms visible",
            "critical": "Apoplexy (sudden vine death) or progressive vine decline over 2-3 years",
        },
        treatment=TreatmentInfo(
            organic=[
                "Trunk surgery (remedial surgery): cut out infected wood and apply wound sealant",
                "Trichoderma-based biocontrol applied to pruning wounds",
            ],
            chemical=[
                "No fully effective chemical treatments exist for established infections",
                "Thiophanate-methyl applied to pruning wounds (preventive)",
                "Boric acid paste applied to pruning wounds",
                "Fosetyl-aluminum (trunk injection - experimental)",
            ],
            cultural=[
                "Delayed pruning (prune during late dormancy to reduce wound susceptibility)",
                "Double pruning: preliminary cut in early winter, final cut in late winter",
                "Remove severely affected vines and replant",
                "Trunk surgery to excise infected tissue (curettage)",
                "Protect pruning wounds with wound sealants or biocontrol agents",
            ],
            biological=[
                "Trichoderma atroviride applied to fresh pruning wounds",
                "Trichoderma harzianum (Biotrichol, Vintec) wound protection",
            ],
            notes="Esca is a chronic trunk disease with no simple cure. Management focuses on prevention and slowing disease progression. Trunk surgery can extend vine life by 5-10 years in many cases.",
        ),
        prevention=[
            "Protect all pruning wounds with Trichoderma-based products or wound sealants",
            "Prune late in the dormant season when wound susceptibility is lower",
            "Avoid large pruning cuts; use minimal pruning when possible",
            "Ensure nursery stock is free of trunk disease pathogens",
            "Train replacement trunks from basal shoots on affected vines",
        ],
        spread_mechanism="Airborne ascospores and conidia infecting pruning wounds. Contaminated nursery stock. Spread through pruning tools.",
    ),
    "grape_leaf_blight": DiseaseInfo(
        disease_id="grape_leaf_blight",
        common_name="Grape Leaf Blight (Isariopsis Leaf Spot)",
        scientific_name="Pseudocercospora vitis (syn. Isariopsis clavispora)",
        pathogen="Pseudocercospora vitis",
        pathogen_type=PathogenType.FUNGAL,
        plant_species="grape",
        description=(
            "Foliar disease of grapevines caused by Pseudocercospora vitis, also known as "
            "Isariopsis leaf spot. Produces dark, angular leaf spots that can cause premature "
            "defoliation. Most common in warm, humid climates. Generally considered a secondary "
            "disease but can be economically important in neglected vineyards."
        ),
        symptoms=[
            "Dark brown to black angular spots on leaves, often vein-limited",
            "Spots 2-25 mm, sometimes with lighter brown centers",
            "Dark sporulation visible on lower leaf surface of spots",
            "Yellow halo may surround spots",
            "Premature leaf drop in severe cases",
            "Primarily affects older leaves in lower canopy",
        ],
        conditions="Warm temperatures (25-30 C), high humidity, frequent rainfall. Most common in late season.",
        severity_criteria={
            "mild": "Few scattered angular spots on lower leaves",
            "moderate": "Moderate leaf spotting across canopy, 10-25% leaf area",
            "severe": "Heavy spotting with early defoliation, 25-50%",
            "critical": "Severe defoliation, compromised vine vigor, >50% leaf loss",
        },
        treatment=TreatmentInfo(
            organic=[
                "Copper-based fungicides",
                "Sulfur sprays (preventive)",
                "Removal of fallen infected leaves",
            ],
            chemical=[
                "Mancozeb (Dithane/Penncozeb) - protectant",
                "Azoxystrobin (Abound) - systemic",
                "Myclobutanil (Rally) - systemic",
            ],
            cultural=[
                "Improve canopy air circulation through leaf pulling",
                "Remove and destroy fallen leaf litter",
                "Maintain balanced vine nutrition (avoid excess nitrogen)",
                "Proper shoot positioning and hedging",
            ],
        ),
        prevention=[
            "Good canopy management for air circulation and spray penetration",
            "Leaf litter management to reduce overwintering inoculum",
            "Include in routine fungicide program where disease is historically present",
            "Avoid excessive vigor that creates dense, humid canopies",
        ],
        spread_mechanism="Conidia dispersed by rain splash and wind from overwintered infected leaves",
    ),
}


def get_disease_info(disease_id: str) -> DiseaseInfo | None:
    """Look up disease information by disease ID.

    Args:
        disease_id: The disease class identifier (e.g., 'tomato_early_blight').

    Returns:
        DiseaseInfo if found, None otherwise.
    """
    return DISEASE_DATABASE.get(disease_id)


def get_all_diseases() -> list[DiseaseInfo]:
    """Return all disease records in the database."""
    return list(DISEASE_DATABASE.values())


def get_diseases_by_species(species: str) -> list[DiseaseInfo]:
    """Filter diseases by plant species.

    Args:
        species: Plant species name (e.g., 'tomato', 'potato').

    Returns:
        List of DiseaseInfo records for the given species.
    """
    return [d for d in DISEASE_DATABASE.values() if d.plant_species == species]


def assess_severity(disease_id: str, affected_percentage: float) -> Severity:
    """Assess disease severity based on percentage of affected leaf area.

    Uses general thresholds that align with the severity_criteria in
    each disease record for a standardized assessment.

    Args:
        disease_id: The disease class identifier.
        affected_percentage: Percentage of leaf area showing symptoms (0-100).

    Returns:
        Severity enum value.
    """
    info = get_disease_info(disease_id)
    if info is None or info.is_healthy:
        return Severity.HEALTHY

    if affected_percentage < 10:
        return Severity.MILD
    elif affected_percentage < 25:
        return Severity.MODERATE
    elif affected_percentage < 50:
        return Severity.SEVERE
    else:
        return Severity.CRITICAL


def get_treatment_summary(disease_id: str) -> dict[str, list[str] | str] | None:
    """Get a concise treatment summary for a disease.

    Args:
        disease_id: The disease class identifier.

    Returns:
        Dictionary with treatment categories, or None if disease not found.
    """
    info = get_disease_info(disease_id)
    if info is None:
        return None

    return {
        "disease": info.common_name,
        "pathogen_type": info.pathogen_type.value,
        "organic_treatments": info.treatment.organic,
        "chemical_treatments": info.treatment.chemical,
        "cultural_practices": info.treatment.cultural,
        "biological_control": info.treatment.biological,
        "prevention": info.prevention,
        "notes": info.treatment.notes,
    }
