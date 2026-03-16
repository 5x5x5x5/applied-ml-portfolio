"""CellVision - Microscopy Image Cell Type Classifier.

A deep learning pipeline for classifying cell types from microscopy images,
with support for H&E stained blood smear analysis.

Cell types supported:
    - Red Blood Cells (Erythrocytes)
    - Neutrophils
    - Lymphocytes
    - Monocytes
    - Eosinophils
    - Basophils
    - Platelets (Thrombocytes)
"""

__version__ = "0.1.0"

CELL_TYPES: list[str] = [
    "red_blood_cell",
    "neutrophil",
    "lymphocyte",
    "monocyte",
    "eosinophil",
    "basophil",
    "platelet",
]

CELL_TYPE_LABELS: dict[str, str] = {
    "red_blood_cell": "Red Blood Cell (Erythrocyte)",
    "neutrophil": "Neutrophil (WBC)",
    "lymphocyte": "Lymphocyte (WBC)",
    "monocyte": "Monocyte (WBC)",
    "eosinophil": "Eosinophil (WBC)",
    "basophil": "Basophil (WBC)",
    "platelet": "Platelet (Thrombocyte)",
}

NUM_CLASSES: int = len(CELL_TYPES)
