# CellVision - Microscopy Image Cell Type Classifier

A deep learning pipeline for classifying cell types from microscopy images of blood smears, built with PyTorch.

## Cell Types

CellVision classifies 7 cell types from H&E-stained peripheral blood smear images:

| Cell Type | Description |
|-----------|-------------|
| **Red Blood Cell** (Erythrocyte) | Biconcave disc, 7-8 um, pale pink with central pallor |
| **Neutrophil** | Multi-lobed nucleus (3-5 lobes), fine cytoplasmic granules |
| **Lymphocyte** | Large round nucleus, thin pale blue cytoplasm rim |
| **Monocyte** | Kidney-shaped nucleus, abundant gray-blue cytoplasm |
| **Eosinophil** | Bilobed nucleus, large red-orange granules |
| **Basophil** | Bilobed nucleus obscured by dark blue/purple granules |
| **Platelet** (Thrombocyte) | Small (2-3 um) anucleate fragments |

## Architecture

Two model options:

- **CellNet**: Custom CNN with 4 conv blocks (32->64->128->256 channels), BatchNorm, dropout, and global average pooling
- **CellNetResNet**: Transfer learning from ImageNet-pretrained ResNet18 with custom classifier head

Both include GradCAM visualization for model interpretability.

## Project Structure

```
09-cell-vision/
├── src/cell_vision/
│   ├── models/cell_classifier.py    # CellNet and ResNet18 models
│   ├── data/dataset.py              # PyTorch Dataset with augmentation
│   ├── preprocessing/image_processor.py  # Stain normalization, segmentation
│   ├── api/main.py                  # FastAPI inference server
│   └── visualization/gradcam.py     # GradCAM implementation
├── scripts/train_model.py           # Training script with argparse
├── app.py                           # Streamlit web application
├── tests/                           # pytest test suite
├── Dockerfile
└── pyproject.toml
```

## Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

## Training

```bash
uv run python scripts/train_model.py \
    --data-dir data/cells \
    --model-type cellnet \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --output-dir models/
```

Data should be organized as:
```
data/cells/
├── red_blood_cell/
├── neutrophil/
├── lymphocyte/
├── monocyte/
├── eosinophil/
├── basophil/
└── platelet/
```

## Inference

### FastAPI Server

```bash
uv run uvicorn cell_vision.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /classify` - Upload image, get cell classification
- `POST /batch-classify` - Multiple images
- `GET /model-info` - Model architecture and metrics

### Streamlit App

```bash
uv run streamlit run app.py
```

## Docker

```bash
docker build -t cellvision .
docker run -p 8000:8000 cellvision

# Streamlit mode
docker run -p 8501:8501 cellvision streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Preprocessing

- **Macenko stain normalization**: Decomposes H&E images into stain vectors using SVD, normalizes to a reference standard
- **Otsu thresholding**: Automatic cell segmentation by finding optimal foreground/background threshold
- **Patch extraction**: Extract tissue patches from whole slide images (WSI) with tissue fraction filtering

## Testing

```bash
uv run pytest tests/ -v
```

## Key Dependencies

- PyTorch + torchvision (deep learning)
- FastAPI + uvicorn (inference API)
- Streamlit (interactive web app)
- scikit-learn (metrics and evaluation)
- Pillow + numpy (image processing)
- matplotlib (visualization)
