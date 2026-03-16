# PlantPathologist

Plant disease detection system using computer vision and deep learning. Identifies 18 diseases across 5 crop species (tomato, potato, corn, apple, grape) from leaf photographs, providing diagnosis with treatment recommendations.

## Architecture

- **Model**: EfficientNet-B0 with multi-task learning (disease + species classification) and temperature-scaled confidence calibration
- **Preprocessing**: Leaf segmentation, color space analysis (chlorosis/necrosis scoring), lesion detection with morphological features
- **Knowledge Base**: Curated plant pathology database with accurate disease descriptions, symptoms, and treatment protocols (organic, chemical, cultural, biological)
- **API**: FastAPI REST endpoints for diagnosis, disease listing, and treatment lookup
- **Frontend**: Streamlit app + standalone HTML/JS mobile-friendly interface with camera capture

## Supported Diseases

| Crop | Diseases |
|------|----------|
| Tomato | Early Blight, Late Blight, Bacterial Spot, Septoria Leaf Spot, Leaf Mold, TYLCV, Target Spot |
| Potato | Early Blight, Late Blight |
| Corn | Northern Leaf Blight, Common Rust, Gray Leaf Spot |
| Apple | Scab, Black Rot, Cedar Apple Rust |
| Grape | Black Rot, Esca, Leaf Blight (Isariopsis) |

## Setup

```bash
cd 11-plant-pathologist
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

### FastAPI Server

```bash
uvicorn plant_pathologist.api.main:app --reload
```

Endpoints:
- `POST /diagnose` - Upload a leaf image for disease diagnosis
- `GET /diseases` - List all detectable diseases (optional `?species=tomato` filter)
- `GET /diseases/{disease_id}/treatment` - Get treatment recommendations

### Streamlit App

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t plant-pathologist .
docker run -p 8000:8000 plant-pathologist
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
11-plant-pathologist/
  src/plant_pathologist/
    models/disease_classifier.py    # EfficientNet-B0 multi-task classifier
    preprocessing/leaf_processor.py # Image preprocessing pipeline
    knowledge/disease_database.py   # Disease knowledge base
    api/main.py                     # FastAPI application
  app.py                            # Streamlit interface
  frontend/                         # Standalone HTML/JS/CSS interface
  tests/                            # pytest test suite
  Dockerfile
  pyproject.toml
```

## Disclaimer

This tool is for educational and preliminary screening purposes. Always consult a certified plant pathologist or agricultural extension agent for definitive diagnosis and treatment decisions.
