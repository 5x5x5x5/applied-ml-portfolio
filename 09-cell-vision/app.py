"""CellVision Streamlit App - Interactive Microscopy Cell Classifier.

Provides a web UI for:
    - Uploading microscopy images for classification
    - Real-time cell type prediction with confidence bars
    - GradCAM overlay visualization for model interpretability
    - Batch processing of multiple images
    - Display of model performance metrics
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from cell_vision import CELL_TYPE_LABELS, CELL_TYPES, __version__
from cell_vision.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from cell_vision.models.cell_classifier import CellClassifier
from cell_vision.visualization.gradcam import overlay_heatmap

logger = logging.getLogger(__name__)

# ---- Page configuration ----

st.set_page_config(
    page_title="CellVision - Cell Type Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Constants ----

IMAGE_SIZE = 224
MODEL_PATH = Path("models/cellnet_best.pt")


# ---- Cached resources ----


@st.cache_resource
def load_classifier(model_type: str = "cellnet") -> CellClassifier:
    """Load and cache the cell classifier model."""
    classifier = CellClassifier(model_type=model_type)
    if MODEL_PATH.exists():
        classifier.load(MODEL_PATH)
        st.sidebar.success(f"Model loaded from {MODEL_PATH}")
    else:
        st.sidebar.warning("No saved model found. Using randomly initialized weights.")
    return classifier


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image into a model-ready tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a displayable numpy array."""
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ---- Sidebar ----

st.sidebar.title("CellVision")
st.sidebar.markdown(f"**Version:** {__version__}")
st.sidebar.markdown("---")

model_type = st.sidebar.selectbox(
    "Model Architecture",
    options=["cellnet", "resnet18"],
    index=0,
    help="CellNet: custom CNN. ResNet18: transfer learning from ImageNet.",
)

top_k = st.sidebar.slider(
    "Top-K Predictions",
    min_value=1,
    max_value=len(CELL_TYPES),
    value=3,
    help="Number of top predictions to display.",
)

show_gradcam = st.sidebar.checkbox(
    "Show GradCAM Overlay",
    value=True,
    help="Visualize which image regions influenced the prediction.",
)

gradcam_alpha = st.sidebar.slider(
    "GradCAM Opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.1,
    help="Blending strength of the GradCAM heatmap overlay.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Cell Types")
for cell_type, label in CELL_TYPE_LABELS.items():
    st.sidebar.markdown(f"- {label}")

# ---- Main content ----

st.title("CellVision: Microscopy Cell Type Classifier")
st.markdown(
    "Upload microscopy images of blood cells to classify them into cell types. "
    "The model identifies **red blood cells**, **white blood cell subtypes** "
    "(neutrophils, lymphocytes, monocytes, eosinophils, basophils), and **platelets**."
)

tab_single, tab_batch, tab_metrics = st.tabs(["Single Image", "Batch Processing", "Model Metrics"])

# ---- Single Image Tab ----

with tab_single:
    st.header("Single Image Classification")

    uploaded_file = st.file_uploader(
        "Upload a microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="single_upload",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        tensor = preprocess_image(image)
        classifier = load_classifier(model_type)

        col_img, col_results = st.columns([1, 1])

        with col_img:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col_results:
            st.subheader("Classification Results")

            with st.spinner("Classifying..."):
                predictions = classifier.predict(tensor, top_k=top_k)

            for pred in predictions:
                st.markdown(f"**{pred['label']}**")
                st.progress(pred["confidence"])
                st.caption(f"Confidence: {pred['confidence']:.1%}")

        # GradCAM visualization
        if show_gradcam:
            st.subheader("GradCAM Explanation")
            st.markdown(
                "The heatmap shows which image regions most influenced the prediction. "
                "Warm colors (red/yellow) indicate high importance."
            )

            with st.spinner("Generating GradCAM..."):
                explanation = classifier.explain(tensor)

            heatmap = explanation["heatmap"]
            display_img = tensor_to_display(tensor)
            overlay = overlay_heatmap(display_img, heatmap, alpha=gradcam_alpha)

            col_orig, col_heat, col_overlay = st.columns(3)
            with col_orig:
                st.image(display_img, caption="Preprocessed", use_container_width=True)
            with col_heat:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(heatmap, cmap="jet")
                ax.set_title("GradCAM Heatmap")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            with col_overlay:
                st.image(overlay, caption="GradCAM Overlay", use_container_width=True)

# ---- Batch Processing Tab ----

with tab_batch:
    st.header("Batch Processing")
    st.markdown("Upload multiple images for batch classification.")

    uploaded_files = st.file_uploader(
        "Upload microscopy images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if uploaded_files:
        classifier = load_classifier(model_type)
        st.markdown(f"**Processing {len(uploaded_files)} images...**")
        progress_bar = st.progress(0)

        all_results = []

        for i, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            tensor = preprocess_image(image)
            predictions = classifier.predict(tensor, top_k=top_k)

            all_results.append(
                {
                    "filename": file.name,
                    "image": image,
                    "predictions": predictions,
                }
            )
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success(f"Classified {len(all_results)} images")

        # Display results in a grid
        cols_per_row = 3
        for row_start in range(0, len(all_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, result_idx in enumerate(
                range(row_start, min(row_start + cols_per_row, len(all_results)))
            ):
                result = all_results[result_idx]
                with cols[col_idx]:
                    st.image(result["image"], caption=result["filename"], use_container_width=True)
                    top_pred = result["predictions"][0]
                    st.markdown(f"**{top_pred['label']}** ({top_pred['confidence']:.1%})")

# ---- Model Metrics Tab ----

with tab_metrics:
    st.header("Model Information & Metrics")

    classifier = load_classifier(model_type)
    info = classifier.get_model_info()

    col_arch, col_params = st.columns(2)

    with col_arch:
        st.subheader("Architecture")
        st.markdown(f"- **Model Type:** {info['model_type']}")
        st.markdown(f"- **Number of Classes:** {info['num_classes']}")
        st.markdown(f"- **Device:** {info['device']}")

    with col_params:
        st.subheader("Parameters")
        st.markdown(f"- **Total Parameters:** {info['total_parameters']:,}")
        st.markdown(f"- **Trainable Parameters:** {info['trainable_parameters']:,}")
        frozen = info["total_parameters"] - info["trainable_parameters"]
        st.markdown(f"- **Frozen Parameters:** {frozen:,}")

    st.subheader("Supported Cell Types")

    cell_descriptions = {
        "red_blood_cell": (
            "Biconcave disc-shaped cells (7-8 um diameter). Most abundant blood cell. "
            "Carries oxygen via hemoglobin. Appears as pale pink with central pallor on H&E stain."
        ),
        "neutrophil": (
            "Most common WBC (60-70%). Multi-lobed nucleus (3-5 lobes). "
            "Fine granules in cytoplasm. First responder in acute inflammation."
        ),
        "lymphocyte": (
            "Second most common WBC (20-40%). Large round/oval nucleus with thin rim of "
            "pale blue cytoplasm. Key role in adaptive immunity (T-cells, B-cells, NK cells)."
        ),
        "monocyte": (
            "Largest WBC (12-20 um). Kidney/horseshoe-shaped nucleus. Abundant gray-blue "
            "cytoplasm. Differentiates into macrophages and dendritic cells in tissues."
        ),
        "eosinophil": (
            "Bilobed nucleus. Large, bright red-orange (eosinophilic) granules. "
            "1-4% of WBCs. Important in parasitic defense and allergic responses."
        ),
        "basophil": (
            "Rarest WBC (<1%). Bilobed nucleus often obscured by large dark blue/purple "
            "(basophilic) granules. Contains histamine and heparin. Role in allergic reactions."
        ),
        "platelet": (
            "Smallest blood element (2-3 um). Anucleate cell fragments from megakaryocytes. "
            "Appear as small purple granules. Essential for hemostasis and blood clotting."
        ),
    }

    for cell_type in CELL_TYPES:
        with st.expander(CELL_TYPE_LABELS[cell_type]):
            st.markdown(cell_descriptions.get(cell_type, ""))
