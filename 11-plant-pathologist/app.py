"""Streamlit application for PlantPathologist disease detection.

Provides an interactive web interface for uploading leaf images,
viewing disease classification results, treatment recommendations,
and browsing the disease knowledge base.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import streamlit as st
from PIL import Image

from plant_pathologist.knowledge.disease_database import (
    Severity,
    assess_severity,
    get_all_diseases,
    get_disease_info,
    get_diseases_by_species,
)
from plant_pathologist.models.disease_classifier import (
    PlantDiseaseClassifier,
    load_model,
)
from plant_pathologist.preprocessing.leaf_processor import LeafProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PlantPathologist - Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d6a4f;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #52796f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2d6a4f;
    }
    .treatment-card {
        background: #f0f7f4;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid #95d5b2;
    }
    .severity-mild { color: #ffc107; font-weight: bold; }
    .severity-moderate { color: #fd7e14; font-weight: bold; }
    .severity-severe { color: #dc3545; font-weight: bold; }
    .severity-critical { color: #721c24; font-weight: bold; background: #f8d7da; padding: 2px 8px; border-radius: 4px; }
    .severity-healthy { color: #28a745; font-weight: bold; }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "diagnosis_history" not in st.session_state:
    st.session_state.diagnosis_history = []

if "model" not in st.session_state:
    st.session_state.model = None

if "processor" not in st.session_state:
    st.session_state.processor = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_classifier() -> PlantDiseaseClassifier:
    """Load and cache the disease classification model."""
    return load_model(checkpoint_path=None, device="cpu")


@st.cache_resource
def load_processor() -> LeafProcessor:
    """Load and cache the leaf preprocessor."""
    return LeafProcessor()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def severity_badge(severity: Severity) -> str:
    """Return an HTML severity badge."""
    css_class = f"severity-{severity.value}"
    labels = {
        Severity.HEALTHY: "HEALTHY",
        Severity.MILD: "MILD",
        Severity.MODERATE: "MODERATE",
        Severity.SEVERE: "SEVERE",
        Severity.CRITICAL: "CRITICAL",
    }
    return f'<span class="{css_class}">{labels.get(severity, severity.value.upper())}</span>'


def confidence_color(conf: float) -> str:
    """Return CSS class based on confidence level."""
    if conf >= 0.8:
        return "confidence-high"
    elif conf >= 0.5:
        return "confidence-medium"
    return "confidence-low"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Diagnose", "Disease Library", "History"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        """
        **PlantPathologist** uses deep learning
        (EfficientNet-B0) to identify plant diseases
        from leaf photographs.

        **Supported crops:**
        - Tomato (8 conditions)
        - Potato (3 conditions)
        - Corn (4 conditions)
        - Apple (4 conditions)
        - Grape (4 conditions)

        **Disclaimer:** This tool is for educational
        and preliminary screening purposes. Always
        consult a certified plant pathologist or
        extension agent for definitive diagnosis.
        """
    )


# ---------------------------------------------------------------------------
# Diagnose Page
# ---------------------------------------------------------------------------

if page == "Diagnose":
    st.markdown('<div class="main-header">PlantPathologist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Plant Disease Detection</div>',
        unsafe_allow_html=True,
    )

    # Input section
    col_upload, col_camera = st.columns(2)

    with col_upload:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a single plant leaf against a simple background.",
        )

    with col_camera:
        st.markdown("### Camera Capture")
        camera_image = st.camera_input(
            "Take a photo of a leaf",
            help="Point your camera at a single leaf with good lighting.",
        )

    # Determine which image to use
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

    if image is not None:
        # Display uploaded image
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True, caption="Input leaf image")

        # Run diagnosis
        with st.spinner("Analyzing leaf image..."):
            model = load_classifier()
            processor = load_processor()

            # Preprocessing
            processed_image = processor.preprocess_for_model(image)
            quality = processor.validate_image_quality(processed_image)
            segmentation = processor.segment_leaf(processed_image)
            color_analysis = processor.analyze_colors(processed_image, segmentation.leaf_mask)
            lesion_info = processor.detect_lesions(processed_image, segmentation.leaf_mask)

            # Classification
            start_time = time.perf_counter()
            result = model.predict(processed_image)
            inference_ms = (time.perf_counter() - start_time) * 1000

        # Image quality warnings
        if not quality.is_valid:
            st.warning("Image quality issues detected:")
            for issue in quality.issues:
                st.markdown(f"- {issue}")

        # Disease info
        disease_info = get_disease_info(result.disease_class)
        severity = assess_severity(result.disease_class, lesion_info.severity_percentage)

        # Results section
        st.markdown("---")
        st.markdown("## Diagnosis Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Plant Species", result.plant_species.capitalize())
        with col2:
            disease_label = disease_info.common_name if disease_info else result.disease_class
            st.metric("Disease", disease_label)
        with col3:
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
        with col4:
            st.metric("Severity", severity.value.capitalize())

        # Detailed diagnosis card
        if disease_info:
            st.markdown(
                f"""
                <div class="diagnosis-card">
                    <h3>{disease_info.common_name}</h3>
                    <p><strong>Scientific name:</strong> <em>{disease_info.scientific_name}</em></p>
                    <p><strong>Pathogen:</strong> {disease_info.pathogen}
                    ({disease_info.pathogen_type.value})</p>
                    <p><strong>Severity:</strong> {severity_badge(severity)}</p>
                    <p>{disease_info.description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Symptoms
            st.markdown("### Symptoms")
            for symptom in disease_info.symptoms:
                st.markdown(f"- {symptom}")

            # Treatment recommendations
            if not disease_info.is_healthy:
                st.markdown("### Treatment Recommendations")

                tab_organic, tab_chemical, tab_cultural, tab_prevention = st.tabs(
                    ["Organic", "Chemical", "Cultural Practices", "Prevention"]
                )

                with tab_organic:
                    st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
                    for t in disease_info.treatment.organic:
                        st.markdown(f"- {t}")
                    if disease_info.treatment.biological:
                        st.markdown("**Biological control:**")
                        for b in disease_info.treatment.biological:
                            st.markdown(f"- {b}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with tab_chemical:
                    st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
                    for t in disease_info.treatment.chemical:
                        st.markdown(f"- {t}")
                    if disease_info.treatment.notes:
                        st.info(disease_info.treatment.notes)
                    st.markdown("</div>", unsafe_allow_html=True)

                with tab_cultural:
                    st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
                    for t in disease_info.treatment.cultural:
                        st.markdown(f"- {t}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with tab_prevention:
                    for p in disease_info.prevention:
                        st.markdown(f"- {p}")

            # Severity criteria
            st.markdown("### Severity Assessment Guide")
            for level, desc in disease_info.severity_criteria.items():
                st.markdown(f"**{level.capitalize()}:** {desc}")

        # Top predictions
        st.markdown("### Top Predictions")
        for disease_class, conf in result.top_k_diseases[:5]:
            info = get_disease_info(disease_class)
            name = info.common_name if info else disease_class
            bar_pct = conf * 100
            st.markdown(f"**{name}** ({bar_pct:.1f}%)")
            st.progress(conf)

        # Preprocessing details (collapsible)
        with st.expander("Preprocessing Details"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Image Quality**")
                st.markdown(f"- Resolution: {quality.width}x{quality.height}")
                st.markdown(f"- Blur score: {quality.blur_score:.1f}")
                st.markdown(f"- Brightness: {quality.mean_brightness:.0f}")
                st.markdown(f"- Valid: {'Yes' if quality.is_valid else 'No'}")

            with col_b:
                st.markdown("**Leaf Analysis**")
                st.markdown(f"- Leaf coverage: {segmentation.leaf_coverage * 100:.1f}%")
                st.markdown(f"- Green ratio: {color_analysis.green_ratio:.2f}")
                st.markdown(f"- Chlorosis score: {color_analysis.chlorosis_score:.2f}")
                st.markdown(f"- Necrosis score: {color_analysis.necrosis_score:.2f}")
                st.markdown(f"- Lesion count: {lesion_info.lesion_count}")
                st.markdown(f"- Lesion area: {lesion_info.total_lesion_area_ratio * 100:.1f}%")
                st.markdown(
                    f"- Concentric rings: {'Yes' if lesion_info.has_concentric_rings else 'No'}"
                )
                st.markdown(f"- Halo present: {'Yes' if lesion_info.has_halo else 'No'}")

            st.markdown(f"**Inference time:** {inference_ms:.1f} ms")

        # Save to history
        st.session_state.diagnosis_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disease": disease_info.common_name if disease_info else result.disease_class,
                "confidence": result.confidence,
                "species": result.plant_species,
                "severity": severity.value,
                "is_healthy": result.is_healthy,
            }
        )

    else:
        # Instructions when no image is uploaded
        st.markdown("---")
        st.markdown("### How to Use")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**1. Capture or Upload**")
            st.markdown(
                "Take a photo of a single leaf or upload an existing image. "
                "Ensure good lighting and a clear view of symptoms."
            )
        with col_b:
            st.markdown("**2. Get Diagnosis**")
            st.markdown(
                "Our AI analyzes the leaf image to identify the plant species "
                "and detect diseases with confidence scores."
            )
        with col_c:
            st.markdown("**3. Treatment Plan**")
            st.markdown(
                "Receive detailed treatment recommendations including organic, "
                "chemical, and cultural management strategies."
            )

        st.markdown("---")
        st.markdown("### Supported Diseases")

        for species in ["tomato", "potato", "corn", "apple", "grape"]:
            diseases = get_diseases_by_species(species)
            disease_names = [d.common_name for d in diseases if not d.is_healthy]
            if disease_names:
                st.markdown(f"**{species.capitalize()}:** {', '.join(disease_names)}")


# ---------------------------------------------------------------------------
# Disease Library Page
# ---------------------------------------------------------------------------

elif page == "Disease Library":
    st.markdown("## Disease Library")
    st.markdown("Browse all detectable plant diseases and their information.")

    # Species filter
    species_filter = st.selectbox(
        "Filter by plant species",
        ["All", "Tomato", "Potato", "Corn", "Apple", "Grape"],
    )

    if species_filter == "All":
        diseases = get_all_diseases()
    else:
        diseases = get_diseases_by_species(species_filter.lower())

    # Show only diseases (not healthy)
    show_healthy = st.checkbox("Include healthy classes", value=False)
    if not show_healthy:
        diseases = [d for d in diseases if not d.is_healthy]

    for disease in diseases:
        with st.expander(f"{disease.common_name} ({disease.plant_species.capitalize()})"):
            st.markdown(f"**Scientific name:** *{disease.scientific_name}*")
            st.markdown(f"**Pathogen:** {disease.pathogen} ({disease.pathogen_type.value})")
            st.markdown(f"**Description:** {disease.description}")

            st.markdown("**Symptoms:**")
            for s in disease.symptoms:
                st.markdown(f"- {s}")

            st.markdown(f"**Favorable conditions:** {disease.conditions}")
            st.markdown(f"**Spread mechanism:** {disease.spread_mechanism}")

            if not disease.is_healthy:
                st.markdown("**Treatment (Organic):**")
                for t in disease.treatment.organic:
                    st.markdown(f"- {t}")

                st.markdown("**Treatment (Chemical):**")
                for t in disease.treatment.chemical:
                    st.markdown(f"- {t}")

                st.markdown("**Prevention:**")
                for p in disease.prevention:
                    st.markdown(f"- {p}")


# ---------------------------------------------------------------------------
# History Page
# ---------------------------------------------------------------------------

elif page == "History":
    st.markdown("## Diagnosis History")
    st.markdown("Review past diagnoses from this session.")

    if not st.session_state.diagnosis_history:
        st.info("No diagnoses yet. Go to the Diagnose page to analyze a leaf image.")
    else:
        # Summary metrics
        total = len(st.session_state.diagnosis_history)
        healthy_count = sum(1 for d in st.session_state.diagnosis_history if d["is_healthy"])
        diseased_count = total - healthy_count

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Diagnoses", total)
        with col2:
            st.metric("Healthy", healthy_count)
        with col3:
            st.metric("Diseased", diseased_count)

        st.markdown("---")

        # History table
        for i, entry in enumerate(reversed(st.session_state.diagnosis_history)):
            severity = entry["severity"]
            status_icon = "+" if entry["is_healthy"] else "-"

            st.markdown(
                f"**{entry['timestamp']}** | "
                f"{entry['species'].capitalize()} | "
                f"{entry['disease']} | "
                f"Confidence: {entry['confidence'] * 100:.1f}% | "
                f"Severity: {severity.capitalize()}"
            )

        if st.button("Clear History"):
            st.session_state.diagnosis_history = []
            st.rerun()
