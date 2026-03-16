"""WildEye Streamlit Dashboard.

Interactive web dashboard for wildlife species classification and
biodiversity monitoring. Features:
    - Image upload with real-time species identification
    - Species confidence breakdown with visual indicators
    - Biodiversity metrics dashboard (Shannon, Simpson, richness)
    - Diel activity pattern heatmaps
    - Geographic map view of sightings
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from wild_eye import __version__

logger = logging.getLogger(__name__)

# -- Page configuration -------------------------------------------------------

st.set_page_config(
    page_title="WildEye - Wildlife Species Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Session state initialization ---------------------------------------------

if "sightings" not in st.session_state:
    st.session_state.sightings = []

if "classification_history" not in st.session_state:
    st.session_state.classification_history = []


# -- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.title("WildEye")
    st.caption(f"Wildlife Species Classifier v{__version__}")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "Species Identification",
            "Biodiversity Dashboard",
            "Activity Patterns",
            "Sightings Map",
        ],
        index=0,
    )

    st.divider()

    st.subheader("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Minimum confidence to report a species detection",
    )

    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="WildEye API endpoint for classification",
    )

    st.divider()
    st.caption("Conservation biology meets AI/ML")
    st.caption("Powered by MobileNetV3 + PyTorch")


# -- Helper functions ---------------------------------------------------------


def classify_image_via_api(
    image_bytes: bytes,
    api_base: str,
    threshold: float,
    camera_id: str = "dashboard",
) -> dict | None:
    """Send an image to the WildEye API for classification.

    Args:
        image_bytes: Raw image bytes.
        api_base: Base URL of the WildEye API.
        threshold: Confidence threshold.
        camera_id: Camera identifier.

    Returns:
        Classification result dict, or None on failure.
    """
    try:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_base}/classify",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                params={
                    "confidence_threshold": threshold,
                    "camera_id": camera_id,
                },
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None


def generate_demo_sightings() -> list[dict]:
    """Generate realistic demo sighting data for dashboard demonstration.

    Simulates a multi-camera study area with ecologically plausible
    detection patterns reflecting real species activity cycles.
    """
    rng = np.random.default_rng(42)

    cameras = {
        "CAM-001": {"lat": 44.4280, "lon": -110.5885, "name": "Yellowstone Meadow"},
        "CAM-002": {"lat": 44.4600, "lon": -110.8300, "name": "Old Faithful Ridge"},
        "CAM-003": {"lat": 44.5500, "lon": -110.4000, "name": "Lamar Valley"},
        "CAM-004": {"lat": 44.3900, "lon": -110.6500, "name": "Hayden Valley"},
        "CAM-005": {"lat": 44.6100, "lon": -110.3800, "name": "Slough Creek"},
    }

    # Species with realistic detection frequencies and activity peaks.
    species_profiles = {
        "white_tailed_deer": {"freq": 0.20, "peak_hours": [6, 18], "nocturnal": False},
        "elk": {"freq": 0.18, "peak_hours": [7, 17], "nocturnal": False},
        "black_bear": {"freq": 0.08, "peak_hours": [10, 15], "nocturnal": False},
        "grizzly_bear": {"freq": 0.04, "peak_hours": [9, 16], "nocturnal": False},
        "gray_wolf": {"freq": 0.05, "peak_hours": [5, 20], "nocturnal": True},
        "coyote": {"freq": 0.10, "peak_hours": [22, 3], "nocturnal": True},
        "red_fox": {"freq": 0.06, "peak_hours": [21, 4], "nocturnal": True},
        "raccoon": {"freq": 0.08, "peak_hours": [23, 2], "nocturnal": True},
        "bald_eagle": {"freq": 0.03, "peak_hours": [10, 14], "nocturnal": False},
        "great_horned_owl": {"freq": 0.02, "peak_hours": [22, 1], "nocturnal": True},
        "snowshoe_hare": {"freq": 0.07, "peak_hours": [20, 5], "nocturnal": True},
        "pronghorn": {"freq": 0.04, "peak_hours": [8, 16], "nocturnal": False},
        "moose": {"freq": 0.03, "peak_hours": [6, 18], "nocturnal": False},
        "bobcat": {"freq": 0.02, "peak_hours": [23, 3], "nocturnal": True},
    }

    sightings = []
    base_date = datetime(2025, 6, 1)

    for _ in range(500):
        # Select species weighted by detection frequency.
        species_list = list(species_profiles.keys())
        weights = [species_profiles[s]["freq"] for s in species_list]
        weights = [w / sum(weights) for w in weights]
        species = rng.choice(species_list, p=weights)

        profile = species_profiles[species]

        # Generate ecologically plausible timestamps.
        day_offset = int(rng.integers(0, 180))
        peak = rng.choice(profile["peak_hours"])
        hour = int(rng.normal(peak, 2)) % 24
        minute = int(rng.integers(0, 60))

        timestamp = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        # Select camera with some spatial clustering.
        camera_id = rng.choice(list(cameras.keys()))
        cam = cameras[camera_id]

        sightings.append(
            {
                "species": species,
                "timestamp": timestamp.isoformat(),
                "camera_id": camera_id,
                "camera_name": cam["name"],
                "latitude": cam["lat"] + float(rng.normal(0, 0.005)),
                "longitude": cam["lon"] + float(rng.normal(0, 0.005)),
                "confidence": float(rng.beta(8, 2)),  # Skewed toward high confidence.
            }
        )

    return sightings


# -- Pages --------------------------------------------------------------------


def page_species_identification() -> None:
    """Species Identification page - upload and classify images."""
    st.header("Species Identification")
    st.write(
        "Upload a camera trap image to identify wildlife species. "
        "The model detects 22 North American species plus empty/human frames."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Camera Trap Image",
            type=["jpg", "jpeg", "png", "tif", "bmp"],
            help="Accepts standard image formats from trail cameras",
        )

        if uploaded_file is not None:
            from PIL import Image

            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Classify Image", type="primary"):
                with st.spinner("Running species classifier..."):
                    uploaded_file.seek(0)
                    result = classify_image_via_api(
                        uploaded_file.read(),
                        api_url,
                        confidence_threshold,
                    )

                if result:
                    st.session_state.classification_history.append(result)
                    st.success("Classification complete!")

    with col2:
        st.subheader("Results")

        if st.session_state.classification_history:
            result = st.session_state.classification_history[-1]

            if result.get("is_empty"):
                st.info("No wildlife detected - empty frame (vegetation trigger)")
            elif result.get("is_human"):
                st.warning("Human detected - excluded from wildlife dataset")
            else:
                st.subheader(f"Top: {result['top_species'].replace('_', ' ').title()}")
                st.metric(
                    "Confidence",
                    f"{result['top_confidence']:.1%}",
                )

                if result.get("species"):
                    st.write("**Detected species:**")
                    for sp in result["species"]:
                        conf = result["probabilities"].get(sp, 0)
                        st.progress(conf, text=f"{sp.replace('_', ' ').title()}: {conf:.1%}")

            # Show full probability breakdown.
            with st.expander("Full probability breakdown"):
                probs = result.get("probabilities", {})
                prob_df = pd.DataFrame(
                    sorted(probs.items(), key=lambda x: x[1], reverse=True),
                    columns=["Species", "Probability"],
                )
                prob_df["Species"] = prob_df["Species"].str.replace("_", " ").str.title()
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.info("Upload an image and click 'Classify' to see results")

        # Demo mode.
        st.divider()
        if st.button("Load Demo Data"):
            st.session_state.sightings = generate_demo_sightings()
            st.success(f"Loaded {len(st.session_state.sightings)} demo sightings")
            st.rerun()


def page_biodiversity_dashboard() -> None:
    """Biodiversity Dashboard page - ecological diversity metrics."""
    st.header("Biodiversity Dashboard")

    sightings = st.session_state.sightings
    if not sightings:
        st.info(
            "No sightings data available. Upload images through the "
            "Species Identification page or load demo data."
        )
        if st.button("Load Demo Data"):
            st.session_state.sightings = generate_demo_sightings()
            st.rerun()
        return

    df = pd.DataFrame(sightings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Overall metrics.
    species_counts = df["species"].value_counts()
    n_species = len(species_counts)
    total_detections = len(df)

    # Shannon diversity index.
    proportions = species_counts / total_detections
    h_prime = float(-np.sum(proportions * np.log(proportions)))

    # Simpson's index.
    simpson = float(1.0 - np.sum(proportions**2))

    # Pielou's evenness.
    j_prime = h_prime / np.log(n_species) if n_species > 1 else 0.0

    # Display metrics.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Species Richness (S)", n_species)
    col2.metric("Shannon Index (H')", f"{h_prime:.3f}")
    col3.metric("Simpson Index (1-D)", f"{simpson:.3f}")
    col4.metric("Pielou's Evenness (J')", f"{j_prime:.3f}")

    st.divider()

    # Species abundance chart.
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Species Abundance (Detection Frequency)")
        chart_data = species_counts.reset_index()
        chart_data.columns = ["Species", "Detections"]
        chart_data["Species"] = chart_data["Species"].str.replace("_", " ").str.title()
        st.bar_chart(chart_data.set_index("Species"), horizontal=True)

    with col_right:
        st.subheader("Detection Timeline")
        df["date"] = df["timestamp"].dt.date
        timeline = df.groupby("date").size().reset_index(name="Detections")
        timeline["date"] = pd.to_datetime(timeline["date"])
        st.line_chart(timeline.set_index("date"))

    st.divider()

    # Per-camera comparison.
    st.subheader("Per-Camera Diversity Comparison")
    camera_stats = []
    for cam_id in df["camera_id"].unique():
        cam_df = df[df["camera_id"] == cam_id]
        cam_counts = cam_df["species"].value_counts()
        cam_n = len(cam_counts)
        cam_total = len(cam_df)
        cam_props = cam_counts / cam_total
        cam_h = float(-np.sum(cam_props * np.log(cam_props)))

        camera_stats.append(
            {
                "Camera": cam_id,
                "Species Richness": cam_n,
                "Detections": cam_total,
                "Shannon H'": round(cam_h, 3),
            }
        )

    st.dataframe(
        pd.DataFrame(camera_stats),
        use_container_width=True,
        hide_index=True,
    )


def page_activity_patterns() -> None:
    """Activity Patterns page - diel activity analysis."""
    st.header("Diel Activity Patterns")
    st.write(
        "Analyse temporal activity patterns to classify species behaviour as "
        "diurnal, nocturnal, crepuscular, or cathemeral."
    )

    sightings = st.session_state.sightings
    if not sightings:
        st.info("No sightings data available. Load demo data from the sidebar.")
        if st.button("Load Demo Data"):
            st.session_state.sightings = generate_demo_sightings()
            st.rerun()
        return

    df = pd.DataFrame(sightings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Species selector.
    all_species = sorted(df["species"].unique())
    selected_species = st.multiselect(
        "Select species to compare",
        options=all_species,
        default=all_species[:4],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if not selected_species:
        st.warning("Select at least one species to view activity patterns.")
        return

    # Build activity heatmap data.
    activity_data = {}
    for species in selected_species:
        sp_df = df[df["species"] == species]
        hourly = sp_df.groupby("hour").size()
        total = hourly.sum()
        activity_data[species.replace("_", " ").title()] = [
            hourly.get(h, 0) / total * 100 if total > 0 else 0 for h in range(24)
        ]

    activity_df = pd.DataFrame(
        activity_data,
        index=[f"{h:02d}:00" for h in range(24)],
    )

    st.subheader("Activity by Hour of Day (% of detections)")
    st.line_chart(activity_df)

    # Activity classification.
    st.divider()
    st.subheader("Diel Activity Classification")

    for species in selected_species:
        sp_df = df[df["species"] == species]
        hourly = sp_df.groupby("hour").size()

        # Classify based on proportion of activity in light/dark periods.
        daytime = sum(hourly.get(h, 0) for h in range(6, 18))
        nighttime = sum(hourly.get(h, 0) for h in list(range(0, 6)) + list(range(18, 24)))
        twilight = sum(hourly.get(h, 0) for h in [5, 6, 7, 17, 18, 19])
        total = daytime + nighttime

        if total == 0:
            pattern = "Insufficient data"
        elif twilight / total > 0.4:
            pattern = "Crepuscular (dawn/dusk active)"
        elif daytime / total > 0.7:
            pattern = "Diurnal (day active)"
        elif nighttime / total > 0.7:
            pattern = "Nocturnal (night active)"
        else:
            pattern = "Cathemeral (active throughout diel cycle)"

        day_pct = daytime / total * 100 if total > 0 else 0
        night_pct = nighttime / total * 100 if total > 0 else 0
        name = species.replace("_", " ").title()

        st.write(f"**{name}**: {pattern} (Day: {day_pct:.0f}% | Night: {night_pct:.0f}%)")


def page_sightings_map() -> None:
    """Sightings Map page - geographic visualization."""
    st.header("Sightings Map")

    sightings = st.session_state.sightings
    if not sightings:
        st.info("No sightings data available. Load demo data to see the map.")
        if st.button("Load Demo Data"):
            st.session_state.sightings = generate_demo_sightings()
            st.rerun()
        return

    df = pd.DataFrame(sightings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Species filter.
    all_species = sorted(df["species"].unique())
    selected = st.multiselect(
        "Filter by species",
        options=all_species,
        default=all_species,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if selected:
        filtered = df[df["species"].isin(selected)]
    else:
        filtered = df

    # Map display.
    map_data = filtered[["latitude", "longitude"]].dropna()

    if not map_data.empty:
        st.map(map_data)
    else:
        st.warning("No geolocated sightings to display.")

    # Recent sightings table.
    st.divider()
    st.subheader("Recent Sightings")

    display_df = (
        filtered[["timestamp", "species", "camera_id", "confidence", "latitude", "longitude"]]
        .sort_values("timestamp", ascending=False)
        .head(50)
    )
    display_df["species"] = display_df["species"].str.replace("_", " ").str.title()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# -- Page routing -------------------------------------------------------------

if page == "Species Identification":
    page_species_identification()
elif page == "Biodiversity Dashboard":
    page_biodiversity_dashboard()
elif page == "Activity Patterns":
    page_activity_patterns()
elif page == "Sightings Map":
    page_sightings_map()
