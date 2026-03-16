"""Ecological analytics for camera trap biodiversity monitoring.

Implements standard community ecology metrics used in wildlife management
and conservation biology research. These metrics quantify species diversity,
temporal activity patterns, spatial occupancy, and interspecific associations
from camera trap detection data.

References:
    - Magurran, A.E. (2004). Measuring Biological Diversity.
    - MacKenzie, D.I. et al. (2006). Occupancy Estimation and Modeling.
    - O'Brien, T.G. et al. (2011). Camera Traps in Animal Ecology.
    - Rovero, F. & Zimmermann, F. (2016). Camera Trapping for Wildlife Research.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Sighting:
    """A single wildlife detection event from a camera trap.

    Attributes:
        species: Species common name (matches SPECIES_LABELS).
        timestamp: UTC datetime of the detection.
        camera_id: Unique identifier for the camera station.
        latitude: Camera station latitude (decimal degrees, WGS84).
        longitude: Camera station longitude (decimal degrees, WGS84).
        confidence: Model confidence score (0-1).
        image_path: Path or S3 URI of the source image.
    """

    species: str
    timestamp: datetime
    camera_id: str
    latitude: float = 0.0
    longitude: float = 0.0
    confidence: float = 0.0
    image_path: str = ""


@dataclass
class BiodiversitySummary:
    """Summary of biodiversity metrics for a camera station or study area.

    Attributes:
        camera_id: Camera station identifier (or 'all' for aggregate).
        species_richness: Number of unique species detected (S).
        shannon_index: Shannon-Wiener diversity index (H').
        simpson_index: Simpson's diversity index (1 - D).
        evenness: Pielou's evenness index (J' = H' / ln(S)).
        total_detections: Total number of detection events.
        species_counts: Per-species detection counts.
        detection_rate: Detections per trap-night.
        survey_effort_nights: Total trap-nights of survey effort.
    """

    camera_id: str = "all"
    species_richness: int = 0
    shannon_index: float = 0.0
    simpson_index: float = 0.0
    evenness: float = 0.0
    total_detections: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)
    detection_rate: float = 0.0
    survey_effort_nights: float = 0.0


def compute_species_richness(sightings: list[Sighting]) -> int:
    """Compute species richness (S) -- the count of unique species detected.

    Species richness is the simplest diversity metric: the total number
    of distinct species observed. It does not account for abundance or
    evenness but provides a baseline for site comparison.

    Args:
        sightings: List of detection events.

    Returns:
        Number of unique species.
    """
    species_set = {s.species for s in sightings if s.species not in ("empty", "human")}
    return len(species_set)


def compute_shannon_index(sightings: list[Sighting]) -> float:
    """Compute the Shannon-Wiener diversity index (H').

    H' = -sum(p_i * ln(p_i)) for each species i

    Where p_i is the proportional abundance of species i. H' increases
    with both species richness and evenness. Typical values for North
    American mammal communities range from 1.5 to 3.5.

    A value of 0 indicates a single-species community. The theoretical
    maximum is ln(S) when all species are equally abundant.

    Args:
        sightings: List of detection events.

    Returns:
        Shannon-Wiener index H'. Returns 0.0 for empty input.
    """
    counts = _get_species_counts(sightings)
    if not counts:
        return 0.0

    total = sum(counts.values())
    if total == 0:
        return 0.0

    h_prime = 0.0
    for count in counts.values():
        if count > 0:
            p_i = count / total
            h_prime -= p_i * np.log(p_i)

    return float(h_prime)


def compute_simpson_index(sightings: list[Sighting]) -> float:
    """Compute Simpson's diversity index (1 - D).

    D = sum(p_i^2), and we return 1 - D so higher values = more diverse.

    Simpson's index gives the probability that two randomly chosen
    individuals belong to different species. It is less sensitive to
    rare species than Shannon's H' and emphasizes dominant species.

    Args:
        sightings: List of detection events.

    Returns:
        Simpson's diversity index (1 - D). Range [0, 1).
    """
    counts = _get_species_counts(sightings)
    if not counts:
        return 0.0

    total = sum(counts.values())
    if total == 0:
        return 0.0

    d = sum((count / total) ** 2 for count in counts.values())
    return float(1.0 - d)


def compute_pielou_evenness(sightings: list[Sighting]) -> float:
    """Compute Pielou's evenness index (J').

    J' = H' / ln(S)

    Measures how evenly individuals are distributed among species.
    J' = 1.0 when all species are equally abundant; approaches 0 when
    one species dominates. Useful for detecting community imbalance
    that may indicate ecological disturbance.

    Args:
        sightings: List of detection events.

    Returns:
        Pielou's evenness J'. Returns 0.0 if fewer than 2 species.
    """
    s = compute_species_richness(sightings)
    if s < 2:
        return 0.0

    h_prime = compute_shannon_index(sightings)
    return float(h_prime / np.log(s))


def compute_activity_pattern(
    sightings: list[Sighting],
    species: str | None = None,
    bin_hours: int = 1,
) -> dict[int, int]:
    """Compute diel activity pattern as a time-of-day histogram.

    Camera traps record activity timestamps, enabling reconstruction of
    circadian activity patterns. This is essential for classifying species
    as diurnal, nocturnal, crepuscular, or cathemeral.

    Standard ecological bins:
        - Diurnal:     06:00-18:00
        - Nocturnal:   18:00-06:00
        - Crepuscular: 05:00-07:00 and 17:00-19:00 peaks

    Args:
        sightings: List of detection events.
        species: Filter to a single species, or None for all wildlife.
        bin_hours: Width of each temporal bin in hours (default 1).

    Returns:
        Dictionary mapping hour-of-day (0-23) to detection count.
    """
    filtered = [
        s
        for s in sightings
        if s.species not in ("empty", "human") and (species is None or s.species == species)
    ]

    bins: dict[int, int] = {h: 0 for h in range(0, 24, bin_hours)}
    for sighting in filtered:
        if sighting.timestamp is not None:
            bin_key = (sighting.timestamp.hour // bin_hours) * bin_hours
            bins[bin_key] = bins.get(bin_key, 0) + 1

    return bins


def compute_activity_overlap(
    sightings: list[Sighting],
    species_a: str,
    species_b: str,
) -> float:
    """Compute temporal activity overlap coefficient between two species.

    Uses the coefficient of overlapping (Dhat1) following Ridout & Linkie
    (2009). Values range from 0 (no overlap) to 1 (identical patterns).

    High overlap between a predator and prey species may indicate active
    hunting behaviour; low overlap may suggest temporal niche partitioning.

    Args:
        sightings: List of detection events.
        species_a: First species label.
        species_b: Second species label.

    Returns:
        Overlap coefficient in [0, 1].
    """
    pattern_a = compute_activity_pattern(sightings, species=species_a)
    pattern_b = compute_activity_pattern(sightings, species=species_b)

    total_a = sum(pattern_a.values())
    total_b = sum(pattern_b.values())

    if total_a == 0 or total_b == 0:
        return 0.0

    # Normalize to probability distributions.
    hours = sorted(set(pattern_a.keys()) | set(pattern_b.keys()))
    overlap = 0.0
    for hour in hours:
        p_a = pattern_a.get(hour, 0) / total_a
        p_b = pattern_b.get(hour, 0) / total_b
        overlap += min(p_a, p_b)

    return float(overlap)


def compute_naive_occupancy(
    sightings: list[Sighting],
    species: str,
    camera_ids: list[str] | None = None,
) -> float:
    """Compute naive occupancy (proportion of sites where species was detected).

    Naive occupancy = (sites with detection) / (total surveyed sites)

    This is an uncorrected estimate that does not account for imperfect
    detection probability. For rigorous occupancy modeling, use the
    single-season occupancy model (MacKenzie et al., 2002) which
    separates occupancy (psi) from detection probability (p).

    Args:
        sightings: List of detection events.
        species: Target species label.
        camera_ids: List of all surveyed camera stations. If None,
            inferred from the sightings data (which may underestimate
            total survey effort if some cameras detected nothing).

    Returns:
        Naive occupancy estimate in [0, 1].
    """
    if camera_ids is None:
        camera_ids = list({s.camera_id for s in sightings})

    if not camera_ids:
        return 0.0

    occupied = {s.camera_id for s in sightings if s.species == species}
    return len(occupied) / len(camera_ids)


def build_detection_history(
    sightings: list[Sighting],
    species: str,
    camera_ids: list[str],
    survey_occasions: list[tuple[datetime, datetime]],
) -> np.ndarray:
    """Build a detection/non-detection history matrix for occupancy modeling.

    Creates the standard occupancy modeling input: a matrix of shape
    (n_sites, n_occasions) where each cell is 1 (detected), 0 (not detected),
    or -1 (not surveyed). This is the input format for single-season and
    multi-season occupancy models (MacKenzie et al., 2002, 2003).

    Args:
        sightings: List of detection events.
        species: Target species.
        camera_ids: Ordered list of camera station IDs (rows).
        survey_occasions: List of (start, end) datetime tuples defining
            each survey occasion (columns).

    Returns:
        Detection history matrix of shape (n_sites, n_occasions).
    """
    n_sites = len(camera_ids)
    n_occasions = len(survey_occasions)
    history = np.zeros((n_sites, n_occasions), dtype=np.int8)

    camera_index = {cid: i for i, cid in enumerate(camera_ids)}

    species_sightings = [s for s in sightings if s.species == species]

    for sighting in species_sightings:
        site_idx = camera_index.get(sighting.camera_id)
        if site_idx is None:
            continue

        for occ_idx, (start, end) in enumerate(survey_occasions):
            if sighting.timestamp and start <= sighting.timestamp <= end:
                history[site_idx, occ_idx] = 1
                break

    return history


def compute_population_trend(
    sightings: list[Sighting],
    species: str,
    period: str = "M",
) -> pd.DataFrame:
    """Compute population index trend over time using relative abundance.

    Uses Relative Abundance Index (RAI) = detections per unit effort,
    calculated as detections per 100 trap-nights. RAI is the most common
    camera trap population index and, while imperfect, correlates with
    true abundance for many species when detection probability is stable.

    Args:
        sightings: List of detection events.
        species: Target species label.
        period: Pandas period alias for temporal aggregation
            ('M' = monthly, 'W' = weekly, 'Q' = quarterly).

    Returns:
        DataFrame with columns: period, detections, trap_nights, rai.
    """
    species_sightings = [s for s in sightings if s.species == species and s.timestamp]
    all_sightings = [s for s in sightings if s.timestamp]

    if not species_sightings or not all_sightings:
        return pd.DataFrame(columns=["period", "detections", "trap_nights", "rai"])

    # Build detection DataFrame.
    df_species = pd.DataFrame(
        [{"timestamp": s.timestamp, "camera_id": s.camera_id} for s in species_sightings]
    )
    df_species["timestamp"] = pd.to_datetime(df_species["timestamp"])
    df_species["period"] = df_species["timestamp"].dt.to_period(period)

    # Build effort DataFrame (all camera activity).
    df_all = pd.DataFrame(
        [{"timestamp": s.timestamp, "camera_id": s.camera_id} for s in all_sightings]
    )
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all["period"] = df_all["timestamp"].dt.to_period(period)

    # Count detections per period.
    detections = df_species.groupby("period").size().reset_index(name="detections")

    # Estimate trap-nights per period (unique camera-days as proxy).
    df_all["date"] = df_all["timestamp"].dt.date
    effort = (
        df_all.groupby("period")
        .apply(lambda g: g[["camera_id", "date"]].drop_duplicates().shape[0])
        .reset_index(name="trap_nights")
    )

    trend = detections.merge(effort, on="period", how="outer").fillna(0)
    trend["detections"] = trend["detections"].astype(int)
    trend["trap_nights"] = trend["trap_nights"].astype(int)

    # RAI = detections per 100 trap-nights.
    trend["rai"] = np.where(
        trend["trap_nights"] > 0,
        (trend["detections"] / trend["trap_nights"]) * 100,
        0.0,
    )

    trend["period"] = trend["period"].astype(str)
    trend = trend.sort_values("period").reset_index(drop=True)
    return trend


def compute_species_co_occurrence(
    sightings: list[Sighting],
    time_window_minutes: int = 30,
) -> pd.DataFrame:
    """Analyse species co-occurrence from temporally proximate detections.

    Two species are considered co-occurring at a site if they are both
    detected at the same camera within a specified time window. This
    can reveal interspecific interactions:
    - Predator-prey: wolf detections shortly after deer
    - Competition: coyote and fox temporal avoidance
    - Commensalism: ravens following predator kills

    Uses the probabilistic co-occurrence framework of Veech (2013) to
    classify species pairs as positive, negative, or random associations.

    Args:
        sightings: List of detection events.
        time_window_minutes: Maximum time gap (minutes) for co-occurrence.

    Returns:
        DataFrame with columns: species_a, species_b, co_occurrences,
        expected, sites_both, association_type.
    """
    wildlife = [s for s in sightings if s.species not in ("empty", "human")]

    # Group sightings by camera.
    by_camera: dict[str, list[Sighting]] = defaultdict(list)
    for s in wildlife:
        by_camera[s.camera_id].append(s)

    # Find co-occurrence events.
    co_occur_counts: Counter[tuple[str, str]] = Counter()
    species_at_site: dict[str, set[str]] = defaultdict(set)

    for camera_id, cam_sightings in by_camera.items():
        sorted_sightings = sorted(cam_sightings, key=lambda s: s.timestamp or datetime.min)

        for s in sorted_sightings:
            species_at_site[camera_id].add(s.species)

        # Check pairwise temporal proximity.
        for i, s_i in enumerate(sorted_sightings):
            for j in range(i + 1, len(sorted_sightings)):
                s_j = sorted_sightings[j]
                if s_i.species == s_j.species:
                    continue
                if s_i.timestamp is None or s_j.timestamp is None:
                    continue

                delta = abs((s_j.timestamp - s_i.timestamp).total_seconds())
                if delta <= time_window_minutes * 60:
                    pair = tuple(sorted([s_i.species, s_j.species]))
                    co_occur_counts[pair] += 1  # type: ignore[arg-type]
                else:
                    # Sightings are sorted by time; no need to check further.
                    break

    # Compute expected co-occurrence under random association.
    all_species = sorted({s.species for s in wildlife})
    n_sites = len(by_camera)

    rows = []
    for sp_a, sp_b in combinations(all_species, 2):
        sites_a = sum(1 for cid, spp in species_at_site.items() if sp_a in spp)
        sites_b = sum(1 for cid, spp in species_at_site.items() if sp_b in spp)
        sites_both = sum(1 for cid, spp in species_at_site.items() if sp_a in spp and sp_b in spp)

        observed = co_occur_counts.get((sp_a, sp_b), 0)
        # Expected co-occurrence under independence.
        expected = (sites_a * sites_b) / n_sites if n_sites > 0 else 0

        if observed > expected * 1.5:
            association = "positive"
        elif observed < expected * 0.5:
            association = "negative"
        else:
            association = "random"

        rows.append(
            {
                "species_a": sp_a,
                "species_b": sp_b,
                "co_occurrences": observed,
                "expected": round(expected, 2),
                "sites_both": sites_both,
                "association_type": association,
            }
        )

    return pd.DataFrame(rows)


def compute_full_biodiversity_summary(
    sightings: list[Sighting],
    camera_id: str | None = None,
    survey_effort_nights: float | None = None,
) -> BiodiversitySummary:
    """Compute a comprehensive biodiversity summary for a camera or study area.

    Aggregates all core diversity metrics into a single summary object
    suitable for dashboard display or API response.

    Args:
        sightings: List of detection events.
        camera_id: Filter to a specific camera, or None for all data.
        survey_effort_nights: Known trap-nights of effort. If None,
            estimated from data as unique camera-days.

    Returns:
        BiodiversitySummary with all computed metrics.
    """
    if camera_id:
        filtered = [s for s in sightings if s.camera_id == camera_id]
    else:
        filtered = list(sightings)

    wildlife = [s for s in filtered if s.species not in ("empty", "human")]

    counts = _get_species_counts(wildlife)

    # Estimate survey effort if not provided.
    if survey_effort_nights is None and wildlife:
        unique_camera_days = set()
        for s in filtered:
            if s.timestamp:
                unique_camera_days.add((s.camera_id, s.timestamp.date()))
        survey_effort_nights = float(len(unique_camera_days))

    total = len(wildlife)

    return BiodiversitySummary(
        camera_id=camera_id or "all",
        species_richness=compute_species_richness(wildlife),
        shannon_index=compute_shannon_index(wildlife),
        simpson_index=compute_simpson_index(wildlife),
        evenness=compute_pielou_evenness(wildlife),
        total_detections=total,
        species_counts=dict(counts),
        detection_rate=((total / survey_effort_nights * 100) if survey_effort_nights else 0.0),
        survey_effort_nights=survey_effort_nights or 0.0,
    )


def _get_species_counts(sightings: list[Sighting]) -> dict[str, int]:
    """Count detections per species, excluding empty and human classes."""
    counts: dict[str, int] = {}
    for s in sightings:
        if s.species not in ("empty", "human"):
            counts[s.species] = counts.get(s.species, 0) + 1
    return counts
