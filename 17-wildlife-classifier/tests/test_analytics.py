"""Tests for biodiversity metrics and ecological analytics.

Validates ecological calculations against known mathematical properties
and edge cases. Uses the standard ecology formulas for Shannon-Wiener,
Simpson, and Pielou indices.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from wild_eye.analytics.biodiversity_metrics import (
    BiodiversitySummary,
    Sighting,
    build_detection_history,
    compute_activity_overlap,
    compute_activity_pattern,
    compute_full_biodiversity_summary,
    compute_naive_occupancy,
    compute_pielou_evenness,
    compute_population_trend,
    compute_shannon_index,
    compute_simpson_index,
    compute_species_co_occurrence,
    compute_species_richness,
)


class TestSpeciesRichness:
    """Tests for species richness computation."""

    def test_richness_with_multiple_species(self, sample_sightings: list[Sighting]) -> None:
        """Should count unique wildlife species correctly."""
        richness = compute_species_richness(sample_sightings)
        assert richness == 10  # 10 species in the fixture

    def test_richness_single_species(self, single_species_sightings: list[Sighting]) -> None:
        """Single-species community should have richness of 1."""
        assert compute_species_richness(single_species_sightings) == 1

    def test_richness_empty(self, empty_sightings: list[Sighting]) -> None:
        """Empty sightings should yield richness of 0."""
        assert compute_species_richness(empty_sightings) == 0

    def test_richness_excludes_empty_and_human(self) -> None:
        """'empty' and 'human' classes should not count as species."""
        sightings = [
            Sighting(species="empty", timestamp=datetime.now(), camera_id="CAM-001"),
            Sighting(species="human", timestamp=datetime.now(), camera_id="CAM-001"),
            Sighting(species="elk", timestamp=datetime.now(), camera_id="CAM-001"),
        ]
        assert compute_species_richness(sightings) == 1


class TestShannonIndex:
    """Tests for Shannon-Wiener diversity index (H')."""

    def test_shannon_positive(self, sample_sightings: list[Sighting]) -> None:
        """H' should be positive for a multi-species community."""
        h = compute_shannon_index(sample_sightings)
        assert h > 0

    def test_shannon_single_species_is_zero(self, single_species_sightings: list[Sighting]) -> None:
        """H' should be 0 for a single-species community (no uncertainty)."""
        h = compute_shannon_index(single_species_sightings)
        assert h == pytest.approx(0.0, abs=1e-10)

    def test_shannon_empty_is_zero(self, empty_sightings: list[Sighting]) -> None:
        """H' should be 0 for empty sightings."""
        assert compute_shannon_index(empty_sightings) == 0.0

    def test_shannon_max_for_equal_distribution(self) -> None:
        """H' should equal ln(S) when all species are equally abundant."""
        n_species = 5
        sightings = []
        base = datetime.now()
        for i in range(n_species):
            for j in range(20):  # Equal abundance
                sightings.append(
                    Sighting(
                        species=f"species_{i}",
                        timestamp=base + timedelta(hours=i * 20 + j),
                        camera_id="CAM-001",
                    )
                )
        h = compute_shannon_index(sightings)
        h_max = np.log(n_species)
        assert h == pytest.approx(h_max, rel=1e-6)

    def test_shannon_less_than_ln_s(self, sample_sightings: list[Sighting]) -> None:
        """H' should always be <= ln(S)."""
        h = compute_shannon_index(sample_sightings)
        s = compute_species_richness(sample_sightings)
        assert h <= np.log(s) + 1e-10


class TestSimpsonIndex:
    """Tests for Simpson's diversity index (1 - D)."""

    def test_simpson_range(self, sample_sightings: list[Sighting]) -> None:
        """Simpson's index should be in [0, 1)."""
        d = compute_simpson_index(sample_sightings)
        assert 0.0 <= d < 1.0

    def test_simpson_single_species(self, single_species_sightings: list[Sighting]) -> None:
        """Single-species community: D=1, so 1-D=0."""
        d = compute_simpson_index(single_species_sightings)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_simpson_empty(self, empty_sightings: list[Sighting]) -> None:
        """Empty sightings should yield 0."""
        assert compute_simpson_index(empty_sightings) == 0.0

    def test_simpson_increases_with_evenness(self) -> None:
        """Simpson's index should be higher when abundances are more even."""
        base = datetime.now()

        # Uneven: one species dominates.
        uneven = [Sighting(species="elk", timestamp=base, camera_id="CAM-001")] * 90 + [
            Sighting(species="deer", timestamp=base, camera_id="CAM-001")
        ] * 10

        # Even: equal abundances.
        even = [Sighting(species="elk", timestamp=base, camera_id="CAM-001")] * 50 + [
            Sighting(species="deer", timestamp=base, camera_id="CAM-001")
        ] * 50

        assert compute_simpson_index(even) > compute_simpson_index(uneven)


class TestPielouEvenness:
    """Tests for Pielou's evenness index (J')."""

    def test_evenness_range(self, sample_sightings: list[Sighting]) -> None:
        """J' should be in (0, 1]."""
        j = compute_pielou_evenness(sample_sightings)
        assert 0.0 < j <= 1.0

    def test_evenness_single_species(self, single_species_sightings: list[Sighting]) -> None:
        """J' is undefined (returns 0) for single-species communities."""
        assert compute_pielou_evenness(single_species_sightings) == 0.0

    def test_evenness_perfect_equality(self) -> None:
        """J' should be 1.0 when all species are equally abundant."""
        base = datetime.now()
        sightings = []
        for i, species in enumerate(["elk", "deer", "wolf"]):
            sightings.extend(
                Sighting(species=species, timestamp=base + timedelta(hours=j), camera_id="CAM-001")
                for j in range(20)
            )
        j = compute_pielou_evenness(sightings)
        assert j == pytest.approx(1.0, rel=1e-6)


class TestActivityPattern:
    """Tests for diel activity pattern analysis."""

    def test_activity_pattern_keys(self, sample_sightings: list[Sighting]) -> None:
        """Activity pattern should have bins for all 24 hours."""
        pattern = compute_activity_pattern(sample_sightings)
        assert len(pattern) == 24
        assert all(h in pattern for h in range(24))

    def test_activity_pattern_species_filter(self, sample_sightings: list[Sighting]) -> None:
        """Filtering by species should only include that species' detections."""
        all_pattern = compute_activity_pattern(sample_sightings)
        elk_pattern = compute_activity_pattern(sample_sightings, species="elk")

        assert sum(elk_pattern.values()) <= sum(all_pattern.values())
        assert sum(elk_pattern.values()) > 0

    def test_activity_pattern_empty(self, empty_sightings: list[Sighting]) -> None:
        """Empty sightings should produce all-zero bins."""
        pattern = compute_activity_pattern(empty_sightings)
        assert all(v == 0 for v in pattern.values())

    def test_nocturnal_species_peak_at_night(self, sample_sightings: list[Sighting]) -> None:
        """Nocturnal species (raccoon) should peak during dark hours."""
        pattern = compute_activity_pattern(sample_sightings, species="raccoon")
        night_detections = sum(pattern.get(h, 0) for h in list(range(0, 6)) + list(range(18, 24)))
        day_detections = sum(pattern.get(h, 0) for h in range(6, 18))
        assert night_detections > day_detections


class TestActivityOverlap:
    """Tests for temporal activity overlap between species."""

    def test_overlap_range(self, sample_sightings: list[Sighting]) -> None:
        """Overlap coefficient should be in [0, 1]."""
        overlap = compute_activity_overlap(sample_sightings, "elk", "coyote")
        assert 0.0 <= overlap <= 1.0

    def test_self_overlap_is_one(self, sample_sightings: list[Sighting]) -> None:
        """A species overlapping with itself should yield 1.0."""
        overlap = compute_activity_overlap(sample_sightings, "elk", "elk")
        assert overlap == pytest.approx(1.0, rel=1e-6)

    def test_overlap_missing_species(self, sample_sightings: list[Sighting]) -> None:
        """Overlap with a nonexistent species should be 0."""
        overlap = compute_activity_overlap(sample_sightings, "elk", "unicorn")
        assert overlap == 0.0


class TestNaiveOccupancy:
    """Tests for naive occupancy estimation."""

    def test_occupancy_range(self, sample_sightings: list[Sighting]) -> None:
        """Naive occupancy should be in [0, 1]."""
        occ = compute_naive_occupancy(sample_sightings, "elk")
        assert 0.0 <= occ <= 1.0

    def test_occupancy_absent_species(self, sample_sightings: list[Sighting]) -> None:
        """Species not in data should have occupancy 0."""
        occ = compute_naive_occupancy(sample_sightings, "wolverine")
        assert occ == 0.0

    def test_occupancy_with_known_cameras(self) -> None:
        """Occupancy denominator should use provided camera list."""
        sightings = [
            Sighting(species="elk", timestamp=datetime.now(), camera_id="CAM-001"),
        ]
        # Only detected at 1 of 5 known cameras.
        occ = compute_naive_occupancy(
            sightings,
            "elk",
            camera_ids=["CAM-001", "CAM-002", "CAM-003", "CAM-004", "CAM-005"],
        )
        assert occ == pytest.approx(0.2)


class TestDetectionHistory:
    """Tests for occupancy model detection history matrix."""

    def test_history_shape(self) -> None:
        """Detection history should have shape (n_sites, n_occasions)."""
        base = datetime(2025, 7, 1)
        sightings = [
            Sighting(species="elk", timestamp=base + timedelta(days=1), camera_id="CAM-001"),
            Sighting(species="elk", timestamp=base + timedelta(days=8), camera_id="CAM-002"),
        ]
        cameras = ["CAM-001", "CAM-002", "CAM-003"]
        occasions = [
            (base, base + timedelta(days=7)),
            (base + timedelta(days=7), base + timedelta(days=14)),
        ]

        history = build_detection_history(sightings, "elk", cameras, occasions)
        assert history.shape == (3, 2)

    def test_history_detections(self) -> None:
        """Detection events should be marked as 1 in the correct cells."""
        base = datetime(2025, 7, 1)
        sightings = [
            Sighting(species="elk", timestamp=base + timedelta(days=2), camera_id="CAM-001"),
        ]
        cameras = ["CAM-001", "CAM-002"]
        occasions = [
            (base, base + timedelta(days=7)),
            (base + timedelta(days=7), base + timedelta(days=14)),
        ]

        history = build_detection_history(sightings, "elk", cameras, occasions)
        assert history[0, 0] == 1  # CAM-001, occasion 1
        assert history[0, 1] == 0  # CAM-001, occasion 2
        assert history[1, 0] == 0  # CAM-002, occasion 1
        assert history[1, 1] == 0  # CAM-002, occasion 2


class TestPopulationTrend:
    """Tests for population trend (Relative Abundance Index) computation."""

    def test_trend_columns(self, sample_sightings: list[Sighting]) -> None:
        """Trend DataFrame should have required columns."""
        trend = compute_population_trend(sample_sightings, "elk")
        expected_cols = {"period", "detections", "trap_nights", "rai"}
        assert expected_cols.issubset(set(trend.columns))

    def test_trend_rai_positive(self, sample_sightings: list[Sighting]) -> None:
        """RAI should be non-negative."""
        trend = compute_population_trend(sample_sightings, "elk")
        assert all(trend["rai"] >= 0)

    def test_trend_empty_species(self, sample_sightings: list[Sighting]) -> None:
        """Non-existent species should return empty DataFrame."""
        trend = compute_population_trend(sample_sightings, "unicorn")
        assert len(trend) == 0


class TestSpeciesCoOccurrence:
    """Tests for species co-occurrence analysis."""

    def test_co_occurrence_columns(self, sample_sightings: list[Sighting]) -> None:
        """Co-occurrence DataFrame should have required columns."""
        df = compute_species_co_occurrence(sample_sightings)
        expected = {
            "species_a",
            "species_b",
            "co_occurrences",
            "expected",
            "sites_both",
            "association_type",
        }
        assert expected.issubset(set(df.columns))

    def test_co_occurrence_association_types(self, sample_sightings: list[Sighting]) -> None:
        """Association type should be one of: positive, negative, random."""
        df = compute_species_co_occurrence(sample_sightings)
        valid_types = {"positive", "negative", "random"}
        assert set(df["association_type"].unique()).issubset(valid_types)

    def test_co_occurrence_empty(self, empty_sightings: list[Sighting]) -> None:
        """Empty sightings should produce empty co-occurrence table."""
        df = compute_species_co_occurrence(empty_sightings)
        assert len(df) == 0


class TestBiodiversitySummary:
    """Tests for the full biodiversity summary computation."""

    def test_summary_fields(self, sample_sightings: list[Sighting]) -> None:
        """Summary should contain all expected fields."""
        summary = compute_full_biodiversity_summary(sample_sightings)
        assert isinstance(summary, BiodiversitySummary)
        assert summary.species_richness > 0
        assert summary.shannon_index > 0
        assert summary.simpson_index > 0
        assert summary.total_detections > 0

    def test_summary_per_camera(self, sample_sightings: list[Sighting]) -> None:
        """Per-camera summary should only include that camera's sightings."""
        summary = compute_full_biodiversity_summary(sample_sightings, camera_id="CAM-001")
        assert summary.camera_id == "CAM-001"
        assert summary.total_detections <= len(sample_sightings)

    def test_summary_empty(self, empty_sightings: list[Sighting]) -> None:
        """Empty sightings should produce zeroed summary."""
        summary = compute_full_biodiversity_summary(empty_sightings)
        assert summary.species_richness == 0
        assert summary.shannon_index == 0.0
        assert summary.total_detections == 0
