"""Tests for VCF file processing pipeline."""

from __future__ import annotations

import pytest

from cloud_genomics.models.variant_classifier import VariantFeatures
from cloud_genomics.pipeline.vcf_processor import (
    QualityFilter,
    QualityThresholds,
    VariantAnnotator,
    VariantType,
    VCFParser,
    VCFProcessor,
    VCFVariant,
    create_sample_vcf,
)


class TestVCFParser:
    """Tests for VCF file parsing."""

    def test_parse_sample_vcf(self, sample_vcf_content: str) -> None:
        """Should parse sample VCF without errors."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        assert len(variants) > 0

    def test_parse_header(self, sample_vcf_content: str) -> None:
        """Should parse VCF header metadata correctly."""
        parser = VCFParser()
        list(parser.parse_string(sample_vcf_content))  # trigger parsing

        header = parser.header
        assert header is not None
        assert header.file_format == "VCFv4.2"
        assert "SAMPLE1" in header.sample_names
        assert "DP" in header.info_fields
        assert "GT" in header.format_fields
        assert header.reference_genome == "GRCh38"

    def test_parse_variant_fields(self, sample_vcf_content: str) -> None:
        """Should parse variant fields correctly."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        first = variants[0]

        assert first.chrom == "chr1"
        assert first.pos == 12345
        assert first.id == "rs1801133"
        assert first.ref == "C"
        assert first.alt == ["T"]
        assert first.qual == 500.0
        assert first.filter_status == ["PASS"]

    def test_parse_info_field(self, sample_vcf_content: str) -> None:
        """Should parse INFO fields into key-value pairs."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        info = variants[0].info

        assert info["DP"] == 100
        assert info["MQ"] == 60
        assert isinstance(info["gnomAD_AF"], float)

    def test_parse_sample_data(self, sample_vcf_content: str) -> None:
        """Should parse FORMAT/sample columns correctly."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        first = variants[0]

        assert "SAMPLE1" in first.sample_data
        sample = first.sample_data["SAMPLE1"]
        assert sample["GT"] == "0/1"
        assert sample["DP"] == "100"
        assert sample["GQ"] == "99"
        assert sample["AD"] == "50,50"

    def test_variant_type_snv(self, sample_vcf_content: str) -> None:
        """SNV should be detected correctly."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        # First variant: C -> T (SNV)
        assert variants[0].variant_type == VariantType.SNV

    def test_variant_type_deletion(self, sample_vcf_content: str) -> None:
        """Deletion should be detected correctly."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        # Last variant: AG -> A (deletion)
        deletion_variants = [v for v in variants if v.variant_type == VariantType.DELETION]
        assert len(deletion_variants) > 0

    def test_variant_key_uniqueness(self, sample_vcf_content: str) -> None:
        """Each variant should have a unique key."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))
        keys = [v.variant_key for v in variants]
        assert len(keys) == len(set(keys))

    def test_is_pass_filter(self, sample_vcf_content: str) -> None:
        """PASS variants should be identified correctly."""
        parser = VCFParser()
        variants = list(parser.parse_string(sample_vcf_content))

        pass_variants = [v for v in variants if v.is_pass]
        non_pass = [v for v in variants if not v.is_pass]

        assert len(pass_variants) > 0
        assert len(non_pass) > 0  # Sample includes LowQual variant

    def test_empty_vcf_raises(self) -> None:
        """VCF without header line should raise ValueError."""
        parser = VCFParser()
        with pytest.raises(ValueError, match="missing.*header"):
            list(parser.parse_string("chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\n"))

    def test_malformed_line_skipped(self) -> None:
        """Malformed data lines should be skipped with a warning."""
        vcf = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\t.\tA\tG\t30\tPASS\tDP=10
bad_line_too_few_columns
chr1\t200\t.\tC\tT\t50\tPASS\tDP=20
"""
        parser = VCFParser()
        variants = list(parser.parse_string(vcf))
        assert len(variants) == 2  # malformed line skipped


class TestQualityFilter:
    """Tests for variant quality filtering."""

    def _make_variant(
        self,
        qual: float = 100.0,
        filter_status: str = "PASS",
        dp: int = 50,
        mq: float = 60.0,
        gq: str = "99",
        ad: str = "25,25",
        gt: str = "0/1",
    ) -> VCFVariant:
        """Create a VCFVariant for filter testing."""
        return VCFVariant(
            chrom="chr1",
            pos=100,
            id=".",
            ref="A",
            alt=["G"],
            qual=qual,
            filter_status=[filter_status],
            info={"DP": dp, "MQ": mq, "FS": 1.0},
            format_fields=["GT", "DP", "GQ", "AD"],
            sample_data={"SAMPLE": {"GT": gt, "DP": str(dp), "GQ": gq, "AD": ad}},
        )

    def test_passing_variant(self) -> None:
        """High-quality variant should pass all filters."""
        variant = self._make_variant()
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert passed
        assert len(failed) == 0

    def test_low_quality_filtered(self) -> None:
        """Low QUAL score should fail quality filter."""
        variant = self._make_variant(qual=5.0)
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert not passed
        assert any("QUAL" in f for f in failed)

    def test_low_depth_filtered(self) -> None:
        """Low depth should fail depth filter."""
        variant = self._make_variant(dp=3)
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert not passed
        assert any("DP" in f for f in failed)

    def test_low_mapping_quality(self) -> None:
        """Low mapping quality should fail MQ filter."""
        variant = self._make_variant(mq=15.0)
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert not passed
        assert any("MQ" in f for f in failed)

    def test_non_pass_filter(self) -> None:
        """Non-PASS filter status should fail when require_pass is True."""
        variant = self._make_variant(filter_status="LowQual")
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert not passed
        assert any("FILTER" in f for f in failed)

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be applied correctly."""
        variant = self._make_variant(qual=15.0, dp=8, filter_status="LowQual")
        lenient = QualityThresholds(
            min_qual=10.0,
            min_depth=5,
            require_pass_filter=False,
        )
        qf = QualityFilter(lenient)
        passed, failed = qf.filter_variant(variant)
        assert passed

    def test_allele_balance_filtered(self) -> None:
        """Extreme allele balance should fail filter."""
        variant = self._make_variant(ad="48,2")  # very skewed
        qf = QualityFilter()
        passed, failed = qf.filter_variant(variant)
        assert not passed
        assert any("AB" in f for f in failed)


class TestVariantAnnotator:
    """Tests for variant annotation."""

    def test_annotate_known_variant(self) -> None:
        """Known rsID should return population frequencies."""
        variant = VCFVariant(
            chrom="chr1",
            pos=12345,
            id="rs1801133",
            ref="C",
            alt=["T"],
            qual=100,
            filter_status=["PASS"],
            info={},
        )
        annotator = VariantAnnotator()
        annotations = annotator.annotate_variant(variant)

        assert annotations["gnomad_af"] == 0.25
        assert "consequence" in annotations

    def test_annotate_with_info_field_frequencies(self) -> None:
        """Frequencies from INFO field should take precedence."""
        variant = VCFVariant(
            chrom="chr1",
            pos=100,
            id=".",
            ref="A",
            alt=["G"],
            qual=100,
            filter_status=["PASS"],
            info={"gnomAD_AF": 0.05, "gnomAD_AF_AFR": 0.03},
        )
        annotator = VariantAnnotator()
        annotations = annotator.annotate_variant(variant)

        assert annotations["gnomad_af"] == 0.05
        assert annotations["gnomad_af_afr"] == 0.03

    def test_annotate_with_predictions(self) -> None:
        """In-silico predictions from INFO should be extracted."""
        variant = VCFVariant(
            chrom="chr1",
            pos=100,
            id=".",
            ref="A",
            alt=["G"],
            qual=100,
            filter_status=["PASS"],
            info={
                "SIFT_score": 0.01,
                "Polyphen2_HDIV_score": 0.99,
                "CADD_phred": 35,
            },
        )
        annotator = VariantAnnotator()
        annotations = annotator.annotate_variant(variant)

        assert annotations["sift_score"] == 0.01
        assert annotations["polyphen2_score"] == 0.99
        assert annotations["cadd_phred"] == 35

    def test_consequence_inference(self) -> None:
        """Should infer consequence from variant type."""
        # SNV -> missense default
        snv = VCFVariant(
            chrom="chr1",
            pos=100,
            id=".",
            ref="A",
            alt=["G"],
            qual=100,
            filter_status=["PASS"],
            info={},
        )
        annotator = VariantAnnotator()
        annotations = annotator.annotate_variant(snv)
        assert annotations["consequence"] == "missense"


class TestVCFProcessor:
    """Tests for the end-to-end VCF processor."""

    def test_process_sample_vcf(self, sample_vcf_content: str) -> None:
        """Should process sample VCF and return results."""
        processor = VCFProcessor()
        results = processor.process_string(sample_vcf_content)

        assert len(results) > 0
        for variant, features in results:
            assert isinstance(variant, VCFVariant)
            assert isinstance(features, VariantFeatures)

    def test_processing_stats(self, sample_vcf_content: str) -> None:
        """Should track processing statistics."""
        processor = VCFProcessor()
        processor.process_string(sample_vcf_content)
        stats = processor.stats

        assert stats.total_variants > 0
        assert stats.passed_filters >= 0
        assert stats.total_variants >= stats.passed_filters
        assert stats.snv_count > 0

    def test_filtered_variants_excluded(self, sample_vcf_content: str) -> None:
        """Low-quality variants should be filtered out."""
        processor = VCFProcessor()
        results = processor.process_string(sample_vcf_content)
        stats = processor.stats

        # Sample VCF has a LowQual variant that should be filtered
        filtered = stats.total_variants - stats.passed_filters
        assert filtered > 0

    def test_lenient_thresholds(
        self,
        sample_vcf_content: str,
        lenient_quality_thresholds: QualityThresholds,
    ) -> None:
        """Lenient thresholds should pass more variants."""
        strict_processor = VCFProcessor()
        strict_results = strict_processor.process_string(sample_vcf_content)

        lenient_processor = VCFProcessor(quality_thresholds=lenient_quality_thresholds)
        lenient_results = lenient_processor.process_string(sample_vcf_content)

        assert len(lenient_results) >= len(strict_results)

    def test_features_populated(self, sample_vcf_content: str) -> None:
        """Extracted features should have non-default values where annotations exist."""
        processor = VCFProcessor()
        results = processor.process_string(sample_vcf_content)

        # First passing variant (rs1801133) should have gnomAD AF annotation
        if results:
            _, features = results[0]
            assert features.variant_type in ("SNV", "insertion", "deletion", "MNV")
            assert features.consequence != ""

    def test_header_accessible(self, sample_vcf_content: str) -> None:
        """VCF header should be accessible after processing."""
        processor = VCFProcessor()
        processor.process_string(sample_vcf_content)

        header = processor.header
        assert header is not None
        assert header.file_format == "VCFv4.2"


class TestCreateSampleVCF:
    """Tests for the sample VCF generator."""

    def test_valid_vcf_format(self) -> None:
        """Generated sample VCF should be parseable."""
        content = create_sample_vcf()
        parser = VCFParser()
        variants = list(parser.parse_string(content))
        assert len(variants) == 5

    def test_includes_different_types(self) -> None:
        """Sample VCF should include SNVs and indels."""
        content = create_sample_vcf()
        parser = VCFParser()
        variants = list(parser.parse_string(content))

        types = {v.variant_type for v in variants}
        assert VariantType.SNV in types
        assert VariantType.DELETION in types
