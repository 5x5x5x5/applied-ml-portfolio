"""VCF (Variant Call Format) file processor.

Parses VCF files, extracts variant features, annotates with population
frequency data, and applies quality filters. Designed for integration
with the CloudGenomics classification pipeline.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from cloud_genomics.models.variant_classifier import VariantFeatures

logger = logging.getLogger(__name__)


class VariantType(str, Enum):
    """Types of genetic variants."""

    SNV = "SNV"
    INSERTION = "insertion"
    DELETION = "deletion"
    MNV = "MNV"
    COMPLEX = "complex"


class FilterStatus(str, Enum):
    """VCF filter status values."""

    PASS = "PASS"
    LOW_QUALITY = "LowQual"
    LOW_DEPTH = "LowDepth"
    STRAND_BIAS = "StrandBias"
    LOW_GQ = "LowGQ"
    FILTERED = "filtered"


@dataclass
class VCFHeader:
    """Parsed VCF file header information."""

    file_format: str = ""
    info_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    format_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    filter_fields: dict[str, str] = field(default_factory=dict)
    contigs: dict[str, dict[str, str]] = field(default_factory=dict)
    sample_names: list[str] = field(default_factory=list)
    raw_headers: list[str] = field(default_factory=list)
    reference_genome: str = ""


@dataclass
class VCFVariant:
    """A single variant record from a VCF file."""

    chrom: str
    pos: int
    id: str  # rsID or "."
    ref: str
    alt: list[str]
    qual: float
    filter_status: list[str]
    info: dict[str, Any]
    format_fields: list[str] = field(default_factory=list)
    sample_data: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def variant_type(self) -> VariantType:
        """Determine the variant type from ref and alt alleles."""
        if len(self.alt) > 1:
            return VariantType.COMPLEX

        alt = self.alt[0]
        ref = self.ref

        if len(ref) == 1 and len(alt) == 1:
            return VariantType.SNV
        if len(ref) > 1 and len(alt) > 1 and len(ref) == len(alt):
            return VariantType.MNV
        if len(alt) > len(ref):
            return VariantType.INSERTION
        if len(ref) > len(alt):
            return VariantType.DELETION
        return VariantType.COMPLEX

    @property
    def variant_key(self) -> str:
        """Generate a unique key for this variant: chr-pos-ref-alt."""
        return f"{self.chrom}-{self.pos}-{self.ref}-{','.join(self.alt)}"

    @property
    def is_pass(self) -> bool:
        """Check if variant passed all filters."""
        return self.filter_status == ["PASS"] or self.filter_status == ["."]


@dataclass
class QualityThresholds:
    """Quality filter thresholds for VCF variant filtering."""

    min_qual: float = 30.0
    min_depth: int = 10
    min_genotype_quality: int = 20
    min_allele_balance: float = 0.2
    max_allele_balance: float = 0.8
    min_mapping_quality: float = 40.0
    max_strand_bias: float = 0.01
    require_pass_filter: bool = True


@dataclass
class ProcessingStats:
    """Statistics from VCF processing."""

    total_variants: int = 0
    passed_filters: int = 0
    failed_quality: int = 0
    failed_depth: int = 0
    failed_genotype_quality: int = 0
    failed_allele_balance: int = 0
    snv_count: int = 0
    insertion_count: int = 0
    deletion_count: int = 0
    mnv_count: int = 0
    complex_count: int = 0
    multiallelic_count: int = 0


class VCFParser:
    """Parser for VCF (Variant Call Format) files.

    Supports VCF 4.1, 4.2, and 4.3 file formats, including gzipped files.
    """

    # Regex patterns for VCF header metadata lines
    _META_PATTERN = re.compile(r"##(\w+)=(.+)")
    _STRUCTURED_META_PATTERN = re.compile(r"##(\w+)=<(.+)>")
    _KEY_VALUE_PATTERN = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^,]+)')

    def __init__(self) -> None:
        self._header: VCFHeader | None = None

    @property
    def header(self) -> VCFHeader | None:
        return self._header

    def parse_file(self, path: str | Path) -> Iterator[VCFVariant]:
        """Parse a VCF file and yield variant records.

        Args:
            path: Path to VCF or VCF.gz file.

        Yields:
            VCFVariant objects for each record.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {path}")

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            yield from self._parse_stream(fh)

    def parse_string(self, vcf_content: str) -> Iterator[VCFVariant]:
        """Parse VCF content from a string.

        Args:
            vcf_content: VCF file content as a string.

        Yields:
            VCFVariant objects.
        """
        yield from self._parse_stream(io.StringIO(vcf_content))

    def _parse_stream(self, stream: TextIO) -> Iterator[VCFVariant]:
        """Parse a VCF stream (file or string).

        Args:
            stream: Text stream containing VCF data.

        Yields:
            VCFVariant objects.
        """
        self._header = VCFHeader()
        header_parsed = False

        for line in stream:
            line = line.rstrip("\n\r")

            if not line:
                continue

            if line.startswith("##"):
                self._parse_meta_line(line)
                continue

            if line.startswith("#CHROM"):
                self._parse_column_header(line)
                header_parsed = True
                continue

            if not header_parsed:
                raise ValueError("VCF file missing #CHROM header line")

            variant = self._parse_variant_line(line)
            if variant is not None:
                yield variant

    def _parse_meta_line(self, line: str) -> None:
        """Parse a VCF metadata header line (##key=value)."""
        assert self._header is not None

        self._header.raw_headers.append(line)

        structured = self._STRUCTURED_META_PATTERN.match(line)
        if structured:
            key = structured.group(1)
            fields_str = structured.group(2)
            fields = self._parse_key_value_pairs(fields_str)

            if key == "INFO":
                field_id = fields.get("ID", "")
                self._header.info_fields[field_id] = fields
            elif key == "FORMAT":
                field_id = fields.get("ID", "")
                self._header.format_fields[field_id] = fields
            elif key == "FILTER":
                field_id = fields.get("ID", "")
                self._header.filter_fields[field_id] = fields.get("Description", "")
            elif key == "contig":
                contig_id = fields.get("ID", "")
                self._header.contigs[contig_id] = fields
            return

        simple = self._META_PATTERN.match(line)
        if simple:
            key, value = simple.group(1), simple.group(2)
            if key == "fileformat":
                self._header.file_format = value
            elif key == "reference":
                self._header.reference_genome = value

    def _parse_key_value_pairs(self, text: str) -> dict[str, str]:
        """Parse comma-separated key=value pairs from structured header lines."""
        result: dict[str, str] = {}
        for match in self._KEY_VALUE_PATTERN.finditer(text):
            key = match.group(1)
            value = match.group(2).strip('"')
            result[key] = value
        return result

    def _parse_column_header(self, line: str) -> None:
        """Parse the #CHROM column header line to extract sample names."""
        assert self._header is not None
        columns = line.lstrip("#").split("\t")
        # Standard VCF columns: CHROM POS ID REF ALT QUAL FILTER INFO [FORMAT SAMPLE...]
        if len(columns) > 9:
            self._header.sample_names = columns[9:]

    def _parse_variant_line(self, line: str) -> VCFVariant | None:
        """Parse a single VCF data line into a VCFVariant.

        Returns None if the line cannot be parsed.
        """
        fields = line.split("\t")
        if len(fields) < 8:
            logger.warning("Skipping malformed VCF line (fewer than 8 columns): %s", line[:80])
            return None

        chrom = fields[0]
        try:
            pos = int(fields[1])
        except ValueError:
            logger.warning("Invalid position value: %s", fields[1])
            return None

        variant_id = fields[2]
        ref = fields[3].upper()
        alt = [a.upper() for a in fields[4].split(",")]

        try:
            qual = float(fields[5]) if fields[5] != "." else 0.0
        except ValueError:
            qual = 0.0

        filter_status = fields[6].split(";") if fields[6] != "." else ["."]

        info = self._parse_info_field(fields[7])

        # Parse FORMAT and sample genotype columns
        format_fields: list[str] = []
        sample_data: dict[str, dict[str, str]] = {}

        if len(fields) > 8:
            format_fields = fields[8].split(":")
            sample_names = self._header.sample_names if self._header else []

            for i, sample_field in enumerate(fields[9:]):
                sample_name = sample_names[i] if i < len(sample_names) else f"SAMPLE_{i}"
                values = sample_field.split(":")
                sample_data[sample_name] = {
                    fmt: val for fmt, val in zip(format_fields, values, strict=False)
                }

        return VCFVariant(
            chrom=chrom,
            pos=pos,
            id=variant_id,
            ref=ref,
            alt=alt,
            qual=qual,
            filter_status=filter_status,
            info=info,
            format_fields=format_fields,
            sample_data=sample_data,
        )

    def _parse_info_field(self, info_str: str) -> dict[str, Any]:
        """Parse the INFO field from a VCF record."""
        if info_str == ".":
            return {}

        info: dict[str, Any] = {}
        for entry in info_str.split(";"):
            if "=" in entry:
                key, value = entry.split("=", 1)
                # Try numeric conversion
                if "," in value:
                    info[key] = value.split(",")
                else:
                    try:
                        info[key] = int(value)
                    except ValueError:
                        try:
                            info[key] = float(value)
                        except ValueError:
                            info[key] = value
            else:
                info[entry] = True  # Flag fields

        return info


class VariantAnnotator:
    """Annotates VCF variants with population frequency and functional data.

    In production, this would query external databases (gnomAD, ClinVar, etc.).
    This implementation provides the annotation interface and mock data for
    development/testing.
    """

    # Population allele frequency lookup (mock data keyed by rsID)
    _MOCK_AF_DATA: dict[str, dict[str, float]] = {
        "rs1801133": {"gnomad_af": 0.25, "af_afr": 0.10, "af_eas": 0.30, "af_nfe": 0.32},
        "rs6025": {"gnomad_af": 0.02, "af_afr": 0.005, "af_eas": 0.001, "af_nfe": 0.05},
        "rs1799963": {"gnomad_af": 0.01, "af_afr": 0.001, "af_eas": 0.0001, "af_nfe": 0.02},
    }

    _CONSEQUENCE_MAP: dict[str, str] = {
        "missense_variant": "missense",
        "synonymous_variant": "synonymous",
        "stop_gained": "nonsense",
        "frameshift_variant": "frameshift",
        "splice_donor_variant": "splice_site",
        "splice_acceptor_variant": "splice_site",
        "splice_region_variant": "splice_region",
        "intron_variant": "intron",
        "5_prime_UTR_variant": "utr_5",
        "3_prime_UTR_variant": "utr_3",
        "intergenic_variant": "intergenic",
        "inframe_insertion": "inframe_insertion",
        "inframe_deletion": "inframe_deletion",
        "start_lost": "start_lost",
        "stop_lost": "stop_lost",
    }

    def annotate_variant(self, variant: VCFVariant) -> dict[str, Any]:
        """Annotate a single variant with functional and frequency data.

        Args:
            variant: Parsed VCF variant record.

        Returns:
            Dictionary of annotations including frequencies, consequences,
            and in-silico predictions.
        """
        annotations: dict[str, Any] = {}

        # Population frequencies
        af_data = self._get_allele_frequencies(variant)
        annotations.update(af_data)

        # Variant consequence from INFO CSQ/ANN field
        annotations["consequence"] = self._extract_consequence(variant)

        # In-silico predictions from INFO field (if present)
        annotations.update(self._extract_predictions(variant))

        # Conservation scores from INFO field
        annotations.update(self._extract_conservation(variant))

        logger.debug(
            "Annotated variant %s with %d annotation fields",
            variant.variant_key,
            len(annotations),
        )
        return annotations

    def _get_allele_frequencies(self, variant: VCFVariant) -> dict[str, float]:
        """Look up population allele frequencies.

        Checks INFO field first, then falls back to mock database.
        """
        result: dict[str, float] = {
            "gnomad_af": 0.0,
            "gnomad_af_afr": 0.0,
            "gnomad_af_eas": 0.0,
            "gnomad_af_nfe": 0.0,
            "gnomad_homozygote_count": 0,
        }

        # Check VCF INFO field for annotated frequencies
        info = variant.info
        if "AF" in info:
            val = info["AF"]
            result["gnomad_af"] = float(val[0]) if isinstance(val, list) else float(val)
        if "gnomAD_AF" in info:
            result["gnomad_af"] = float(info["gnomAD_AF"])
        if "gnomAD_AF_AFR" in info:
            result["gnomad_af_afr"] = float(info["gnomAD_AF_AFR"])
        if "gnomAD_AF_EAS" in info:
            result["gnomad_af_eas"] = float(info["gnomAD_AF_EAS"])
        if "gnomAD_AF_NFE" in info:
            result["gnomad_af_nfe"] = float(info["gnomAD_AF_NFE"])
        if "gnomAD_nhomalt" in info:
            result["gnomad_homozygote_count"] = int(info["gnomAD_nhomalt"])

        # Fall back to mock lookup by rsID
        if result["gnomad_af"] == 0.0 and variant.id in self._MOCK_AF_DATA:
            mock = self._MOCK_AF_DATA[variant.id]
            result["gnomad_af"] = mock.get("gnomad_af", 0.0)
            result["gnomad_af_afr"] = mock.get("af_afr", 0.0)
            result["gnomad_af_eas"] = mock.get("af_eas", 0.0)
            result["gnomad_af_nfe"] = mock.get("af_nfe", 0.0)

        return result

    def _extract_consequence(self, variant: VCFVariant) -> str:
        """Extract the most severe consequence from INFO annotations."""
        info = variant.info

        # Check for VEP CSQ field
        if "CSQ" in info:
            csq = info["CSQ"]
            if isinstance(csq, str):
                parts = csq.split("|")
                if len(parts) > 1:
                    raw_consequence = parts[1]
                    return self._CONSEQUENCE_MAP.get(raw_consequence, "missense")

        # Check for SnpEff ANN field
        if "ANN" in info:
            ann = info["ANN"]
            if isinstance(ann, str):
                parts = ann.split("|")
                if len(parts) > 1:
                    raw_consequence = parts[1]
                    return self._CONSEQUENCE_MAP.get(raw_consequence, "missense")

        # Infer from variant type
        vtype = variant.variant_type
        if vtype == VariantType.SNV:
            return "missense"  # conservative default for SNVs
        if vtype == VariantType.INSERTION:
            if len(variant.alt[0]) - len(variant.ref) % 3 != 0:
                return "frameshift"
            return "inframe_insertion"
        if vtype == VariantType.DELETION:
            if len(variant.ref) - len(variant.alt[0]) % 3 != 0:
                return "frameshift"
            return "inframe_deletion"

        return "missense"

    def _extract_predictions(self, variant: VCFVariant) -> dict[str, float]:
        """Extract in-silico pathogenicity predictions from INFO field."""
        predictions: dict[str, float] = {}
        info = variant.info

        prediction_keys = {
            "SIFT_score": "sift_score",
            "Polyphen2_HDIV_score": "polyphen2_score",
            "CADD_phred": "cadd_phred",
            "REVEL_score": "revel_score",
            "MutationTaster_score": "mutation_taster_score",
            "SpliceAI_DS_AG": "splice_ai_score",
            "phyloP100way_vertebrate": "phylop_score",
            "phastCons100way_vertebrate": "phastcons_score",
            "GERP_RS": "gerp_score",
        }

        for vcf_key, feature_key in prediction_keys.items():
            if vcf_key in info:
                try:
                    val = info[vcf_key]
                    if isinstance(val, list):
                        val = val[0]
                    if val != "." and val != "":
                        predictions[feature_key] = float(val)
                except (ValueError, TypeError):
                    pass

        return predictions

    def _extract_conservation(self, variant: VCFVariant) -> dict[str, float]:
        """Extract conservation scores from INFO field."""
        conservation: dict[str, float] = {}
        info = variant.info

        if "phyloP" in info:
            conservation["phylop_score"] = float(info["phyloP"])
        if "phastCons" in info:
            conservation["phastcons_score"] = float(info["phastCons"])
        if "GERP" in info:
            conservation["gerp_score"] = float(info["GERP"])

        return conservation


class QualityFilter:
    """Applies quality-based filters to VCF variants.

    Implements GATK-style hard filtering with configurable thresholds.
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = thresholds or QualityThresholds()

    @property
    def thresholds(self) -> QualityThresholds:
        return self._thresholds

    def filter_variant(self, variant: VCFVariant) -> tuple[bool, list[str]]:
        """Apply quality filters to a variant.

        Args:
            variant: VCF variant record to filter.

        Returns:
            Tuple of (passed: bool, failed_filters: list[str]).
        """
        failed: list[str] = []
        t = self._thresholds

        # PASS filter check
        if t.require_pass_filter and not variant.is_pass:
            failed.append(f"FILTER={';'.join(variant.filter_status)}")

        # Quality score
        if variant.qual < t.min_qual:
            failed.append(f"QUAL={variant.qual}<{t.min_qual}")

        # Depth check from INFO field
        depth = variant.info.get("DP", 0)
        if isinstance(depth, int) and depth < t.min_depth:
            failed.append(f"DP={depth}<{t.min_depth}")

        # Mapping quality from INFO
        mq = variant.info.get("MQ", 60.0)
        if isinstance(mq, (int, float)) and mq < t.min_mapping_quality:
            failed.append(f"MQ={mq}<{t.min_mapping_quality}")

        # Strand bias (Fisher strand)
        fs = variant.info.get("FS", 0.0)
        if (
            isinstance(fs, (int, float)) and fs > -10 * np.log10(t.max_strand_bias)
            if t.max_strand_bias > 0
            else False
        ):
            # FS > 60 typically indicates strong strand bias
            if fs > 60:
                failed.append(f"FS={fs}>60")

        # Per-sample quality checks
        for sample_name, sample in variant.sample_data.items():
            # Genotype quality
            gq = sample.get("GQ", "99")
            try:
                gq_val = int(gq)
                if gq_val < t.min_genotype_quality:
                    failed.append(f"{sample_name}:GQ={gq_val}<{t.min_genotype_quality}")
            except ValueError:
                pass

            # Allele balance for heterozygous calls
            ad = sample.get("AD", "")
            gt = sample.get("GT", "")
            if ad and "/" in gt or "|" in gt:
                alleles = gt.replace("|", "/").split("/")
                if len(set(alleles)) > 1:  # Heterozygous
                    try:
                        depths = [int(d) for d in ad.split(",")]
                        total = sum(depths)
                        if total > 0 and len(depths) > 1:
                            balance = depths[1] / total
                            if balance < t.min_allele_balance or balance > t.max_allele_balance:
                                failed.append(
                                    f"{sample_name}:AB={balance:.2f} "
                                    f"outside [{t.min_allele_balance},{t.max_allele_balance}]"
                                )
                    except (ValueError, ZeroDivisionError):
                        pass

        passed = len(failed) == 0
        return passed, failed


# Need numpy for the strand bias calculation
import numpy as np


class VCFProcessor:
    """End-to-end VCF file processing pipeline.

    Orchestrates parsing, quality filtering, annotation, and feature
    extraction to produce model-ready VariantFeatures.
    """

    def __init__(
        self,
        quality_thresholds: QualityThresholds | None = None,
    ) -> None:
        self._parser = VCFParser()
        self._annotator = VariantAnnotator()
        self._quality_filter = QualityFilter(quality_thresholds)
        self._stats = ProcessingStats()

    @property
    def stats(self) -> ProcessingStats:
        return self._stats

    @property
    def header(self) -> VCFHeader | None:
        return self._parser.header

    def process_file(self, path: str | Path) -> list[tuple[VCFVariant, VariantFeatures]]:
        """Process a VCF file and extract variant features.

        Args:
            path: Path to the VCF file.

        Returns:
            List of (variant, features) tuples for variants passing filters.
        """
        self._stats = ProcessingStats()
        results: list[tuple[VCFVariant, VariantFeatures]] = []

        for variant in self._parser.parse_file(path):
            result = self._process_single_variant(variant)
            if result is not None:
                results.append(result)

        logger.info(
            "Processed VCF file %s: %d total, %d passed filters",
            path,
            self._stats.total_variants,
            self._stats.passed_filters,
        )
        return results

    def process_string(self, vcf_content: str) -> list[tuple[VCFVariant, VariantFeatures]]:
        """Process VCF content from a string.

        Args:
            vcf_content: VCF file content as string.

        Returns:
            List of (variant, features) tuples for variants passing filters.
        """
        self._stats = ProcessingStats()
        results: list[tuple[VCFVariant, VariantFeatures]] = []

        for variant in self._parser.parse_string(vcf_content):
            result = self._process_single_variant(variant)
            if result is not None:
                results.append(result)

        logger.info(
            "Processed VCF string: %d total, %d passed filters",
            self._stats.total_variants,
            self._stats.passed_filters,
        )
        return results

    def _process_single_variant(
        self, variant: VCFVariant
    ) -> tuple[VCFVariant, VariantFeatures] | None:
        """Process a single variant through filtering and annotation."""
        self._stats.total_variants += 1

        # Track variant types
        vtype = variant.variant_type
        if vtype == VariantType.SNV:
            self._stats.snv_count += 1
        elif vtype == VariantType.INSERTION:
            self._stats.insertion_count += 1
        elif vtype == VariantType.DELETION:
            self._stats.deletion_count += 1
        elif vtype == VariantType.MNV:
            self._stats.mnv_count += 1
        else:
            self._stats.complex_count += 1

        if len(variant.alt) > 1:
            self._stats.multiallelic_count += 1

        # Quality filter
        passed, failed_filters = self._quality_filter.filter_variant(variant)
        if not passed:
            if any("QUAL" in f for f in failed_filters):
                self._stats.failed_quality += 1
            if any("DP" in f for f in failed_filters):
                self._stats.failed_depth += 1
            if any("GQ" in f for f in failed_filters):
                self._stats.failed_genotype_quality += 1
            if any("AB" in f for f in failed_filters):
                self._stats.failed_allele_balance += 1
            logger.debug("Variant %s failed filters: %s", variant.variant_key, failed_filters)
            return None

        self._stats.passed_filters += 1

        # Annotate
        annotations = self._annotator.annotate_variant(variant)

        # Convert to VariantFeatures
        features = self._build_features(variant, annotations)

        return variant, features

    def _build_features(self, variant: VCFVariant, annotations: dict[str, Any]) -> VariantFeatures:
        """Build VariantFeatures from a variant and its annotations."""
        vtype_map = {
            VariantType.SNV: "SNV",
            VariantType.INSERTION: "insertion",
            VariantType.DELETION: "deletion",
            VariantType.MNV: "MNV",
            VariantType.COMPLEX: "SNV",
        }

        return VariantFeatures(
            phylop_score=annotations.get("phylop_score", 0.0),
            phastcons_score=annotations.get("phastcons_score", 0.0),
            gerp_score=annotations.get("gerp_score", 0.0),
            gnomad_af=annotations.get("gnomad_af", 0.0),
            gnomad_af_afr=annotations.get("gnomad_af_afr", 0.0),
            gnomad_af_eas=annotations.get("gnomad_af_eas", 0.0),
            gnomad_af_nfe=annotations.get("gnomad_af_nfe", 0.0),
            gnomad_homozygote_count=int(annotations.get("gnomad_homozygote_count", 0)),
            sift_score=annotations.get("sift_score", 1.0),
            polyphen2_score=annotations.get("polyphen2_score", 0.0),
            cadd_phred=annotations.get("cadd_phred", 0.0),
            revel_score=annotations.get("revel_score", 0.0),
            mutation_taster_score=annotations.get("mutation_taster_score", 0.0),
            in_protein_domain=bool(annotations.get("in_protein_domain", False)),
            domain_conservation=annotations.get("domain_conservation", 0.0),
            distance_to_active_site=annotations.get("distance_to_active_site", -1.0),
            pfam_domain_count=int(annotations.get("pfam_domain_count", 0)),
            variant_type=vtype_map.get(variant.variant_type, "SNV"),
            consequence=annotations.get("consequence", "missense"),
            exon_number=int(annotations.get("exon_number", 0)),
            total_exons=int(annotations.get("total_exons", 0)),
            amino_acid_change_blosum62=annotations.get("amino_acid_change_blosum62", 0.0),
            grantham_distance=annotations.get("grantham_distance", 0.0),
            splice_ai_score=annotations.get("splice_ai_score", 0.0),
            max_splice_distance=int(annotations.get("max_splice_distance", 0)),
        )


def create_sample_vcf() -> str:
    """Generate a sample VCF file content for testing.

    Returns:
        Valid VCF 4.2 content string with example variants.
    """
    return """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS Mapping Quality">
##INFO=<ID=FS,Number=1,Type=Float,Description="Phred-scaled p-value using Fisher exact test for strand bias">
##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="gnomAD global allele frequency">
##INFO=<ID=SIFT_score,Number=1,Type=Float,Description="SIFT score">
##INFO=<ID=Polyphen2_HDIV_score,Number=1,Type=Float,Description="PolyPhen2 HDIV score">
##INFO=<ID=CADD_phred,Number=1,Type=Float,Description="CADD Phred score">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr17,length=83257441>
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1
chr1\t12345\trs1801133\tC\tT\t500\tPASS\tDP=100;AF=0.5;MQ=60;FS=1.2;gnomAD_AF=0.25;SIFT_score=0.8;Polyphen2_HDIV_score=0.1;CADD_phred=8.5\tGT:DP:GQ:AD\t0/1:100:99:50,50
chr17\t7675088\t.\tG\tA\t200\tPASS\tDP=50;AF=0.3;MQ=55;FS=2.0;SIFT_score=0.01;Polyphen2_HDIV_score=0.99;CADD_phred=35\tGT:DP:GQ:AD\t0/1:50:80:35,15
chr1\t55505599\trs6025\tC\tT\t80\tPASS\tDP=30;AF=0.4;MQ=50;FS=0.5;gnomAD_AF=0.02;SIFT_score=0.03;Polyphen2_HDIV_score=0.95;CADD_phred=28\tGT:DP:GQ:AD\t0/1:30:60:18,12
chr1\t100\t.\tA\tG\t10\tLowQual\tDP=5;AF=0.5;MQ=20;FS=50\tGT:DP:GQ:AD\t0/1:5:10:3,2
chr17\t43094464\t.\tAG\tA\t300\tPASS\tDP=80;AF=0.45;MQ=58;FS=1.0;CADD_phred=22\tGT:DP:GQ:AD\t0/1:80:95:44,36
"""
