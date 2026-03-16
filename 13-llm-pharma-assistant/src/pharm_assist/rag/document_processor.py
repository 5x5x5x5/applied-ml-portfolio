"""Document ingestion pipeline for pharmaceutical documents.

Handles parsing of PDF drug labels, clinical guidelines, and FDA documents
with multiple chunking strategies, metadata extraction, and text normalization.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

logger = structlog.get_logger(__name__)


class ChunkStrategy(str, Enum):
    """Available chunking strategies for document processing."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SECTION_BASED = "section_based"


class DocumentType(str, Enum):
    """Pharmaceutical document types for metadata classification."""

    DRUG_LABEL = "drug_label"
    CLINICAL_GUIDELINE = "clinical_guideline"
    FDA_REGULATORY = "fda_regulatory"
    RESEARCH_PAPER = "research_paper"
    INTERACTION_DATABASE = "interaction_database"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a pharmaceutical document."""

    source_file: str
    document_type: DocumentType
    drug_name: str | None = None
    manufacturer: str | None = None
    section_type: str | None = None
    document_date: str | None = None
    ndc_code: str | None = None
    page_number: int | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    document_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a flat dictionary for ChromaDB storage."""
        result: dict[str, Any] = {
            "source_file": self.source_file,
            "document_type": self.document_type.value,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "document_hash": self.document_hash,
        }
        if self.drug_name:
            result["drug_name"] = self.drug_name
        if self.manufacturer:
            result["manufacturer"] = self.manufacturer
        if self.section_type:
            result["section_type"] = self.section_type
        if self.document_date:
            result["document_date"] = self.document_date
        if self.ndc_code:
            result["ndc_code"] = self.ndc_code
        if self.page_number is not None:
            result["page_number"] = self.page_number
        return result


@dataclass
class ProcessedChunk:
    """A processed text chunk with metadata, ready for embedding."""

    chunk_id: str
    text: str
    metadata: DocumentMetadata
    token_count: int
    parent_chunk_id: str | None = None


# Standard FDA drug label sections used for section-based chunking
FDA_LABEL_SECTIONS = [
    "HIGHLIGHTS OF PRESCRIBING INFORMATION",
    "BOXED WARNING",
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "DOSAGE FORMS AND STRENGTHS",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "DRUG INTERACTIONS",
    "USE IN SPECIFIC POPULATIONS",
    "OVERDOSAGE",
    "DESCRIPTION",
    "CLINICAL PHARMACOLOGY",
    "NONCLINICAL TOXICOLOGY",
    "CLINICAL STUDIES",
    "HOW SUPPLIED",
    "STORAGE AND HANDLING",
    "PATIENT COUNSELING INFORMATION",
    "MEDICATION GUIDE",
]


class TokenCounter:
    """Utility for counting tokens using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count tokens in the given text."""
        return len(self._encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])


class TextNormalizer:
    """Cleans and normalizes pharmaceutical document text."""

    # Common OCR artifacts and noise patterns in FDA documents
    _NOISE_PATTERNS = [
        (r"\x0c", "\n"),  # Form feeds
        (r"\r\n", "\n"),  # Windows line endings
        (r"[ \t]+\n", "\n"),  # Trailing whitespace
        (r"\n{4,}", "\n\n\n"),  # Excessive blank lines
        (r"(?<!\n)\n(?!\n)(?=[a-z])", " "),  # Spurious line breaks mid-sentence
        (r"[ \t]{2,}", " "),  # Multiple spaces/tabs
    ]

    @staticmethod
    def normalize(text: str) -> str:
        """Clean and normalize raw document text."""
        if not text or not text.strip():
            return ""

        result = text
        for pattern, replacement in TextNormalizer._NOISE_PATTERNS:
            result = re.sub(pattern, replacement, result)

        # Normalize unicode characters common in pharma docs
        replacements = {
            "\u2018": "'",
            "\u2019": "'",  # Smart single quotes
            "\u201c": '"',
            "\u201d": '"',  # Smart double quotes
            "\u2013": "-",
            "\u2014": "--",  # En-dash, Em-dash
            "\u2022": "*",  # Bullet
            "\u00b0": " degrees ",  # Degree symbol
            "\u00b5": "micro",  # Micro sign
            "\u2265": ">=",
            "\u2264": "<=",  # Comparison operators
        }
        for char, repl in replacements.items():
            result = result.replace(char, repl)

        return result.strip()


class MetadataExtractor:
    """Extracts pharmaceutical metadata from document text and filename."""

    # Common drug name patterns at the start of FDA labels
    _DRUG_NAME_PATTERNS = [
        r"(?i)^(?:HIGHLIGHTS\s+OF\s+PRESCRIBING\s+INFORMATION\s+)?(\w[\w\s-]{2,30}?)(?:\s*\()",
        r"(?i)^(\w[\w\s-]{2,30}?)\s*(?:tablets?|capsules?|injection|solution|cream|ointment)",
        r"(?i)(?:brand\s+name|drug\s+name|product\s+name)\s*:\s*(\w[\w\s-]{2,30})",
    ]

    _MANUFACTURER_PATTERNS = [
        r"(?i)(?:manufactured|marketed|distributed)\s+(?:by|for)\s*:?\s*(.+?)(?:\n|$)",
        r"(?i)(?:mfr|manufacturer)\s*:\s*(.+?)(?:\n|$)",
    ]

    _DATE_PATTERNS = [
        r"(?i)(?:revised|updated|effective|approved)\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(?i)(?:revised|updated|effective|approved)\s*:?\s*(\w+\s+\d{4})",
        r"(?i)(?:date)\s*:?\s*(\d{4}-\d{2}-\d{2})",
    ]

    _NDC_PATTERN = r"NDC\s*(?:Code\s*)?:?\s*(\d{4,5}-\d{3,4}-\d{1,2})"

    @classmethod
    def extract(cls, text: str, source_file: str) -> DocumentMetadata:
        """Extract metadata from document text and file path."""
        doc_type = cls._detect_document_type(text, source_file)
        drug_name = cls._extract_drug_name(text, source_file)
        manufacturer = cls._extract_pattern(text, cls._MANUFACTURER_PATTERNS)
        doc_date = cls._extract_pattern(text, cls._DATE_PATTERNS)
        ndc_match = re.search(cls._NDC_PATTERN, text)
        ndc_code = ndc_match.group(1) if ndc_match else None
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        return DocumentMetadata(
            source_file=source_file,
            document_type=doc_type,
            drug_name=drug_name,
            manufacturer=manufacturer,
            document_date=doc_date,
            ndc_code=ndc_code,
            document_hash=doc_hash,
        )

    @classmethod
    def _detect_document_type(cls, text: str, source_file: str) -> DocumentType:
        """Determine the document type from content and filename."""
        lower_text = text[:2000].lower()
        lower_file = source_file.lower()

        if "drug_label" in lower_file or "prescribing information" in lower_text:
            return DocumentType.DRUG_LABEL
        if "guideline" in lower_file or "clinical guideline" in lower_text:
            return DocumentType.CLINICAL_GUIDELINE
        if "fda" in lower_file or "federal register" in lower_text:
            return DocumentType.FDA_REGULATORY
        if "interaction" in lower_file:
            return DocumentType.INTERACTION_DATABASE
        if any(kw in lower_text for kw in ("abstract", "doi:", "pubmed")):
            return DocumentType.RESEARCH_PAPER
        # Fall back to drug label if we detect typical label sections
        if any(section.lower() in lower_text for section in FDA_LABEL_SECTIONS[:5]):
            return DocumentType.DRUG_LABEL
        return DocumentType.UNKNOWN

    @classmethod
    def _extract_drug_name(cls, text: str, source_file: str) -> str | None:
        """Extract drug name from text or infer from filename."""
        # Try filename first - most reliable for sample data
        filename = Path(source_file).stem
        if filename and filename.lower() not in ("document", "doc", "file", "unknown"):
            return filename.replace("_", " ").replace("-", " ").title()

        # Try patterns in text
        for pattern in cls._DRUG_NAME_PATTERNS:
            match = re.search(pattern, text[:1000])
            if match:
                name = match.group(1).strip()
                if 2 < len(name) < 40:
                    return name.title()
        return None

    @classmethod
    def _extract_pattern(cls, text: str, patterns: list[str]) -> str | None:
        """Try multiple regex patterns and return the first match."""
        for pattern in patterns:
            match = re.search(pattern, text[:5000])
            if match:
                return match.group(1).strip()
        return None


class DocumentProcessor:
    """Main document processing pipeline for pharmaceutical documents.

    Supports PDF parsing, multiple chunking strategies, metadata extraction,
    text normalization, and token counting for optimal chunk sizes.
    """

    def __init__(
        self,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_token_count: int = 1024,
    ) -> None:
        self._chunk_strategy = chunk_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_token_count = max_token_count
        self._token_counter = TokenCounter()
        self._normalizer = TextNormalizer()
        self._metadata_extractor = MetadataExtractor()

        logger.info(
            "document_processor.initialized",
            strategy=chunk_strategy.value,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

    def process_file(self, file_path: str | Path) -> list[ProcessedChunk]:
        """Process a single file and return chunks with metadata."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("document_processor.file_not_found", path=str(file_path))
            raise FileNotFoundError(f"Document not found: {file_path}")

        raw_text = self._read_file(file_path)
        if not raw_text:
            logger.warning("document_processor.empty_file", path=str(file_path))
            return []

        return self.process_text(raw_text, source_file=str(file_path))

    def process_text(self, text: str, source_file: str = "unknown") -> list[ProcessedChunk]:
        """Process raw text into chunks with metadata."""
        normalized = self._normalizer.normalize(text)
        if not normalized:
            return []

        base_metadata = self._metadata_extractor.extract(normalized, source_file)
        chunks_text = self._chunk_text(normalized)
        total_chunks = len(chunks_text)

        processed: list[ProcessedChunk] = []
        parent_id: str | None = None

        for idx, chunk_text in enumerate(chunks_text):
            section_type = self._detect_section(chunk_text)
            token_count = self._token_counter.count(chunk_text)

            # Enforce token limit
            if token_count > self._max_token_count:
                chunk_text = self._token_counter.truncate_to_tokens(
                    chunk_text, self._max_token_count
                )
                token_count = self._max_token_count

            chunk_metadata = DocumentMetadata(
                source_file=base_metadata.source_file,
                document_type=base_metadata.document_type,
                drug_name=base_metadata.drug_name,
                manufacturer=base_metadata.manufacturer,
                section_type=section_type,
                document_date=base_metadata.document_date,
                ndc_code=base_metadata.ndc_code,
                chunk_index=idx,
                total_chunks=total_chunks,
                document_hash=base_metadata.document_hash,
            )

            chunk_id = self._generate_chunk_id(source_file, idx)

            # For section-based chunking, first chunk in each section is a parent
            if self._chunk_strategy == ChunkStrategy.SECTION_BASED and section_type:
                parent_id = chunk_id

            processed.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                    token_count=token_count,
                    parent_chunk_id=parent_id if parent_id != chunk_id else None,
                )
            )

        logger.info(
            "document_processor.processed",
            source=source_file,
            chunks=total_chunks,
            drug_name=base_metadata.drug_name,
            doc_type=base_metadata.document_type.value,
        )
        return processed

    def process_directory(self, dir_path: str | Path) -> list[ProcessedChunk]:
        """Process all supported files in a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_chunks: list[ProcessedChunk] = []
        extensions = {".txt", ".pdf", ".md"}

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                try:
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
                except Exception:
                    logger.exception(
                        "document_processor.file_error",
                        path=str(file_path),
                    )

        logger.info(
            "document_processor.directory_processed",
            directory=str(dir_path),
            total_chunks=len(all_chunks),
        )
        return all_chunks

    def _read_file(self, file_path: Path) -> str:
        """Read text from a file, handling PDF and plain text."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(file_path)
        # Plain text, markdown, etc.
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _read_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            pages: list[str] = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Page {page_num}]\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            logger.error("document_processor.pypdf_not_installed")
            raise
        except Exception:
            logger.exception("document_processor.pdf_read_error", path=str(file_path))
            raise

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks using the configured strategy."""
        if self._chunk_strategy == ChunkStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(text)
        elif self._chunk_strategy == ChunkStrategy.RECURSIVE:
            return self._recursive_chunk(text)
        elif self._chunk_strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(text)
        elif self._chunk_strategy == ChunkStrategy.SECTION_BASED:
            return self._section_based_chunk(text)
        else:
            return self._recursive_chunk(text)

    def _fixed_size_chunk(self, text: str) -> list[str]:
        """Split text into fixed-size chunks with overlap using token counting."""
        splitter = TokenTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            encoding_name="cl100k_base",
        )
        return splitter.split_text(text)

    def _recursive_chunk(self, text: str) -> list[str]:
        """Recursively split text using hierarchical separators."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size * 4,  # Character-based, so multiply
            chunk_overlap=self._chunk_overlap * 4,
            separators=[
                "\n\n\n",  # Major section breaks
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence breaks
                "; ",  # Clause breaks
                ", ",  # Phrase breaks
                " ",  # Word breaks
            ],
            length_function=self._token_counter.count,
        )
        return splitter.split_text(text)

    def _semantic_chunk(self, text: str) -> list[str]:
        """Split text based on semantic boundaries (paragraphs, sections).

        Falls back to recursive chunking but respects section headers
        as hard boundaries that should never be split across chunks.
        """
        # Split at section-level boundaries first
        section_pattern = r"\n(?=[A-Z][A-Z\s]{3,}(?:\n|$))"
        sections = re.split(section_pattern, text)

        chunks: list[str] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size * 4,
            chunk_overlap=self._chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", " "],
            length_function=self._token_counter.count,
        )

        for section in sections:
            section = section.strip()
            if not section:
                continue
            token_count = self._token_counter.count(section)
            if token_count <= self._chunk_size:
                chunks.append(section)
            else:
                sub_chunks = splitter.split_text(section)
                chunks.extend(sub_chunks)

        return chunks if chunks else [text]

    def _section_based_chunk(self, text: str) -> list[str]:
        """Split text based on FDA drug label section headers."""
        # Build pattern that matches known section headings
        escaped = [re.escape(s) for s in FDA_LABEL_SECTIONS]
        pattern = r"(?i)\n\s*(?:\d+\.?\s*)?(" + "|".join(escaped) + r")\s*\n"

        parts = re.split(pattern, text)
        if len(parts) <= 1:
            # No section headers found, fall back to recursive
            return self._recursive_chunk(text)

        chunks: list[str] = []
        current_header = ""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size * 4,
            chunk_overlap=self._chunk_overlap * 4,
            length_function=self._token_counter.count,
        )

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Check if this part is a section header
            if part.upper() in [s.upper() for s in FDA_LABEL_SECTIONS]:
                current_header = part
                continue

            section_text = f"{current_header}\n\n{part}" if current_header else part
            token_count = self._token_counter.count(section_text)

            if token_count <= self._chunk_size:
                chunks.append(section_text)
            else:
                sub_chunks = splitter.split_text(section_text)
                chunks.extend(sub_chunks)

        return chunks if chunks else [text]

    @staticmethod
    def _detect_section(chunk_text: str) -> str | None:
        """Detect which FDA label section a chunk belongs to."""
        first_lines = chunk_text[:200].upper()
        for section in FDA_LABEL_SECTIONS:
            if section.upper() in first_lines:
                return section
        return None

    @staticmethod
    def _generate_chunk_id(source: str, index: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{source}::chunk::{index}"
        return hashlib.md5(raw.encode()).hexdigest()
