"""Shared test fixtures for PharmAssistAI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pharm_assist.llm.guardrails import SafetyGuardrails
from pharm_assist.rag.document_processor import (
    ChunkStrategy,
    DocumentProcessor,
    ProcessedChunk,
)
from pharm_assist.rag.vector_store import CollectionName, VectorStore


@pytest.fixture
def sample_aspirin_text() -> str:
    """Return a snippet of the aspirin drug label for testing."""
    return """HIGHLIGHTS OF PRESCRIBING INFORMATION
ASPIRIN (acetylsalicylic acid) tablets, for oral use

INDICATIONS AND USAGE
Aspirin is a nonsteroidal anti-inflammatory drug indicated for:
- Temporary relief of headache, pain and fever
- Cardiovascular event risk reduction

DOSAGE AND ADMINISTRATION
- For analgesic/antipyretic use: 325 mg to 650 mg every 4 to 6 hours as needed.
- Do not exceed 4,000 mg in 24 hours.
- For cardiovascular indications: 75 mg to 325 mg once daily.

CONTRAINDICATIONS
- Known allergy to NSAIDs
- Patients with asthma, rhinitis, and nasal polyps
- Children or teenagers with viral infections (risk of Reye syndrome)

ADVERSE REACTIONS
Common (>=1%):
- Gastrointestinal: Dyspepsia, heartburn, nausea, vomiting
- Hematologic: Increased bleeding time
- CNS: Dizziness, headache, tinnitus

DRUG INTERACTIONS
Anticoagulants (e.g., warfarin): Aspirin can increase anticoagulant effect and risk of bleeding.
Methotrexate: Aspirin can inhibit renal clearance of methotrexate.
Other NSAIDs: Concurrent use increases the risk of GI bleeding.
"""


@pytest.fixture
def sample_metformin_text() -> str:
    """Return a snippet of the metformin drug label for testing."""
    return """HIGHLIGHTS OF PRESCRIBING INFORMATION
METFORMIN HYDROCHLORIDE tablets, for oral use

BOXED WARNING: LACTIC ACIDOSIS
Metformin-associated lactic acidosis has resulted in death. Risk factors include
renal impairment, age 65 years or greater, and excessive alcohol intake.

INDICATIONS AND USAGE
Metformin is a biguanide indicated to improve glycemic control in adults and
pediatric patients 10 years and older with type 2 diabetes mellitus.

DOSAGE AND ADMINISTRATION
- Adults: Starting dose 500 mg twice daily with meals. Maximum dose: 2550 mg/day.
- Pediatric: Starting dose 500 mg twice daily. Maximum dose: 2000 mg/day.

ADVERSE REACTIONS
Most common (>=5%): Diarrhea (53.2%), nausea/vomiting (25.5%), flatulence (12.1%).
"""


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """Create a DocumentProcessor with default settings."""
    return DocumentProcessor(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=256,
        chunk_overlap=32,
    )


@pytest.fixture
def processed_chunks(
    document_processor: DocumentProcessor, sample_aspirin_text: str
) -> list[ProcessedChunk]:
    """Process sample aspirin text into chunks."""
    return document_processor.process_text(sample_aspirin_text, source_file="aspirin.txt")


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """Create a VectorStore using a temporary directory."""
    return VectorStore(persist_directory=str(tmp_path / "chromadb"))


@pytest.fixture
def populated_vector_store(
    vector_store: VectorStore, processed_chunks: list[ProcessedChunk]
) -> VectorStore:
    """Create a VectorStore with sample data indexed."""
    vector_store.index_chunks(processed_chunks, CollectionName.DRUG_LABELS)
    return vector_store


@pytest.fixture
def guardrails() -> SafetyGuardrails:
    """Create a SafetyGuardrails instance."""
    return SafetyGuardrails(strict_mode=True, inject_disclaimer=True)


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client for testing without API calls."""
    client = MagicMock()

    # Mock messages.create
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response about aspirin.")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    client.messages.create.return_value = mock_response

    return client


@pytest.fixture
def sample_drug_label_file(tmp_path: Path, sample_aspirin_text: str) -> Path:
    """Write sample aspirin text to a temporary file."""
    file_path = tmp_path / "aspirin.txt"
    file_path.write_text(sample_aspirin_text, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_drug_labels_dir(
    tmp_path: Path, sample_aspirin_text: str, sample_metformin_text: str
) -> Path:
    """Create a temporary directory with multiple drug label files."""
    labels_dir = tmp_path / "drug_labels"
    labels_dir.mkdir()
    (labels_dir / "aspirin.txt").write_text(sample_aspirin_text, encoding="utf-8")
    (labels_dir / "metformin.txt").write_text(sample_metformin_text, encoding="utf-8")
    return labels_dir
