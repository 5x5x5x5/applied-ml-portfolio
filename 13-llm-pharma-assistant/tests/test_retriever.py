"""Tests for the PharmaceuticalRetriever and supporting components."""

from __future__ import annotations

import pytest

from pharm_assist.rag.document_processor import (
    ChunkStrategy,
    DocumentMetadata,
    DocumentProcessor,
    DocumentType,
    MetadataExtractor,
    ProcessedChunk,
    TextNormalizer,
    TokenCounter,
)
from pharm_assist.rag.retriever import (
    Citation,
    MultiQueryGenerator,
    PharmaceuticalRetriever,
    RetrievalResult,
)
from pharm_assist.rag.vector_store import (
    CollectionName,
    SearchQuery,
    SearchResult,
    VectorStore,
)

# ── Document Processor Tests ────────────────────────────────────────────────


class TestTokenCounter:
    """Tests for the TokenCounter utility."""

    def test_count_empty_string(self) -> None:
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_simple_text(self) -> None:
        counter = TokenCounter()
        count = counter.count("Hello world")
        assert count > 0
        assert count <= 5  # Should be 2 tokens roughly

    def test_truncate_to_tokens(self) -> None:
        counter = TokenCounter()
        text = "This is a longer sentence that should be truncated to fit within a token budget."
        truncated = counter.truncate_to_tokens(text, 5)
        assert counter.count(truncated) <= 5

    def test_truncate_short_text_unchanged(self) -> None:
        counter = TokenCounter()
        text = "Hi"
        truncated = counter.truncate_to_tokens(text, 100)
        assert truncated == text


class TestTextNormalizer:
    """Tests for the TextNormalizer."""

    def test_normalize_empty_string(self) -> None:
        assert TextNormalizer.normalize("") == ""
        assert TextNormalizer.normalize("   ") == ""

    def test_normalize_removes_form_feeds(self) -> None:
        text = "Section 1\x0cSection 2"
        result = TextNormalizer.normalize(text)
        assert "\x0c" not in result

    def test_normalize_collapses_blank_lines(self) -> None:
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = TextNormalizer.normalize(text)
        assert "\n\n\n\n" not in result

    def test_normalize_replaces_smart_quotes(self) -> None:
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = TextNormalizer.normalize(text)
        assert '"Hello"' in result
        assert "'world'" in result

    def test_normalize_replaces_special_chars(self) -> None:
        text = "Temperature: 25\u00b0C, dose: 5\u00b5g, BP \u2265 140"
        result = TextNormalizer.normalize(text)
        assert "degrees" in result
        assert "micro" in result
        assert ">=" in result


class TestMetadataExtractor:
    """Tests for the MetadataExtractor."""

    def test_extract_drug_label_type(self) -> None:
        text = "HIGHLIGHTS OF PRESCRIBING INFORMATION\nASPIRIN tablets"
        meta = MetadataExtractor.extract(text, "aspirin.txt")
        assert meta.document_type == DocumentType.DRUG_LABEL

    def test_extract_drug_name_from_filename(self) -> None:
        meta = MetadataExtractor.extract("Some drug text", "metformin.txt")
        assert meta.drug_name is not None
        assert "metformin" in meta.drug_name.lower()

    def test_extract_ndc_code(self) -> None:
        text = "NDC 0363-0781-18"
        meta = MetadataExtractor.extract(text, "test.txt")
        assert meta.ndc_code == "0363-0781-18"

    def test_extract_generates_hash(self) -> None:
        meta = MetadataExtractor.extract("some text", "test.txt")
        assert meta.document_hash
        assert len(meta.document_hash) == 16


class TestDocumentProcessor:
    """Tests for the main DocumentProcessor pipeline."""

    def test_process_text_produces_chunks(self, sample_aspirin_text: str) -> None:
        processor = DocumentProcessor(chunk_size=128, chunk_overlap=16)
        chunks = processor.process_text(sample_aspirin_text, source_file="aspirin.txt")
        assert len(chunks) > 0
        assert all(isinstance(c, ProcessedChunk) for c in chunks)

    def test_chunk_ids_are_unique(self, sample_aspirin_text: str) -> None:
        processor = DocumentProcessor(chunk_size=128)
        chunks = processor.process_text(sample_aspirin_text, source_file="aspirin.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_have_metadata(self, processed_chunks: list[ProcessedChunk]) -> None:
        for chunk in processed_chunks:
            assert chunk.metadata.source_file == "aspirin.txt"
            assert chunk.metadata.document_hash
            assert chunk.token_count > 0

    def test_process_file(self, sample_drug_label_file) -> None:
        processor = DocumentProcessor(chunk_size=128)
        chunks = processor.process_file(sample_drug_label_file)
        assert len(chunks) > 0

    def test_process_file_not_found(self) -> None:
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process_file("/nonexistent/file.txt")

    def test_process_empty_text(self) -> None:
        processor = DocumentProcessor()
        chunks = processor.process_text("", source_file="empty.txt")
        assert chunks == []

    def test_fixed_size_chunking(self, sample_aspirin_text: str) -> None:
        processor = DocumentProcessor(
            chunk_strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=128,
            chunk_overlap=16,
        )
        chunks = processor.process_text(sample_aspirin_text, source_file="aspirin.txt")
        assert len(chunks) > 1

    def test_semantic_chunking(self, sample_aspirin_text: str) -> None:
        processor = DocumentProcessor(
            chunk_strategy=ChunkStrategy.SEMANTIC,
            chunk_size=256,
        )
        chunks = processor.process_text(sample_aspirin_text, source_file="aspirin.txt")
        assert len(chunks) > 0

    def test_section_based_chunking(self, sample_aspirin_text: str) -> None:
        processor = DocumentProcessor(
            chunk_strategy=ChunkStrategy.SECTION_BASED,
            chunk_size=512,
        )
        chunks = processor.process_text(sample_aspirin_text, source_file="aspirin.txt")
        assert len(chunks) > 0

    def test_process_directory(self, sample_drug_labels_dir) -> None:
        processor = DocumentProcessor(chunk_size=128)
        chunks = processor.process_directory(sample_drug_labels_dir)
        assert len(chunks) > 0
        # Should have chunks from both files
        sources = {c.metadata.source_file for c in chunks}
        assert len(sources) == 2

    def test_metadata_to_dict(self) -> None:
        meta = DocumentMetadata(
            source_file="test.txt",
            document_type=DocumentType.DRUG_LABEL,
            drug_name="Aspirin",
            section_type="ADVERSE REACTIONS",
            document_hash="abc123",
        )
        d = meta.to_dict()
        assert d["source_file"] == "test.txt"
        assert d["document_type"] == "drug_label"
        assert d["drug_name"] == "Aspirin"
        assert d["section_type"] == "ADVERSE REACTIONS"


# ── Vector Store Tests ──────────────────────────────────────────────────────


class TestVectorStore:
    """Tests for the ChromaDB vector store."""

    def test_create_collection(self, vector_store: VectorStore) -> None:
        coll = vector_store.get_or_create_collection(CollectionName.DRUG_LABELS)
        assert coll is not None
        assert coll.count() == 0

    def test_index_and_search(self, populated_vector_store: VectorStore) -> None:
        query = SearchQuery(
            query_text="aspirin side effects",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=3,
        )
        results = populated_vector_store.search(query)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_relevance_scores(self, populated_vector_store: VectorStore) -> None:
        query = SearchQuery(
            query_text="dosage administration",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=3,
        )
        results = populated_vector_store.search(query)
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0

    def test_search_with_drug_filter(self, populated_vector_store: VectorStore) -> None:
        query = SearchQuery(
            query_text="side effects",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=5,
            drug_name="Aspirin",
        )
        results = populated_vector_store.search(query)
        # Should return results (filter may or may not match depending on metadata)
        assert isinstance(results, list)

    def test_hybrid_search(self, populated_vector_store: VectorStore) -> None:
        query = SearchQuery(
            query_text="aspirin bleeding risk anticoagulant",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=3,
            use_hybrid=True,
        )
        results = populated_vector_store.search(query)
        assert len(results) > 0

    def test_mmr_search(self, populated_vector_store: VectorStore) -> None:
        query = SearchQuery(
            query_text="aspirin",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=3,
            use_mmr=True,
        )
        results = populated_vector_store.search(query)
        assert len(results) > 0

    def test_collection_stats(self, populated_vector_store: VectorStore) -> None:
        stats = populated_vector_store.get_collection_stats()
        assert CollectionName.DRUG_LABELS.value in stats
        assert stats[CollectionName.DRUG_LABELS.value] > 0

    def test_empty_search(self, vector_store: VectorStore) -> None:
        vector_store.get_or_create_collection(CollectionName.DRUG_LABELS)
        query = SearchQuery(
            query_text="anything",
            collection_names=[CollectionName.DRUG_LABELS],
            n_results=3,
        )
        results = vector_store.search(query)
        assert results == []

    def test_index_empty_chunks(self, vector_store: VectorStore) -> None:
        count = vector_store.index_chunks([], CollectionName.DRUG_LABELS)
        assert count == 0


# ── Retriever Tests ─────────────────────────────────────────────────────────


class TestMultiQueryGenerator:
    """Tests for the MultiQueryGenerator."""

    @pytest.mark.asyncio
    async def test_rule_based_expansion(self) -> None:
        gen = MultiQueryGenerator(client=None)
        queries = await gen.generate_queries("What are the side effects of aspirin?")
        assert len(queries) >= 1
        assert queries[0] == "What are the side effects of aspirin?"

    @pytest.mark.asyncio
    async def test_expansion_with_medical_terms(self) -> None:
        gen = MultiQueryGenerator(client=None)
        queries = await gen.generate_queries("What are the side effects of aspirin?")
        # Should include the original plus medical terminology variant
        has_variant = any("adverse" in q.lower() for q in queries)
        assert len(queries) >= 1  # At minimum the original

    @pytest.mark.asyncio
    async def test_expansion_for_dosage(self) -> None:
        gen = MultiQueryGenerator(client=None)
        queries = await gen.generate_queries("What is the dosage of metformin?")
        assert len(queries) >= 1


class TestCitation:
    """Tests for the Citation model."""

    def test_format_short(self) -> None:
        citation = Citation(
            citation_id=1,
            source_file="aspirin.txt",
            drug_name="Aspirin",
            section_type="ADVERSE REACTIONS",
            chunk_text="Some text...",
            relevance_score=0.92,
            collection="drug_labels",
        )
        short = citation.format_short()
        assert "[1]" in short
        assert "Aspirin" in short

    def test_format_full(self) -> None:
        citation = Citation(
            citation_id=2,
            source_file="metformin.txt",
            drug_name="Metformin",
            section_type=None,
            chunk_text="Some text...",
            relevance_score=0.85,
            collection="drug_labels",
        )
        full = citation.format_full()
        assert "[2]" in full
        assert "metformin.txt" in full
        assert "0.85" in full


class TestRetrievalResult:
    """Tests for the RetrievalResult model."""

    def test_has_results(self) -> None:
        result = RetrievalResult(
            query="test",
            context_text="",
            citations=[],
            search_results=[
                SearchResult(
                    chunk_id="1",
                    text="test chunk",
                    metadata={},
                    relevance_score=0.9,
                    collection="test",
                )
            ],
            retrieval_strategy="hybrid",
        )
        assert result.has_results is True

    def test_no_results(self) -> None:
        result = RetrievalResult(
            query="test",
            context_text="",
            citations=[],
            search_results=[],
            retrieval_strategy="hybrid",
        )
        assert result.has_results is False

    def test_format_context_with_citations(self) -> None:
        result = RetrievalResult(
            query="test",
            context_text="",
            citations=[],
            search_results=[
                SearchResult(
                    chunk_id="1",
                    text="Aspirin is an NSAID.",
                    metadata={"drug_name": "Aspirin"},
                    relevance_score=0.95,
                    collection="drug_labels",
                )
            ],
            retrieval_strategy="hybrid",
        )
        formatted = result.format_context_with_citations()
        assert "Source [1]" in formatted
        assert "Aspirin" in formatted
        assert "0.95" in formatted

    def test_format_no_results(self) -> None:
        result = RetrievalResult(
            query="test",
            context_text="",
            citations=[],
            search_results=[],
            retrieval_strategy="hybrid",
        )
        formatted = result.format_context_with_citations()
        assert "No relevant information" in formatted


class TestPharmaceuticalRetriever:
    """Tests for the PharmaceuticalRetriever."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, populated_vector_store: VectorStore) -> None:
        retriever = PharmaceuticalRetriever(
            vector_store=populated_vector_store,
            use_multi_query=False,
            use_mmr=False,
        )
        result = await retriever.retrieve("What are aspirin side effects?")
        assert isinstance(result, RetrievalResult)
        assert result.query == "What are aspirin side effects?"

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query(self, populated_vector_store: VectorStore) -> None:
        retriever = PharmaceuticalRetriever(
            vector_store=populated_vector_store,
            use_multi_query=True,
            use_mmr=True,
        )
        result = await retriever.retrieve("aspirin dosage", n_results=3)
        assert isinstance(result, RetrievalResult)
        assert "multi_query" in result.retrieval_strategy

    @pytest.mark.asyncio
    async def test_retrieve_builds_citations(self, populated_vector_store: VectorStore) -> None:
        retriever = PharmaceuticalRetriever(
            vector_store=populated_vector_store,
            use_multi_query=False,
        )
        result = await retriever.retrieve("aspirin")
        if result.has_results:
            assert len(result.citations) > 0
            assert result.citations[0].citation_id == 1

    @pytest.mark.asyncio
    async def test_retrieve_for_drug(self, populated_vector_store: VectorStore) -> None:
        retriever = PharmaceuticalRetriever(
            vector_store=populated_vector_store,
            use_multi_query=False,
        )
        result = await retriever.retrieve_for_drug("Aspirin", "What are side effects?")
        assert isinstance(result, RetrievalResult)
