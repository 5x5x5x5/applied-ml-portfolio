"""Advanced retrieval strategies for pharmaceutical question answering.

Implements multi-query retrieval, parent document retrieval, contextual compression,
and source attribution for accurate citations in RAG answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from pharm_assist.rag.vector_store import CollectionName, SearchQuery, SearchResult, VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class Citation:
    """A source citation linking an answer segment to its source document."""

    citation_id: int
    source_file: str
    drug_name: str | None
    section_type: str | None
    chunk_text: str
    relevance_score: float
    collection: str

    def format_short(self) -> str:
        """Format citation as a short inline reference."""
        parts = [f"[{self.citation_id}]"]
        if self.drug_name:
            parts.append(self.drug_name)
        if self.section_type:
            parts.append(f"({self.section_type})")
        return " ".join(parts)

    def format_full(self) -> str:
        """Format citation as a full reference entry."""
        lines = [f"[{self.citation_id}] Source: {self.source_file}"]
        if self.drug_name:
            lines.append(f"    Drug: {self.drug_name}")
        if self.section_type:
            lines.append(f"    Section: {self.section_type}")
        lines.append(f"    Relevance: {self.relevance_score:.2f}")
        lines.append(f"    Collection: {self.collection}")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Complete retrieval result with context, citations, and metadata."""

    query: str
    context_text: str
    citations: list[Citation]
    search_results: list[SearchResult]
    retrieval_strategy: str
    total_tokens_estimated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        """Check if any relevant results were found."""
        return len(self.search_results) > 0

    def format_context_with_citations(self) -> str:
        """Format the retrieval context with inline citation markers."""
        if not self.search_results:
            return "No relevant information found in the knowledge base."

        sections: list[str] = []
        for i, result in enumerate(self.search_results, start=1):
            header = f"--- Source [{i}]"
            if result.drug_name:
                header += f" | {result.drug_name}"
            if result.section_type:
                header += f" | {result.section_type}"
            header += f" (relevance: {result.relevance_score:.2f}) ---"

            sections.append(f"{header}\n{result.text}")

        return "\n\n".join(sections)

    def format_citations_block(self) -> str:
        """Format all citations as a reference block."""
        if not self.citations:
            return ""
        entries = [c.format_full() for c in self.citations]
        return "\n\nSources:\n" + "\n\n".join(entries)


class MultiQueryGenerator:
    """Generates multiple query variations to improve retrieval recall.

    Uses the Anthropic API to rephrase a user question into several alternative
    formulations that may match different relevant documents.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client

    async def generate_queries(self, original_query: str, n_queries: int = 3) -> list[str]:
        """Generate alternative query formulations."""
        queries = [original_query]

        # If no API client, use rule-based query expansion
        if self._client is None:
            return self._rule_based_expansion(original_query, n_queries)

        try:
            prompt = (
                f"You are a pharmaceutical information specialist. Given the following question, "
                f"generate {n_queries} alternative phrasings that could help retrieve relevant "
                f"information from a pharmaceutical knowledge base. Each alternative should "
                f"approach the question from a different angle (e.g., using medical terminology, "
                f"focusing on different aspects, being more specific or more general).\n\n"
                f"Original question: {original_query}\n\n"
                f"Provide exactly {n_queries} alternative phrasings, one per line. "
                f"Do not number them or add any other text."
            )

            message = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text
            alternatives = [
                line.strip()
                for line in response_text.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            queries.extend(alternatives[:n_queries])

        except Exception:
            logger.exception("multi_query.generation_failed")
            queries.extend(self._rule_based_expansion(original_query, n_queries)[1:])

        logger.info("multi_query.generated", original=original_query, total=len(queries))
        return queries

    @staticmethod
    def _rule_based_expansion(query: str, n_queries: int) -> list[str]:
        """Generate query variations using rule-based transformations."""
        queries = [query]
        lower_query = query.lower()

        # Add medical terminology variant
        medical_terms = {
            "side effects": "adverse reactions",
            "adverse reactions": "side effects",
            "dosage": "dosing regimen and administration",
            "dose": "dosage and administration",
            "interactions": "drug-drug interactions and contraindications",
            "pregnancy": "use in specific populations pregnancy",
            "children": "pediatric use",
            "elderly": "geriatric use",
            "liver": "hepatic impairment",
            "kidney": "renal impairment",
            "allergic": "hypersensitivity reactions",
            "overdose": "overdosage and toxicity",
            "storage": "storage and handling conditions",
            "how it works": "mechanism of action clinical pharmacology",
            "what is it for": "indications and usage",
        }

        for term, replacement in medical_terms.items():
            if term in lower_query:
                queries.append(query.lower().replace(term, replacement))
                break

        # Add a more specific variant
        if "?" in query:
            queries.append(query.replace("?", "") + " prescribing information")

        # Add a broader variant
        drug_pattern = re.search(r"\b([A-Z][a-z]+(?:in|ol|ide|ine|ate|one))\b", query)
        if drug_pattern:
            drug = drug_pattern.group(1)
            queries.append(f"{drug} FDA drug label prescribing information")

        return queries[: n_queries + 1]


class ContextualCompressor:
    """Compresses retrieved chunks to extract only the most relevant portions.

    Uses the Anthropic API to identify and extract the specific sentences or
    paragraphs within each chunk that directly answer the user's question.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client

    async def compress(
        self,
        query: str,
        results: list[SearchResult],
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Compress search results to keep only relevant portions."""
        if not results or self._client is None:
            return results[:max_results]

        compressed: list[SearchResult] = []

        for result in results[:max_results]:
            try:
                prompt = (
                    f"Given the following pharmaceutical document excerpt and a user's question, "
                    f"extract ONLY the sentences that are directly relevant to answering the question. "
                    f"If nothing is relevant, respond with 'NOT_RELEVANT'.\n\n"
                    f"Question: {query}\n\n"
                    f"Document excerpt:\n{result.text}\n\n"
                    f"Relevant sentences:"
                )

                message = self._client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

                extracted = message.content[0].text.strip()

                if extracted and extracted != "NOT_RELEVANT":
                    compressed.append(
                        SearchResult(
                            chunk_id=result.chunk_id,
                            text=extracted,
                            metadata=result.metadata,
                            relevance_score=result.relevance_score,
                            collection=result.collection,
                        )
                    )
                else:
                    logger.debug(
                        "compressor.filtered_irrelevant",
                        chunk_id=result.chunk_id,
                    )

            except Exception:
                logger.exception("compressor.error", chunk_id=result.chunk_id)
                compressed.append(result)  # Keep original on error

        return compressed


class PharmaceuticalRetriever:
    """Advanced retrieval orchestrator for pharmaceutical questions.

    Combines multi-query retrieval, parent document retrieval, contextual
    compression, and source attribution into a single pipeline.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        anthropic_client: anthropic.Anthropic | None = None,
        default_n_results: int = 5,
        use_multi_query: bool = True,
        use_compression: bool = False,
        use_mmr: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._client = anthropic_client
        self._default_n_results = default_n_results
        self._use_multi_query = use_multi_query
        self._use_compression = use_compression
        self._use_mmr = use_mmr
        self._multi_query_gen = MultiQueryGenerator(anthropic_client)
        self._compressor = ContextualCompressor(anthropic_client)

        logger.info(
            "retriever.initialized",
            multi_query=use_multi_query,
            compression=use_compression,
            mmr=use_mmr,
        )

    async def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        drug_name: str | None = None,
        section_type: str | None = None,
        collection_names: list[CollectionName] | None = None,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline for a pharmaceutical question.

        Steps:
        1. Generate multiple query variations (if enabled)
        2. Search the vector store with each query
        3. Deduplicate and merge results
        4. Apply contextual compression (if enabled)
        5. Build citations and format context
        """
        effective_n = n_results or self._default_n_results
        strategy_parts: list[str] = []

        # Step 1: Multi-query generation
        queries: list[str]
        if self._use_multi_query:
            queries = await self._multi_query_gen.generate_queries(query)
            strategy_parts.append("multi_query")
        else:
            queries = [query]

        # Step 2: Search with each query
        all_results: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()

        for q in queries:
            search_query = SearchQuery(
                query_text=q,
                collection_names=collection_names,
                n_results=effective_n,
                drug_name=drug_name,
                section_type=section_type,
                use_mmr=self._use_mmr,
                use_hybrid=True,
            )
            results = self._vector_store.search(search_query)

            for result in results:
                if result.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(result.chunk_id)
                    all_results.append(result)

        if self._use_mmr:
            strategy_parts.append("mmr")

        strategy_parts.append("hybrid")

        # Step 3: Sort merged results by relevance
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Step 4: Parent document retrieval - try to get full sections
        expanded = await self._expand_parent_context(all_results[:effective_n])
        if expanded:
            strategy_parts.append("parent_doc")

        # Step 5: Contextual compression
        if self._use_compression and self._client:
            all_results = await self._compressor.compress(query, all_results, effective_n)
            strategy_parts.append("compression")

        # Trim to final count
        final_results = all_results[:effective_n]

        # Step 6: Build citations
        citations = self._build_citations(final_results)

        # Step 7: Format context
        retrieval_result = RetrievalResult(
            query=query,
            context_text="",
            citations=citations,
            search_results=final_results,
            retrieval_strategy="+".join(strategy_parts),
            total_tokens_estimated=sum(len(r.text.split()) for r in final_results),
        )
        retrieval_result.context_text = retrieval_result.format_context_with_citations()

        logger.info(
            "retriever.complete",
            query_preview=query[:80],
            results=len(final_results),
            strategy=retrieval_result.retrieval_strategy,
        )
        return retrieval_result

    async def _expand_parent_context(self, results: list[SearchResult]) -> list[SearchResult]:
        """Attempt to expand chunks to their parent document section.

        If a chunk has a parent_chunk_id in metadata, retrieves the parent chunk
        and replaces the child with the fuller context.
        """
        expanded: list[SearchResult] = []
        for result in results:
            parent_id = result.metadata.get("parent_chunk_id")
            if parent_id:
                # Try to retrieve parent chunk from the same collection
                try:
                    collection = self._vector_store.get_or_create_collection(result.collection)
                    parent_data = collection.get(
                        ids=[parent_id], include=["documents", "metadatas"]
                    )
                    if parent_data and parent_data["documents"]:
                        parent_text = parent_data["documents"][0]
                        expanded.append(
                            SearchResult(
                                chunk_id=parent_id,
                                text=parent_text,
                                metadata=parent_data["metadatas"][0]
                                if parent_data["metadatas"]
                                else result.metadata,
                                relevance_score=result.relevance_score,
                                collection=result.collection,
                            )
                        )
                        continue
                except Exception:
                    logger.debug("retriever.parent_lookup_failed", parent_id=parent_id)

            expanded.append(result)
        return expanded

    @staticmethod
    def _build_citations(results: list[SearchResult]) -> list[Citation]:
        """Build citation objects from search results."""
        citations: list[Citation] = []
        for i, result in enumerate(results, start=1):
            citations.append(
                Citation(
                    citation_id=i,
                    source_file=result.source_file,
                    drug_name=result.drug_name,
                    section_type=result.section_type,
                    chunk_text=result.text[:300],
                    relevance_score=result.relevance_score,
                    collection=result.collection,
                )
            )
        return citations

    async def retrieve_for_drug(
        self,
        drug_name: str,
        question: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """Convenience method: retrieve information about a specific drug."""
        return await self.retrieve(
            query=question,
            n_results=n_results,
            drug_name=drug_name,
            collection_names=[CollectionName.DRUG_LABELS],
        )

    async def retrieve_interactions(
        self,
        drug_a: str,
        drug_b: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """Retrieve drug interaction information between two drugs."""
        query = (
            f"Drug interactions between {drug_a} and {drug_b}. "
            f"What are the potential adverse effects, contraindications, and "
            f"dosage adjustments when {drug_a} is taken together with {drug_b}?"
        )
        return await self.retrieve(
            query=query,
            n_results=n_results,
            collection_names=[
                CollectionName.DRUG_LABELS,
                CollectionName.INTERACTIONS,
            ],
        )

    async def retrieve_guidelines(
        self,
        condition: str,
        n_results: int = 5,
    ) -> RetrievalResult:
        """Retrieve clinical guideline information for a condition."""
        query = (
            f"Clinical guidelines and treatment recommendations for {condition}. "
            f"What are the evidence-based pharmacotherapy options?"
        )
        return await self.retrieve(
            query=query,
            n_results=n_results,
            collection_names=[CollectionName.CLINICAL_GUIDELINES],
        )
