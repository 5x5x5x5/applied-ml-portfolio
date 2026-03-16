"""ChromaDB vector store for pharmaceutical document indexing and retrieval.

Provides document indexing with metadata filtering, hybrid search (semantic + keyword),
collection management, re-ranking, and MMR for result diversity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from pharm_assist.rag.document_processor import ProcessedChunk

logger = structlog.get_logger(__name__)


class CollectionName(str, Enum):
    """Pre-defined collection names for different document types."""

    DRUG_LABELS = "drug_labels"
    CLINICAL_GUIDELINES = "clinical_guidelines"
    REGULATORY = "regulatory_documents"
    INTERACTIONS = "drug_interactions"
    GENERAL = "general_pharma"


@dataclass
class SearchResult:
    """A single search result with text, metadata, and relevance score."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    relevance_score: float
    collection: str

    @property
    def drug_name(self) -> str | None:
        """Convenience accessor for drug name metadata."""
        return self.metadata.get("drug_name")

    @property
    def section_type(self) -> str | None:
        """Convenience accessor for section type metadata."""
        return self.metadata.get("section_type")

    @property
    def source_file(self) -> str:
        """Convenience accessor for source file metadata."""
        return self.metadata.get("source_file", "unknown")


@dataclass
class SearchQuery:
    """Parameters for a vector store search."""

    query_text: str
    collection_names: list[CollectionName] | None = None
    n_results: int = 5
    where_filter: dict[str, Any] | None = None
    drug_name: str | None = None
    section_type: str | None = None
    min_relevance: float = 0.0
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    use_hybrid: bool = True


class VectorStore:
    """ChromaDB-backed vector store for pharmaceutical documents.

    Supports multiple collections, metadata filtering, hybrid search,
    MMR diversity re-ranking, and cross-collection queries.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        if host:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("vector_store.connected_remote", host=host, port=port)
        else:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("vector_store.initialized_local", path=persist_directory)

        self._collections: dict[str, chromadb.Collection] = {}

    def get_or_create_collection(self, name: CollectionName | str) -> chromadb.Collection:
        """Get or create a named collection."""
        collection_name = name.value if isinstance(name, CollectionName) else name
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "vector_store.collection_ready",
                name=collection_name,
                count=self._collections[collection_name].count(),
            )
        return self._collections[collection_name]

    def index_chunks(
        self,
        chunks: list[ProcessedChunk],
        collection_name: CollectionName | str = CollectionName.GENERAL,
    ) -> int:
        """Index processed document chunks into the vector store.

        Returns the number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata.to_dict())

        # ChromaDB handles batching internally, but we split large batches
        batch_size = 500
        total_indexed = 0

        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            try:
                collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
                total_indexed += end - start
            except Exception:
                logger.exception(
                    "vector_store.index_error",
                    batch_start=start,
                    batch_end=end,
                )

        logger.info(
            "vector_store.indexed",
            collection=collection_name
            if isinstance(collection_name, str)
            else collection_name.value,
            chunks=total_indexed,
            total_in_collection=collection.count(),
        )
        return total_indexed

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute a search across one or more collections.

        Supports metadata filtering, hybrid search, and MMR re-ranking.
        """
        collections_to_search = self._resolve_collections(query.collection_names)

        if not collections_to_search:
            logger.warning("vector_store.no_collections")
            return []

        all_results: list[SearchResult] = []

        for coll_name, collection in collections_to_search.items():
            if collection.count() == 0:
                continue

            where_filter = self._build_where_filter(query)
            n_fetch = query.n_results * 3 if query.use_mmr else query.n_results

            try:
                if query.use_hybrid:
                    results = self._hybrid_search(
                        collection, query.query_text, n_fetch, where_filter
                    )
                else:
                    results = self._semantic_search(
                        collection, query.query_text, n_fetch, where_filter
                    )

                for r in results:
                    r.collection = coll_name
                all_results.extend(results)

            except Exception:
                logger.exception("vector_store.search_error", collection=coll_name)

        # Filter by minimum relevance
        if query.min_relevance > 0:
            all_results = [r for r in all_results if r.relevance_score >= query.min_relevance]

        # Sort by relevance
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Apply MMR for diversity if requested
        if query.use_mmr and len(all_results) > query.n_results:
            all_results = self._apply_mmr(all_results, query.n_results, query.mmr_lambda)

        # Trim to requested count
        all_results = all_results[: query.n_results]

        logger.info(
            "vector_store.search_complete",
            query_preview=query.query_text[:80],
            results=len(all_results),
        )
        return all_results

    def _semantic_search(
        self,
        collection: chromadb.Collection,
        query_text: str,
        n_results: int,
        where_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Pure semantic (embedding-based) search."""
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, collection.count()),
        }
        if where_filter:
            kwargs["where"] = where_filter

        raw = collection.query(**kwargs)
        return self._parse_chroma_results(raw)

    def _hybrid_search(
        self,
        collection: chromadb.Collection,
        query_text: str,
        n_results: int,
        where_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Hybrid search combining semantic similarity with keyword matching.

        Performs semantic search, then boosts results that contain query keywords.
        """
        # Semantic search first
        semantic_results = self._semantic_search(collection, query_text, n_results, where_filter)

        # Extract keywords for boosting
        keywords = self._extract_keywords(query_text)
        if not keywords:
            return semantic_results

        # Boost scores for keyword matches
        for result in semantic_results:
            text_lower = result.text.lower()
            keyword_hits = sum(1 for kw in keywords if kw in text_lower)
            if keyword_hits > 0:
                boost = min(0.15, keyword_hits * 0.05)
                result.relevance_score = min(1.0, result.relevance_score + boost)

        # Re-sort after boosting
        semantic_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return semantic_results

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from query text for hybrid search."""
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "between",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "how",
            "when",
            "where",
            "why",
            "me",
            "my",
            "i",
            "we",
            "our",
            "you",
            "your",
            "he",
            "she",
            "they",
            "them",
            "their",
        }
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [w for w in words if w not in stop_words]

    def _apply_mmr(
        self,
        results: list[SearchResult],
        k: int,
        lambda_param: float,
    ) -> list[SearchResult]:
        """Apply Maximal Marginal Relevance to diversify results.

        Balances relevance to query with diversity among selected results.
        lambda_param controls the trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
        """
        if len(results) <= k:
            return results

        selected: list[SearchResult] = [results[0]]
        candidates = list(results[1:])

        while len(selected) < k and candidates:
            best_score = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(candidates):
                # Relevance component
                relevance = candidate.relevance_score

                # Diversity component: max similarity to already-selected results
                max_sim = max(self._text_similarity(candidate.text, sel.text) for sel in selected)

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts as a proxy for diversity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    def _parse_chroma_results(raw: dict[str, Any]) -> list[SearchResult]:
        """Parse ChromaDB query results into SearchResult objects."""
        results: list[SearchResult] = []

        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        documents = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [0.0] * len(ids)

        for i, chunk_id in enumerate(ids):
            # ChromaDB returns distances; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            distance = distances[i] if i < len(distances) else 0.0
            relevance = max(0.0, 1.0 - distance)

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    relevance_score=relevance,
                    collection="",
                )
            )

        return results

    def _resolve_collections(
        self, names: list[CollectionName] | None
    ) -> dict[str, chromadb.Collection]:
        """Resolve collection names to ChromaDB collection objects."""
        if names is None:
            # Search all existing collections
            existing = self._client.list_collections()
            result = {}
            for coll in existing:
                coll_name = coll.name if hasattr(coll, "name") else str(coll)
                result[coll_name] = self.get_or_create_collection(coll_name)
            return result

        return {name.value: self.get_or_create_collection(name) for name in names}

    @staticmethod
    def _build_where_filter(query: SearchQuery) -> dict[str, Any] | None:
        """Build a ChromaDB where filter from search query parameters."""
        conditions: list[dict[str, Any]] = []

        if query.drug_name:
            conditions.append({"drug_name": {"$eq": query.drug_name}})
        if query.section_type:
            conditions.append({"section_type": {"$eq": query.section_type}})
        if query.where_filter:
            conditions.append(query.where_filter)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def delete_collection(self, name: CollectionName | str) -> None:
        """Delete a collection from the vector store."""
        collection_name = name.value if isinstance(name, CollectionName) else name
        try:
            self._client.delete_collection(collection_name)
            self._collections.pop(collection_name, None)
            logger.info("vector_store.collection_deleted", name=collection_name)
        except Exception:
            logger.exception("vector_store.delete_error", name=collection_name)

    def get_collection_stats(self) -> dict[str, int]:
        """Return document counts for all collections."""
        stats: dict[str, int] = {}
        existing = self._client.list_collections()
        for coll in existing:
            coll_name = coll.name if hasattr(coll, "name") else str(coll)
            collection = self.get_or_create_collection(coll_name)
            stats[coll_name] = collection.count()
        return stats

    def reset(self) -> None:
        """Delete all collections. Use with caution."""
        existing = self._client.list_collections()
        for coll in existing:
            coll_name = coll.name if hasattr(coll, "name") else str(coll)
            self._client.delete_collection(coll_name)
        self._collections.clear()
        logger.warning("vector_store.reset_complete")
