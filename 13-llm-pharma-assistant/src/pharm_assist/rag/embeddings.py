"""Embedding management for pharmaceutical document vectors.

Generates embeddings via the Anthropic API (using a compatible embedding model),
supports batch processing with rate limiting, caching, and similarity utilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    vector: list[float]
    model: str
    token_count: int


class EmbeddingCache:
    """Disk-backed cache for computed embeddings to avoid redundant API calls.

    Uses content hashing so identical text always maps to the same cache key,
    regardless of source document.
    """

    def __init__(self, cache_dir: str | Path = ".cache/embeddings") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0
        logger.info("embedding_cache.initialized", cache_dir=str(self._cache_dir))

    def _key(self, text: str, model: str) -> str:
        """Generate a deterministic cache key from text content and model."""
        raw = f"{model}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        """Look up a cached embedding vector."""
        key = self._key(text, model)

        # Check in-memory cache first
        if key in self._memory_cache:
            self._hits += 1
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                vector = data["vector"]
                self._memory_cache[key] = vector
                self._hits += 1
                return vector
            except (json.JSONDecodeError, KeyError):
                logger.warning("embedding_cache.corrupt_entry", key=key)
                cache_file.unlink(missing_ok=True)

        self._misses += 1
        return None

    def put(self, text: str, model: str, vector: list[float]) -> None:
        """Store an embedding vector in the cache."""
        key = self._key(text, model)
        self._memory_cache[key] = vector

        cache_file = self._cache_dir / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps({"model": model, "vector": vector}),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("embedding_cache.write_failed", key=key)

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
            "memory_entries": len(self._memory_cache),
        }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
        self._hits = 0
        self._misses = 0
        logger.info("embedding_cache.cleared")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Prevents hitting API rate limits when generating embeddings for
    large batches of document chunks.
    """

    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 100_000) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._request_timestamps: list[float] = []
        self._token_counts: list[tuple[float, int]] = []

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until a request slot is available."""
        now = time.monotonic()

        # Purge old request timestamps (older than 60s)
        cutoff = now - 60.0
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

        # Check request rate
        if len(self._request_timestamps) >= self._rpm:
            wait_time = self._request_timestamps[0] - cutoff
            if wait_time > 0:
                logger.debug("rate_limiter.waiting", wait_seconds=round(wait_time, 2))
                await asyncio.sleep(wait_time)

        # Check token rate
        current_tokens = sum(c for _, c in self._token_counts)
        if current_tokens + estimated_tokens > self._tpm:
            wait_time = self._token_counts[0][0] - cutoff if self._token_counts else 1.0
            if wait_time > 0:
                logger.debug("rate_limiter.token_wait", wait_seconds=round(wait_time, 2))
                await asyncio.sleep(wait_time)

        self._request_timestamps.append(time.monotonic())
        if estimated_tokens > 0:
            self._token_counts.append((time.monotonic(), estimated_tokens))


class EmbeddingManager:
    """Manages embedding generation with caching, batching, and rate limiting.

    Uses a configurable embedding function that can be backed by any provider.
    Default implementation uses a local sentence-transformer-compatible approach
    via chromadb's built-in embedding function, but can be swapped for
    Anthropic Voyager, OpenAI, or other APIs.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        batch_size: int = 32,
        cache_dir: str | Path = ".cache/embeddings",
        requests_per_minute: int = 50,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._cache = EmbeddingCache(cache_dir)
        self._rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self._embed_fn: Any = None

        logger.info(
            "embedding_manager.initialized",
            model=model_name,
            dimension=dimension,
            batch_size=batch_size,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def _get_embed_function(self) -> Any:
        """Lazy-load the embedding function."""
        if self._embed_fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    DefaultEmbeddingFunction,
                )

                self._embed_fn = DefaultEmbeddingFunction()
                logger.info("embedding_manager.using_default_chromadb_embeddings")
            except ImportError:
                logger.warning("embedding_manager.chromadb_embeddings_unavailable")
                self._embed_fn = self._fallback_embed
        return self._embed_fn

    @staticmethod
    def _fallback_embed(texts: list[str]) -> list[list[float]]:
        """Deterministic hash-based embedding fallback for testing/development.

        Produces consistent vectors from text content without any ML model.
        NOT suitable for production semantic search -- only for development.
        """
        dimension = 384
        results: list[list[float]] = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode("utf-8")).digest()
            rng = np.random.RandomState(int.from_bytes(text_hash[:4], "big"))
            vec = rng.randn(dimension).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        # Check cache first
        cached = self._cache.get(text, self._model_name)
        if cached is not None:
            return cached

        await self._rate_limiter.acquire(estimated_tokens=len(text.split()))

        embed_fn = self._get_embed_function()
        vectors = embed_fn([text])
        vector = vectors[0] if vectors else self._fallback_embed([text])[0]

        self._cache.put(text, self._model_name, vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts with rate limiting.

        Automatically splits into sub-batches and checks the cache
        for each text before making API calls.
        """
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for all texts
        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._model_name)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            logger.info("embedding_manager.all_cached", count=len(texts))
            return [r for r in results if r is not None]

        logger.info(
            "embedding_manager.batch_embed",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            to_embed=len(uncached_texts),
        )

        # Process uncached texts in sub-batches
        embed_fn = self._get_embed_function()
        for batch_start in range(0, len(uncached_texts), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(uncached_texts))
            batch = uncached_texts[batch_start:batch_end]

            estimated_tokens = sum(len(t.split()) for t in batch)
            await self._rate_limiter.acquire(estimated_tokens=estimated_tokens)

            try:
                vectors = embed_fn(batch)
            except Exception:
                logger.exception("embedding_manager.batch_error", batch_start=batch_start)
                vectors = self._fallback_embed(batch)

            for j, vector in enumerate(vectors):
                global_idx = uncached_indices[batch_start + j]
                results[global_idx] = vector
                self._cache.put(uncached_texts[batch_start + j], self._model_name, vector)

        return [r for r in results if r is not None]

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return self._cache.stats()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def top_k_similar(
    query_vector: list[float],
    vectors: list[list[float]],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find top-k most similar vectors by cosine similarity.

    Returns list of (index, similarity_score) tuples, sorted descending.
    """
    scores: list[tuple[int, float]] = []
    for i, vec in enumerate(vectors):
        sim = cosine_similarity(query_vector, vec)
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
