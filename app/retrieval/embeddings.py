"""Embedding generation for vector search.

Supports OpenAI text-embedding-3-small and local sentence-transformers
(all-MiniLM-L6-v2).
"""

import hashlib
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Model identifiers
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class EmbeddingService:
    """Generate text embeddings via OpenAI or local sentence-transformers."""

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Either "text-embedding-3-small" (OpenAI) or
                "all-MiniLM-L6-v2" (local).
            api_key: API key for OpenAI. Required for OpenAI model;
                ignored for local model.
        """
        self.model_name = model_name
        self.api_key = api_key
        self._local_model = None
        self._cache: dict[str, List[float]] = {}
        self._is_openai = model_name.lower() in (
            "text-embedding-3-small",
            "text-embedding-3-large",
        )
        logger.info(
            "EmbeddingService initialized: model=%s, provider=%s",
            model_name,
            "openai" if self._is_openai else "local",
        )

    def _get_local_model(self):
        """Lazy-load sentence-transformers model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.model_name)
                logger.info("Loaded local embedding model: %s", self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._local_model

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        embeddings: List[List[float]] = []
        total_tokens = 0

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            for attempt in range(MAX_RETRIES):
                try:
                    t0 = time.perf_counter()
                    resp = client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        encoding_format="float",
                    )
                    elapsed = time.perf_counter() - t0
                    batch_embeddings = [e.embedding for e in resp.data]
                    usage = resp.usage
                    tokens = usage.total_tokens if usage else 0
                    total_tokens += tokens
                    logger.info(
                        "OpenAI embedding batch %d-%d: %d texts, %d tokens, %.3fs",
                        i,
                        i + len(batch),
                        len(batch),
                        tokens,
                        elapsed,
                    )
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            "OpenAI API error (retry %d/%d): %s",
                            attempt + 1,
                            MAX_RETRIES,
                            e,
                        )
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error("OpenAI API failed after %d retries: %s", MAX_RETRIES, e)
                        raise

        if total_tokens > 0:
            logger.info("OpenAI embedding total tokens: %d", total_tokens)
        return embeddings

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via sentence-transformers."""
        model = self._get_local_model()
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            t0 = time.perf_counter()
            batch_embeddings = model.encode(batch, convert_to_numpy=False)
            elapsed = time.perf_counter() - t0
            if isinstance(batch_embeddings[0], list):
                pass  # already list
            else:
                batch_embeddings = [list(e) for e in batch_embeddings]
            embeddings.extend(batch_embeddings)
            logger.info(
                "Local embedding batch %d-%d: %d texts, %.3fs",
                i,
                i + len(batch),
                len(batch),
                elapsed,
            )
        return embeddings

    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.
            use_cache: If True, return cached embedding when available.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            Exception: On API or model failure after retries.
        """
        if not text or not text.strip():
            # Return zeros for empty text (some models handle this, others need fallback)
            text = " "  # minimal non-empty to avoid API issues
        if use_cache:
            key = self._cache_key(text)
            if key in self._cache:
                return self._cache[key]
        t0 = time.perf_counter()
        embeddings = self.embed_batch([text])
        elapsed = time.perf_counter() - t0
        vec = embeddings[0]
        if use_cache:
            self._cache[key] = vec
        logger.debug("Single embed latency: %.3fs, dim=%d", elapsed, len(vec))
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Processes in batches of 100. Logs progress for large batches.
        Caches results for potential reuse in single-text calls.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        # Normalize empty texts
        normalized = [t.strip() if t and t.strip() else " " for t in texts]
        if len(normalized) > BATCH_SIZE:
            logger.info("Embedding batch: %d texts (in batches of %d)", len(normalized), BATCH_SIZE)
        t0 = time.perf_counter()
        if self._is_openai:
            embeddings = self._embed_openai(normalized)
        else:
            embeddings = self._embed_local(normalized)
        elapsed = time.perf_counter() - t0
        logger.info("embed_batch completed: %d embeddings in %.3fs", len(embeddings), elapsed)
        # Cache for potential reuse
        for t, vec in zip(normalized, embeddings):
            self._cache[self._cache_key(t)] = vec
        return embeddings
