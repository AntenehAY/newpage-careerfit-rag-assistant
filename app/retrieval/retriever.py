"""Smart retrieval with MMR diversity reranking and LLM-ready context formatting."""

import math
import time
from typing import Any, List, Optional

from app.config import settings
from app.utils.logging import get_logger
from app.utils.metrics import get_metrics

logger = get_logger(__name__)
metrics = get_metrics()

# Estimate ~4 chars per token for common models
CHARS_PER_TOKEN = 4


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length (~4 chars per token)."""
    return max(0, len(text) // CHARS_PER_TOKEN)


class Retriever:
    """Retrieves relevant document chunks with MMR diversity and formats context for LLMs."""

    def __init__(self, vector_store: Any, top_k: Optional[int] = None) -> None:
        """Initialize the retriever.

        Args:
            vector_store: VectorStore instance for similarity search.
            top_k: Default number of results to retrieve. Uses settings.top_k_results if None.
        """
        self.vector_store = vector_store
        self.top_k = top_k if top_k is not None else settings.top_k_results
        logger.info("Retriever initialized: top_k=%d", self.top_k)

    def retrieve_for_query(
        self,
        query: str,
        filter_doc_type: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        """Retrieve relevant chunks for a query with optional filters and MMR diversity.

        Args:
            query: User search query.
            filter_doc_type: Optional filter (e.g. "resume", "job_description").
            filter_doc_id: Optional filter by document ID.
            top_k: Override default top_k for this query.

        Returns:
            List of dicts: [{id, text, metadata, score}, ...] with MMR reranking applied.
        """
        k = top_k if top_k is not None else self.top_k
        filter_metadata: dict[str, Any] = {}
        if filter_doc_type:
            filter_metadata["doc_type"] = filter_doc_type
        if filter_doc_id:
            filter_metadata["doc_id"] = filter_doc_id

        t0 = time.perf_counter()
        # Fetch more candidates for MMR diversity (2–3x top_k)
        fetch_k = min(max(k * 3, k), 50)
        results = self.vector_store.search(
            query=query,
            top_k=fetch_k,
            filter_metadata=filter_metadata if filter_metadata else None,
        )

        if not results:
            elapsed = time.perf_counter() - t0
            logger.info("retrieve_for_query: 0 results in {:.3f}s for query='{}'", elapsed, query[:50])
            metrics.record_retrieval(query=query[:100], duration=elapsed, chunks_returned=0)
            return []

        reranked = self._apply_mmr(results, query=query, lambda_param=0.5)
        final = reranked[:k]

        elapsed = time.perf_counter() - t0
        logger.info(
            "retrieve_for_query: {} chunks in {:.3f}s (query='{}')",
            len(final),
            elapsed,
            query[:50],
        )
        metrics.record_retrieval(query=query[:100], duration=elapsed, chunks_returned=len(final))
        return final

    def _apply_mmr(
        self,
        results: List[dict[str, Any]],
        query: str,
        lambda_param: float = 0.5,
    ) -> List[dict[str, Any]]:
        """Rerank results for diversity using Maximal Marginal Relevance.

        Balances relevance (similarity to query) vs novelty (distance to already selected).

        Args:
            results: Raw search results with id, text, metadata, score.
            query: Original query text (used to compute query embedding for MMR).
            lambda_param: Balance: 1.0 = pure relevance, 0.0 = pure diversity. Default 0.5.

        Returns:
            Reordered list of results.
        """
        if len(results) <= 1:
            return results

        query_emb = self.vector_store.embedding_service.embed_text(query, use_cache=True)
        texts = [r["text"] for r in results]
        doc_embeddings = self.vector_store.embedding_service.embed_batch(texts)

        selected: List[int] = []
        remaining = list(range(len(results)))

        # First: select the most relevant (highest score)
        best_idx = max(remaining, key=lambda i: results[i]["score"])
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Iteratively select next: argmax [lambda * sim(q,d) - (1-lambda) * max sim(d, s)]
        while remaining:
            best_mmr = -1.0
            best_i = remaining[0]
            for i in remaining:
                sim_q = _cosine_similarity(query_emb, doc_embeddings[i])
                max_sim_to_selected = max(
                    _cosine_similarity(doc_embeddings[i], doc_embeddings[s])
                    for s in selected
                )
                mmr_score = lambda_param * sim_q - (1 - lambda_param) * max_sim_to_selected
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_i = i
            selected.append(best_i)
            remaining.remove(best_i)

        return [results[i] for i in selected]

    def get_context_for_llm(
        self,
        query: str,
        max_tokens: int = 4000,
        filter_doc_type: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Retrieve chunks and format context for LLM consumption within token limit.

        Args:
            query: User query.
            max_tokens: Maximum context tokens. Default 4000.
            filter_doc_type: Optional filter by document type.
            filter_doc_id: Optional filter by document ID.
            top_k: Override default top_k.

        Returns:
            Dict with keys: context (formatted string), sources (list of source refs), total_chunks (int).
        """
        chunks = self.retrieve_for_query(
            query=query,
            filter_doc_type=filter_doc_type,
            filter_doc_id=filter_doc_id,
            top_k=top_k,
        )

        if not chunks:
            return {
                "context": "",
                "sources": [],
                "total_chunks": 0,
            }

        # Fit within token limit by truncating chunks
        result_chunks: List[dict[str, Any]] = []
        used_tokens = 0
        for c in chunks:
            formatted = self._format_context([c])
            chunk_tokens = _estimate_tokens(formatted)
            if used_tokens + chunk_tokens <= max_tokens:
                result_chunks.append(c)
                used_tokens += chunk_tokens
            else:
                break

        context = self._format_context(result_chunks)
        sources = [
            {
                "doc_id": c["metadata"].get("doc_id", ""),
                "doc_name": c["metadata"].get("doc_id", "unknown"),
                "doc_type": c["metadata"].get("doc_type", "unknown"),
                "chunk_text": c["text"],
                "relevance_score": c["score"],
            }
            for c in result_chunks
        ]

        logger.info(
            "get_context_for_llm: {} chunks, ~{} tokens",
            len(result_chunks),
            _estimate_tokens(context),
        )

        return {
            "context": context,
            "sources": sources,
            "total_chunks": len(result_chunks),
        }

    def _format_context(self, chunks: List[dict[str, Any]]) -> str:
        """Format chunks as structured context with source attribution.

        Template: "Document: {doc_name} (Type: {doc_type})\n{text}\n---"

        Args:
            chunks: List of dicts with id, text, metadata, score.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        parts: List[str] = []
        for c in chunks:
            meta = c.get("metadata", {})
            doc_id = meta.get("doc_id", "unknown")
            doc_type = meta.get("doc_type", "unknown")
            text = c.get("text", "")
            # Use doc_id as doc_name when doc_name not in metadata
            doc_name = meta.get("doc_name", doc_id)
            block = f"Document: {doc_name} (Type: {doc_type})\n{text}\n---"
            parts.append(block)

        return "\n".join(parts)
