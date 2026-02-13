"""Dependency injection for embedding service, vector store, retriever, and RAG chain.

Uses FastAPI Depends with singleton caching for expensive resources.
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import Depends

from app.config import settings
from app.rag.chain import RAGChain
from app.rag.guardrails import RateLimiter
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ChromaDB collection name for career documents
COLLECTION_NAME = "careerfit_chunks"


def get_embedding_service() -> EmbeddingService:
    """Load and return configured embedding service.

    Uses singleton pattern (cached per process) to avoid re-initializing
    the embedding model on each request.

    Returns:
        EmbeddingService instance configured with model and API key from config.
    """
    return _get_embedding_service_cached()


@lru_cache(maxsize=1)
def _get_embedding_service_cached() -> EmbeddingService:
    """Cached singleton for EmbeddingService."""
    svc = EmbeddingService(
        model_name=settings.embedding_model_name,
        api_key=settings.openai_api_key,
    )
    logger.info("EmbeddingService singleton initialized: model=%s", settings.embedding_model_name)
    return svc


# Module-level singleton for vector store (shared across requests)
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> VectorStore:
    """Initialize ChromaDB vector store with config path.

    Singleton pattern ensures a single persistent connection to the vector DB.

    Args:
        embedding_service: Injected embedding service for generating vectors.

    Returns:
        VectorStore instance configured with ChromaDB at vector_db_path.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            collection_name=COLLECTION_NAME,
            persist_directory=settings.vector_db_path,
            embedding_service=embedding_service,
        )
        logger.info("VectorStore singleton initialized: path=%s", settings.vector_db_path)
    return _vector_store_instance


def get_retriever(
    vector_store: VectorStore = Depends(get_vector_store),
) -> Retriever:
    """Create retriever with vector store.

    Args:
        vector_store: Injected vector store for similarity search.

    Returns:
        Retriever instance with MMR diversity and context formatting.
    """
    return Retriever(vector_store=vector_store, top_k=settings.top_k_results)


def get_rag_chain(
    retriever: Retriever = Depends(get_retriever),
) -> RAGChain:
    """Initialize RAG chain with Claude Sonnet 4 and rate limiter.

    Args:
        retriever: Injected retriever for context retrieval.

    Returns:
        RAGChain instance with LLM and guardrails.
    """
    rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
    return RAGChain(
        retriever=retriever,
        llm_model=settings.llm_model_name,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        rate_limiter=rate_limiter,
    )
