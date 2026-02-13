"""End-to-end document ingestion pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from app.config import settings

if TYPE_CHECKING:
    from app.retrieval.vector_store import VectorStore
from app.models import ChunkMetadata

from .chunker import chunk_document
from .parsers import parse_document

logger = logging.getLogger(__name__)


def ingest_document(
    file_path: str,
    doc_type: Literal["resume", "job_description"],
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    store_in_vector_db: bool = False,
    vector_store: "VectorStore | None" = None,
) -> list[ChunkMetadata]:
    """Ingest a document: parse, chunk, and return chunks with full metadata.

    Steps:
    1. Parse document (PDF or DOCX) via parsers.py
    2. Chunk text via chunker.py with configurable size/overlap
    3. Optionally embed and store chunks in vector DB (when store_in_vector_db=True)
    4. Return list of ChunkMetadata with doc_id, doc_type, char_start, char_end, etc.

    On any error, returns an empty list and logs the failure (does not crash).

    Args:
        file_path: Path to the document file (.pdf or .docx).
        doc_type: Type of document: "resume" or "job_description".
        doc_id: Unique identifier for the document.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        store_in_vector_db: If True, embed chunks and store in ChromaDB after ingestion.
        vector_store: Optional VectorStore to use. If provided, store_in_vector_db uses it.

    Returns:
        List of ChunkMetadata. Empty list on parse/chunk error.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    t0 = time.perf_counter()
    path = Path(file_path)

    try:
        # Step 1: Parse
        t_parse = time.perf_counter()
        text = parse_document(str(path))
        parse_elapsed = time.perf_counter() - t_parse
        logger.info("Stage 1 (parse) completed in %.3fs for %s", parse_elapsed, path.name)

        if not text or not text.strip():
            logger.warning("Document parsed to empty text: %s", path.name)
            return []

        # Step 2: Chunk
        t_chunk = time.perf_counter()
        metadata = {"doc_id": doc_id, "doc_type": doc_type}
        chunks = chunk_document(text, metadata, chunk_size, chunk_overlap)
        chunk_elapsed = time.perf_counter() - t_chunk
        logger.info("Stage 2 (chunk) completed in %.3fs: %d chunks", chunk_elapsed, len(chunks))

        # Step 3 (optional): Store in vector DB
        if store_in_vector_db and chunks:
            try:
                vs = vector_store
                if vs is None:
                    from app.retrieval.embeddings import EmbeddingService
                    from app.retrieval.vector_store import VectorStore

                    embedding_svc = EmbeddingService(
                        model_name=settings.embedding_model_name,
                        api_key=settings.openai_api_key,
                    )
                    vs = VectorStore(
                        collection_name="careerfit_chunks",
                        persist_directory=settings.vector_db_path,
                        embedding_service=embedding_svc,
                    )
                chunk_texts = [text[c.char_start : c.char_end] for c in chunks]
                vs.add_documents(chunks, chunk_texts)
                logger.info("Stored %d chunks in vector DB", len(chunks))
            except Exception as e:
                logger.exception("Vector DB storage failed (chunks still returned): %s", e)

        total_elapsed = time.perf_counter() - t0
        logger.info("Ingestion complete for %s in %.3fs total", path.name, total_elapsed)
        return chunks

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return []
    except ValueError as e:
        logger.error("Ingestion failed for %s: %s", path.name, e)
        return []
    except Exception as e:
        logger.exception("Unexpected error ingesting %s: %s", path.name, e)
        return []
