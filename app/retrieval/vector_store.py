"""ChromaDB vector store wrapper for document chunks."""

import logging
from pathlib import Path
from typing import Any, List, Optional

from app.models import ChunkMetadata

from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with metadata filtering support."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_service: EmbeddingService,
        chroma_client: Optional[Any] = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Path for persistent storage.
            embedding_service: Service for generating embeddings.
            chroma_client: Optional ChromaDB client (e.g. EphemeralClient for tests).
                If None, creates PersistentClient with persist_directory.
        """
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).expanduser().resolve())
        self.embedding_service = embedding_service
        self._client = None
        self._collection = None
        self._init_client(chroma_client)

    def _init_client(self, chroma_client: Optional[Any] = None) -> None:
        """Initialize ChromaDB client and get/create collection."""
        try:
            import chromadb

            if chroma_client is not None:
                self._client = chroma_client
            else:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "VectorStore initialized: collection=%s, path=%s",
                self.collection_name,
                self.persist_directory,
            )
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB: %s", e)
            raise

    @property
    def collection(self):
        """Lazy access to collection (for tests)."""
        return self._collection

    def add_documents(self, chunks: List[ChunkMetadata], texts: List[str]) -> None:
        """Generate embeddings and store chunks in ChromaDB.

        Args:
            chunks: Chunk metadata (chunk_id, doc_id, doc_type, etc.).
            texts: Corresponding chunk text for each ChunkMetadata.

        Raises:
            ValueError: If chunks and texts lengths differ.
        """
        if len(chunks) != len(texts):
            raise ValueError(
                f"chunks length ({len(chunks)}) must equal texts length ({len(texts)})"
            )
        if not chunks:
            logger.warning("add_documents called with empty chunks")
            return
        ids = [c.chunk_id for c in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        metadatas: List[dict[str, Any]] = []
        for c in chunks:
            meta: dict[str, Any] = {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type,
                "chunk_index": c.chunk_index,
            }
            if c.page_number is not None:
                meta["page_number"] = c.page_number
            if c.section is not None:
                meta["section"] = c.section
            if c.char_start is not None:
                meta["char_start"] = c.char_start
            if c.char_end is not None:
                meta["char_end"] = c.char_end
            metadatas.append(meta)
        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info("add_documents: stored %d chunks in collection %s", len(chunks), self.collection_name)
        except Exception as e:
            logger.exception("add_documents failed: %s", e)
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return.
            filter_metadata: Optional metadata filter (e.g. {"doc_type": "resume", "doc_id": "abc"}).

        Returns:
            List of dicts: [{id, text, metadata, score}, ...]. Higher score = more similar.
        """
        query_embedding = self.embedding_service.embed_text(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        try:
            result = self._collection.query(**kwargs)
        except Exception as e:
            logger.exception("search failed: %s", e)
            raise
        # ChromaDB returns: ids, documents, metadatas, distances
        # For cosine, lower distance = more similar; we convert to score (1 - distance) for [0,1]
        ids_list = result.get("ids", [[]])[0]
        docs_list = result.get("documents", [[]])[0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]
        results: List[dict[str, Any]] = []
        for i, doc_id in enumerate(ids_list):
            meta = metas_list[i] if metas_list and i < len(metas_list) else {}
            doc_text = docs_list[i] if docs_list and i < len(docs_list) else ""
            dist = dists_list[i] if dists_list and i < len(dists_list) else 0.0
            # Cosine distance: 0 = identical, 2 = opposite. Score: 1 - (dist/2) for [0,1] approx
            score = max(0.0, 1.0 - (dist / 2.0)) if dists_list else 1.0
            results.append(
                {
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta or {},
                    "score": score,
                }
            )
        return results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document.

        Args:
            doc_id: Document ID to delete chunks for.
        """
        try:
            # ChromaDB delete with where filter
            self._collection.delete(where={"doc_id": doc_id})
            logger.info("delete_by_doc_id: removed chunks for doc_id=%s", doc_id)
        except Exception as e:
            # Fallback: get ids by doc_id then delete by ids
            try:
                got = self._collection.get(where={"doc_id": doc_id})
                ids_to_del = got.get("ids", [])
                if ids_to_del:
                    self._collection.delete(ids=ids_to_del)
                    logger.info("delete_by_doc_id: removed %d chunks for doc_id=%s", len(ids_to_del), doc_id)
                else:
                    logger.info("delete_by_doc_id: no chunks found for doc_id=%s", doc_id)
            except Exception as e2:
                logger.exception("delete_by_doc_id failed: %s", e2)
                raise

    def get_collection_stats(self) -> dict[str, Any]:
        """Return collection statistics.

        Returns:
            dict with keys like count, name, etc.
        """
        try:
            count = self._collection.count()
            return {
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.warning("get_collection_stats failed: %s", e)
            return {
                "count": 0,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "error": str(e),
            }
