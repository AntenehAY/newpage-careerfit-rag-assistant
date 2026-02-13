"""Integration tests: Upload document -> parse -> chunk -> embed -> store -> query."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pipeline import ingest_document
from app.models import ChunkMetadata
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


@pytest.mark.integration
class TestIngestionToVector:
    """Full document lifecycle: parse, chunk, embed, store, retrieve."""

    @pytest.fixture
    def chroma_client(self):
        """ChromaDB EphemeralClient."""
        try:
            import chromadb

            return chromadb.EphemeralClient()
        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")

    @pytest.fixture
    def mock_embedding(self):
        """Mock EmbeddingService (OpenAI) to avoid API calls."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [
            [0.1 * (i % 10)] * dim for i in range(len(texts))
        ]
        return svc

    @pytest.fixture
    def vector_store(self, chroma_client, mock_embedding):
        """VectorStore with real ChromaDB, mocked embeddings."""
        return VectorStore(
            collection_name="integration_test",
            persist_directory="/tmp/integration_vectordb",
            embedding_service=mock_embedding,
            chroma_client=chroma_client,
        )

    def test_upload_parse_chunk_embed_store(
        self, sample_resume_pdf: Path, vector_store
    ):
        """Ingest PDF -> chunks -> store in vector DB."""
        chunks = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="resume-integration-1",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        assert len(chunks) >= 1
        assert all(c.doc_type == "resume" for c in chunks)
        stats = vector_store.get_collection_stats()
        assert stats["count"] == len(chunks)

    def test_query_vector_store_after_ingestion(
        self, sample_jd_docx: Path, vector_store
    ):
        """Ingest DOCX, then search returns relevant chunks."""
        chunks = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-integration-1",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        assert len(chunks) >= 1

        results = vector_store.search("Python experience", top_k=3)
        assert len(results) >= 1
        assert any("Python" in r.get("text", "") for r in results)

    def test_delete_removes_from_store(
        self, sample_resume_pdf: Path, vector_store
    ):
        """Delete by doc_id removes chunks."""
        chunks = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="doc-to-delete",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        assert vector_store.get_collection_stats()["count"] == len(chunks)

        vector_store.delete_by_doc_id("doc-to-delete")
        assert vector_store.get_collection_stats()["count"] == 0

    def test_full_lifecycle_resume_and_jd(
        self, sample_resume_pdf: Path, sample_jd_docx: Path, vector_store
    ):
        """Upload resume + JD, query both, verify retrieval, delete."""
        c1 = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="res-lifecycle",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        c2 = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-lifecycle",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        assert len(c1) >= 1 and len(c2) >= 1

        retriever = Retriever(vector_store=vector_store, top_k=5)
        ctx = retriever.get_context_for_llm("Python and AWS skills")
        assert ctx["total_chunks"] >= 1
        assert "context" in ctx

        vector_store.delete_by_doc_id("res-lifecycle")
        vector_store.delete_by_doc_id("jd-lifecycle")
        assert vector_store.get_collection_stats()["count"] == 0
