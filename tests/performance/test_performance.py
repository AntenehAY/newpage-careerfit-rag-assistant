"""Performance tests - benchmarks (marked slow)."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.pipeline import ingest_document
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


@pytest.mark.slow
@pytest.mark.performance
class TestPerformance:
    """Ingestion time, retrieval latency, concurrent requests."""

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
        """Mock EmbeddingService for fast tests."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [
            [0.0] * dim for _ in texts
        ]
        return svc

    def test_ingestion_time_small_doc(self, sample_resume_txt: Path):
        """Ingestion of small document completes in reasonable time."""
        text = sample_resume_txt.read_text()
        metadata = {"doc_id": "perf-1", "doc_type": "resume"}
        start = time.perf_counter()
        chunks = chunk_document(text, metadata, chunk_size=512, chunk_overlap=50)
        elapsed = time.perf_counter() - start
        assert len(chunks) >= 1
        assert elapsed < 1.0, f"Chunking took {elapsed:.2f}s (expected <1s)"

    def test_ingestion_time_medium_doc(self, sample_jd_docx: Path):
        """Ingestion of medium DOCX completes in reasonable time."""
        start = time.perf_counter()
        chunks = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="perf-jd",
        )
        elapsed = time.perf_counter() - start
        assert len(chunks) >= 1
        assert elapsed < 5.0, f"Ingestion took {elapsed:.2f}s (expected <5s)"

    def test_retrieval_latency(
        self, chroma_client, mock_embedding, sample_resume_pdf: Path
    ):
        """Retrieval from vector store completes quickly."""
        store = VectorStore(
            collection_name="perf_test",
            persist_directory="/tmp/perf_vectordb",
            embedding_service=mock_embedding,
            chroma_client=chroma_client,
        )
        chunks = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="perf-res",
            store_in_vector_db=True,
            vector_store=store,
        )
        assert len(chunks) >= 1

        retriever = Retriever(store, top_k=5)
        start = time.perf_counter()
        results = retriever.retrieve_for_query("Python skills")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Retrieval took {elapsed:.2f}s (expected <1s)"
        assert isinstance(results, list)
