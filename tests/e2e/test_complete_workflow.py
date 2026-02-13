"""E2E tests: Full workflow from empty state to cleanup."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.api.dependencies import get_rag_chain, get_vector_store
from app.api.document_registry import DocumentRegistry
from app.api.routes import get_registry
from app.main import app
from app.models import QueryResponse, SourceReference
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from fastapi.testclient import TestClient


@pytest.mark.e2e
class TestCompleteWorkflow:
    """Full workflow: no docs -> upload resume + 2 JDs -> 5 query types -> delete -> cleanup."""

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
        """Mock EmbeddingService (no OpenAI calls)."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [
            [0.1 * (i % 5)] * dim for i in range(len(texts))
        ]
        return svc

    @pytest.fixture
    def vector_store(self, chroma_client, mock_embedding):
        """VectorStore for E2E."""
        return VectorStore(
            collection_name="e2e_careerfit",
            persist_directory="/tmp/e2e_vectordb",
            embedding_service=mock_embedding,
            chroma_client=chroma_client,
        )

    @pytest.fixture
    def registry(self, tmp_path):
        """Fresh document registry."""
        return DocumentRegistry(registry_path=tmp_path / "e2e_registry.json")

    @pytest.fixture
    def uploads_dir(self, tmp_path):
        """Temporary uploads directory."""
        d = tmp_path / "e2e_uploads"
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def mock_rag_chain(self):
        """RAG chain that returns realistic responses per query type."""
        def answer_fn(query, **kwargs):
            q = (query or "").lower()
            if "skill" in q or "gap" in q or "missing" in q:
                ans = "Skill gap: You have Python, SQL. Consider highlighting AWS experience."
            elif "experience" in q or "align" in q or "match" in q:
                ans = "Experience alignment: Your 5+ years Python aligns well with the 3+ required."
            elif "interview" in q or "question" in q or "expect" in q:
                ans = "Interview prep: Expect questions about Python projects and teamwork."
            elif "filter" in q:
                ans = "Filtered results show resume-only context."
            else:
                ans = "General career: Your background is strong for this role."
            return QueryResponse(
                answer=ans,
                sources=[
                    SourceReference(
                        doc_id="d1",
                        doc_name="resume.pdf",
                        doc_type="resume",
                        chunk_text="Sample chunk",
                        relevance_score=0.9,
                    ),
                ],
                confidence=0.85,
                generated_at=datetime.now(timezone.utc),
            )

        chain = MagicMock()
        chain.answer_query.side_effect = answer_fn
        return chain

    @pytest.fixture
    def client(self, vector_store, registry, uploads_dir, mock_rag_chain):
        """TestClient with real vector store, registry, mocked RAG."""
        app.dependency_overrides[get_vector_store] = lambda: vector_store
        app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
        app.dependency_overrides[get_registry] = lambda: registry

        with patch("app.api.routes.UPLOADS_DIR", uploads_dir):
            yield TestClient(app)

        app.dependency_overrides.clear()

    @pytest.fixture
    def sample_pdf(self):
        """Minimal valid PDF."""
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"

    def test_complete_workflow(
        self,
        client,
        sample_pdf,
        sample_resume_pdf: Path,
        sample_jd_docx: Path,
        vector_store,
        registry,
        uploads_dir,
    ):
        """Full workflow: upload -> 5 query types -> delete -> verify cleanup."""
        # 1. Start with no documents
        list_resp = client.get("/api/documents")
        assert list_resp.status_code == 200
        assert list_resp.json() == []

        # 2. Upload resume + 2 job descriptions
        # Use real fixture files for ingestion; API saves to uploads_dir
        from app.ingestion.pipeline import ingest_document

        # Ingest via pipeline directly (bypass API for file handling)
        resume_chunks = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="resume-e2e",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        # Simulate registry entries and upload files
        from app.models import DocumentUpload

        for doc_id, fname, dtype in [
            ("resume-e2e", "resume.pdf", "resume"),
            ("jd1-e2e", "jd1.docx", "job_description"),
            ("jd2-e2e", "jd2.docx", "job_description"),
        ]:
            registry.add(
                DocumentUpload(
                    file_id=doc_id,
                    file_name=fname,
                    file_type=dtype,
                    file_size=100,
                    uploaded_at=datetime.now(timezone.utc),
                    status="completed",
                )
            )
            (uploads_dir / f"{doc_id}.pdf").write_bytes(sample_pdf)

        ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd1-e2e",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd2-e2e",
            store_in_vector_db=True,
            vector_store=vector_store,
        )

        # 3. Ask 5 different query types
        queries = [
            ("What skills am I missing for this role?", "skill_gap"),
            ("How does my experience align with the job?", "experience"),
            ("What interview questions should I expect?", "interview"),
            ("What are my strengths for this role?", "general"),
            ("Summarize my resume", "general"),
        ]
        for query, _ in queries:
            resp = client.post("/api/query", json={"query": query})
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "sources" in data
            assert isinstance(data["sources"], list)
            assert len(data["answer"]) > 10

        # 4. Delete documents
        for doc_id in ["resume-e2e", "jd1-e2e", "jd2-e2e"]:
            resp = client.delete(f"/api/documents/{doc_id}")
            assert resp.status_code == 200
            assert resp.json()["success"] is True

        # 5. Verify cleanup
        list_after = client.get("/api/documents")
        assert list_after.status_code == 200
        assert list_after.json() == []
        assert vector_store.get_collection_stats()["count"] == 0
