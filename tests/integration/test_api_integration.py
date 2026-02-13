"""Integration tests: API endpoints with real services (mocked LLM/embeddings)."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_chain, get_vector_store
from app.api.document_registry import DocumentRegistry
from app.api.routes import get_registry
from app.main import app
from app.models import DocumentUpload, QueryResponse, SourceReference
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


@pytest.mark.integration
class TestAPIIntegration:
    """API layer with real vector store and registry; mocked LLM and embeddings."""

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
        """Mock EmbeddingService."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [
            [0.0] * dim for _ in texts
        ]
        return svc

    @pytest.fixture
    def real_vector_store(self, chroma_client, mock_embedding):
        """Real VectorStore for integration."""
        return VectorStore(
            collection_name="api_integration_test",
            persist_directory="/tmp/api_integration_vectordb",
            embedding_service=mock_embedding,
            chroma_client=chroma_client,
        )

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Temporary document registry."""
        return DocumentRegistry(registry_path=tmp_path / "registry.json")

    @pytest.fixture
    def temp_uploads_dir(self, tmp_path):
        """Temporary uploads directory."""
        uploads = tmp_path / "uploads"
        uploads.mkdir(parents=True)
        return uploads

    @pytest.fixture
    def sample_pdf_bytes(self):
        """Valid minimal PDF for upload."""
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"

    @pytest.fixture
    def mock_rag_response(self):
        """Mock RAG chain response."""
        chain = MagicMock()
        chain.answer_query.return_value = QueryResponse(
            answer="Based on your documents, you have Python and SQL skills.",
            sources=[
                SourceReference(
                    doc_id="d1",
                    doc_name="resume.pdf",
                    doc_type="resume",
                    chunk_text="Python, SQL",
                    relevance_score=0.9,
                ),
            ],
            confidence=0.85,
            generated_at=datetime.now(timezone.utc),
        )
        return chain

    @pytest.fixture
    def client(
        self,
        real_vector_store,
        temp_registry,
        temp_uploads_dir,
        mock_rag_response,
    ):
        """TestClient with real vector store, temp registry, mocked RAG."""
        app.dependency_overrides[get_vector_store] = lambda: real_vector_store
        app.dependency_overrides[get_rag_chain] = lambda: mock_rag_response
        app.dependency_overrides[get_registry] = lambda: temp_registry

        with patch("app.api.routes.UPLOADS_DIR", temp_uploads_dir):
            yield TestClient(app)

        app.dependency_overrides.clear()

    def test_upload_query_delete_workflow(
        self,
        client,
        sample_pdf_bytes,
        temp_registry,
        real_vector_store,
        temp_uploads_dir,
    ):
        """Upload -> query -> delete full workflow."""
        # Upload
        response = client.post(
            "/api/upload",
            files={"file": ("resume.pdf", sample_pdf_bytes, "application/pdf")},
            data={"doc_type": "resume"},
        )
        assert response.status_code == 200
        data = response.json()
        file_id = data["file_id"]
        assert file_id
        assert data["file_name"] == "resume.pdf"

        # List documents
        list_resp = client.get("/api/documents")
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert any(d["file_id"] == file_id for d in docs)

        # Query (uses mock RAG)
        query_resp = client.post(
            "/api/query",
            json={"query": "What skills do I have?"},
        )
        assert query_resp.status_code == 200
        assert "Python" in query_resp.json().get("answer", "")

        # Delete
        delete_resp = client.delete(f"/api/documents/{file_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["success"] is True

        # Verify cleanup
        list_after = client.get("/api/documents")
        assert not any(d["file_id"] == file_id for d in list_after.json())

    def test_upload_invalid_file_returns_400(self, client):
        """Upload .txt file returns 400."""
        response = client.post(
            "/api/upload",
            files={"file": ("doc.txt", b"plain text", "text/plain")},
            data={"doc_type": "resume"},
        )
        assert response.status_code == 400

    def test_query_without_docs_returns_200(self, client, mock_rag_response):
        """Query with empty store still returns 200 (RAG handles fallback)."""
        # Mock returns fallback for empty context
        from app.rag.guardrails import get_fallback_response

        mock_rag_response.answer_query.return_value = get_fallback_response(
            "No relevant information found"
        )

        response = client.post(
            "/api/query",
            json={"query": "What skills am I missing?"},
        )
        assert response.status_code == 200
        assert "upload" in response.json().get("answer", "").lower()

    def test_delete_nonexistent_returns_404(self, client):
        """Delete non-existent document returns 404."""
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404

    def test_stats_reflects_vector_store(self, client):
        """GET /api/stats returns vector store info."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "vector_db" in data
        assert "documents_by_type" in data
