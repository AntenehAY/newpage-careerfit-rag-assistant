"""E2E tests: Error scenarios and recovery."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_chain, get_vector_store
from app.api.document_registry import DocumentRegistry
from app.api.routes import get_registry
from app.main import app
from app.rag.guardrails import get_fallback_response


@pytest.mark.e2e
class TestErrorScenarios:
    """Upload invalid file, query without docs, rate limit, API errors."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore."""
        mock = MagicMock()
        mock.add_documents = MagicMock()
        mock.delete_by_doc_id = MagicMock()
        mock.get_collection_stats.return_value = {
            "count": 0,
            "collection_name": "test",
            "persist_directory": "/tmp",
        }
        return mock

    @pytest.fixture
    def mock_rag_chain(self):
        """Mock RAG chain."""
        chain = MagicMock()
        chain.answer_query.return_value = get_fallback_response(
            "No relevant information found"
        )
        return chain

    @pytest.fixture
    def registry(self, tmp_path):
        """Empty registry."""
        return DocumentRegistry(registry_path=tmp_path / "err_registry.json")

    @pytest.fixture
    def uploads_dir(self, tmp_path):
        """Temp uploads."""
        d = tmp_path / "err_uploads"
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def client(self, mock_vector_store, mock_rag_chain, registry, uploads_dir):
        """TestClient with mocks."""
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
        app.dependency_overrides[get_registry] = lambda: registry

        with patch("app.api.routes.UPLOADS_DIR", uploads_dir):
            yield TestClient(app)

        app.dependency_overrides.clear()

    def test_upload_invalid_file_returns_400(self, client):
        """Upload invalid file type (.txt) returns 400."""
        response = client.post(
            "/api/upload",
            files={"file": ("bad.txt", b"not a pdf", "text/plain")},
            data={"doc_type": "resume"},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json().get("detail", "")

    def test_upload_empty_file_returns_400(self, client):
        """Upload empty file returns 400."""
        response = client.post(
            "/api/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            data={"doc_type": "resume"},
        )
        assert response.status_code == 400

    def test_query_without_documents_returns_fallback(self, client, mock_rag_chain):
        """Query with no documents returns helpful fallback."""
        mock_rag_chain.answer_query.return_value = get_fallback_response(
            "No relevant information found"
        )
        response = client.post(
            "/api/query",
            json={"query": "What skills am I missing?"},
        )
        assert response.status_code == 200
        answer = response.json().get("answer", "").lower()
        assert "upload" in answer or "documents" in answer

    def test_query_rate_limit_returns_429(self, client):
        """When LLM raises rate limit, API returns 429."""
        mock_chain = MagicMock()
        mock_chain.answer_query.side_effect = RuntimeError(
            "The AI service is currently busy. Please try again."
        )
        app.dependency_overrides[get_rag_chain] = lambda: mock_chain
        try:
            response = client.post(
                "/api/query",
                json={"query": "What are my skills?"},
            )
            assert response.status_code == 429
        finally:
            app.dependency_overrides.pop(get_rag_chain, None)

    def test_query_rate_limit_fallback_returns_200(self, client):
        """When RAG returns rate limit fallback (not exception), 200."""
        fallback = get_fallback_response("Rate limit exceeded")
        mock_chain = MagicMock()
        mock_chain.answer_query.return_value = fallback
        app.dependency_overrides[get_rag_chain] = lambda: mock_chain
        try:
            response = client.post(
                "/api/query",
                json={"query": "Skills?"},
            )
            assert response.status_code == 200
            assert "wait" in response.json().get("answer", "").lower() or "rate" in response.json().get("answer", "").lower()
        finally:
            app.dependency_overrides.pop(get_rag_chain, None)

    def test_delete_nonexistent_returns_404(self, client):
        """Delete non-existent document returns 404."""
        response = client.delete("/api/documents/nonexistent-123")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_query_too_short_handled(self, client, mock_rag_chain):
        """Very short query gets handled (fallback or validation)."""
        mock_rag_chain.answer_query.return_value = get_fallback_response(
            "Query too short"
        )
        response = client.post("/api/query", json={"query": "Hi"})
        assert response.status_code in (200, 400)
