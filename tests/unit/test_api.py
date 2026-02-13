"""API unit tests - route handlers with mocked dependencies."""

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
from app.rag.guardrails import get_fallback_response


@pytest.fixture
def temp_uploads_dir(tmp_path):
    """Temporary uploads directory for tests."""
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True)
    return uploads


@pytest.fixture
def temp_registry(tmp_path):
    """DocumentRegistry with temp storage."""
    registry_path = tmp_path / "document_registry.json"
    return DocumentRegistry(registry_path=registry_path)


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with minimal interface."""
    mock = MagicMock()
    mock.add_documents = MagicMock()
    mock.delete_by_doc_id = MagicMock()
    mock.get_collection_stats = MagicMock(
        return_value={
            "count": 0,
            "collection_name": "careerfit_chunks",
            "persist_directory": "/tmp/test",
        }
    )
    return mock


@pytest.fixture
def mock_rag_chain():
    """Mock RAGChain that returns predefined QueryResponse."""
    mock = MagicMock()
    mock.answer_query = MagicMock(
        return_value=QueryResponse(
            answer="This is a test answer.",
            sources=[
                SourceReference(
                    doc_id="doc1",
                    doc_name="resume.pdf",
                    doc_type="resume",
                    chunk_text="Sample chunk text",
                    relevance_score=0.9,
                )
            ],
            confidence=0.85,
            generated_at=datetime.now(timezone.utc),
        )
    )
    return mock


@pytest.fixture
def client(mock_vector_store, mock_rag_chain, temp_registry, temp_uploads_dir):
    """TestClient with overridden dependencies and patched paths."""
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
    app.dependency_overrides[get_registry] = lambda: temp_registry

    with patch("app.api.routes.UPLOADS_DIR", temp_uploads_dir):
        yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid PDF bytes for upload tests."""
    return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"


@pytest.mark.unit
def test_upload_valid_pdf(client, sample_pdf_bytes):
    """POST /api/upload with valid PDF file."""
    response = client.post(
        "/api/upload",
        files={"file": ("resume.pdf", sample_pdf_bytes, "application/pdf")},
        data={"doc_type": "resume"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["file_name"] == "resume.pdf"
    assert data["file_type"] == "resume"
    assert data["status"] in ("completed", "processing", "failed")


@pytest.mark.unit
def test_upload_invalid_file_type(client):
    """POST /api/upload with invalid file type (.txt) returns 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("document.txt", b"plain text content", "text/plain")},
        data={"doc_type": "resume"},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json().get("detail", "")


@pytest.mark.unit
def test_upload_invalid_doc_type(client, sample_pdf_bytes):
    """POST /api/upload with invalid doc_type returns 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("resume.pdf", sample_pdf_bytes, "application/pdf")},
        data={"doc_type": "invalid"},
    )
    assert response.status_code == 400
    assert "doc_type" in response.json().get("detail", "")


@pytest.mark.unit
def test_query_valid(client):
    """POST /api/query with valid request."""
    response = client.post(
        "/api/query",
        json={"query": "What skills does this resume highlight?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "This is a test answer" in data.get("answer", "")


@pytest.mark.unit
def test_query_with_filters(client):
    """POST /api/query with filter_doc_type and filter_doc_id."""
    response = client.post(
        "/api/query",
        json={
            "query": "Summarize experience",
            "filter_doc_type": "resume",
            "filter_doc_id": "abc-123",
            "max_results": 10,
        },
    )
    assert response.status_code == 200


@pytest.mark.unit
def test_query_invalid_too_short(client):
    """POST /api/query with query too short returns fallback (RAG handles it)."""
    response = client.post("/api/query", json={"query": "ab"})
    assert response.status_code in (200, 400)


@pytest.mark.unit
def test_query_rate_limit(client):
    """POST /api/query when RAG returns rate limit fallback yields 200."""
    fallback = get_fallback_response("Rate limit exceeded")
    mock_chain = MagicMock()
    mock_chain.answer_query = MagicMock(return_value=fallback)

    app.dependency_overrides[get_rag_chain] = lambda: mock_chain
    try:
        response = client.post(
            "/api/query", json={"query": "What are my skills?"}
        )
        assert response.status_code == 200
        answer = response.json().get("answer", "").lower()
        assert "wait" in answer or "rate" in answer
    finally:
        app.dependency_overrides.pop(get_rag_chain, None)


@pytest.mark.unit
def test_query_raises_429_when_llm_rate_limited(client):
    """POST /api/query when LLM raises rate limit returns 429."""
    mock_chain = MagicMock()
    mock_chain.answer_query = MagicMock(
        side_effect=RuntimeError(
            "The AI service is currently busy. Please try again."
        )
    )

    app.dependency_overrides[get_rag_chain] = lambda: mock_chain
    try:
        response = client.post(
            "/api/query", json={"query": "What are my skills?"}
        )
        assert response.status_code == 429
    finally:
        app.dependency_overrides.pop(get_rag_chain, None)


@pytest.mark.unit
def test_list_documents_empty(client):
    """GET /api/documents returns empty list when no uploads."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.unit
def test_list_documents_after_upload(client, temp_registry):
    """GET /api/documents returns uploaded documents."""
    doc = DocumentUpload(
        file_id="test-id-123",
        file_name="resume.pdf",
        file_type="resume",
        file_size=100,
        uploaded_at=datetime.now(timezone.utc),
        status="completed",
    )
    temp_registry.add(doc)

    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any(d["file_id"] == "test-id-123" for d in data)


@pytest.mark.unit
def test_delete_document_not_found(client):
    """DELETE /api/documents/{doc_id} returns 404 when document not found."""
    response = client.delete("/api/documents/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.unit
def test_delete_document_success(client, temp_registry, temp_uploads_dir):
    """DELETE /api/documents/{doc_id} removes document."""
    doc_id = "doc-to-delete"
    doc = DocumentUpload(
        file_id=doc_id,
        file_name="resume.pdf",
        file_type="resume",
        file_size=100,
        uploaded_at=datetime.now(timezone.utc),
        status="completed",
    )
    temp_registry.add(doc)
    (temp_uploads_dir / f"{doc_id}.pdf").write_bytes(b"dummy")

    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == f"Document {doc_id} deleted"


@pytest.mark.unit
def test_stats(client, mock_vector_store):
    """GET /api/stats returns system stats."""
    mock_vector_store.get_collection_stats.return_value = {
        "count": 42,
        "collection_name": "careerfit_chunks",
        "persist_directory": "/data/vectordb",
    }

    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert data["total_chunks"] == 42
    assert "vector_db" in data


@pytest.mark.unit
def test_health_simple(client):
    """GET /health returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.unit
def test_health_api_with_components(client):
    """GET /api/health returns status and components."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert data["status"] in ("ok", "degraded")


@pytest.mark.unit
def test_root_returns_html(client):
    """GET / returns HTML status page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Career Intelligence Assistant" in response.text
    assert "ok" in response.text.lower()
