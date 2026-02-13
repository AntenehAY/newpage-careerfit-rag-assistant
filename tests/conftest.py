"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --- Path fixtures ---


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to tests/fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_resume_pdf(fixtures_dir: Path) -> Path:
    """Ensure sample_resume.pdf exists; create if missing."""
    path = fixtures_dir / "sample_resume.pdf"
    if not path.exists():
        from tests.create_fixtures import create_sample_pdf

        create_sample_pdf()
    return path


@pytest.fixture(scope="session")
def sample_jd_docx(fixtures_dir: Path) -> Path:
    """Ensure sample_jd.docx exists; create if missing."""
    path = fixtures_dir / "sample_jd.docx"
    if not path.exists():
        from tests.create_fixtures import create_sample_docx

        create_sample_docx()
    return path


@pytest.fixture
def sample_resume_txt(fixtures_dir: Path) -> Path:
    """Path to sample_resume.txt."""
    return fixtures_dir / "sample_resume.txt"


# --- Temporary file management ---


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Minimal valid PDF bytes for upload tests."""
    return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"


# --- ChromaDB / Vector Store fixtures ---


@pytest.fixture
def chroma_ephemeral_client():
    """ChromaDB EphemeralClient for isolated tests."""
    try:
        import chromadb

        return chromadb.EphemeralClient()
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")


@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService returning fixed-dim vectors."""
    from app.retrieval.embeddings import EmbeddingService

    svc = MagicMock(spec=EmbeddingService)
    dim = 384
    svc.embed_text.return_value = [0.0] * dim
    svc.embed_batch.side_effect = lambda texts: [[0.0] * dim for _ in texts]
    return svc


@pytest.fixture
def temp_vector_store(chroma_ephemeral_client, mock_embedding_service):
    """VectorStore with temp ChromaDB, auto-cleanup."""
    from app.retrieval.vector_store import VectorStore

    store = VectorStore(
        collection_name="test_careerfit",
        persist_directory="/tmp/test_vectordb",
        embedding_service=mock_embedding_service,
        chroma_client=chroma_ephemeral_client,
    )
    yield store
    # Cleanup: delete collection or reset
    try:
        store.delete_by_doc_id("__cleanup__")
    except Exception:
        pass


# --- API client fixture (for integration/e2e) ---


@pytest.fixture
def api_client():
    """FastAPI TestClient for API tests."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


# --- DocumentRegistry fixture ---


@pytest.fixture
def temp_registry(tmp_path):
    """DocumentRegistry with temporary storage."""
    from app.api.document_registry import DocumentRegistry

    registry_path = tmp_path / "document_registry.json"
    return DocumentRegistry(registry_path=registry_path)


# --- Sample job description text (for fixtures) ---


@pytest.fixture
def sample_jd_text() -> str:
    """Sample job description text for creating fixtures."""
    return """Senior Backend Engineer

Requirements:
- 5+ years of Python experience
- Experience with AWS or GCP
- Strong SQL and NoSQL knowledge

Responsibilities:
- Design and implement RESTful APIs
- Collaborate with product and frontend teams
- Mentor junior engineers
"""
