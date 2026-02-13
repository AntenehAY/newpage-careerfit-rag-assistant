"""Step 2 verification: test app/config.py and app/models.py."""

from datetime import datetime

import pytest

from app.config import settings
from app.models import (
    ChunkMetadata,
    DocumentUpload,
    QueryRequest,
    QueryResponse,
    SourceReference,
)


@pytest.mark.unit
class TestConfig:
    """Verify app/config.py loads and validates correctly."""

    def test_settings_singleton(self):
        """settings is a valid singleton instance."""
        from app.config import settings as s2

        assert settings is s2

    def test_required_api_keys_present(self):
        """ANTHROPIC_API_KEY and OPENAI_API_KEY are loaded."""
        assert settings.anthropic_api_key
        assert settings.openai_api_key
        assert len(settings.anthropic_api_key) > 10
        assert len(settings.openai_api_key) > 10

    def test_llm_defaults(self):
        """LLM defaults match spec."""
        assert settings.llm_model_name == "claude-sonnet-4-20250514"
        assert settings.embedding_model_name == "text-embedding-3-small"
        assert settings.llm_temperature == 0.3
        assert settings.llm_max_tokens == 2000

    def test_chunking_defaults(self):
        """Chunking defaults match spec."""
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50

    def test_rag_defaults(self):
        """RAG defaults match spec."""
        assert settings.top_k_results == 5
        assert settings.vector_db_path == "./data/vectordb"

    def test_app_metadata(self):
        """App metadata defaults match spec."""
        assert settings.app_name == "Career Intelligence Assistant"
        assert settings.log_level == "INFO"


@pytest.mark.unit
class TestModels:
    """Verify app/models.py schemas work correctly."""

    def test_document_upload(self):
        """DocumentUpload model validates and serializes."""
        doc = DocumentUpload(
            file_id="abc-123",
            file_name="resume.pdf",
            file_type="resume",
            file_size=1024,
            uploaded_at=datetime(2025, 2, 13, 12, 0, 0),
            status="completed",
        )
        assert doc.file_id == "abc-123"
        assert doc.file_type == "resume"
        data = doc.model_dump()
        assert data["file_name"] == "resume.pdf"

    def test_chunk_metadata(self):
        """ChunkMetadata model with optional fields."""
        chunk = ChunkMetadata(
            chunk_id="ch-1",
            doc_id="doc-1",
            doc_type="job_description",
            chunk_index=0,
            page_number=1,
            section="Requirements",
            char_start=0,
            char_end=500,
        )
        assert chunk.section == "Requirements"
        assert chunk.page_number == 1

    def test_chunk_metadata_minimal(self):
        """ChunkMetadata without optional fields."""
        chunk = ChunkMetadata(
            chunk_id="ch-2",
            doc_id="doc-2",
            doc_type="resume",
            chunk_index=1,
            char_start=100,
            char_end=600,
        )
        assert chunk.page_number is None
        assert chunk.section is None

    def test_query_request(self):
        """QueryRequest validates constraints."""
        req = QueryRequest(
            query="What skills are required?",
            filter_doc_type="job_description",
            max_results=10,
        )
        assert req.query == "What skills are required?"
        assert req.max_results == 10

    def test_query_request_max_length(self):
        """QueryRequest rejects query over 500 chars."""
        with pytest.raises(ValueError):
            QueryRequest(query="x" * 501)

    def test_source_reference(self):
        """SourceReference validates relevance_score range."""
        src = SourceReference(
            doc_id="d1",
            doc_name="Job Posting.pdf",
            doc_type="job_description",
            chunk_text="5+ years Python experience",
            relevance_score=0.92,
        )
        assert src.relevance_score == 0.92

    def test_query_response_roundtrip(self):
        """QueryResponse with all fields serializes correctly."""
        resp = QueryResponse(
            answer="The resume highlights Python and SQL.",
            sources=[
                SourceReference(
                    doc_id="r1",
                    doc_name="resume.pdf",
                    doc_type="resume",
                    chunk_text="Skills: Python, SQL",
                    relevance_score=0.88,
                ),
            ],
            confidence=0.9,
            generated_at=datetime.now(),
        )
        data = resp.model_dump()
        assert "answer" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["doc_name"] == "resume.pdf"
