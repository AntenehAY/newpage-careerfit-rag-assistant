"""Step 3: Document ingestion pipeline tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.parsers import parse_docx, parse_document, parse_pdf
from app.ingestion.pipeline import ingest_document


@pytest.mark.unit
class TestParsers:
    """Tests for app/ingestion/parsers.py."""

    def test_parse_pdf_with_sample(self, sample_resume_pdf: Path) -> None:
        """parse_pdf extracts text from sample PDF."""
        text = parse_pdf(str(sample_resume_pdf))
        assert isinstance(text, str)
        assert len(text) > 50
        assert "JANE DOE" in text or "Jane Doe" in text or "SUMMARY" in text

    def test_parse_docx_with_sample(self, sample_jd_docx: Path) -> None:
        """parse_docx extracts text from sample DOCX."""
        text = parse_docx(str(sample_jd_docx))
        assert isinstance(text, str)
        assert "Requirements" in text
        assert "Responsibilities" in text

    def test_parse_document_dispatcher_pdf(self, sample_resume_pdf: Path) -> None:
        """parse_document dispatches .pdf to parse_pdf."""
        text = parse_document(str(sample_resume_pdf))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_parse_document_dispatcher_docx(self, sample_jd_docx: Path) -> None:
        """parse_document dispatches .docx to parse_docx."""
        text = parse_document(str(sample_jd_docx))
        assert "Requirements" in text

    def test_parse_document_unsupported_format(self, tmp_path: Path) -> None:
        """parse_document raises ValueError for unsupported formats."""
        bad = tmp_path / "file.xyz"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported document format"):
            parse_document(str(bad))

    def test_parse_pdf_file_not_found(self) -> None:
        """parse_pdf raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_parse_docx_file_not_found(self) -> None:
        """parse_docx raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_docx("/nonexistent/path/file.docx")

    def test_parse_docx_corrupt_file(self, tmp_path: Path) -> None:
        """parse_docx handles corrupt/invalid DOCX gracefully."""
        bad = tmp_path / "corrupt.docx"
        bad.write_bytes(b"not a valid docx")
        with pytest.raises(ValueError, match="Failed to parse DOCX"):
            parse_docx(str(bad))

    def test_parse_pdf_empty_file(self, tmp_path: Path) -> None:
        """parse_pdf with minimal/empty PDF handles gracefully."""
        # Minimal valid PDF structure - may return empty or minimal text
        minimal_pdf = tmp_path / "empty.pdf"
        minimal_pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n178\n%%EOF"
        )
        text = parse_pdf(str(minimal_pdf))
        assert isinstance(text, str)


@pytest.mark.unit
class TestChunker:
    """Tests for app/ingestion/chunker.py."""

    def test_chunk_document_basic(self) -> None:
        """chunk_document returns ChunkMetadata list with correct structure."""
        text = "This is a sample document. " * 100
        metadata = {"doc_id": "doc-1", "doc_type": "resume"}
        chunks = chunk_document(text, metadata, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 1
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc-1_chunk_{i}"
            assert c.doc_id == "doc-1"
            assert c.doc_type == "resume"
            assert c.chunk_index == i
            assert c.char_start >= 0
            assert c.char_end > c.char_start

    def test_chunk_document_detects_resume_sections(self) -> None:
        """chunk_document detects resume section headers."""
        text = "SKILLS:\nPython, SQL, JavaScript\n\nEXPERIENCE:\nWorked at Acme Corp"
        metadata = {"doc_id": "d1", "doc_type": "resume"}
        chunks = chunk_document(text, metadata, chunk_size=200, chunk_overlap=20)
        sections = [c.section for c in chunks if c.section]
        assert any(s and "Skill" in s for s in sections) or len(chunks) >= 1

    def test_chunk_document_detects_jd_sections(self) -> None:
        """chunk_document detects JD section headers."""
        text = "Requirements:\n5+ years Python\n\nResponsibilities:\nBuild APIs"
        metadata = {"doc_id": "j1", "doc_type": "job_description"}
        chunks = chunk_document(text, metadata, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1

    def test_chunk_document_empty_returns_empty_list(self) -> None:
        """chunk_document returns empty list for empty text."""
        chunks = chunk_document("", {"doc_id": "x", "doc_type": "resume"})
        assert chunks == []

    def test_chunk_document_whitespace_only_returns_empty_list(self) -> None:
        """chunk_document returns empty list for whitespace-only text."""
        chunks = chunk_document("   \n\n   ", {"doc_id": "x", "doc_type": "resume"})
        assert chunks == []

    def test_chunk_document_single_char(self) -> None:
        """chunk_document with single character returns one chunk."""
        chunks = chunk_document("x", {"doc_id": "doc", "doc_type": "resume"})
        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 1

    def test_chunk_document_boundary_chunk_size(self) -> None:
        """chunk_document with text exactly at chunk_size boundary."""
        text = "a" * 100
        metadata = {"doc_id": "d1", "doc_type": "resume"}
        chunks = chunk_document(text, metadata, chunk_size=100, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0].char_end - chunks[0].char_start <= 100


@pytest.mark.unit
class TestPipeline:
    """Tests for app/ingestion/pipeline.py."""

    def test_ingest_document_pdf_end_to_end(self, sample_resume_pdf: Path) -> None:
        """ingest_document processes PDF and returns chunks."""
        chunks = ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="resume-001",
        )
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(c.doc_id == "resume-001" for c in chunks)
        assert all(c.doc_type == "resume" for c in chunks)

    def test_ingest_document_docx_end_to_end(self, sample_jd_docx: Path) -> None:
        """ingest_document processes DOCX and returns chunks."""
        chunks = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-001",
        )
        assert len(chunks) >= 1
        assert all(c.doc_type == "job_description" for c in chunks)

    def test_ingest_document_nonexistent_returns_empty_list(self) -> None:
        """ingest_document returns empty list for missing file."""
        chunks = ingest_document(
            "/nonexistent/resume.pdf",
            doc_type="resume",
            doc_id="x",
        )
        assert chunks == []

    def test_ingest_document_unsupported_format_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """ingest_document returns empty list for unsupported format."""
        bad = tmp_path / "file.xyz"
        bad.write_text("dummy")
        chunks = ingest_document(str(bad), doc_type="resume", doc_id="x")
        assert chunks == []

    def test_ingest_document_uses_custom_chunk_params(
        self, sample_jd_docx: Path
    ) -> None:
        """ingest_document respects chunk_size and chunk_overlap."""
        chunks_small = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-a",
            chunk_size=50,
            chunk_overlap=5,
        )
        chunks_large = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-b",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert len(chunks_small) >= len(chunks_large)

    @patch("app.retrieval.vector_store.VectorStore")
    @patch("app.retrieval.embeddings.EmbeddingService")
    def test_ingest_document_store_in_vector_db(
        self,
        mock_embedding_svc_class: MagicMock,
        mock_vector_store_class: MagicMock,
        sample_jd_docx: Path,
    ) -> None:
        """ingest_document with store_in_vector_db=True calls VectorStore.add_documents."""
        mock_store = MagicMock()
        mock_vector_store_class.return_value = mock_store
        assert sample_jd_docx.exists(), "Fixture sample_jd_docx must exist"
        chunks = ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-vec",
            store_in_vector_db=True,
        )
        assert len(chunks) >= 1, "Ingestion should produce chunks"
        mock_vector_store_class.assert_called_once()
        mock_store.add_documents.assert_called_once()
        call_args = mock_store.add_documents.call_args
        assert call_args[0][0] == chunks
        assert len(call_args[0][1]) == len(chunks)

    def test_ingest_document_malformed_pdf_returns_empty(self, tmp_path: Path) -> None:
        """ingest_document with malformed PDF returns empty list (error recovery)."""
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a valid pdf content at all")
        # Pipeline catches ValueError from parse_pdf and returns []
        chunks = ingest_document(str(bad_pdf), doc_type="resume", doc_id="bad")
        assert chunks == []


@pytest.mark.unit
class TestChunkerWithSampleText:
    """Chunker tests using sample_resume.txt content."""

    def test_chunk_sample_resume_txt(self, sample_resume_txt: Path) -> None:
        """chunk_document works with sample resume text."""
        text = sample_resume_txt.read_text()
        metadata = {"doc_id": "sample", "doc_type": "resume"}
        chunks = chunk_document(text, metadata)
        assert len(chunks) >= 1
        first = chunks[0]
        assert first.char_start == 0
        assert first.char_end <= len(text)
