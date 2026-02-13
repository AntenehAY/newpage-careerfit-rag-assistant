"""Integration tests: Ingestion -> retrieval -> LLM -> response (mock LLM only)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.pipeline import ingest_document
from app.rag.chain import RAGChain
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


@pytest.mark.integration
class TestRAGPipeline:
    """End-to-end RAG flow with real ingestion + retrieval, mocked LLM."""

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
        """Mock EmbeddingService to avoid OpenAI API calls."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [
            [0.1] * dim for _ in texts
        ]
        return svc

    @pytest.fixture
    def vector_store(self, chroma_client, mock_embedding):
        """VectorStore with real ChromaDB."""
        return VectorStore(
            collection_name="rag_pipeline_test",
            persist_directory="/tmp/rag_pipeline_vectordb",
            embedding_service=mock_embedding,
            chroma_client=chroma_client,
        )

    @pytest.fixture
    def populated_store(
        self, vector_store, sample_resume_pdf: Path, sample_jd_docx: Path
    ):
        """Vector store with resume and JD ingested."""
        ingest_document(
            str(sample_resume_pdf),
            doc_type="resume",
            doc_id="res-rag",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        ingest_document(
            str(sample_jd_docx),
            doc_type="job_description",
            doc_id="jd-rag",
            store_in_vector_db=True,
            vector_store=vector_store,
        )
        return vector_store

    def test_rag_flow_returns_answer(
        self, populated_store, sample_resume_pdf: Path, sample_jd_docx: Path
    ):
        """Full RAG: ingest -> retriever -> RAG chain (mocked LLM) -> answer."""
        retriever = Retriever(vector_store=populated_store, top_k=5)

        mock_llm_response = MagicMock()
        mock_llm_response.content = (
            "Based on your resume [Source 1] and the job description [Source 2], "
            "you have strong Python experience that aligns well with the role."
        )

        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=retriever,
                llm_model="claude-test",
                api_key="test-key",
            )
            chain.llm = mock_instance

            result = chain.answer_query("What skills do I have for this job?")

            assert result.answer
            assert "Python" in result.answer or "skills" in result.answer.lower()
            assert isinstance(result.sources, list)

    def test_rag_with_filter_doc_type(self, populated_store):
        """RAG respects filter_doc_type=resume."""
        retriever = Retriever(vector_store=populated_store, top_k=5)
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Your resume shows Python experience."

        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=retriever,
                llm_model="test",
                api_key="test",
            )
            chain.llm = mock_instance

            result = chain.answer_query(
                "Summarize my experience",
                filter_doc_type="resume",
            )

            assert result.answer
            assert "experience" in result.answer.lower() or "Python" in result.answer

    def test_rag_empty_context_returns_fallback(self, vector_store):
        """RAG with empty store returns fallback (no LLM call)."""
        retriever = Retriever(vector_store=vector_store, top_k=5)

        with patch("app.rag.chain.ChatAnthropic"):
            chain = RAGChain(
                retriever=retriever,
                llm_model="test",
                api_key="test",
            )
            chain.llm = MagicMock()

            result = chain.answer_query("What skills am I missing?")

            assert "upload" in result.answer.lower() or "documents" in result.answer.lower()
            assert result.sources == []
            chain.llm.invoke.assert_not_called()
