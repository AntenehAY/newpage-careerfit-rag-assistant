"""Tests for RAG pipeline: prompts, query detection, chain, and response parsing."""

from unittest.mock import MagicMock, patch

import pytest

from app.models import QueryResponse, SourceReference
from app.rag.chain import RAGChain
from app.rag.prompts import (
    EXPERIENCE_ALIGNMENT_PROMPT,
    GENERAL_CAREER_PROMPT,
    INTERVIEW_PREP_PROMPT,
    SKILL_GAP_PROMPT,
    get_prompt_for_type,
)


# --- Fixtures ---


@pytest.fixture
def mock_retriever():
    """Retriever that returns controlled context and sources."""
    ret = MagicMock()
    ret.get_context_for_llm.return_value = {
        "context": """Document: resume_1 (Type: resume)
Candidate has 5 years Python, SQL, data analysis experience.
---
Document: jd_1 (Type: job_description)
Job requires: Python, 3+ years, teamwork, leadership.
---""",
        "sources": [
            {
                "doc_id": "res_1",
                "doc_name": "resume_1",
                "doc_type": "resume",
                "chunk_text": "Candidate has 5 years Python, SQL, data analysis experience.",
                "relevance_score": 0.92,
            },
            {
                "doc_id": "jd_1",
                "doc_name": "jd_1",
                "doc_type": "job_description",
                "chunk_text": "Job requires: Python, 3+ years, teamwork, leadership.",
                "relevance_score": 0.88,
            },
        ],
        "total_chunks": 2,
    }
    return ret


@pytest.fixture
def mock_llm_response():
    """Simulated LLM response with citations."""
    msg = MagicMock()
    msg.content = """Based on the documents [Source 1], [Source 2]:

**Skills the job requires:**
- Python, 3+ years, teamwork, leadership [Source 2]

**Skills you have:**
- 5 years Python, SQL, data analysis [Source 1]

**Gaps:** Leadership experience could be highlighted more explicitly.
"""
    return msg


@pytest.fixture
def empty_context_retriever():
    """Retriever that returns empty context."""
    ret = MagicMock()
    ret.get_context_for_llm.return_value = {
        "context": "",
        "sources": [],
        "total_chunks": 0,
    }
    return ret


# --- Prompt Template Formatting ---


class TestPromptTemplates:
    """Test prompt template formatting and structure."""

    def test_skill_gap_prompt_has_context_and_query(self):
        """SKILL_GAP_PROMPT accepts context and query variables."""
        msgs = SKILL_GAP_PROMPT.format_messages(
            context="Test context here.",
            query="What skills am I missing?",
        )
        assert len(msgs) >= 2
        full_text = " ".join(str(m) for m in msgs)
        assert "Test context here" in full_text
        assert "What skills am I missing?" in full_text

    def test_experience_alignment_prompt_formats(self):
        """EXPERIENCE_ALIGNMENT_PROMPT formats correctly."""
        msgs = EXPERIENCE_ALIGNMENT_PROMPT.format_messages(
            context="Resume and JD content.",
            query="How does my experience align?",
        )
        assert len(msgs) >= 2
        full_text = " ".join(str(m) for m in msgs)
        assert "Resume and JD content" in full_text
        assert "How does my experience align?" in full_text

    def test_interview_prep_prompt_formats(self):
        """INTERVIEW_PREP_PROMPT formats correctly."""
        msgs = INTERVIEW_PREP_PROMPT.format_messages(
            context="Document context.",
            query="What interview questions should I expect?",
        )
        assert len(msgs) >= 2
        full_text = " ".join(str(m) for m in msgs)
        assert "Document context" in full_text
        assert "interview questions" in full_text

    def test_general_career_prompt_formats(self):
        """GENERAL_CAREER_PROMPT formats correctly."""
        msgs = GENERAL_CAREER_PROMPT.format_messages(
            context="Some career context.",
            query="What strengths should I highlight?",
        )
        assert len(msgs) >= 2
        full_text = " ".join(str(m) for m in msgs)
        assert "Some career context" in full_text
        assert "strengths" in full_text

    def test_get_prompt_for_type_returns_correct_template(self):
        """get_prompt_for_type returns correct template per type."""
        assert get_prompt_for_type("skill_gap") == SKILL_GAP_PROMPT
        assert get_prompt_for_type("experience_alignment") == EXPERIENCE_ALIGNMENT_PROMPT
        assert get_prompt_for_type("interview_prep") == INTERVIEW_PREP_PROMPT
        assert get_prompt_for_type("general") == GENERAL_CAREER_PROMPT
        assert get_prompt_for_type("unknown") == GENERAL_CAREER_PROMPT


# --- Query Type Detection ---


class TestQueryTypeDetection:
    """Test _detect_query_type heuristics."""

    @pytest.fixture
    def chain_with_mock_llm(self, mock_retriever):
        """RAGChain with mocked LLM (no real API calls)."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            MockLLM.return_value = mock_instance
            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
                temperature=0.3,
            )
            chain.llm = mock_instance
            return chain

    def test_detect_skill_gap(self, chain_with_mock_llm):
        """Queries about skills/gaps map to skill_gap."""
        assert chain_with_mock_llm._detect_query_type("What skills am I missing?") == "skill_gap"
        assert chain_with_mock_llm._detect_query_type("skill gaps") == "skill_gap"
        assert chain_with_mock_llm._detect_query_type("qualifications required") == "skill_gap"

    def test_detect_experience_alignment(self, chain_with_mock_llm):
        """Queries about experience/alignment map to experience_alignment."""
        assert chain_with_mock_llm._detect_query_type("How does my experience align?") == "experience_alignment"
        assert chain_with_mock_llm._detect_query_type("Do I match the job?") == "experience_alignment"
        assert chain_with_mock_llm._detect_query_type("years of background") == "experience_alignment"

    def test_detect_interview_prep(self, chain_with_mock_llm):
        """Queries about interviews map to interview_prep."""
        assert chain_with_mock_llm._detect_query_type("What interview questions should I expect?") == "interview_prep"
        assert chain_with_mock_llm._detect_query_type("prepare for interview") == "interview_prep"
        assert chain_with_mock_llm._detect_query_type("talking points and weaknesses") == "interview_prep"

    def test_detect_general(self, chain_with_mock_llm):
        """Unrelated queries map to general."""
        assert chain_with_mock_llm._detect_query_type("Hello") == "general"
        assert chain_with_mock_llm._detect_query_type("Summarize my resume") == "general"


# --- Response Parsing ---


class TestResponseParsing:
    """Test _parse_response extraction and source mapping."""

    @pytest.fixture
    def chain_with_mock_llm(self, mock_retriever):
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
                temperature=0.3,
            )
            chain.llm = MagicMock()
            return chain

    def test_parse_extracts_answer(self, chain_with_mock_llm, mock_retriever):
        """_parse_response returns the full answer text."""
        out = chain_with_mock_llm._parse_response(
            "This is the answer [Source 1].",
            mock_retriever.get_context_for_llm.return_value["sources"],
        )
        assert out["answer"] == "This is the answer [Source 1]."
        assert "answer" in out
        assert "sources" in out

    def test_parse_maps_citations_to_sources(self, chain_with_mock_llm):
        """Cited sources are included as SourceReference objects."""
        sources_raw = [
            {"doc_id": "d1", "doc_name": "Doc1", "doc_type": "resume", "chunk_text": "Text1", "relevance_score": 0.9},
            {"doc_id": "d2", "doc_name": "Doc2", "doc_type": "jd", "chunk_text": "Text2", "relevance_score": 0.8},
        ]
        out = chain_with_mock_llm._parse_response("Answer [Source 2] and [Source 1].", sources_raw)
        refs = out["sources"]
        assert len(refs) == 2
        assert all(isinstance(r, SourceReference) for r in refs)
        doc_ids = {r.doc_id for r in refs}
        assert doc_ids == {"d1", "d2"}

    def test_parse_includes_confidence(self, chain_with_mock_llm):
        """_parse_response includes optional confidence."""
        sources_raw = [
            {"doc_id": "d1", "doc_name": "D1", "doc_type": "resume", "chunk_text": "T", "relevance_score": 0.9},
        ]
        out = chain_with_mock_llm._parse_response("Answer [Source 1].", sources_raw)
        assert out.get("confidence") is not None
        assert 0 <= out["confidence"] <= 1.0


# --- RAG Chain End-to-End ---


class TestRAGChain:
    """Test RAGChain answer_query with mocked retriever and LLM."""

    def test_answer_query_returns_query_response(self, mock_retriever, mock_llm_response):
        """answer_query returns QueryResponse with answer and sources."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
                temperature=0.3,
            )
            chain.llm = mock_instance

            result = chain.answer_query("What skills am I missing?")

            assert isinstance(result, QueryResponse)
            assert result.answer
            assert "Python" in result.answer or "skills" in result.answer.lower()
            assert isinstance(result.sources, list)
            assert result.generated_at is not None

    def test_answer_query_empty_context_returns_fallback(self, empty_context_retriever):
        """answer_query with empty context returns helpful fallback message."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            chain = RAGChain(
                retriever=empty_context_retriever,
                llm_model="claude-test",
                api_key="test-key",
                temperature=0.3,
            )
            chain.llm = MagicMock()

            result = chain.answer_query("Any question?")

            assert isinstance(result, QueryResponse)
            assert "uploaded" in result.answer.lower() or "documents" in result.answer.lower()
            assert result.sources == []
            chain.llm.invoke.assert_not_called()

    def test_answer_query_passes_filters_to_retriever(self, mock_retriever, mock_llm_response):
        """answer_query passes filter_doc_type and filter_doc_id to retriever."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
            )
            chain.llm = mock_instance

            chain.answer_query(
                "Skills?",
                filter_doc_type="resume",
                filter_doc_id="doc_123",
            )

            mock_retriever.get_context_for_llm.assert_called_once()
            call_kw = mock_retriever.get_context_for_llm.call_args[1]
            assert call_kw["filter_doc_type"] == "resume"
            assert call_kw["filter_doc_id"] == "doc_123"

    def test_query_response_structure(self, mock_retriever, mock_llm_response):
        """QueryResponse has required fields: answer, sources, generated_at."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
            )
            chain.llm = mock_instance

            result = chain.answer_query("Test query")

            assert hasattr(result, "answer")
            assert hasattr(result, "sources")
            assert hasattr(result, "generated_at")
            assert result.answer
            assert isinstance(result.sources, list)
            for s in result.sources:
                assert isinstance(s, SourceReference)
                assert s.doc_id
                assert s.doc_type
                assert s.chunk_text
                assert 0 <= s.relevance_score <= 1.0


# --- Streaming ---


class TestRAGStreaming:
    """Test answer_query_stream."""

    def test_stream_yields_text_chunks(self, mock_retriever):
        """answer_query_stream yields text chunks from LLM."""
        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.stream.return_value = iter([
                MagicMock(content="First "),
                MagicMock(content="chunk. "),
                MagicMock(content="Done."),
            ])
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="claude-test",
                api_key="test-key",
            )
            chain.llm = mock_instance

            chunks = list(chain.answer_query_stream("Stream this"))

            assert chunks == ["First ", "chunk. ", "Done."]

    def test_stream_empty_context_yields_fallback(self, empty_context_retriever):
        """answer_query_stream with empty context yields fallback message."""
        with patch("app.rag.chain.ChatAnthropic"):
            chain = RAGChain(
                retriever=empty_context_retriever,
                llm_model="claude-test",
                api_key="test-key",
            )
            chain.llm = MagicMock()

            chunks = list(chain.answer_query_stream("Anything"))

            assert len(chunks) == 1
            assert "uploaded" in chunks[0].lower() or "documents" in chunks[0].lower()
            chain.llm.stream.assert_not_called()
