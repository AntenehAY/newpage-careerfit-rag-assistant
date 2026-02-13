"""Tests for guardrails: input/output validation, rate limiting, content safety."""

from unittest.mock import MagicMock, patch

import pytest

from app.models import QueryResponse
from app.rag.chain import RAGChain
from app.rag.guardrails import (
    RateLimiter,
    check_content_safety,
    get_fallback_response,
    validate_query,
    validate_response,
)


# --- Query Validation ---


class TestValidateQuery:
    """Test validate_query function."""

    def test_valid_query(self):
        """Valid queries pass validation."""
        result = validate_query("What skills am I missing for this role?")
        assert result["valid"] is True
        assert result["reason"] is None
        assert result["sanitized_query"] == "What skills am I missing for this role?"

    def test_valid_query_min_length(self):
        """Query with exactly 3 chars passes."""
        result = validate_query("Why")
        assert result["valid"] is True
        assert result["sanitized_query"] == "Why"

    def test_query_too_short(self):
        """Query with < 3 chars fails."""
        result = validate_query("Hi")
        assert result["valid"] is False
        assert result["reason"] == "Query too short"
        assert result["sanitized_query"] == "Hi"

    def test_query_too_short_single_char(self):
        """Single character query fails."""
        result = validate_query("?")
        assert result["valid"] is False
        assert result["reason"] == "Query too short"

    def test_query_too_long(self):
        """Query over 500 chars fails."""
        long_query = "a" * 501
        result = validate_query(long_query)
        assert result["valid"] is False
        assert result["reason"] == "Query too long"
        assert len(result["sanitized_query"]) == 500

    def test_query_exactly_max_length(self):
        """Query with exactly 500 chars passes."""
        result = validate_query("a" * 500)
        assert result["valid"] is True

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        result = validate_query("  What skills?  ")
        assert result["valid"] is True
        assert result["sanitized_query"] == "What skills?"

    def test_injection_attempt_sql(self):
        """SQL injection patterns are rejected."""
        result = validate_query("What skills; DROP TABLE users;")
        assert result["valid"] is False
        assert result["reason"] == "Inappropriate content detected"

    def test_injection_attempt_union(self):
        """UNION SELECT injection is rejected."""
        result = validate_query("skills UNION SELECT * FROM users")
        assert result["valid"] is False

    def test_injection_attempt_script(self):
        """Script tag injection is rejected."""
        result = validate_query("What skills <script>alert(1)</script>")
        assert result["valid"] is False

    def test_non_string_input(self):
        """Non-string input fails with Invalid input."""
        result = validate_query(123)  # type: ignore
        assert result["valid"] is False
        assert result["reason"] == "Invalid input"
        assert result["sanitized_query"] == ""


# --- Response Validation ---


class TestValidateResponse:
    """Test validate_response function."""

    def test_valid_response(self):
        """Valid response with citations passes."""
        response = "Based on the documents [Source 1], you have strong Python skills."
        sources = [
            {"doc_id": "d1", "doc_name": "Resume", "doc_type": "resume", "chunk_text": "x", "relevance_score": 0.9},
        ]
        result = validate_response(response, sources)
        assert result["valid"] is True
        assert result["safe_response"] == response

    def test_empty_response(self):
        """Empty response fails."""
        result = validate_response("", [])
        assert result["valid"] is False
        assert result["reason"] == "Invalid response"
        assert result["safe_response"] == ""

    def test_whitespace_only_response(self):
        """Whitespace-only response fails."""
        result = validate_response("   \n\t  ", [])
        assert result["valid"] is False

    def test_non_string_response(self):
        """Non-string response fails."""
        result = validate_response(123, [])  # type: ignore
        assert result["valid"] is False
        assert result["reason"] == "Invalid response"

    def test_response_with_inappropriate_content(self):
        """Response with profanity fails."""
        response = "You have great skills. piece of shit extra"
        sources = [{"doc_id": "d1", "doc_name": "x", "doc_type": "resume", "chunk_text": "x", "relevance_score": 0.9}]
        result = validate_response(response, sources)
        assert result["valid"] is False


# --- Rate Limiting ---


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_allows_requests_within_limit(self):
        """Requests within limit are allowed."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check_rate_limit("user1") is True
        assert limiter.check_rate_limit("user1") is True
        assert limiter.check_rate_limit("user1") is True

    def test_blocks_when_limit_exceeded(self):
        """Fourth request in window is blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check_rate_limit("u1") is True
        assert limiter.check_rate_limit("u1") is True
        assert limiter.check_rate_limit("u1") is True
        assert limiter.check_rate_limit("u1") is False

    def test_per_user_limits(self):
        """Different users have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check_rate_limit("user_a") is True
        assert limiter.check_rate_limit("user_a") is True
        assert limiter.check_rate_limit("user_a") is False
        assert limiter.check_rate_limit("user_b") is True
        assert limiter.check_rate_limit("user_b") is True
        assert limiter.check_rate_limit("user_b") is False

    def test_default_user_id(self):
        """Works with default user_id."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check_rate_limit() is True
        assert limiter.check_rate_limit() is False

    def test_reset_clears_user(self):
        """Reset clears count for specified user."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check_rate_limit("u1") is True
        assert limiter.check_rate_limit("u1") is False
        limiter.reset("u1")
        assert limiter.check_rate_limit("u1") is True

    def test_reset_all_clears_everyone(self):
        """Reset with no user clears all."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check_rate_limit("a") is True
        assert limiter.check_rate_limit("b") is True
        assert limiter.check_rate_limit("a") is False
        limiter.reset()
        assert limiter.check_rate_limit("a") is True


# --- Content Safety ---


class TestCheckContentSafety:
    """Test check_content_safety function."""

    def test_empty_text_safe(self):
        """Empty text is safe."""
        result = check_content_safety("")
        assert result["safe"] is True
        assert result["issues"] == []

    def test_normal_text_safe(self):
        """Normal career-related text is safe."""
        result = check_content_safety("What skills does this job require?")
        assert result["safe"] is True

    def test_profanity_detected(self):
        """Profanity triggers issues."""
        result = check_content_safety("This is piece of shit")
        assert result["safe"] is False
        assert "inappropriate_content" in result["issues"]

    def test_injection_detected(self):
        """SQL injection patterns trigger issues."""
        result = check_content_safety("SELECT * FROM users")
        assert result["safe"] is False
        assert "suspicious_pattern" in result["issues"]


# --- Fallback Responses ---


class TestGetFallbackResponse:
    """Test get_fallback_response function."""

    def test_returns_query_response(self):
        """Returns proper QueryResponse."""
        resp = get_fallback_response("Query too long")
        assert isinstance(resp, QueryResponse)
        assert resp.answer
        assert resp.sources == []
        assert resp.confidence == 0.0
        assert resp.generated_at is not None

    def test_query_too_long_message(self):
        """Query too long has appropriate message."""
        resp = get_fallback_response("Query too long")
        assert "500" in resp.answer or "limit" in resp.answer.lower()

    def test_query_too_short_message(self):
        """Query too short has appropriate message."""
        resp = get_fallback_response("Query too short")
        assert "3" in resp.answer or "characters" in resp.answer.lower()

    def test_inappropriate_content_message(self):
        """Inappropriate content has professional message."""
        resp = get_fallback_response("Inappropriate content detected")
        assert "rephrase" in resp.answer.lower() or "professional" in resp.answer.lower()

    def test_rate_limit_message(self):
        """Rate limit has appropriate message."""
        resp = get_fallback_response("Rate limit exceeded")
        assert "try again" in resp.answer.lower() or "wait" in resp.answer.lower()

    def test_no_relevant_info_message(self):
        """No relevant info suggests uploading documents."""
        resp = get_fallback_response("No relevant information found")
        assert "upload" in resp.answer.lower() or "documents" in resp.answer.lower()

    def test_unknown_reason_uses_invalid_input(self):
        """Unknown reason falls back to Invalid input message."""
        resp = get_fallback_response("SomeUnknownReason")
        assert resp.answer
        assert "rephras" in resp.answer.lower() or "invalid" in resp.answer.lower()


# --- Integration with RAGChain ---


class TestGuardrailsIntegration:
    """Test guardrails integrated with RAGChain."""

    @pytest.fixture
    def mock_retriever(self):
        """Retriever returning valid context."""
        ret = MagicMock()
        ret.get_context_for_llm.return_value = {
            "context": "Resume: 5 years Python. JD: Requires Python.",
            "sources": [
                {"doc_id": "d1", "doc_name": "Resume", "doc_type": "resume", "chunk_text": "x", "relevance_score": 0.9},
            ],
            "total_chunks": 1,
        }
        return ret

    @pytest.fixture
    def mock_llm_response(self):
        """Valid LLM response."""
        msg = MagicMock()
        msg.content = "You have strong Python experience [Source 1]."
        return msg

    def test_invalid_query_returns_fallback_without_retrieval(self, mock_retriever):
        """Too short query returns fallback, retriever not called."""
        with patch("app.rag.chain.ChatAnthropic"):
            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="test",
                api_key="test",
            )
            chain.llm = MagicMock()

            result = chain.answer_query("Hi")

            mock_retriever.get_context_for_llm.assert_not_called()
            assert isinstance(result, QueryResponse)
            assert "3" in result.answer or "characters" in result.answer.lower()
            assert result.sources == []

    def test_injection_query_returns_fallback(self, mock_retriever):
        """Injection attempt returns fallback without calling retriever."""
        with patch("app.rag.chain.ChatAnthropic"):
            chain = RAGChain(retriever=mock_retriever, llm_model="test", api_key="test")
            chain.llm = MagicMock()

            result = chain.answer_query("skills; DROP TABLE users")

            mock_retriever.get_context_for_llm.assert_not_called()
            assert isinstance(result, QueryResponse)
            assert result.sources == []

    def test_rate_limit_returns_fallback(self, mock_retriever, mock_llm_response):
        """When rate limited, returns fallback without processing."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check_rate_limit("u1")  # Use up the one allowed request

        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="test",
                api_key="test",
                rate_limiter=limiter,
            )
            chain.llm = mock_instance

            result = chain.answer_query("What skills do I have?", user_id="u1")

            mock_retriever.get_context_for_llm.assert_not_called()
            assert "try again" in result.answer.lower() or "wait" in result.answer.lower()

    def test_valid_query_with_rate_limiter_succeeds(self, mock_retriever, mock_llm_response):
        """Valid query with rate limiter (under limit) succeeds."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_llm_response
            MockLLM.return_value = mock_instance

            chain = RAGChain(
                retriever=mock_retriever,
                llm_model="test",
                api_key="test",
                rate_limiter=limiter,
            )
            chain.llm = mock_instance

            result = chain.answer_query("What skills am I missing?")

            mock_retriever.get_context_for_llm.assert_called_once()
            assert result.answer
            assert "Python" in result.answer

    def test_output_validation_rejects_unsafe_response(self, mock_retriever):
        """When LLM returns inappropriate content, fallback is returned."""
        bad_msg = MagicMock()
        bad_msg.content = "You have great skills. piece of shit and more"

        with patch("app.rag.chain.ChatAnthropic") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = bad_msg
            MockLLM.return_value = mock_instance

            chain = RAGChain(retriever=mock_retriever, llm_model="test", api_key="test")
            chain.llm = mock_instance

            result = chain.answer_query("What skills do I have?")

            # Output validation should catch the profanity and return fallback
            assert isinstance(result, QueryResponse)
            assert "try again" in result.answer.lower() or "issue" in result.answer.lower()

    def test_stream_invalid_query_yields_fallback(self, mock_retriever):
        """Stream with invalid query yields fallback message."""
        with patch("app.rag.chain.ChatAnthropic"):
            chain = RAGChain(retriever=mock_retriever, llm_model="test", api_key="test")
            chain.llm = MagicMock()

            chunks = list(chain.answer_query_stream("x"))

            mock_retriever.get_context_for_llm.assert_not_called()
            assert len(chunks) == 1
            assert "3" in chunks[0] or "characters" in chunks[0].lower()
