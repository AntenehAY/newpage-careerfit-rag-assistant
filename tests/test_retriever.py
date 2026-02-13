"""Tests for the Retriever class: retrieval, MMR, context formatting, token limits."""

from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.retriever import (
    CHARS_PER_TOKEN,
    Retriever,
    _cosine_similarity,
    _estimate_tokens,
)


# --- Fixtures ---


@pytest.fixture
def mock_vector_store():
    """Vector store with mocked search and embedding_service."""
    store = MagicMock()
    store.embedding_service = MagicMock()
    # Default: embed_text returns a simple vector, embed_batch returns list of same
    dim = 4
    store.embedding_service.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]
    store.embedding_service.embed_batch.side_effect = (
        lambda texts: [[0.1 * (i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]
    )
    return store


@pytest.fixture
def sample_search_results():
    """Sample search results as returned by VectorStore.search."""
    return [
        {"id": "c1", "text": "Chunk one about Python.", "metadata": {"doc_id": "d1", "doc_type": "resume"}, "score": 0.9},
        {"id": "c2", "text": "Chunk two about Java.", "metadata": {"doc_id": "d1", "doc_type": "resume"}, "score": 0.8},
        {"id": "c3", "text": "Chunk three about SQL.", "metadata": {"doc_id": "d2", "doc_type": "job_description"}, "score": 0.75},
    ]


# --- Basic Retrieval ---


class TestRetrieverBasic:
    """Test basic retrieval behavior."""

    def test_init_uses_settings_top_k(self, mock_vector_store):
        """Retriever uses settings.top_k_results when top_k not provided."""
        with patch("app.retrieval.retriever.settings") as mock_settings:
            mock_settings.top_k_results = 7
            r = Retriever(mock_vector_store)
            assert r.top_k == 7

    def test_init_uses_provided_top_k(self, mock_vector_store):
        """Retriever uses provided top_k when given."""
        r = Retriever(mock_vector_store, top_k=10)
        assert r.top_k == 10

    def test_retrieve_for_query_calls_search(self, mock_vector_store, sample_search_results):
        """retrieve_for_query calls vector_store.search with query and top_k."""
        mock_vector_store.search.return_value = sample_search_results[:2]
        r = Retriever(mock_vector_store, top_k=5)
        results = r.retrieve_for_query("Python skills")
        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["query"] == "Python skills"
        assert call_kwargs["top_k"] >= 5  # May fetch more for MMR
        assert len(results) <= 5

    def test_retrieve_returns_empty_list_when_no_results(self, mock_vector_store):
        """retrieve_for_query returns [] when search returns no results."""
        mock_vector_store.search.return_value = []
        r = Retriever(mock_vector_store, top_k=5)
        results = r.retrieve_for_query("nonexistent")
        assert results == []


# --- Filters ---


class TestRetrieverFilters:
    """Test retrieval with doc_type and doc_id filters."""

    def test_retrieve_with_doc_type_filter(self, mock_vector_store, sample_search_results):
        """retrieve_for_query passes filter_doc_type to search."""
        filtered = [s for s in sample_search_results if s["metadata"]["doc_type"] == "resume"]
        mock_vector_store.search.return_value = filtered
        r = Retriever(mock_vector_store, top_k=5)
        r.retrieve_for_query("skills", filter_doc_type="resume")
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["filter_metadata"] == {"doc_type": "resume"}

    def test_retrieve_with_doc_id_filter(self, mock_vector_store, sample_search_results):
        """retrieve_for_query passes filter_doc_id to search."""
        mock_vector_store.search.return_value = sample_search_results[:1]
        r = Retriever(mock_vector_store, top_k=5)
        r.retrieve_for_query("Python", filter_doc_id="d1")
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["filter_metadata"] == {"doc_id": "d1"}

    def test_retrieve_with_both_filters(self, mock_vector_store, sample_search_results):
        """retrieve_for_query combines filter_doc_type and filter_doc_id."""
        mock_vector_store.search.return_value = []
        r = Retriever(mock_vector_store, top_k=5)
        r.retrieve_for_query("test", filter_doc_type="resume", filter_doc_id="d1")
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["filter_metadata"] == {"doc_type": "resume", "doc_id": "d1"}


# --- MMR Reranking ---


class TestRetrieverMMR:
    """Test MMR reranking behavior."""

    def test_apply_mmr_reorders_results(self, mock_vector_store, sample_search_results):
        """_apply_mmr produces reordered list (same items, different order when diversity matters)."""
        r = Retriever(mock_vector_store, top_k=5)
        # Use distinct embeddings so MMR can diversify
        mock_vector_store.embedding_service.embed_text.return_value = [1.0, 0.0, 0.0, 0.0]
        mock_vector_store.embedding_service.embed_batch.side_effect = (
            lambda texts: [
                [0.9, 0.1, 0.0, 0.0],
                [0.3, 0.9, 0.0, 0.0],
                [0.1, 0.1, 0.9, 0.0],
            ][: len(texts)]
        )
        reranked = r._apply_mmr(sample_search_results, query="skills", lambda_param=0.5)
        assert len(reranked) == 3
        assert set(c["id"] for c in reranked) == {"c1", "c2", "c3"}

    def test_apply_mmr_single_result_unchanged(self, mock_vector_store):
        """_apply_mmr with single result returns it unchanged."""
        r = Retriever(mock_vector_store)
        single = [{"id": "c1", "text": "Only chunk", "metadata": {}, "score": 0.9}]
        result = r._apply_mmr(single, query="test", lambda_param=0.5)
        assert result == single
        mock_vector_store.embedding_service.embed_text.assert_not_called()
        mock_vector_store.embedding_service.embed_batch.assert_not_called()

    def test_apply_mmr_empty_returns_empty(self, mock_vector_store):
        """_apply_mmr with empty list returns empty list."""
        r = Retriever(mock_vector_store)
        result = r._apply_mmr([], query="test", lambda_param=0.5)
        assert result == []


# --- Context Formatting ---


class TestRetrieverContextFormatting:
    """Test _format_context and get_context_for_llm output."""

    def test_format_context_structure(self, mock_vector_store, sample_search_results):
        """_format_context produces correct template: Document: X (Type: Y)\n{text}\n---"""
        r = Retriever(mock_vector_store)
        formatted = r._format_context(sample_search_results[:2])
        assert "Document:" in formatted
        assert "(Type: resume)" in formatted
        assert "Chunk one about Python." in formatted
        assert "Chunk two about Java." in formatted
        assert "---" in formatted

    def test_format_context_empty_returns_empty(self, mock_vector_store):
        """_format_context with empty list returns empty string."""
        r = Retriever(mock_vector_store)
        assert r._format_context([]) == ""

    def test_format_context_source_attribution(self, mock_vector_store):
        """_format_context includes doc_id and doc_type from metadata."""
        chunks = [
            {"id": "x", "text": "Content", "metadata": {"doc_id": "res_123", "doc_type": "resume"}, "score": 0.9},
        ]
        r = Retriever(mock_vector_store)
        formatted = r._format_context(chunks)
        assert "res_123" in formatted
        assert "resume" in formatted


# --- get_context_for_llm ---


class TestGetContextForLLM:
    """Test get_context_for_llm."""

    def test_get_context_returns_dict_with_keys(self, mock_vector_store, sample_search_results):
        """get_context_for_llm returns dict with context, sources, total_chunks."""
        mock_vector_store.search.return_value = sample_search_results[:2]
        r = Retriever(mock_vector_store, top_k=5)
        out = r.get_context_for_llm("skills")
        assert "context" in out
        assert "sources" in out
        assert "total_chunks" in out
        assert isinstance(out["context"], str)
        assert isinstance(out["sources"], list)
        assert out["total_chunks"] == 2

    def test_get_context_empty_results(self, mock_vector_store):
        """get_context_for_llm with no results returns empty context."""
        mock_vector_store.search.return_value = []
        r = Retriever(mock_vector_store)
        out = r.get_context_for_llm("nothing")
        assert out["context"] == ""
        assert out["sources"] == []
        assert out["total_chunks"] == 0

    def test_get_context_sources_have_required_fields(self, mock_vector_store, sample_search_results):
        """Sources in get_context_for_llm include doc_id, doc_name, doc_type, chunk_text, relevance_score."""
        mock_vector_store.search.return_value = sample_search_results[:1]
        r = Retriever(mock_vector_store)
        out = r.get_context_for_llm("Python")
        assert len(out["sources"]) == 1
        s = out["sources"][0]
        assert "doc_id" in s
        assert "doc_name" in s
        assert "doc_type" in s
        assert "chunk_text" in s
        assert "relevance_score" in s


# --- Token Limit Enforcement ---


class TestTokenLimit:
    """Test token limit enforcement."""

    def test_estimate_tokens(self):
        """_estimate_tokens uses ~4 chars per token."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("abcd") == 1  # 4 chars -> 1 token
        assert _estimate_tokens("a" * 40) == 10

    def test_get_context_respects_max_tokens(self, mock_vector_store):
        """get_context_for_llm truncates chunks to fit max_tokens."""
        # Create chunks that would exceed a small token limit
        long_chunk = {"id": "c1", "text": "x" * 2000, "metadata": {"doc_id": "d1", "doc_type": "resume"}, "score": 0.9}
        mock_vector_store.search.return_value = [long_chunk]
        r = Retriever(mock_vector_store)
        out = r.get_context_for_llm("test", max_tokens=100)
        # 100 tokens * 4 = 400 chars. One long chunk + header would exceed; should truncate
        estimated = _estimate_tokens(out["context"])
        assert estimated <= 100 + CHARS_PER_TOKEN  # Allow small overshoot from rounding

    def test_get_context_includes_chunks_within_limit(self, mock_vector_store, sample_search_results):
        """get_context_for_llm includes all chunks when under limit."""
        mock_vector_store.search.return_value = sample_search_results
        r = Retriever(mock_vector_store)
        out = r.get_context_for_llm("skills", max_tokens=4000)
        assert out["total_chunks"] == 3
        assert "Chunk one" in out["context"]
        assert "Chunk two" in out["context"]
        assert "Chunk three" in out["context"]


# --- Cosine Similarity ---


class TestCosineSimilarity:
    """Test _cosine_similarity helper."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_empty_returns_zero(self):
        """Empty vectors return 0."""
        assert _cosine_similarity([], []) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0
