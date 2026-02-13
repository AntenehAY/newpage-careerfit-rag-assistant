"""Tests for retrieval modules: embeddings and vector store."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models import ChunkMetadata
from app.retrieval.embeddings import (
    BATCH_SIZE,
    EmbeddingService,
    LOCAL_MODEL,
    OPENAI_MODEL,
)


@pytest.mark.unit
class TestEmbeddingServiceOpenAI:
    """Test EmbeddingService with OpenAI model (mocked)."""

    def test_init_openai(self):
        """EmbeddingService initializes for OpenAI model."""
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        assert svc.model_name == OPENAI_MODEL
        assert svc._is_openai is True

    def test_init_local(self):
        """EmbeddingService initializes for local model."""
        svc = EmbeddingService(model_name=LOCAL_MODEL, api_key="")
        assert svc.model_name == LOCAL_MODEL
        assert svc._is_openai is False

    @patch("openai.OpenAI")
    def test_embed_text_openai(self, mock_openai_class):
        """embed_text returns embedding vector via mocked OpenAI."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 384)],
            usage=MagicMock(total_tokens=10),
        )
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        result = svc.embed_text("Hello world")
        assert len(result) == 384
        assert result[0] == 0.1
        mock_client.embeddings.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_embed_text_caches_result(self, mock_openai_class):
        """embed_text caches and reuses result for same text."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.5] * 384)],
            usage=MagicMock(total_tokens=5),
        )
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        r1 = svc.embed_text("cached text", use_cache=True)
        r2 = svc.embed_text("cached text", use_cache=True)
        assert r1 == r2
        mock_client.embeddings.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_embed_batch_openai(self, mock_openai_class):
        """embed_batch processes texts and returns list of embeddings."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1] * 384),
                MagicMock(embedding=[0.2] * 384),
                MagicMock(embedding=[0.3] * 384),
            ],
            usage=MagicMock(total_tokens=30),
        )
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        texts = ["a", "b", "c"]
        result = svc.embed_batch(texts)
        assert len(result) == 3
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3
        mock_client.embeddings.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_embed_batch_processes_in_chunks(self, mock_openai_class):
        """embed_batch processes large lists in batches of 100."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        def make_resp(*args, **kwargs):
            batch = kwargs.get("input", args[1] if len(args) > 1 else [])
            return MagicMock(
                data=[MagicMock(embedding=[0.0] * 384) for _ in batch],
                usage=MagicMock(total_tokens=len(batch) * 10),
            )

        mock_client.embeddings.create.side_effect = make_resp
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        texts = ["x"] * (BATCH_SIZE + 50)
        result = svc.embed_batch(texts)
        assert len(result) == BATCH_SIZE + 50
        assert mock_client.embeddings.create.call_count == 2

    def test_embed_batch_empty_returns_empty(self):
        """embed_batch with empty list returns empty list without API call."""
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        result = svc.embed_batch([])
        assert result == []

    @patch("openai.OpenAI")
    def test_embed_text_empty_string(self, mock_openai_class):
        """embed_text with empty string returns empty or minimal embedding."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.0] * 384)],
            usage=MagicMock(total_tokens=0),
        )
        svc = EmbeddingService(model_name=OPENAI_MODEL, api_key="sk-test")
        result = svc.embed_text("")
        assert isinstance(result, list)
        assert len(result) == 384


@pytest.mark.unit
class TestEmbeddingServiceLocal:
    """Test EmbeddingService with local sentence-transformers."""

    @pytest.mark.skipif(
        True,
        reason="sentence-transformers optional; run with -k 'not Local' to skip",
    )
    def test_embed_text_local(self):
        """embed_text works with local model."""
        svc = EmbeddingService(model_name=LOCAL_MODEL, api_key="")
        result = svc.embed_text("test sentence")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.skipif(
        True,
        reason="sentence-transformers optional",
    )
    def test_embed_batch_local(self):
        """embed_batch works with local model."""
        svc = EmbeddingService(model_name=LOCAL_MODEL, api_key="")
        result = svc.embed_batch(["a", "b"])
        assert len(result) == 2
        assert len(result[0]) == len(result[1])


@pytest.mark.unit
class TestVectorStore:
    """Test VectorStore with EphemeralClient."""

    @pytest.fixture
    def chroma_client(self, tmp_path):
        """Use ChromaDB EphemeralClient for in-memory tests."""
        try:
            import chromadb

            return chromadb.EphemeralClient()
        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")

    @pytest.fixture
    def mock_embedding_service(self):
        """Embedding service that returns fixed-dim vectors."""
        svc = MagicMock(spec=EmbeddingService)
        dim = 384
        svc.embed_text.return_value = [0.0] * dim
        svc.embed_batch.side_effect = lambda texts: [[0.0] * dim for _ in texts]
        return svc

    @pytest.fixture
    def sample_chunks(self):
        """Sample ChunkMetadata for tests."""
        return [
            ChunkMetadata(
                chunk_id="doc1_chunk_0",
                doc_id="doc1",
                doc_type="resume",
                chunk_index=0,
                page_number=None,
                section="Experience",
                char_start=0,
                char_end=100,
            ),
            ChunkMetadata(
                chunk_id="doc1_chunk_1",
                doc_id="doc1",
                doc_type="resume",
                chunk_index=1,
                page_number=None,
                section="Skills",
                char_start=100,
                char_end=200,
            ),
        ]

    def test_vector_store_init(self, chroma_client, mock_embedding_service):
        """VectorStore initializes with EphemeralClient."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_col",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        assert store.collection_name == "test_col"
        assert store._collection is not None

    def test_add_documents(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """add_documents stores chunks with embeddings."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_add",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        texts = ["First chunk text.", "Second chunk text."]
        store.add_documents(sample_chunks, texts)
        mock_embedding_service.embed_batch.assert_called_once_with(texts)
        stats = store.get_collection_stats()
        assert stats["count"] == 2

    def test_search_basic(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """search returns similar chunks."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_search",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        texts = [
            "Python developer with 5 years experience.",
            "Java and SQL skills.",
        ]
        store.add_documents(sample_chunks, texts)
        results = store.search("programming skills", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_search_with_filter(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """search with filter_metadata filters results."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_search_filter",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        texts = ["Resume content.", "Job description content."]
        chunks = [
            ChunkMetadata(
                chunk_id="r_chunk_0",
                doc_id="res1",
                doc_type="resume",
                chunk_index=0,
                page_number=None,
                section=None,
                char_start=0,
                char_end=50,
            ),
            ChunkMetadata(
                chunk_id="jd_chunk_0",
                doc_id="jd1",
                doc_type="job_description",
                chunk_index=0,
                page_number=None,
                section=None,
                char_start=0,
                char_end=50,
            ),
        ]
        store.add_documents(chunks, texts)
        results = store.search(
            "skills",
            top_k=5,
            filter_metadata={"doc_type": "resume"},
        )
        assert all(r["metadata"].get("doc_type") == "resume" for r in results)

    def test_delete_by_doc_id(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """delete_by_doc_id removes chunks for document."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_delete",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        texts = ["Chunk one.", "Chunk two."]
        store.add_documents(sample_chunks, texts)
        assert store.get_collection_stats()["count"] == 2
        store.delete_by_doc_id("doc1")
        assert store.get_collection_stats()["count"] == 0

    def test_get_collection_stats(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """get_collection_stats returns count and info."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_stats",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        stats = store.get_collection_stats()
        assert "count" in stats
        assert "collection_name" in stats
        assert stats["count"] == 0
        store.add_documents(sample_chunks, ["a", "b"])
        stats = store.get_collection_stats()
        assert stats["count"] == 2

    def test_add_documents_length_mismatch_raises(
        self, chroma_client, mock_embedding_service, sample_chunks
    ):
        """add_documents raises when chunks and texts lengths differ."""
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_mismatch",
            persist_directory="/tmp/dummy",
            embedding_service=mock_embedding_service,
            chroma_client=chroma_client,
        )
        with pytest.raises(ValueError, match="must equal"):
            store.add_documents(sample_chunks, ["only one text"])
