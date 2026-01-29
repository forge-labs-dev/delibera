"""Tests for the evidence retrieval system."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from delibera.retrieval import (
    EmbeddingRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievalResult,
    WebRetriever,
    create_retriever,
)
from delibera.retrieval.base import EmbeddingError, WebSearchError


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_local_result(self) -> None:
        """Test creating a local retrieval result."""
        result = RetrievalResult(
            source="evidence/doc.txt",
            excerpt="This is a test excerpt.",
            score=0.85,
            source_type="local",
            method="embedding",
        )

        assert result.source == "evidence/doc.txt"
        assert result.excerpt == "This is a test excerpt."
        assert result.score == 0.85
        assert result.source_type == "local"
        assert result.method == "embedding"
        assert result.metadata == {}

    def test_create_web_result(self) -> None:
        """Test creating a web retrieval result."""
        result = RetrievalResult(
            source="https://example.com/article",
            excerpt="Web article content.",
            score=0.9,
            source_type="web",
            method="grounding",
            metadata={"title": "Example Article"},
        )

        assert result.source == "https://example.com/article"
        assert result.source_type == "web"
        assert result.method == "grounding"
        assert result.metadata["title"] == "Example Article"


class TestKeywordRetriever:
    """Tests for KeywordRetriever."""

    @pytest.fixture
    def evidence_dir(self, tmp_path: Path) -> Path:
        """Create a temp evidence directory with test files."""
        evidence = tmp_path / "evidence"
        evidence.mkdir()

        (evidence / "uv_notes.txt").write_text(
            "UV is a fast Python package manager.\n"
            "It provides reproducible builds and is very fast.\n"
            "UV is compatible with pip requirements files."
        )

        (evidence / "graphql_guide.txt").write_text(
            "GraphQL is a query language for APIs.\n"
            "It provides a complete description of the data in your API.\n"
            "GraphQL offers efficient data fetching."
        )

        (evidence / "rest_api.md").write_text(
            "REST is an architectural style for APIs.\n"
            "REST uses standard HTTP methods.\n"
            "RESTful APIs are stateless."
        )

        return evidence

    def test_retrieve_finds_relevant_docs(self, evidence_dir: Path) -> None:
        """Test that keyword retrieval finds relevant documents."""
        retriever = KeywordRetriever(evidence_dir)

        results = retriever.retrieve("fast Python package manager")

        assert len(results) > 0
        # UV file should be found since it contains "fast" and "Python"
        sources = [r.source for r in results]
        assert any("uv_notes" in s for s in sources)

    def test_retrieve_graphql_query(self, evidence_dir: Path) -> None:
        """Test retrieval for GraphQL-related query."""
        retriever = KeywordRetriever(evidence_dir)

        results = retriever.retrieve("GraphQL API query language")

        assert len(results) > 0
        # GraphQL guide should rank high
        assert results[0].source == "graphql_guide.txt"
        assert results[0].source_type == "local"
        assert results[0].method == "keyword"

    def test_retrieve_no_matches(self, evidence_dir: Path) -> None:
        """Test retrieval with no matching documents."""
        retriever = KeywordRetriever(evidence_dir)

        results = retriever.retrieve("xyzabc123 nonexistent term")

        assert len(results) == 0

    def test_retrieve_max_results(self, evidence_dir: Path) -> None:
        """Test max_results parameter."""
        retriever = KeywordRetriever(evidence_dir)

        results = retriever.retrieve("API", max_results=1)

        assert len(results) <= 1

    def test_empty_query(self, evidence_dir: Path) -> None:
        """Test empty query returns no results."""
        retriever = KeywordRetriever(evidence_dir)

        results = retriever.retrieve("")

        assert len(results) == 0

    def test_tokenize(self, evidence_dir: Path) -> None:
        """Test tokenization method."""
        retriever = KeywordRetriever(evidence_dir)

        tokens = retriever._tokenize("Hello, World! How are you?")

        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens
        # Single char tokens should be excluded
        assert "i" not in retriever._tokenize("I am")


class TestEmbeddingRetriever:
    """Tests for EmbeddingRetriever with mocked API."""

    @pytest.fixture
    def evidence_dir(self, tmp_path: Path) -> Path:
        """Create a temp evidence directory."""
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        (evidence / "test.txt").write_text("Test content for embedding.")
        return evidence

    @pytest.fixture
    def mock_embedding_api(self) -> MagicMock:
        """Create a mock for the embedding API."""
        with patch("urllib.request.urlopen") as mock:
            # Return a mock embedding
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(
                {"embedding": {"values": [0.1] * 768}}
            ).encode("utf-8")
            mock_response.__enter__ = lambda s: mock_response
            mock_response.__exit__ = MagicMock(return_value=False)
            mock.return_value = mock_response
            yield mock

    def test_init_without_api_key(self, evidence_dir: Path) -> None:
        """Test that init fails without API key."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(EmbeddingError, match="GEMINI_API_KEY"),
        ):
            EmbeddingRetriever(evidence_dir)

    def test_init_with_api_key(self, evidence_dir: Path) -> None:
        """Test init with API key."""
        retriever = EmbeddingRetriever(evidence_dir, api_key="test-key")
        assert retriever._api_key == "test-key"

    def test_get_embedding(self, evidence_dir: Path, mock_embedding_api: MagicMock) -> None:
        """Test getting embedding from API."""
        retriever = EmbeddingRetriever(evidence_dir, api_key="test-key")

        embedding = retriever._get_embedding("test text")

        assert len(embedding) == 768
        assert all(v == 0.1 for v in embedding)

    def test_index_evidence(self, evidence_dir: Path, mock_embedding_api: MagicMock) -> None:
        """Test indexing evidence files."""
        retriever = EmbeddingRetriever(evidence_dir, api_key="test-key")

        count = retriever.index_evidence()

        assert count == 1
        assert len(retriever._index) == 1

    def test_retrieve(self, evidence_dir: Path, mock_embedding_api: MagicMock) -> None:
        """Test retrieval with mocked embeddings."""
        retriever = EmbeddingRetriever(evidence_dir, api_key="test-key")

        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].source_type == "local"
        assert results[0].method == "embedding"


class TestWebRetriever:
    """Tests for WebRetriever with mocked API."""

    @pytest.fixture
    def mock_grounding_api(self) -> MagicMock:
        """Create a mock for the grounding API."""
        with patch("urllib.request.urlopen") as mock:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Summary text"}]},
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {
                                        "web": {
                                            "uri": "https://example.com/1",
                                            "title": "Example 1",
                                        }
                                    },
                                    {
                                        "web": {
                                            "uri": "https://example.com/2",
                                            "title": "Example 2",
                                        }
                                    },
                                ],
                                "groundingSupports": [
                                    {
                                        "segment": {"text": "Excerpt from source 1"},
                                        "groundingChunkIndices": [0],
                                    },
                                    {
                                        "segment": {"text": "Excerpt from source 2"},
                                        "groundingChunkIndices": [1],
                                    },
                                ],
                            },
                        }
                    ]
                }
            ).encode("utf-8")
            mock_response.__enter__ = lambda s: mock_response
            mock_response.__exit__ = MagicMock(return_value=False)
            mock.return_value = mock_response
            yield mock

    def test_init_without_api_key(self) -> None:
        """Test that init fails without API key."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(WebSearchError, match="GEMINI_API_KEY"),
        ):
            WebRetriever()

    def test_init_with_api_key(self) -> None:
        """Test init with API key."""
        retriever = WebRetriever(api_key="test-key")
        assert retriever._api_key == "test-key"

    def test_retrieve(self, mock_grounding_api: MagicMock) -> None:
        """Test web retrieval with mocked API."""
        retriever = WebRetriever(api_key="test-key")

        results = retriever.retrieve("What is Python?")

        assert len(results) == 2
        assert results[0].source == "https://example.com/1"
        assert results[0].source_type == "web"
        assert results[0].method == "grounding"
        assert results[0].metadata["title"] == "Example 1"

    def test_retrieve_max_results(self, mock_grounding_api: MagicMock) -> None:
        """Test max_results limits output."""
        retriever = WebRetriever(api_key="test-key")

        results = retriever.retrieve("test query", max_results=1)

        assert len(results) == 1


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_rrf_fusion(self) -> None:
        """Test that RRF fusion combines results correctly."""
        # Create mock retrievers
        mock_local = MagicMock()
        mock_local.retrieve.return_value = [
            RetrievalResult(
                source="local/doc1.txt",
                excerpt="Local doc 1",
                score=0.9,
                source_type="local",
                method="embedding",
            ),
            RetrievalResult(
                source="local/doc2.txt",
                excerpt="Local doc 2",
                score=0.7,
                source_type="local",
                method="embedding",
            ),
        ]

        mock_web = MagicMock()
        mock_web.retrieve.return_value = [
            RetrievalResult(
                source="https://example.com/1",
                excerpt="Web doc 1",
                score=0.95,
                source_type="web",
                method="grounding",
            ),
            RetrievalResult(
                source="local/doc1.txt",  # Same as local
                excerpt="Web reference to local",
                score=0.8,
                source_type="web",
                method="grounding",
            ),
        ]

        retriever = HybridRetriever([mock_local, mock_web])

        results = retriever.retrieve("test query", max_results=3)

        # local/doc1.txt should rank highest (found by both)
        assert len(results) <= 3
        assert results[0].source == "local/doc1.txt"
        assert results[0].method == "hybrid"
        assert "original_method" in results[0].metadata

    def test_handles_retriever_failures(self) -> None:
        """Test that failed retrievers are skipped."""
        mock_working = MagicMock()
        mock_working.retrieve.return_value = [
            RetrievalResult(
                source="doc.txt",
                excerpt="Working result",
                score=0.9,
                source_type="local",
                method="keyword",
            ),
        ]

        mock_failing = MagicMock()
        mock_failing.retrieve.side_effect = Exception("API error")

        retriever = HybridRetriever([mock_failing, mock_working])

        # Should still return results from working retriever
        results = retriever.retrieve("test")

        assert len(results) == 1
        assert results[0].source == "doc.txt"


class TestCreateRetriever:
    """Tests for the create_retriever factory function."""

    @pytest.fixture
    def evidence_dir(self, tmp_path: Path) -> Path:
        """Create a temp evidence directory."""
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        (evidence / "test.txt").write_text("Test content.")
        return evidence

    def test_create_keyword_retriever(self, evidence_dir: Path) -> None:
        """Test creating a keyword retriever."""
        retriever = create_retriever("keyword", evidence_dir=evidence_dir)

        assert isinstance(retriever, KeywordRetriever)

    def test_create_embedding_retriever(self, evidence_dir: Path) -> None:
        """Test creating an embedding retriever."""
        retriever = create_retriever("embedding", evidence_dir=evidence_dir, api_key="test-key")

        assert isinstance(retriever, EmbeddingRetriever)

    def test_create_web_retriever(self) -> None:
        """Test creating a web retriever."""
        retriever = create_retriever("web", api_key="test-key")

        assert isinstance(retriever, WebRetriever)

    def test_create_hybrid_retriever(self, evidence_dir: Path) -> None:
        """Test creating a hybrid retriever."""
        retriever = create_retriever("hybrid", evidence_dir=evidence_dir, api_key="test-key")

        assert isinstance(retriever, HybridRetriever)

    def test_missing_evidence_dir(self) -> None:
        """Test error when evidence_dir is required but missing."""
        with pytest.raises(ValueError, match="evidence_dir required"):
            create_retriever("embedding", api_key="test-key")

    def test_unknown_method(self, evidence_dir: Path) -> None:
        """Test error for unknown method."""
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            create_retriever("unknown", evidence_dir=evidence_dir)  # type: ignore
