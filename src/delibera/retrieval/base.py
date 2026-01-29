"""Base types and protocols for evidence retrieval.

This module defines the core abstractions for the multi-source
evidence retrieval system, including result types and the
retriever protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class RetrievalResult:
    """A single evidence retrieval result.

    Attributes:
        source: File path (local) or URL (web).
        excerpt: Relevant text snippet from the source.
        score: Relevance score (higher is more relevant).
        source_type: Whether the result is from local files or web.
        method: The retrieval method used (embedding, keyword, grounding).
        metadata: Additional metadata about the result.
    """

    source: str
    excerpt: str
    score: float
    source_type: Literal["local", "web"]
    method: Literal["embedding", "keyword", "grounding", "hybrid"]
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceRetriever(Protocol):
    """Protocol for evidence retrieval implementations.

    Retrievers search for relevant evidence given a query and return
    ranked results. Different implementations may use embeddings,
    keyword matching, web search, or combinations thereof.
    """

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant evidence for a query.

        Args:
            query: The search query (typically derived from question + context).
            max_results: Maximum number of results to return.

        Returns:
            List of retrieval results, sorted by relevance (highest first).
        """
        ...


class RetrieverError(Exception):
    """Base exception for retriever errors."""

    pass


class EmbeddingError(RetrieverError):
    """Error computing or using embeddings."""

    pass


class WebSearchError(RetrieverError):
    """Error performing web search."""

    pass


class IndexError(RetrieverError):
    """Error with the evidence index."""

    pass
