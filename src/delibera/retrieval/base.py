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


class VerificationError(RetrieverError):
    """Error during verification."""

    pass


@dataclass
class VerificationResult:
    """Result of verifying a retrieval result.

    Attributes:
        original: The original retrieval result being verified.
        verified: Whether the result passed verification.
        confidence: Confidence score (0.0-1.0) in the verification.
        method: The verification method used.
        details: Additional verification details.
        actual_url: The resolved URL (if redirect was followed).
        actual_content: Content fetched from the actual source (if available).
    """

    original: RetrievalResult
    verified: bool
    confidence: float
    method: Literal["fetch", "cross_reference", "llm_check"]
    details: dict[str, Any] = field(default_factory=dict)
    actual_url: str | None = None
    actual_content: str | None = None


class EvidenceVerifier(Protocol):
    """Protocol for evidence verification implementations.

    Verifiers check if retrieval results are accurate and trustworthy.
    Different implementations may fetch actual content, cross-reference
    sources, or use LLMs for fact-checking.
    """

    def verify(
        self,
        result: RetrievalResult,
    ) -> VerificationResult:
        """Verify a single retrieval result.

        Args:
            result: The retrieval result to verify.

        Returns:
            Verification result with confidence and details.
        """
        ...

    def verify_batch(
        self,
        results: list[RetrievalResult],
    ) -> list[VerificationResult]:
        """Verify multiple retrieval results.

        Args:
            results: List of retrieval results to verify.

        Returns:
            List of verification results in the same order.
        """
        ...
