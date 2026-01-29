"""Multi-source evidence retrieval system.

This module provides pluggable evidence retrieval with support for:
- Embedding-based semantic search (local files)
- Keyword-based search (local files, no API needed)
- Web search via Gemini's Google Search grounding
- Hybrid search combining local and web results

Example usage:
    from delibera.retrieval import create_retriever

    # Create a hybrid retriever (local + web)
    retriever = create_retriever(
        method="hybrid",
        evidence_dir=Path("evidence/"),
    )

    # Retrieve evidence for a query
    results = retriever.retrieve("What are the benefits of GraphQL?")
    for r in results:
        print(f"{r.source}: {r.excerpt[:100]}...")
"""

from delibera.retrieval.base import (
    EmbeddingError,
    EvidenceRetriever,
    EvidenceVerifier,
    IndexError,
    RetrievalResult,
    RetrieverError,
    VerificationError,
    VerificationResult,
    WebSearchError,
)
from delibera.retrieval.embedding import EmbeddingRetriever
from delibera.retrieval.hybrid import HybridRetriever, create_retriever
from delibera.retrieval.keyword import KeywordRetriever
from delibera.retrieval.verify import (
    CrossReferenceVerifier,
    FetchVerifier,
    LLMVerifier,
    create_verifier,
)
from delibera.retrieval.web import WebRetriever

__all__ = [
    # Base types
    "RetrievalResult",
    "EvidenceRetriever",
    "VerificationResult",
    "EvidenceVerifier",
    # Errors
    "RetrieverError",
    "EmbeddingError",
    "WebSearchError",
    "IndexError",
    "VerificationError",
    # Retrievers
    "EmbeddingRetriever",
    "KeywordRetriever",
    "WebRetriever",
    "HybridRetriever",
    # Verifiers
    "FetchVerifier",
    "LLMVerifier",
    "CrossReferenceVerifier",
    # Factories
    "create_retriever",
    "create_verifier",
]
