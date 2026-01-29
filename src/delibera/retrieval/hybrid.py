"""Hybrid evidence retrieval combining multiple sources.

This module implements a hybrid retriever that combines results from
local (embedding/keyword) and web (grounding) search using
Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path
from typing import Literal

from delibera.retrieval.base import EvidenceRetriever, RetrievalResult


class HybridRetriever:
    """Combines multiple retrievers using Reciprocal Rank Fusion.

    RRF is a simple but effective method for combining ranked lists
    from different retrieval systems. It's robust to different
    score scales and doesn't require score normalization.

    The RRF score for a document is: sum(1 / (k + rank_i)) for each
    ranker i that returns the document. k is typically 60.
    """

    def __init__(
        self,
        retrievers: list[EvidenceRetriever],
        k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            retrievers: List of retrievers to combine.
            k: RRF constant (default 60, higher = more uniform weighting).
        """
        self._retrievers = retrievers
        self._k = k

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve and fuse results from all retrievers.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of fused retrieval results sorted by RRF score.
        """
        # Collect results from all retrievers
        all_results: dict[str, RetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        for retriever in self._retrievers:
            try:
                results = retriever.retrieve(query, max_results=max_results * 2)
            except Exception:
                # Skip failed retrievers
                continue

            for rank, result in enumerate(results):
                source = result.source

                # Update RRF score
                rrf_scores[source] = rrf_scores.get(source, 0) + 1 / (self._k + rank)

                # Keep the result with highest original score
                if source not in all_results or result.score > all_results[source].score:
                    all_results[source] = result

        # Sort by RRF score
        sorted_sources = sorted(rrf_scores.keys(), key=lambda s: -rrf_scores[s])

        # Build final results with updated scores
        final_results: list[RetrievalResult] = []
        for source in sorted_sources[:max_results]:
            original = all_results[source]
            final_results.append(
                RetrievalResult(
                    source=original.source,
                    excerpt=original.excerpt,
                    score=rrf_scores[source],  # Use RRF score
                    source_type=original.source_type,
                    method="hybrid",
                    metadata={
                        **original.metadata,
                        "original_method": original.method,
                        "original_score": original.score,
                    },
                )
            )

        return final_results


def create_retriever(
    method: Literal["embedding", "keyword", "web", "hybrid"],
    evidence_dir: Path | None = None,
    api_key: str | None = None,
) -> EvidenceRetriever:
    """Factory function to create a retriever by method name.

    Args:
        method: The retrieval method to use.
        evidence_dir: Directory containing local evidence files.
            Required for embedding, keyword, and hybrid methods.
        api_key: Gemini API key. If None, reads from environment.

    Returns:
        A retriever instance.

    Raises:
        ValueError: If method is invalid or required args are missing.
    """
    if method == "embedding":
        if evidence_dir is None:
            raise ValueError("evidence_dir required for embedding retriever")
        from delibera.retrieval.embedding import EmbeddingRetriever

        return EmbeddingRetriever(evidence_dir, api_key=api_key)

    elif method == "keyword":
        if evidence_dir is None:
            raise ValueError("evidence_dir required for keyword retriever")
        from delibera.retrieval.keyword import KeywordRetriever

        return KeywordRetriever(evidence_dir)

    elif method == "web":
        from delibera.retrieval.web import WebRetriever

        return WebRetriever(api_key=api_key)

    elif method == "hybrid":
        if evidence_dir is None:
            raise ValueError("evidence_dir required for hybrid retriever")
        from delibera.retrieval.embedding import EmbeddingRetriever
        from delibera.retrieval.web import WebRetriever

        return HybridRetriever(
            retrievers=[
                EmbeddingRetriever(evidence_dir, api_key=api_key),
                WebRetriever(api_key=api_key),
            ]
        )

    else:
        raise ValueError(f"Unknown retrieval method: {method}")
