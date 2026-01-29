"""Keyword-based evidence retrieval.

This module implements keyword/token matching search over local
evidence files. It provides a simpler alternative to embedding
search that works without API calls.
"""

from pathlib import Path

from delibera.retrieval.base import RetrievalResult


class KeywordRetriever:
    """Keyword-based search over local evidence files.

    Uses token matching to find relevant documents. Each document
    is scored by the sum of query token occurrences.

    This is a simple retriever that doesn't require API calls,
    useful as a fallback or for testing.
    """

    def __init__(self, evidence_dir: Path) -> None:
        """Initialize the keyword retriever.

        Args:
            evidence_dir: Directory containing evidence files (.txt, .md).
        """
        self._evidence_dir = evidence_dir

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        return [w.lower().strip("?.,!\"':;") for w in text.split() if len(w) > 1]

    def _score_document(self, doc_text: str, query_tokens: list[str]) -> int:
        """Score a document by counting query token occurrences."""
        doc_lower = doc_text.lower()
        return sum(doc_lower.count(token) for token in query_tokens)

    def _extract_snippet(self, text: str, query_tokens: list[str], max_length: int = 300) -> str:
        """Extract a snippet around the first query match."""
        text_lower = text.lower()

        # Find first matching token
        best_pos = len(text)
        for token in query_tokens:
            pos = text_lower.find(token)
            if 0 <= pos < best_pos:
                best_pos = pos

        if best_pos == len(text):
            # No match found, return start of document
            return text[:max_length] + ("..." if len(text) > max_length else "")

        # Extract context around match
        start = max(0, best_pos - 50)
        end = min(len(text), best_pos + max_length - 50)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant evidence using keyword matching.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of retrieval results sorted by relevance.
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Score all documents
        scored: list[tuple[str, str, int]] = []

        for path in self._evidence_dir.glob("**/*.txt"):
            text = path.read_text()
            score = self._score_document(text, query_tokens)
            if score > 0:
                rel_path = str(path.relative_to(self._evidence_dir))
                scored.append((rel_path, text, score))

        for path in self._evidence_dir.glob("**/*.md"):
            text = path.read_text()
            score = self._score_document(text, query_tokens)
            if score > 0:
                rel_path = str(path.relative_to(self._evidence_dir))
                scored.append((rel_path, text, score))

        # Sort by score (highest first)
        scored.sort(key=lambda x: -x[2])

        # Build results
        results: list[RetrievalResult] = []
        for rel_path, text, score in scored[:max_results]:
            snippet = self._extract_snippet(text, query_tokens)

            results.append(
                RetrievalResult(
                    source=rel_path,
                    excerpt=snippet,
                    score=float(score),
                    source_type="local",
                    method="keyword",
                )
            )

        return results
