"""Web-based evidence retrieval using Gemini Search grounding.

This module implements web search over the internet using
Gemini's built-in Google Search grounding capability.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from delibera.retrieval.base import RetrievalResult, WebSearchError

# Model that supports grounding
GROUNDING_MODEL = "gemini-2.0-flash"


def _get_api_key() -> str:
    """Get the Gemini API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise WebSearchError("GEMINI_API_KEY environment variable not set")
    return api_key


class WebRetriever:
    """Web search using Gemini's Google Search grounding.

    Uses the google_search_retrieval tool to find relevant web content.
    This provides real-time information from the web that may not be
    in local evidence files.

    Note: Web search grounding costs $35/1K queries.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the web retriever.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        self._api_key = api_key or _get_api_key()

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant evidence from web search.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of retrieval results from web sources.

        Raises:
            WebSearchError: If the web search fails.
        """
        # Build API URL
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GROUNDING_MODEL}:generateContent?key={self._api_key}"
        )

        # Build request with google_search tool (grounding)
        body: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"Find factual information and evidence about: {query}\n\n"
                                "Provide a brief summary of the most relevant findings."
                            )
                        }
                    ]
                }
            ],
            "tools": [{"google_search": {}}],
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise WebSearchError(f"Web search API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise WebSearchError(f"Connection failed: {e}") from e

        return self._parse_grounding_response(response_data, max_results)

    def _parse_grounding_response(
        self, response: dict[str, Any], max_results: int
    ) -> list[RetrievalResult]:
        """Parse grounding metadata from Gemini response.

        Args:
            response: The API response.
            max_results: Maximum results to return.

        Returns:
            List of retrieval results extracted from grounding chunks.
        """
        results: list[RetrievalResult] = []

        # Extract candidates
        candidates = response.get("candidates", [])
        if not candidates:
            return results

        candidate = candidates[0]

        # Extract grounding metadata
        grounding_metadata = candidate.get("groundingMetadata", {})

        # Get grounding chunks (web sources)
        grounding_chunks = grounding_metadata.get("groundingChunks", [])

        # Get grounding supports (links content to sources)
        grounding_supports = grounding_metadata.get("groundingSupports", [])

        # Build excerpt map from supports
        source_excerpts: dict[int, list[str]] = {}
        for support in grounding_supports:
            segment = support.get("segment", {})
            text = segment.get("text", "")
            indices = support.get("groundingChunkIndices", [])
            for idx in indices:
                if idx not in source_excerpts:
                    source_excerpts[idx] = []
                if text:
                    source_excerpts[idx].append(text)

        # Build results from chunks
        for i, chunk in enumerate(grounding_chunks[:max_results]):
            web_info = chunk.get("web", {})
            uri = web_info.get("uri", "")
            title = web_info.get("title", "")

            if not uri:
                continue

            # Get excerpt from supports or use title
            excerpts = source_excerpts.get(i, [])
            excerpt = " ".join(excerpts) if excerpts else title

            # Use rank-based scoring (first result = highest score)
            score = 1.0 - (i * 0.1)

            results.append(
                RetrievalResult(
                    source=uri,
                    excerpt=excerpt[:500] if excerpt else "",
                    score=max(score, 0.1),  # Minimum score of 0.1
                    source_type="web",
                    method="grounding",
                    metadata={"title": title},
                )
            )

        # If no grounding chunks but we have a response, extract from content
        if not results and candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if parts:
                text = parts[0].get("text", "")
                if text:
                    # Return a single result with the generated summary
                    results.append(
                        RetrievalResult(
                            source="web:gemini-summary",
                            excerpt=text[:500],
                            score=0.5,
                            source_type="web",
                            method="grounding",
                            metadata={"note": "No specific sources found"},
                        )
                    )

        return results
