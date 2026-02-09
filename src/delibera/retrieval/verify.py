"""Evidence verification implementations.

This module provides verifiers for checking the accuracy and trustworthiness
of retrieved evidence, particularly from web sources.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from delibera.retrieval.base import (
    RetrievalResult,
    VerificationError,
    VerificationResult,
)


def _get_api_key() -> str:
    """Get the Gemini API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise VerificationError("GEMINI_API_KEY environment variable not set")
    return api_key


class FetchVerifier:
    """Verifies web results by fetching the actual URL content.

    For Gemini grounding URLs, this follows the redirect to get the
    actual source URL and fetches content to verify the excerpt matches.
    """

    def __init__(self, timeout: int = 30) -> None:
        """Initialize the fetch verifier.

        Args:
            timeout: Request timeout in seconds.
        """
        self._timeout = timeout

    def verify(self, result: RetrievalResult) -> VerificationResult:
        """Verify a result by fetching its source URL.

        Args:
            result: The retrieval result to verify.

        Returns:
            Verification result with actual URL and content.
        """
        if result.source_type != "web":
            # Local files don't need URL verification
            return VerificationResult(
                original=result,
                verified=True,
                confidence=1.0,
                method="fetch",
                details={"reason": "local_file"},
            )

        try:
            actual_url, content = self._fetch_with_redirect(result.source)

            # Check if excerpt appears in content
            excerpt_lower = result.excerpt.lower()[:100]  # First 100 chars
            content_lower = content.lower()

            # Simple containment check
            excerpt_found = excerpt_lower in content_lower

            # Calculate confidence based on match
            if excerpt_found:
                confidence = 0.9
                verified = True
            else:
                # Partial match - check for key terms
                terms = excerpt_lower.split()[:5]
                matches = sum(1 for t in terms if t in content_lower)
                confidence = matches / max(len(terms), 1) * 0.7
                verified = confidence > 0.3

            return VerificationResult(
                original=result,
                verified=verified,
                confidence=confidence,
                method="fetch",
                details={
                    "excerpt_found": excerpt_found,
                    "content_length": len(content),
                },
                actual_url=actual_url,
                actual_content=content[:2000] if content else None,
            )

        except Exception as e:
            return VerificationResult(
                original=result,
                verified=False,
                confidence=0.0,
                method="fetch",
                details={"error": str(e)},
            )

    def verify_batch(self, results: list[RetrievalResult]) -> list[VerificationResult]:
        """Verify multiple results."""
        return [self.verify(r) for r in results]

    def _fetch_with_redirect(self, url: str) -> tuple[str, str]:
        """Fetch URL content, following redirects.

        Args:
            url: The URL to fetch.

        Returns:
            Tuple of (final_url, content).
        """
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; Delibera/1.0; "
                    "+https://github.com/forge-labs-dev/delibera)"
                )
            },
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            final_url = resp.geturl()
            content = resp.read().decode("utf-8", errors="replace")

        return final_url, content


class LLMVerifier:
    """Verifies evidence using an LLM to fact-check claims.

    Uses Gemini to analyze whether the excerpt actually supports
    the claims being made in the deliberation.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the LLM verifier.

        Args:
            api_key: Gemini API key. If None, reads from environment.
        """
        self._api_key = api_key or _get_api_key()

    def verify(
        self,
        result: RetrievalResult,
        claim: str | None = None,
    ) -> VerificationResult:
        """Verify a result using LLM fact-checking.

        Args:
            result: The retrieval result to verify.
            claim: Optional claim to verify against the evidence.

        Returns:
            Verification result with LLM analysis.
        """
        prompt = self._build_prompt(result, claim)

        try:
            response = self._call_llm(prompt)
            analysis = self._parse_response(response)

            return VerificationResult(
                original=result,
                verified=analysis["verified"],
                confidence=analysis["confidence"],
                method="llm_check",
                details={
                    "reasoning": analysis["reasoning"],
                    "claim_checked": claim,
                },
            )
        except Exception as e:
            return VerificationResult(
                original=result,
                verified=False,
                confidence=0.0,
                method="llm_check",
                details={"error": str(e)},
            )

    def verify_batch(self, results: list[RetrievalResult]) -> list[VerificationResult]:
        """Verify multiple results."""
        return [self.verify(r) for r in results]

    def _build_prompt(self, result: RetrievalResult, claim: str | None) -> str:
        """Build the verification prompt."""
        base = f"""Analyze this evidence excerpt for accuracy and reliability.

Source: {result.source}
Excerpt: {result.excerpt}

"""
        if claim:
            base += f"""Claim to verify: {claim}

Does this evidence support the claim? """

        base += """Respond in JSON format:
{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""
        return base

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the Gemini API."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={self._api_key}"
        )

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return result

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse the LLM response."""
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]

            # Extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return {
                    "verified": bool(data.get("verified", False)),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": str(data.get("reasoning", "")),
                }
        except (KeyError, json.JSONDecodeError, ValueError):
            pass

        return {"verified": False, "confidence": 0.0, "reasoning": "Parse error"}


class CrossReferenceVerifier:
    """Verifies evidence by cross-referencing with additional searches.

    Performs multiple web searches to check if the same information
    appears consistently across different sources.
    """

    def __init__(self, api_key: str | None = None, min_sources: int = 2) -> None:
        """Initialize the cross-reference verifier.

        Args:
            api_key: Gemini API key for web search.
            min_sources: Minimum sources needed for verification.
        """
        self._api_key = api_key or _get_api_key()
        self._min_sources = min_sources

    def verify(self, result: RetrievalResult) -> VerificationResult:
        """Verify by searching for corroborating sources.

        Args:
            result: The retrieval result to verify.

        Returns:
            Verification result with cross-reference details.
        """
        # Extract key facts from excerpt to search for
        key_terms = self._extract_key_terms(result.excerpt)
        if not key_terms:
            return VerificationResult(
                original=result,
                verified=False,
                confidence=0.0,
                method="cross_reference",
                details={"reason": "no_key_terms"},
            )

        try:
            # Search for corroborating sources
            query = " ".join(key_terms[:5])
            sources = self._search_for_corroboration(query)

            # Count unique domains that mention similar content
            matching_sources = self._count_matching_sources(sources, result.excerpt)

            verified = matching_sources >= self._min_sources
            confidence = min(matching_sources / 3, 1.0)  # Cap at 1.0

            return VerificationResult(
                original=result,
                verified=verified,
                confidence=confidence,
                method="cross_reference",
                details={
                    "matching_sources": matching_sources,
                    "search_query": query,
                },
            )
        except Exception as e:
            return VerificationResult(
                original=result,
                verified=False,
                confidence=0.0,
                method="cross_reference",
                details={"error": str(e)},
            )

    def verify_batch(self, results: list[RetrievalResult]) -> list[VerificationResult]:
        """Verify multiple results."""
        return [self.verify(r) for r in results]

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for searching."""
        # Simple extraction - remove common words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "to",
            "from",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "of",
            "as",
            "it",
            "its",
        }

        words = text.lower().split()
        key_terms = [
            w.strip(".,!?\"'()[]{}:;") for w in words if w.lower() not in stop_words and len(w) > 3
        ]
        return key_terms[:10]

    def _search_for_corroboration(self, query: str) -> list[dict[str, Any]]:
        """Search for sources that might corroborate the information."""
        from delibera.retrieval.web import WebRetriever

        retriever = WebRetriever(api_key=self._api_key)
        results = retriever.retrieve(query, max_results=5)

        return [{"source": r.source, "excerpt": r.excerpt} for r in results]

    def _count_matching_sources(self, sources: list[dict[str, Any]], original_excerpt: str) -> int:
        """Count sources that contain similar information."""
        original_terms = set(self._extract_key_terms(original_excerpt))
        if not original_terms:
            return 0

        matching = 0
        for source in sources:
            source_terms = set(self._extract_key_terms(source["excerpt"]))
            overlap = len(original_terms & source_terms)
            if overlap >= 2:  # At least 2 matching key terms
                matching += 1

        return matching


def create_verifier(
    method: str,
    api_key: str | None = None,
) -> FetchVerifier | LLMVerifier | CrossReferenceVerifier:
    """Factory function to create a verifier by method name.

    Args:
        method: The verification method (fetch, llm, cross_reference).
        api_key: Optional API key for LLM-based verifiers.

    Returns:
        A verifier instance.

    Raises:
        ValueError: If method is unknown.
    """
    if method == "fetch":
        return FetchVerifier()
    elif method == "llm":
        return LLMVerifier(api_key=api_key)
    elif method == "cross_reference":
        return CrossReferenceVerifier(api_key=api_key)
    else:
        raise ValueError(f"Unknown verification method: {method}")
