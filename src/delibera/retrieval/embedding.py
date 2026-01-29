"""Embedding-based evidence retrieval using Gemini.

This module implements semantic search over local evidence files
using the Gemini embedding API (gemini-embedding-001).
"""

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from delibera.retrieval.base import EmbeddingError, RetrievalResult

# Gemini embedding model
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # Use smaller dimension for efficiency
MAX_INPUT_TOKENS = 2048

# Cache file location
CACHE_DIR = Path(".delibera/cache")
CACHE_FILE = CACHE_DIR / "embeddings.json"


def _get_api_key() -> str:
    """Get the Gemini API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY environment variable not set")
    return api_key


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


class EmbeddingRetriever:
    """Semantic search over local evidence using Gemini embeddings.

    This retriever:
    1. Pre-computes embeddings for all evidence files
    2. Caches embeddings to disk for performance
    3. Uses cosine similarity to find relevant documents

    Attributes:
        evidence_dir: Directory containing evidence files.
    """

    def __init__(
        self,
        evidence_dir: Path,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the embedding retriever.

        Args:
            evidence_dir: Directory containing evidence files (.txt, .md).
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            cache_dir: Directory for caching embeddings. Defaults to .delibera/cache.
        """
        self._evidence_dir = evidence_dir
        self._api_key = api_key or _get_api_key()
        self._cache_dir = cache_dir or CACHE_DIR
        self._index: dict[str, list[float]] = {}
        self._file_hashes: dict[str, str] = {}

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using Gemini API.

        Args:
            text: Text to embed (truncated to MAX_INPUT_TOKENS).

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If API call fails.
        """
        # Truncate text to fit token limit (rough approximation: 4 chars per token)
        max_chars = MAX_INPUT_TOKENS * 4
        truncated = text[:max_chars]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EMBEDDING_MODEL}:embedContent?key={self._api_key}"
        )

        body = {
            "content": {"parts": [{"text": truncated}]},
            "outputDimensionality": EMBEDDING_DIMENSIONS,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise EmbeddingError(f"Embedding API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise EmbeddingError(f"Connection failed: {e}") from e

        try:
            return response_data["embedding"]["values"]  # type: ignore[no-any-return]
        except KeyError as e:
            raise EmbeddingError(f"Invalid API response: {response_data}") from e

    def _load_cache(self) -> dict[str, Any]:
        """Load embedding cache from disk."""
        if not CACHE_FILE.exists():
            return {"model": EMBEDDING_MODEL, "files": {}}

        try:
            return json.loads(CACHE_FILE.read_text())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            return {"model": EMBEDDING_MODEL, "files": {}}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        """Save embedding cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache))

    def index_evidence(self) -> int:
        """Pre-compute embeddings for all evidence files.

        Reads from cache when available, only re-embeds changed files.

        Returns:
            Number of files indexed.
        """
        cache = self._load_cache()

        # Check if cache uses same model
        if cache.get("model") != EMBEDDING_MODEL:
            cache = {"model": EMBEDDING_MODEL, "files": {}}

        files_indexed = 0
        for path in self._evidence_dir.glob("**/*.txt"):
            rel_path = str(path.relative_to(self._evidence_dir))
            file_hash = _file_hash(path)

            # Check cache
            cached = cache.get("files", {}).get(rel_path, {})
            if cached.get("hash") == file_hash and "embedding" in cached:
                # Use cached embedding
                self._index[rel_path] = cached["embedding"]
                self._file_hashes[rel_path] = file_hash
            else:
                # Compute new embedding
                text = path.read_text()
                embedding = self._get_embedding(text)
                self._index[rel_path] = embedding
                self._file_hashes[rel_path] = file_hash

                # Update cache
                if "files" not in cache:
                    cache["files"] = {}
                cache["files"][rel_path] = {
                    "hash": file_hash,
                    "embedding": embedding,
                }

            files_indexed += 1

        # Also index .md files
        for path in self._evidence_dir.glob("**/*.md"):
            rel_path = str(path.relative_to(self._evidence_dir))
            file_hash = _file_hash(path)

            cached = cache.get("files", {}).get(rel_path, {})
            if cached.get("hash") == file_hash and "embedding" in cached:
                self._index[rel_path] = cached["embedding"]
                self._file_hashes[rel_path] = file_hash
            else:
                text = path.read_text()
                embedding = self._get_embedding(text)
                self._index[rel_path] = embedding
                self._file_hashes[rel_path] = file_hash

                if "files" not in cache:
                    cache["files"] = {}
                cache["files"][rel_path] = {
                    "hash": file_hash,
                    "embedding": embedding,
                }

            files_indexed += 1

        # Save updated cache
        self._save_cache(cache)

        return files_indexed

    def _extract_excerpt(self, path: Path, query: str, max_length: int = 300) -> str:
        """Extract relevant excerpt from a file.

        Uses a simple keyword-based approach to find the most relevant section.

        Args:
            path: Path to the file.
            query: The search query.
            max_length: Maximum excerpt length.

        Returns:
            Excerpt string.
        """
        text = path.read_text()

        # Find best matching section using query keywords
        query_words = [w.lower().strip("?.,!") for w in query.split() if len(w) > 2]

        lines = text.split("\n")
        best_score = -1
        best_start = 0

        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for w in query_words if w in line_lower)
            if score > best_score:
                best_score = score
                best_start = i

        # Extract context around best line
        context_lines = lines[max(0, best_start - 1) : best_start + 3]
        excerpt = "\n".join(context_lines)

        if len(excerpt) > max_length:
            excerpt = excerpt[:max_length] + "..."

        return excerpt.strip()

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant evidence using embedding similarity.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of retrieval results sorted by relevance.
        """
        if not self._index:
            self.index_evidence()

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Score all documents
        scored: list[tuple[str, float]] = []
        for rel_path, doc_embedding in self._index.items():
            score = _cosine_similarity(query_embedding, doc_embedding)
            scored.append((rel_path, score))

        # Sort by score (highest first)
        scored.sort(key=lambda x: -x[1])

        # Build results
        results: list[RetrievalResult] = []
        for rel_path, score in scored[:max_results]:
            full_path = self._evidence_dir / rel_path
            excerpt = self._extract_excerpt(full_path, query)

            results.append(
                RetrievalResult(
                    source=rel_path,
                    excerpt=excerpt,
                    score=score,
                    source_type="local",
                    method="embedding",
                )
            )

        return results
