"""Local docs tools for reading and searching evidence files.

This module provides tools for working with local evidence packs:
- docs.read: Read files from an allowlisted directory
- docs.search: Search for relevant files/snippets within an evidence pack

Both tools operate within a configured evidence_root directory (evidence pack).
"""

from pathlib import Path
from typing import Any

from delibera.tools.spec import RiskLevel, ToolCapability, ToolExecutionError

# Maximum file size to read (1MB)
MAX_FILE_SIZE = 1024 * 1024

# Maximum number of files to scan during search
MAX_SEARCH_FILES = 200

# Default evidence root directory (relative to repo root)
DEFAULT_EVIDENCE_ROOT = Path("evidence")

# Maximum snippet length for search results
MAX_SNIPPET_LENGTH = 200


class DocsReadTool:
    """Tool for reading local documentation/evidence files.

    Reads files from a restricted directory to support evidence collection.
    Safety constraints:
    - Only reads from allowlisted root directory
    - Rejects absolute paths and path traversal
    - Maximum file size limit
    - UTF-8 encoding with error replacement

    Capability: inspect (not discovery)
    During CLAIM_CHECK, access is restricted to already-cited sources.

    Input: {"path": "evidence/file.txt"}
    Output: {"text": "...file contents..."}
    """

    def __init__(self, evidence_root: Path | None = None) -> None:
        """Initialize the docs.read tool.

        Args:
            evidence_root: Root directory for evidence files.
                Defaults to ./evidence relative to current working directory.
        """
        self._evidence_root = evidence_root or DEFAULT_EVIDENCE_ROOT

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "docs.read"

    @property
    def risk_level(self) -> RiskLevel:
        """Risk classification of this tool."""
        return RiskLevel.LOW

    @property
    def capability(self) -> ToolCapability:
        """docs.read is an inspect tool.

        It reads from a fixed local directory, not searching for new sources.
        During CLAIM_CHECK, it should only be allowed for already-cited
        evidence sources (enforced by policy).
        """
        return ToolCapability.INSPECT

    @property
    def is_discovery(self) -> bool:
        """Whether this tool discovers new information."""
        return self.capability == ToolCapability.DISCOVERY

    def validate_input(self, tool_input: dict[str, Any]) -> None:
        """Validate the input before execution.

        Args:
            tool_input: The input dictionary to validate.

        Raises:
            ValueError: If the input is invalid.
        """
        if "path" not in tool_input:
            raise ValueError("Missing required field: 'path'")

        path = tool_input["path"]
        if not isinstance(path, str):
            raise ValueError("Field 'path' must be a string")

        if not path.strip():
            raise ValueError("Field 'path' cannot be empty")

        # Reject absolute paths
        if Path(path).is_absolute():
            raise ValueError("Absolute paths are not allowed")

        # Reject path traversal
        if ".." in path:
            raise ValueError("Path traversal (..) is not allowed")

    def execute(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Read a file from the evidence directory.

        Args:
            tool_input: Dictionary with 'path' key.

        Returns:
            Dictionary with 'text' key containing file contents.

        Raises:
            ToolExecutionError: If the file cannot be read.
        """
        self.validate_input(tool_input)
        relative_path = tool_input["path"]

        # Resolve the full path
        full_path = self._resolve_path(relative_path)

        # Validate the path is within evidence root
        try:
            full_path.resolve().relative_to(self._evidence_root.resolve())
        except ValueError as e:
            raise ToolExecutionError(
                self.name,
                f"Path '{relative_path}' is outside evidence directory",
            ) from e

        # Check file exists
        if not full_path.exists():
            raise ToolExecutionError(
                self.name,
                f"File not found: {relative_path}",
            )

        if not full_path.is_file():
            raise ToolExecutionError(
                self.name,
                f"Path is not a file: {relative_path}",
            )

        # Reject symlinks to prevent escaping evidence directory
        if full_path.is_symlink():
            raise ToolExecutionError(
                self.name,
                f"Symlinks are not allowed: {relative_path}",
            )

        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ToolExecutionError(
                self.name,
                f"File too large: {file_size} bytes (max {MAX_FILE_SIZE})",
            )

        # Read file with UTF-8 encoding
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise ToolExecutionError(
                self.name,
                f"Failed to read file: {e}",
            ) from e

        return {"text": text}

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against the evidence root.

        The path can be specified with or without the evidence/ prefix.

        Args:
            relative_path: The relative path from input.

        Returns:
            Full resolved Path object.
        """
        path = Path(relative_path)

        # If path starts with evidence/, use it directly
        # Otherwise, prepend evidence root
        if path.parts and path.parts[0] == self._evidence_root.name:
            return path
        else:
            return self._evidence_root / path


class DocsSearchTool:
    """Tool for searching local documentation/evidence files.

    Searches within an evidence pack (allowlisted directory) for files
    containing query terms. Returns deterministic results with snippets.

    Capability: discovery
    This tool is DENIED during CLAIM_CHECK to enforce evidence-local restriction.

    Scoring:
    - Count of query token matches in file text (case-insensitive)
    - Ties broken by path lexicographic order

    Input: {"query": "search terms", "max_results": 5}
    Output: {"results": [{"path": "...", "score": 1.0, "snippet": "..."}, ...]}
    """

    def __init__(self, evidence_root: Path | None = None) -> None:
        """Initialize the docs.search tool.

        Args:
            evidence_root: Root directory for evidence files.
                Defaults to ./evidence relative to current working directory.
        """
        self._evidence_root = evidence_root or DEFAULT_EVIDENCE_ROOT

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "docs.search"

    @property
    def risk_level(self) -> RiskLevel:
        """Risk classification of this tool."""
        return RiskLevel.LOW

    @property
    def capability(self) -> ToolCapability:
        """docs.search is a discovery tool.

        It searches for new information, so it is denied during CLAIM_CHECK.
        """
        return ToolCapability.DISCOVERY

    @property
    def is_discovery(self) -> bool:
        """Whether this tool discovers new information."""
        return self.capability == ToolCapability.DISCOVERY

    def validate_input(self, tool_input: dict[str, Any]) -> None:
        """Validate the input before execution.

        Args:
            tool_input: The input dictionary to validate.

        Raises:
            ValueError: If the input is invalid.
        """
        if "query" not in tool_input:
            raise ValueError("Missing required field: 'query'")

        query = tool_input["query"]
        if not isinstance(query, str):
            raise ValueError("Field 'query' must be a string")

        if not query.strip():
            raise ValueError("Field 'query' cannot be empty")

        max_results = tool_input.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("Field 'max_results' must be a positive integer")

    def execute(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Search for files matching the query.

        Args:
            tool_input: Dictionary with 'query' and optional 'max_results' keys.

        Returns:
            Dictionary with 'results' list of matches.

        Raises:
            ToolExecutionError: If search fails.
        """
        self.validate_input(tool_input)
        query = tool_input["query"]
        max_results = tool_input.get("max_results", 5)

        # Check evidence root exists
        if not self._evidence_root.exists():
            return {"results": [], "warning": "Evidence directory not found"}

        # Tokenize query (simple whitespace split, lowercase)
        query_tokens = query.lower().split()
        if not query_tokens:
            return {"results": []}

        # Collect all text files in evidence root
        results: list[dict[str, Any]] = []
        file_count = 0

        for file_path in self._iter_text_files():
            file_count += 1
            if file_count > MAX_SEARCH_FILES:
                break

            # Read and score the file
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            score = self._score_text(text, query_tokens)
            if score > 0:
                # Get relative path from evidence root
                try:
                    rel_path = file_path.relative_to(self._evidence_root)
                except ValueError:
                    rel_path = file_path

                # Extract snippet around first match
                snippet = self._extract_snippet(text, query_tokens)

                results.append(
                    {
                        "path": str(rel_path),
                        "score": score,
                        "snippet": snippet,
                    }
                )

        # Sort by score (descending) then path (ascending) for determinism
        results.sort(key=lambda r: (-r["score"], r["path"]))

        # Limit results
        results = results[:max_results]

        output: dict[str, Any] = {"results": results}
        if file_count > MAX_SEARCH_FILES:
            output["warning"] = f"Search limited to {MAX_SEARCH_FILES} files"

        return output

    def _iter_text_files(self) -> "list[Path]":
        """Iterate over text files in evidence root.

        Returns:
            Sorted list of text file paths (sorted for determinism).
        """
        if not self._evidence_root.is_dir():
            return []

        # Collect files and sort for deterministic ordering
        files: list[Path] = []
        for path in self._evidence_root.rglob("*"):
            if not path.is_file():
                continue
            if path.is_symlink():
                continue
            # Skip files that are too large
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            # Check for text-like extensions or no extension
            suffix = path.suffix.lower()
            if suffix in ("", ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv"):
                files.append(path)

        # Sort for determinism
        files.sort()
        return files

    def _score_text(self, text: str, query_tokens: list[str]) -> int:
        """Score text based on query token matches.

        Args:
            text: The file content.
            query_tokens: Lowercased query tokens.

        Returns:
            Total count of token matches in text.
        """
        text_lower = text.lower()
        score = 0
        for token in query_tokens:
            # Count occurrences of each token
            score += text_lower.count(token)
        return score

    def _extract_snippet(self, text: str, query_tokens: list[str]) -> str:
        """Extract a snippet around the first query match.

        Args:
            text: The file content.
            query_tokens: Lowercased query tokens.

        Returns:
            Snippet string (max MAX_SNIPPET_LENGTH chars).
        """
        text_lower = text.lower()

        # Find first occurrence of any token
        first_pos = len(text)
        for token in query_tokens:
            pos = text_lower.find(token)
            if pos != -1 and pos < first_pos:
                first_pos = pos

        if first_pos == len(text):
            # No match found, return start of text
            return text[:MAX_SNIPPET_LENGTH].strip()

        # Extract context around the match
        start = max(0, first_pos - 50)
        end = min(len(text), first_pos + MAX_SNIPPET_LENGTH - 50)

        snippet = text[start:end]

        # Clean up boundaries
        if start > 0:
            # Trim to word boundary
            space_pos = snippet.find(" ")
            if 0 < space_pos < 20:
                snippet = "..." + snippet[space_pos + 1 :]

        if end < len(text):
            # Trim to word boundary
            space_pos = snippet.rfind(" ")
            if space_pos > len(snippet) - 20:
                snippet = snippet[:space_pos] + "..."

        return snippet.strip()
