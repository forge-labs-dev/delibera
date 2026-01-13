"""Local docs tool for reading evidence files.

This tool reads files from an allowlisted directory (evidence/)
to support claim validation with local evidence.
"""

from pathlib import Path
from typing import Any

from delibera.tools.spec import RiskLevel, ToolExecutionError

# Maximum file size to read (1MB)
MAX_FILE_SIZE = 1024 * 1024

# Default evidence root directory (relative to repo root)
DEFAULT_EVIDENCE_ROOT = Path("evidence")


class DocsReadTool:
    """Tool for reading local documentation/evidence files.

    Reads files from a restricted directory to support evidence collection.
    Safety constraints:
    - Only reads from allowlisted root directory
    - Rejects absolute paths and path traversal
    - Maximum file size limit
    - UTF-8 encoding with error replacement

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
    def is_discovery(self) -> bool:
        """docs.read is NOT a discovery tool.

        It reads from a fixed local directory, not searching for new sources.
        However, during CLAIM_CHECK, it should only be allowed for
        already-cited evidence sources (enforced by policy).
        """
        return False

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
