"""LLM abstraction layer for Delibera.

Provides a stable interface for LLM providers with structured request/response types.
LLMs are used only in WORK steps and never during validation.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class LLMMessage:
    """A single message in an LLM conversation.

    Attributes:
        role: The role of the message sender.
        content: The message content.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMRequest:
    """A request to an LLM provider.

    Attributes:
        model: The model identifier (e.g., "gemini-1.5-pro").
        messages: The conversation messages.
        temperature: Sampling temperature (0.0-2.0). None uses provider default.
        max_output_tokens: Maximum tokens in response. None uses provider default.
        response_format: Expected response format ("json" or "text").
        json_schema: Optional JSON schema for structured output.
        metadata: Trace tags for logging (run_id, node_id, step_id, role).
    """

    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_format: Literal["json", "text"] = "text"
    json_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_format": self.response_format,
            "json_schema": self.json_schema,
            "metadata": self.metadata,
        }


@dataclass
class LLMUsage:
    """Token usage information from an LLM response.

    Attributes:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens (input + output).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}
        if self.input_tokens is not None:
            result["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            result["output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens
        return result


@dataclass
class LLMResponse:
    """A response from an LLM provider.

    Attributes:
        text: The raw text response.
        parsed_json: Parsed JSON if response_format was "json" and parsing succeeded.
        usage: Token usage information if available.
        provider: The provider name (e.g., "gemini").
        model: The actual model used.
        latency_ms: Response latency in milliseconds.
    """

    text: str
    parsed_json: dict[str, Any] | None = None
    usage: LLMUsage | None = None
    provider: str = ""
    model: str = ""
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "parsed_json": self.parsed_json,
            "usage": self.usage.to_dict() if self.usage else None,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


class LLMClient(Protocol):
    """Protocol defining the LLM client interface.

    Implementations must provide a synchronous generate method.
    """

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: The LLM request with messages and parameters.

        Returns:
            The LLM response with text and optional parsed JSON.

        Raises:
            LLMError: If the request fails.
        """
        ...


class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(self, message: str, provider: str = "", model: str = "") -> None:
        """Initialize LLM error.

        Args:
            message: Error message.
            provider: The provider that raised the error.
            model: The model that was being used.
        """
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Raised when rate limits are exceeded."""

    pass


class LLMAuthError(LLMError):
    """Raised when authentication fails."""

    pass


class LLMInvalidResponseError(LLMError):
    """Raised when the LLM response is invalid or unparseable."""

    pass


class LLMConnectionError(LLMError):
    """Raised when connection to the LLM provider fails."""

    pass
