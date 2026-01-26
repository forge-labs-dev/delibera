"""Redaction helpers for LLM trace logging.

Provides utilities to redact sensitive information from prompts and responses
before logging to traces. This ensures privacy while maintaining auditability.
"""

import re
from typing import Any

from delibera.llm.base import LLMMessage, LLMRequest, LLMResponse

# Patterns for redaction (compiled for performance)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_API_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|token|secret|password)[=:\s]+['\"]?[\w-]{16,}['\"]?", re.IGNORECASE
)
_LONG_NUMBER_PATTERN = re.compile(r"\b\d{10,}\b")
_FILE_PATH_PATTERN = re.compile(r"(?:/[a-zA-Z0-9._-]+){3,}")
_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")

# Redaction placeholder
REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    """Redact sensitive information from text.

    Removes:
    - Email addresses
    - API keys and secrets
    - Long numbers (10+ digits)
    - File paths (3+ segments)
    - URLs

    Args:
        text: The text to redact.

    Returns:
        The redacted text.
    """
    result = text
    result = _EMAIL_PATTERN.sub(REDACTED, result)
    result = _API_KEY_PATTERN.sub(REDACTED, result)
    result = _LONG_NUMBER_PATTERN.sub(REDACTED, result)
    result = _FILE_PATH_PATTERN.sub(REDACTED, result)
    result = _URL_PATTERN.sub(REDACTED, result)
    return result


def summarize_messages(messages: list[LLMMessage]) -> dict[str, Any]:
    """Summarize messages for trace logging without storing full content.

    Args:
        messages: The messages to summarize.

    Returns:
        A summary dict with:
        - message_count: Total number of messages
        - chars_by_role: Character counts per role
        - user_preview: First 80 chars of user message (redacted)
    """
    chars_by_role: dict[str, int] = {"system": 0, "user": 0, "assistant": 0}
    user_preview = ""

    for msg in messages:
        chars_by_role[msg.role] += len(msg.content)
        if msg.role == "user" and not user_preview:
            redacted = redact_text(msg.content)
            user_preview = redacted[:80] + "..." if len(redacted) > 80 else redacted

    return {
        "message_count": len(messages),
        "chars_by_role": chars_by_role,
        "user_preview": user_preview,
    }


def create_llm_request_trace(request: LLMRequest) -> dict[str, Any]:
    """Create a trace-safe representation of an LLM request.

    Does NOT include full prompt content. Only includes:
    - Model and parameters
    - Message summary
    - Metadata

    Args:
        request: The LLM request to trace.

    Returns:
        A dict safe for trace logging.
    """
    return {
        "model": request.model,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "response_format": request.response_format,
        "message_summary": summarize_messages(request.messages),
        "metadata": request.metadata,
    }


def create_llm_response_trace(response: LLMResponse) -> dict[str, Any]:
    """Create a trace-safe representation of an LLM response.

    Includes:
    - Provider and model info
    - Usage statistics
    - Latency
    - Output length
    - JSON keys if parsed (not values)

    Args:
        response: The LLM response to trace.

    Returns:
        A dict safe for trace logging.
    """
    result: dict[str, Any] = {
        "provider": response.provider,
        "model": response.model,
        "output_length": len(response.text),
        "latency_ms": response.latency_ms,
    }

    if response.usage:
        result["usage"] = response.usage.to_dict()

    if response.parsed_json is not None:
        # Only include keys, not values
        result["json_keys"] = list(response.parsed_json.keys())

    return result


def create_llm_error_trace(
    error: Exception,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Create a trace-safe representation of an LLM error.

    Args:
        error: The exception that occurred.
        provider: The LLM provider.
        model: The model being used.

    Returns:
        A dict safe for trace logging.
    """
    return {
        "provider": provider,
        "model": model,
        "error_type": type(error).__name__,
        "error_message": redact_text(str(error)),
    }
