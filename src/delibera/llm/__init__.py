"""LLM subsystem for Delibera.

Provides abstraction for LLM providers with Gemini as the first implementation.
LLMs are used only in WORK steps and never during validation.
"""

from delibera.llm.base import (
    LLMAuthError,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from delibera.llm.gemini import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY_ENV, GeminiClient
from delibera.llm.prompts import (
    PROPOSER_JSON_SCHEMA,
    build_proposer_system_prompt,
    build_proposer_user_prompt,
)
from delibera.llm.redaction import (
    create_llm_error_trace,
    create_llm_request_trace,
    create_llm_response_trace,
    redact_text,
    summarize_messages,
)

__all__ = [
    # Base types
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "LLMClient",
    # Errors
    "LLMError",
    "LLMRateLimitError",
    "LLMAuthError",
    "LLMInvalidResponseError",
    "LLMConnectionError",
    # Gemini
    "GeminiClient",
    "DEFAULT_GEMINI_MODEL",
    "GEMINI_API_KEY_ENV",
    # Prompts
    "PROPOSER_JSON_SCHEMA",
    "build_proposer_system_prompt",
    "build_proposer_user_prompt",
    # Redaction
    "redact_text",
    "summarize_messages",
    "create_llm_request_trace",
    "create_llm_response_trace",
    "create_llm_error_trace",
]
