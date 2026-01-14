"""Gemini LLM client implementation.

Provides a GeminiClient that implements the LLMClient interface using
the Google Gemini API via the official SDK.
"""

import json
import os
import time
from typing import Any

from delibera.llm.base import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

# Default model for Gemini
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Environment variable for API key
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


def _get_api_key() -> str:
    """Get the Gemini API key from environment.

    Returns:
        The API key.

    Raises:
        LLMAuthError: If the API key is not set.
    """
    api_key = os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        raise LLMAuthError(
            f"{GEMINI_API_KEY_ENV} environment variable not set",
            provider="gemini",
        )
    return api_key


class GeminiClient:
    """LLM client for Google Gemini API.

    Uses the official google-generativeai SDK when available.
    Falls back to HTTP requests if SDK is not installed.

    Attributes:
        model: The default model to use.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Gemini client.

        Args:
            model: Default model name. Defaults to gemini-1.5-flash.
            api_key: API key. If None, reads from GEMINI_API_KEY env var.

        Raises:
            LLMAuthError: If API key is not available.
        """
        self.model = model
        self._api_key = api_key or _get_api_key()
        self._sdk_available = False
        self._genai: Any = None

        # Try to import the SDK
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._genai = genai
            self._sdk_available = True
        except ImportError:
            # SDK not available, will use HTTP fallback
            pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            request: The LLM request.

        Returns:
            The LLM response.

        Raises:
            LLMError: If the request fails.
        """
        model = request.model or self.model
        start_time = time.monotonic()

        try:
            if self._sdk_available:
                response = self._generate_with_sdk(request, model)
            else:
                response = self._generate_with_http(request, model)

            latency_ms = int((time.monotonic() - start_time) * 1000)
            response.latency_ms = latency_ms

            # Parse JSON if requested
            if request.response_format == "json" and response.parsed_json is None:
                response.parsed_json = self._parse_json_response(response.text)

            return response

        except LLMError:
            raise
        except Exception as e:
            raise LLMConnectionError(
                f"Failed to connect to Gemini: {e}",
                provider="gemini",
                model=model,
            ) from e

    def _generate_with_sdk(self, request: LLMRequest, model: str) -> LLMResponse:
        """Generate using the official SDK.

        Args:
            request: The LLM request.
            model: The model to use.

        Returns:
            The LLM response.
        """
        genai = self._genai

        # Build generation config
        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            generation_config["max_output_tokens"] = request.max_output_tokens

        # Set response MIME type for JSON
        if request.response_format == "json":
            generation_config["response_mime_type"] = "application/json"

        # Create the model
        try:
            genai_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config if generation_config else None,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to create Gemini model: {e}",
                provider="gemini",
                model=model,
            ) from e

        # Build content from messages
        contents = self._build_sdk_contents(request)

        # Generate
        try:
            response = genai_model.generate_content(contents)
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise LLMRateLimitError(str(e), provider="gemini", model=model) from e
            if "auth" in error_str or "key" in error_str or "permission" in error_str:
                raise LLMAuthError(str(e), provider="gemini", model=model) from e
            raise LLMError(str(e), provider="gemini", model=model) from e

        # Extract text
        try:
            text = response.text
        except ValueError as e:
            # Response was blocked or empty
            raise LLMInvalidResponseError(
                f"Gemini response was blocked or empty: {e}",
                provider="gemini",
                model=model,
            ) from e

        # Extract usage if available
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            usage = LLMUsage(
                input_tokens=getattr(metadata, "prompt_token_count", None),
                output_tokens=getattr(metadata, "candidates_token_count", None),
                total_tokens=getattr(metadata, "total_token_count", None),
            )

        return LLMResponse(
            text=text,
            provider="gemini",
            model=model,
            usage=usage,
        )

    def _generate_with_http(self, request: LLMRequest, model: str) -> LLMResponse:
        """Generate using direct HTTP requests.

        This is a fallback when the SDK is not available.

        Args:
            request: The LLM request.
            model: The model to use.

        Returns:
            The LLM response.
        """
        import urllib.error
        import urllib.request

        # Build API URL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._api_key}"

        # Build request body
        contents = self._build_http_contents(request)
        body: dict[str, Any] = {"contents": contents}

        # Add generation config
        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_output_tokens
        if request.response_format == "json":
            generation_config["responseMimeType"] = "application/json"

        if generation_config:
            body["generationConfig"] = generation_config

        # Make request
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429:
                raise LLMRateLimitError(
                    f"Rate limit exceeded: {error_body}",
                    provider="gemini",
                    model=model,
                ) from e
            if e.code in (401, 403):
                raise LLMAuthError(
                    f"Authentication failed: {error_body}",
                    provider="gemini",
                    model=model,
                ) from e
            raise LLMError(
                f"HTTP error {e.code}: {error_body}",
                provider="gemini",
                model=model,
            ) from e
        except urllib.error.URLError as e:
            raise LLMConnectionError(
                f"Connection failed: {e}",
                provider="gemini",
                model=model,
            ) from e

        # Extract text from response
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise LLMInvalidResponseError(
                    "No candidates in response",
                    provider="gemini",
                    model=model,
                )
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise LLMInvalidResponseError(
                    "No parts in response content",
                    provider="gemini",
                    model=model,
                )
            text = parts[0].get("text", "")
        except (KeyError, IndexError) as e:
            raise LLMInvalidResponseError(
                f"Failed to parse response: {e}",
                provider="gemini",
                model=model,
            ) from e

        # Extract usage
        usage = None
        usage_metadata = response_data.get("usageMetadata", {})
        if usage_metadata:
            usage = LLMUsage(
                input_tokens=usage_metadata.get("promptTokenCount"),
                output_tokens=usage_metadata.get("candidatesTokenCount"),
                total_tokens=usage_metadata.get("totalTokenCount"),
            )

        return LLMResponse(
            text=text,
            provider="gemini",
            model=model,
            usage=usage,
        )

    def _build_sdk_contents(self, request: LLMRequest) -> list[Any]:
        """Build SDK-compatible content list from messages.

        Args:
            request: The LLM request.

        Returns:
            Content list for SDK.
        """
        contents: list[Any] = []

        # Combine system and user messages
        combined_prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                combined_prompt += msg.content + "\n\n"
            elif msg.role == "user":
                combined_prompt += msg.content
            elif msg.role == "assistant":
                # For multi-turn, include assistant responses
                if combined_prompt:
                    contents.append({"role": "user", "parts": [{"text": combined_prompt}]})
                    combined_prompt = ""
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        if combined_prompt:
            contents.append({"role": "user", "parts": [{"text": combined_prompt}]})

        return contents

    def _build_http_contents(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Build HTTP API-compatible content list from messages.

        Args:
            request: The LLM request.

        Returns:
            Content list for HTTP API.
        """
        contents: list[dict[str, Any]] = []

        # Combine system and user messages
        combined_prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                combined_prompt += msg.content + "\n\n"
            elif msg.role == "user":
                combined_prompt += msg.content
            elif msg.role == "assistant":
                if combined_prompt:
                    contents.append({"role": "user", "parts": [{"text": combined_prompt}]})
                    combined_prompt = ""
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        if combined_prompt:
            contents.append({"role": "user", "parts": [{"text": combined_prompt}]})

        return contents

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from response text.

        Handles cases where the model wraps JSON in markdown code blocks.

        Args:
            text: The response text.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMInvalidResponseError: If JSON parsing fails.
        """
        # Try direct parse first
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence (with optional language)
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise LLMInvalidResponseError(
                f"Failed to parse JSON response: {e}. Text: {text[:100]}...",
                provider="gemini",
                model=self.model,
            ) from e
