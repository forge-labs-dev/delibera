"""LLM-backed Planner agent for Delibera.

Uses an LLM to generate contextually relevant branch labels
based on the deliberation question.
"""

from typing import TYPE_CHECKING, Any

from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_planner_system_prompt, build_planner_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient

# Default branch labels used as fallback when LLM returns insufficient branches.
_DEFAULT_BRANCHES = [
    "Option A: Direct migration approach",
    "Option B: Incremental adoption approach",
    "Option C: Conservative wait-and-see approach",
]


class PlannerLLM:
    """LLM-backed planner agent.

    Generates contextually relevant branch labels using an LLM
    with structured JSON output.

    Attributes:
        llm_client: The LLM client to use for generation.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.2 for consistency).
        max_output_tokens: Maximum output tokens (default 400).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 400,
    ) -> None:
        """Initialize the LLM planner.

        Args:
            llm_client: The LLM client for generation.
            model: Model name (None uses client default).
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens.
        """
        self._llm_client = llm_client
        self._model = model or ""
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate branch labels for a deliberation question using LLM.

        Args:
            context: Must contain:
                - "question": The deliberation question
                Optional:
                - "constraints": List of user constraints

        Returns:
            Dict with "branches" list and metadata, matching PlannerStub format.
        """
        question = context.get("question", "")
        constraints = context.get("constraints", [])

        # Build messages
        system_prompt = build_planner_system_prompt()
        user_prompt = build_planner_user_prompt(
            question=question,
            constraints=constraints,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Build request with metadata for tracing
        request = LLMRequest(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            response_format="json",
            metadata={
                "role": "planner",
                "step": "PLAN",
            },
        )

        # Generate response
        response = self._llm_client.generate(request)

        # Parse and normalize the response
        parsed = response.parsed_json or {}
        return self._normalize_output(parsed)

    def _normalize_output(
        self,
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize LLM output to match expected planner format.

        Applies deterministic post-processing:
        - Strips whitespace from branch labels
        - Drops empty entries
        - Truncates long labels
        - Ensures at least 2 branches (falls back to defaults)
        - Caps at 5 branches

        Args:
            parsed: The parsed JSON from LLM.

        Returns:
            Normalized output dict with "branches" key.
        """
        # Extract and normalize branches
        branches_raw = parsed.get("branches", [])
        if not isinstance(branches_raw, list):
            branches_raw = []

        branches: list[str] = []
        for item in branches_raw[:5]:
            label = str(item).strip()
            if label:
                # Truncate long labels
                if len(label) > 100:
                    label = label[:97] + "..."
                branches.append(label)

        # Must have at least 2 branches; fall back to defaults if not
        if len(branches) < 2:
            branches = list(_DEFAULT_BRANCHES)

        # Extract reasoning
        reasoning = str(parsed.get("reasoning", "")).strip()

        return {
            "branches": branches,
            "role": "planner",
            "step": "PLAN",
            "reasoning": reasoning,
            "llm_generated": True,
        }
