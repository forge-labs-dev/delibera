"""LLM-backed Proposer agent for Delibera.

Uses an LLM to generate structured proposals with recommendations,
rationale, claims, and confidence scores.
"""

from typing import TYPE_CHECKING, Any

from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_proposer_system_prompt, build_proposer_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient
    from delibera.tools import ToolCallback


class ProposerLLM:
    """LLM-backed proposer agent.

    Generates proposals using an LLM with structured JSON output.

    Attributes:
        llm_client: The LLM client to use for generation.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.2 for consistency).
        max_output_tokens: Maximum output tokens (default 800).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> None:
        """Initialize the LLM proposer.

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

    def execute(
        self,
        context: dict[str, Any],
        tool: "ToolCallback | None" = None,  # noqa: ARG002 - unused but kept for interface
    ) -> dict[str, Any]:
        """Generate a proposal for a branch using LLM.

        Args:
            context: Must contain:
                - "label": Branch label
                - "question": Original question
                - "node_id": Node ID for tracing
                Optional:
                - "evidence_snippets": List of evidence excerpts
                - "constraints": List of user constraints
            tool: Optional tool callback (not used by LLM proposer).

        Returns:
            Dict with proposal details matching ProposerStub output format.
        """
        label = context.get("label", "")
        question = context.get("question", "")
        node_id = context.get("node_id", "")
        evidence_snippets = context.get("evidence_snippets", [])
        constraints = context.get("constraints", [])

        # Build messages
        system_prompt = build_proposer_system_prompt()
        user_prompt = build_proposer_user_prompt(
            question=question,
            option_label=label,
            evidence_snippets=evidence_snippets,
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
                "node_id": node_id,
                "role": "proposer",
                "step": "PROPOSE",
            },
        )

        # Generate response
        response = self._llm_client.generate(request)

        # Parse and normalize the response
        parsed = response.parsed_json or {}
        return self._normalize_output(parsed, label)

    def _normalize_output(
        self,
        parsed: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        """Normalize LLM output to match expected format.

        Applies deterministic post-processing:
        - Strips whitespace
        - Drops empty entries
        - Truncates long strings
        - Clamps confidence to [0, 1]

        Args:
            parsed: The parsed JSON from LLM.
            label: The option label for fallback.

        Returns:
            Normalized output dict.
        """
        # Extract and normalize recommendation
        recommendation = str(parsed.get("recommendation", "")).strip()
        if not recommendation:
            recommendation = f"Recommendation for {label}"

        # Extract and normalize rationale (max 6)
        rationale_raw = parsed.get("rationale", [])
        if not isinstance(rationale_raw, list):
            rationale_raw = [str(rationale_raw)] if rationale_raw else []
        rationale = []
        for item in rationale_raw[:6]:
            text = str(item).strip()
            if text:
                # Truncate long rationale items
                if len(text) > 300:
                    text = text[:297] + "..."
                rationale.append(text)

        # Extract and normalize claims (max 8)
        claims_raw = parsed.get("claims", [])
        if not isinstance(claims_raw, list):
            claims_raw = []
        claims = []
        valid_types = {"fact", "inference", "plan", "value"}
        for item in claims_raw[:8]:
            if not isinstance(item, dict):
                continue
            claim_type = str(item.get("type", "inference")).lower()
            if claim_type not in valid_types:
                claim_type = "inference"
            claim_text = str(item.get("text", "")).strip()
            if claim_text:
                # Truncate long claims
                if len(claim_text) > 200:
                    claim_text = claim_text[:197] + "..."
                claims.append({"type": claim_type, "text": claim_text})

        # Extract and clamp confidence
        confidence_raw = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Extract and normalize sub_branches (max 2)
        sub_raw = parsed.get("sub_branches", [])
        if not isinstance(sub_raw, list):
            sub_raw = []
        sub_branches = []
        for item in sub_raw[:2]:
            text = str(item).strip()
            if text:
                if len(text) > 200:
                    text = text[:197] + "..."
                sub_branches.append(text)
        # Fallback: generate generic sub-branches from the label
        if len(sub_branches) < 2:
            option_letter = label[0] if label else "X"
            sub_branches = [
                f"Subplan {option_letter}.1: Detailed implementation",
                f"Subplan {option_letter}.2: Alternative approach",
            ]

        # Build output in format compatible with ProposerStub
        # Extract facts for the "facts" field expected by engine
        fact_texts = [c["text"] for c in claims if c["type"] == "fact"]

        return {
            "proposal": recommendation,
            "summary": rationale[0] if rationale else f"Proposal for {label}",
            "pros": rationale[:3],  # First 3 rationale items as pros
            "cons": rationale[3:5] if len(rationale) > 3 else [],  # Next as cons
            "facts": fact_texts,
            "score": confidence,  # Use confidence as initial score
            "role": "proposer",
            # LLM-specific fields
            "recommendation": recommendation,
            "rationale": rationale,
            "claims": claims,
            "confidence": confidence,
            "llm_generated": True,
            # Sub-branches for multi-level tree expansion
            "sub_branches": sub_branches,
        }


class LLMNotAllowedError(Exception):
    """Raised when LLM is used in a step where it's not allowed."""

    pass


def check_llm_allowed_in_step(step_kind: str) -> None:
    """Check if LLM usage is allowed in the given step.

    LLMs are only allowed in WORK steps, never in validate steps.

    Args:
        step_kind: The step kind ("work", "validate", "score").

    Raises:
        LLMNotAllowedError: If LLM is not allowed in this step.
    """
    if step_kind != "work":
        raise LLMNotAllowedError(
            f"LLM usage is not allowed in '{step_kind}' steps. "
            "LLMs can only be used in 'work' steps."
        )
