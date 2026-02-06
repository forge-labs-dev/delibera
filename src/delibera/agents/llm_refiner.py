"""LLM-backed Refiner agent for Delibera.

Uses an LLM to improve proposals by addressing objections,
strengthening weak claims, and incorporating new evidence.
"""

from typing import TYPE_CHECKING, Any

from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_refiner_system_prompt, build_refiner_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient


class RefinerLLM:
    """LLM-backed refiner agent.

    Improves proposals by addressing weaknesses, objections, and
    unsupported claims using an LLM.

    Attributes:
        llm_client: The LLM client to use for generation.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.2 for consistency).
        max_output_tokens: Maximum output tokens (default 600).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 600,
    ) -> None:
        """Initialize the LLM refiner.

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
        """Refine a proposal using LLM.

        Args:
            context: Must contain:
                - "node_id": Node ID
                - "artifact": Current artifact dict
                - "claims": List of claim dicts with status and text
                - "round": Current refinement round number
                Optional:
                - "question": The deliberation question
                - "objections": List of open objection dicts

        Returns:
            Dict with refined artifact and metadata, matching RefinerStub format.
        """
        node_id = context.get("node_id", "")
        artifact = context.get("artifact", {})
        claims = context.get("claims", [])
        round_num = context.get("round", 1)
        question = context.get("question", "")
        objections = context.get("objections", [])

        # Build LLM request
        system_prompt = build_refiner_system_prompt()
        user_prompt = build_refiner_user_prompt(
            question=question,
            artifact=artifact,
            claims=claims,
            objections=objections,
            round_num=round_num,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        request = LLMRequest(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            response_format="json",
            metadata={
                "node_id": node_id,
                "role": "refiner",
                "step": "REFINE",
            },
        )

        # Generate response
        response = self._llm_client.generate(request)
        parsed = response.parsed_json or {}

        return self._normalize_output(parsed, artifact, claims, round_num)

    def _normalize_output(
        self,
        parsed: dict[str, Any],
        original_artifact: dict[str, Any],
        claims: list[dict[str, Any]],
        round_num: int,
    ) -> dict[str, Any]:
        """Normalize LLM output to match RefinerStub format.

        Merges LLM improvements into the original artifact to maintain
        backward compatibility with the orchestrator.

        Args:
            parsed: The parsed JSON from LLM.
            original_artifact: The original artifact to refine.
            claims: The claims list for counting addressed weaknesses.
            round_num: Current refinement round.

        Returns:
            Normalized output dict with refined artifact.
        """
        # Extract improvements
        improved_summary = str(parsed.get("improved_summary", "")).strip()
        improvements_raw = parsed.get("improvements", [])
        if not isinstance(improvements_raw, list):
            improvements_raw = []
        improvements = [
            str(item).strip()[:300] for item in improvements_raw[:6] if str(item).strip()
        ]

        # Extract addressed weaknesses
        addressed_raw = parsed.get("addressed_weaknesses", [])
        if not isinstance(addressed_raw, list):
            addressed_raw = []
        addressed = [str(item).strip() for item in addressed_raw if str(item).strip()]

        # Extract confidence delta
        try:
            confidence_delta = float(parsed.get("confidence_delta", 0.0))
        except (TypeError, ValueError):
            confidence_delta = 0.0
        confidence_delta = max(-0.1, min(0.1, confidence_delta))

        # Count weak and unsupported claims that were addressed
        weak_ids = {c.get("claim_id") for c in claims if c.get("status") == "weak"}
        unsupported_ids = {c.get("claim_id") for c in claims if c.get("status") == "unsupported"}
        weak_addressed = len(weak_ids & set(addressed))
        unsupported_addressed = len(unsupported_ids & set(addressed))

        # Build refined artifact by copying original and applying improvements
        refined_artifact = dict(original_artifact)

        if improved_summary:
            refined_artifact["summary"] = improved_summary
        elif not refined_artifact.get("summary"):
            refined_artifact["summary"] = f"Refined in round {round_num}"

        refined_artifact["refinement_round"] = round_num
        refined_artifact["refinement_notes"] = improvements

        # Adjust score based on confidence delta
        original_score = refined_artifact.get("score", 0.5)
        try:
            original_score = float(original_score)
        except (TypeError, ValueError):
            original_score = 0.5
        refined_artifact["score"] = max(0.0, min(1.0, original_score + confidence_delta))

        return {
            "artifact": refined_artifact,
            "refinement_notes": improvements,
            "weak_addressed": weak_addressed,
            "unsupported_addressed": unsupported_addressed,
            "role": "refiner",
            "step": "REFINE",
            "llm_generated": True,
        }
