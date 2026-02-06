"""LLM-backed Red-Teamer agent for Delibera.

Uses an LLM to generate meaningful objections based on actual
proposal content, claims, and evidence status.
"""

import hashlib
from typing import TYPE_CHECKING, Any

from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_redteam_system_prompt, build_redteam_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient


class RedTeamLLM:
    """LLM-backed red-team agent.

    Analyzes proposals and generates meaningful objections with
    severity classification using an LLM.

    Attributes:
        llm_client: The LLM client to use for generation.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.4 for variety).
        max_output_tokens: Maximum output tokens (default 600).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        temperature: float = 0.4,
        max_output_tokens: int = 600,
    ) -> None:
        """Initialize the LLM red-teamer.

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
        """Generate objections for a proposal using LLM.

        Args:
            context: Must contain:
                - "node_id": Node ID
                - "artifact": The node's artifact dict
                - "claims": List of claim dicts
                - "evidence_count": Number of evidence items
                Optional:
                - "question": The deliberation question

        Returns:
            Dict with objections list and metadata, matching RedTeamStub format.
        """
        node_id = context.get("node_id", "")
        artifact = context.get("artifact", {})
        claims = context.get("claims", [])
        evidence_count = context.get("evidence_count", 0)
        question = context.get("question", "")

        # Build LLM request
        system_prompt = build_redteam_system_prompt()
        user_prompt = build_redteam_user_prompt(
            question=question,
            artifact=artifact,
            claims=claims,
            evidence_count=evidence_count,
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
                "role": "redteam",
                "step": "REDTEAM",
            },
        )

        # Generate response
        response = self._llm_client.generate(request)
        parsed = response.parsed_json or {}

        return self._normalize_output(parsed, node_id)

    def _normalize_output(
        self,
        parsed: dict[str, Any],
        node_id: str,
    ) -> dict[str, Any]:
        """Normalize LLM output to match RedTeamStub format.

        Args:
            parsed: The parsed JSON from LLM.
            node_id: Node ID for generating objection IDs.

        Returns:
            Normalized output dict with objections list.
        """
        objections_raw = parsed.get("objections", [])
        if not isinstance(objections_raw, list):
            objections_raw = []

        objections: list[dict[str, str]] = []
        for item in objections_raw[:4]:
            if not isinstance(item, dict):
                continue

            target = str(item.get("target", "artifact")).strip()
            if not target:
                target = "artifact"

            severity = str(item.get("severity", "nonblocking")).lower()
            if severity not in ("blocking", "nonblocking"):
                severity = "nonblocking"

            rationale = str(item.get("rationale", "")).strip()
            if not rationale:
                continue

            # Truncate long rationale
            if len(rationale) > 300:
                rationale = rationale[:297] + "..."

            # Generate deterministic objection ID
            objection_id = _make_objection_id(node_id, target, severity, rationale)

            objections.append(
                {
                    "objection_id": objection_id,
                    "target": target,
                    "severity": severity,
                    "status": "open",
                    "rationale": rationale,
                }
            )

        # Extract analysis
        analysis = str(parsed.get("analysis", "")).strip()

        return {
            "objections": objections,
            "analysis": analysis,
            "role": "redteam",
            "step": "REDTEAM",
            "llm_generated": True,
        }


def _make_objection_id(
    node_id: str,
    target: str,
    severity: str,
    rationale: str,
) -> str:
    """Generate a deterministic objection ID.

    Args:
        node_id: The node ID.
        target: The objection target.
        severity: The objection severity.
        rationale: The objection rationale.

    Returns:
        A 10-character hex hash.
    """
    key = f"{node_id}:{target}:{severity}:{rationale}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]
