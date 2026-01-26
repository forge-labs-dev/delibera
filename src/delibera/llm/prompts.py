"""Prompt helpers and JSON schema definitions for LLM agents.

Provides shared prompt templates and JSON schema instructions for
structured LLM outputs in Delibera.
"""

from typing import Any

# JSON schema for Proposer output
PROPOSER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "description": "A concise recommendation for this option",
        },
        "rationale": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key reasons supporting this recommendation (max 6)",
            "maxItems": 6,
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["fact", "inference", "plan", "value"],
                        "description": "The type of claim",
                    },
                    "text": {
                        "type": "string",
                        "description": "The claim text",
                    },
                },
                "required": ["type", "text"],
            },
            "description": "Claims made in this proposal (max 8)",
            "maxItems": 8,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in this recommendation (0.0 to 1.0)",
        },
    },
    "required": ["recommendation", "rationale", "claims", "confidence"],
}


def build_proposer_system_prompt() -> str:
    """Build the system prompt for the Proposer agent.

    Returns:
        The system prompt string.
    """
    return """You are a proposal generator for a structured deliberation system.

Your task is to analyze an option and produce a structured proposal.

RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.
2. Follow the exact schema provided.
3. Be specific and actionable in your recommendation.
4. Provide clear, evidence-based rationale.
5. Classify claims correctly:
   - fact: Verifiable statements about the world
   - inference: Logical conclusions from facts
   - plan: Proposed actions or steps
   - value: Judgments about importance or quality
6. Keep rationale concise (max 6 items).
7. Keep claims focused (max 8 items).
8. Confidence should reflect certainty (0.0-1.0).

OUTPUT FORMAT (JSON only):
{
  "recommendation": "string",
  "rationale": ["string", ...],
  "claims": [{"type": "fact|inference|plan|value", "text": "string"}, ...],
  "confidence": 0.0-1.0
}"""


def build_proposer_user_prompt(
    question: str,
    option_label: str,
    evidence_snippets: list[str] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """Build the user prompt for the Proposer agent.

    Args:
        question: The deliberation question.
        option_label: The label of the option being proposed.
        evidence_snippets: Optional evidence excerpts to consider.
        constraints: Optional constraints from user.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
        f"\nOPTION: {option_label}",
    ]

    if constraints:
        parts.append("\nCONSTRAINTS:")
        for c in constraints:
            parts.append(f"- {c}")

    if evidence_snippets:
        parts.append("\nAVAILABLE EVIDENCE:")
        for i, snippet in enumerate(evidence_snippets[:5], 1):  # Limit to 5
            # Truncate long snippets
            truncated = snippet[:300] + "..." if len(snippet) > 300 else snippet
            parts.append(f"{i}. {truncated}")

    parts.append("\nGenerate a structured proposal for this option. Output JSON only.")

    return "\n".join(parts)


def get_proposer_schema_string() -> str:
    """Get the JSON schema as a formatted string for prompt inclusion.

    Returns:
        JSON schema as string.
    """
    import json

    return json.dumps(PROPOSER_JSON_SCHEMA, indent=2)
