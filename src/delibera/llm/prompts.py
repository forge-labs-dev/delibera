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
        "sub_branches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2 alternative sub-plans for implementing this option",
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": ["recommendation", "rationale", "claims", "confidence", "sub_branches"],
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
9. Propose exactly 2 alternative sub-plans for implementing this option.

OUTPUT FORMAT (JSON only):
{
  "recommendation": "string",
  "rationale": ["string", ...],
  "claims": [{"type": "fact|inference|plan|value", "text": "string"}, ...],
  "confidence": 0.0-1.0,
  "sub_branches": ["Sub-plan 1: description", "Sub-plan 2: description"]
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


# JSON schema for Planner output
PLANNER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "branches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Distinct approaches to the question (2-5 options)",
            "minItems": 2,
            "maxItems": 5,
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why these branches cover the decision space",
        },
    },
    "required": ["branches", "reasoning"],
}


def build_planner_system_prompt() -> str:
    """Build the system prompt for the Planner agent.

    Returns:
        The system prompt string.
    """
    return """You are a deliberation planner for a structured decision-making system.

Your task is to analyze a question and propose 2-5 distinct approaches to explore.

RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.
2. Follow the exact schema provided.
3. Generate 2-5 meaningfully different approaches to the question.
4. Each branch should represent a genuinely distinct strategy, not variations of the same idea.
5. Branch labels should be concise but descriptive (under 80 characters).
6. Cover the decision space well: include both bold and conservative options.
7. Explain briefly why these branches were chosen.

OUTPUT FORMAT (JSON only):
{
  "branches": ["Approach 1: description", "Approach 2: description", ...],
  "reasoning": "Brief explanation of the branch selection"
}"""


def build_planner_user_prompt(
    question: str,
    constraints: list[str] | None = None,
) -> str:
    """Build the user prompt for the Planner agent.

    Args:
        question: The deliberation question.
        constraints: Optional constraints from user.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
    ]

    if constraints:
        parts.append("\nCONSTRAINTS:")
        for c in constraints:
            parts.append(f"- {c}")

    parts.append(
        "\nPropose 2-5 distinct approaches to explore for this question. Output JSON only."
    )

    return "\n".join(parts)


# JSON schema for Researcher output
RESEARCHER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A targeted search query",
                    },
                    "intent": {
                        "type": "string",
                        "description": "What this query aims to find",
                    },
                },
                "required": ["query", "intent"],
            },
            "description": "Search queries to find relevant evidence (2-5)",
            "minItems": 1,
            "maxItems": 5,
        },
    },
    "required": ["queries"],
}


def build_researcher_system_prompt() -> str:
    """Build the system prompt for the Researcher agent.

    Returns:
        The system prompt string.
    """
    return """You are a research agent for a structured deliberation system.

Your task is to generate targeted search queries to find evidence relevant to a specific option.

RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.
2. Generate 2-5 focused search queries.
3. Each query should target a different aspect of the option.
4. Include an intent explaining what the query aims to find.
5. Queries should be specific enough to return useful results.
6. Consider: factual data, benchmarks, case studies, expert opinions, implementation details.

OUTPUT FORMAT (JSON only):
{
  "queries": [
    {"query": "search query text", "intent": "what this query aims to find"},
    ...
  ]
}"""


def build_researcher_user_prompt(
    question: str,
    label: str,
    proposal: str | None = None,
) -> str:
    """Build the user prompt for the Researcher agent.

    Args:
        question: The deliberation question.
        label: The option label being researched.
        proposal: Optional proposal summary for context.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
        f"\nOPTION: {label}",
    ]

    if proposal:
        truncated = proposal[:500] + "..." if len(proposal) > 500 else proposal
        parts.append(f"\nPROPOSAL SUMMARY: {truncated}")

    parts.append(
        "\nGenerate 2-5 targeted search queries to find evidence for this option. Output JSON only."
    )

    return "\n".join(parts)


# JSON schema for Red-Teamer output
REDTEAM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The claim_id to target, or 'artifact' for overall concerns",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["blocking", "nonblocking"],
                        "description": "blocking = fatal flaw; nonblocking = concern worth noting",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Explanation of the concern",
                    },
                },
                "required": ["target", "severity", "rationale"],
            },
            "description": "Objections found in the proposal (1-4)",
            "minItems": 1,
            "maxItems": 4,
        },
        "analysis": {
            "type": "string",
            "description": "Brief overall assessment of the proposal",
        },
    },
    "required": ["objections", "analysis"],
}


def build_redteam_system_prompt() -> str:
    """Build the system prompt for the Red-Teamer agent.

    Returns:
        The system prompt string.
    """
    return """You are a red-team adversary for a structured deliberation system.

Your task is to critically analyze a proposal and find weaknesses,
logical gaps, unsupported claims, and risks.

RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.
2. Generate 1-4 objections.
3. Classify each objection:
   - blocking: Fatal flaw that must be addressed (missing evidence, logical error, critical risk)
   - nonblocking: Concern worth considering (tradeoff, alternative unexplored, minor gap)
4. Target specific claims by their ID when possible, or "artifact" for overall concerns.
5. Be constructive: explain WHY something is a problem.
6. Focus on the most significant issues, not every minor detail.

OUTPUT FORMAT (JSON only):
{
  "objections": [
    {"target": "claim_id", "severity": "blocking|nonblocking",
     "rationale": "explanation"},
    ...
  ],
  "analysis": "Brief overall assessment"
}"""


def build_redteam_user_prompt(
    question: str,
    artifact: dict[str, Any],
    claims: list[dict[str, Any]],
    evidence_count: int,
) -> str:
    """Build the user prompt for the Red-Teamer agent.

    Args:
        question: The deliberation question.
        artifact: The node's artifact dict.
        claims: List of claim dicts with claim_id, claim_type, status.
        evidence_count: Number of evidence items attached.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
        "\nPROPOSAL:",
    ]

    if isinstance(artifact, dict):
        rec = artifact.get("recommendation", artifact.get("summary", ""))
        if rec:
            truncated = str(rec)[:300]
            parts.append(f"  Recommendation: {truncated}")
        rationale = artifact.get("rationale", [])
        if isinstance(rationale, list):
            for r in rationale[:4]:
                parts.append(f"  - {str(r)[:200]}")

    if claims:
        parts.append("\nCLAIMS:")
        for c in claims[:10]:
            cid = c.get("claim_id", "?")
            ctype = c.get("claim_type", "?")
            status = c.get("status", "?")
            parts.append(f"  [{cid}] ({ctype}, {status})")

    parts.append(f"\nEVIDENCE: {evidence_count} items attached")
    parts.append("\nAnalyze this proposal and identify weaknesses. Output JSON only.")

    return "\n".join(parts)


# JSON schema for Refiner output
REFINER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "improved_summary": {
            "type": "string",
            "description": "Updated summary incorporating improvements",
        },
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific improvements made",
            "maxItems": 6,
        },
        "addressed_weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of weak/unsupported claims addressed",
        },
        "confidence_delta": {
            "type": "number",
            "minimum": -0.1,
            "maximum": 0.1,
            "description": "How much to adjust confidence (-0.1 to +0.1)",
        },
    },
    "required": ["improved_summary", "improvements"],
}


def build_refiner_system_prompt() -> str:
    """Build the system prompt for the Refiner agent.

    Returns:
        The system prompt string.
    """
    return """You are a refinement agent for a structured deliberation system.

Your task is to improve a proposal by addressing its weaknesses and strengthening claims.

RULES:
1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.
2. Address weak and unsupported claims specifically.
3. Respond to objections constructively.
4. Keep improvements focused and specific.
5. Don't fundamentally change the proposal's direction.
6. Confidence delta should be small (-0.1 to +0.1).

OUTPUT FORMAT (JSON only):
{
  "improved_summary": "Updated summary incorporating improvements",
  "improvements": ["specific improvement 1", ...],
  "addressed_weaknesses": ["claim_id_1", ...],
  "confidence_delta": 0.0
}"""


def build_refiner_user_prompt(
    question: str,
    artifact: dict[str, Any],
    claims: list[dict[str, Any]],
    objections: list[dict[str, Any]] | None = None,
    round_num: int = 1,
) -> str:
    """Build the user prompt for the Refiner agent.

    Args:
        question: The deliberation question.
        artifact: Current artifact dict.
        claims: List of claim dicts with claim_id, status, text.
        objections: Optional list of open objection dicts.
        round_num: Current refinement round number.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
        f"\nROUND: {round_num}",
        "\nCURRENT PROPOSAL:",
    ]

    if isinstance(artifact, dict):
        summary = artifact.get("summary", "")
        if summary:
            parts.append(f"  Summary: {str(summary)[:500]}")
        rec = artifact.get("recommendation", "")
        if rec:
            parts.append(f"  Recommendation: {str(rec)[:300]}")

    weak = [c for c in claims if c.get("status") in ("weak", "unsupported")]
    if weak:
        parts.append("\nWEAK/UNSUPPORTED CLAIMS TO ADDRESS:")
        for c in weak[:6]:
            cid = c.get("claim_id", "?")
            status = c.get("status", "?")
            text = str(c.get("text", ""))[:200]
            parts.append(f"  [{cid}] ({status}): {text}")

    if objections:
        parts.append("\nOBJECTIONS TO ADDRESS:")
        for o in objections[:4]:
            severity = o.get("severity", "?")
            rationale = str(o.get("rationale", ""))[:200]
            parts.append(f"  [{severity}] {rationale}")

    parts.append("\nImprove this proposal by addressing the weaknesses. Output JSON only.")

    return "\n".join(parts)


# JSON schema for Claim Validator output
VALIDATOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "The ID of the claim being assessed",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["supported", "weak", "unsupported", "contradicted"],
                        "description": "Assessment of evidence support",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the support relationship",
                    },
                    "supporting_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of evidence that support this claim",
                    },
                    "contradicting_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of evidence that contradict this claim",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the assessment",
                    },
                    "needs_more_evidence": {
                        "type": "boolean",
                        "description": "Whether this claim needs additional evidence",
                    },
                },
                "required": [
                    "claim_id",
                    "status",
                    "confidence",
                    "supporting_evidence_ids",
                    "reasoning",
                ],
            },
            "description": "Per-claim assessments",
        },
    },
    "required": ["assessments"],
}


def build_validator_system_prompt() -> str:
    """Build the system prompt for the Claim Validator agent.

    Returns:
        The system prompt string.
    """
    return (
        "You are a claim validator for a structured deliberation system.\n"
        "\n"
        "Your task is to assess whether available evidence supports, "
        "weakens, or contradicts each claim.\n"
        "\n"
        "RULES:\n"
        "1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.\n"
        "2. Assess EVERY claim provided.\n"
        "3. For each claim, determine:\n"
        "   - supported: Evidence clearly and directly supports the claim\n"
        "   - weak: Evidence provides only indirect or partial support\n"
        "   - unsupported: No evidence supports the claim\n"
        "   - contradicted: Evidence directly contradicts the claim\n"
        "4. Match evidence semantically, not just by keyword overlap.\n"
        "5. Consider paraphrasing and indirect support.\n"
        "6. Score confidence from 0.0 (no support) to 1.0 (strong support).\n"
        "7. Note which claims need additional evidence.\n"
        "8. Plan and value claims that don't require evidence should be "
        'marked as "supported" with confidence 1.0.\n'
        "\n"
        "OUTPUT FORMAT (JSON only):\n"
        "{\n"
        '  "assessments": [\n'
        "    {\n"
        '      "claim_id": "id",\n'
        '      "status": "supported|weak|unsupported|contradicted",\n'
        '      "confidence": 0.0-1.0,\n'
        '      "supporting_evidence_ids": ["ev_id1", ...],\n'
        '      "contradicting_evidence_ids": ["ev_id2", ...],\n'
        '      "reasoning": "brief explanation",\n'
        '      "needs_more_evidence": true/false\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def build_validator_user_prompt(
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:
    """Build the user prompt for the Claim Validator agent.

    Args:
        claims: List of claim dicts with claim_id, claim_type, text.
        evidence: List of evidence dicts with evidence_id, source, excerpt.

    Returns:
        The user prompt string.
    """
    parts = ["CLAIMS TO VALIDATE:"]

    for c in claims[:15]:
        cid = c.get("claim_id", "?")
        ctype = c.get("claim_type", "?")
        text = str(c.get("text", ""))[:300]
        parts.append(f"  [{cid}] ({ctype}): {text}")

    if evidence:
        parts.append("\nAVAILABLE EVIDENCE:")
        for e in evidence[:10]:
            eid = e.get("evidence_id", "?")
            source = str(e.get("source", "?"))[:100]
            excerpt = str(e.get("excerpt", ""))[:300]
            parts.append(f"  [{eid}] (source: {source}): {excerpt}")
    else:
        parts.append("\nAVAILABLE EVIDENCE: None")

    parts.append("\nAssess whether the evidence supports each claim. Output JSON only.")

    return "\n".join(parts)


# JSON schema for Synthesizer output
SYNTHESIZER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "description": "A coherent synthesized recommendation",
        },
        "summary": {
            "type": "string",
            "description": "Brief summary of the synthesized recommendation",
        },
        "rationale": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key reasons supporting this synthesis (max 6)",
            "maxItems": 6,
        },
        "key_tradeoffs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key tradeoffs or tensions between the options (max 4)",
            "maxItems": 4,
        },
    },
    "required": ["recommendation", "summary", "rationale", "key_tradeoffs"],
}


def build_synthesizer_system_prompt() -> str:
    """Build the system prompt for the Synthesizer (LLM reduce) agent.

    Returns:
        The system prompt string.
    """
    return (
        "You are a decision synthesis expert for a structured deliberation system.\n"
        "\n"
        "Your task is to synthesize multiple competing recommendations from a "
        "structured deliberation into a single coherent recommendation.\n"
        "\n"
        "RULES:\n"
        "1. Output ONLY valid JSON. No markdown, no explanations, no code blocks.\n"
        "2. If options agree, combine their strengths into a unified recommendation.\n"
        "3. If options conflict, acknowledge the tension explicitly and explain "
        "the tradeoff.\n"
        "4. NEVER simply concatenate contradictory recommendations.\n"
        "5. Prioritize higher-scoring options but incorporate insights from all.\n"
        "6. Be specific about which aspects of each option you adopt and why.\n"
        "7. List the key tradeoffs the decision-maker should be aware of.\n"
        "\n"
        "OUTPUT FORMAT (JSON only):\n"
        "{\n"
        '  "recommendation": "A coherent synthesized recommendation",\n'
        '  "summary": "Brief summary",\n'
        '  "rationale": ["reason 1", ...],\n'
        '  "key_tradeoffs": ["tradeoff 1", ...]\n'
        "}"
    )


def build_synthesizer_user_prompt(
    question: str,
    survivors: list[dict[str, Any]],
) -> str:
    """Build the user prompt for the Synthesizer agent.

    Args:
        question: The deliberation question.
        survivors: List of survivor dicts, each with label, score,
                   recommendation, rationale, pros, cons.

    Returns:
        The user prompt string.
    """
    parts = [
        f"QUESTION: {question}",
        f"\nThe deliberation produced {len(survivors)} surviving options (ranked by score):\n",
    ]

    for i, s in enumerate(survivors, 1):
        label = s.get("label", "?")
        score = s.get("score", 0.0)
        rec = str(s.get("recommendation", ""))[:500]
        parts.append(f"--- OPTION {i}: {label} (score: {score:.2f}) ---")
        parts.append(f"Recommendation: {rec}")

        rationale = s.get("rationale", [])
        if rationale:
            parts.append("Rationale:")
            for r in rationale[:4]:
                parts.append(f"  - {str(r)[:200]}")

        pros = s.get("pros", [])
        if pros:
            parts.append("Pros:")
            for p in pros[:3]:
                parts.append(f"  + {str(p)[:200]}")

        cons = s.get("cons", [])
        if cons:
            parts.append("Cons:")
            for c in cons[:3]:
                parts.append(f"  - {str(c)[:200]}")

        parts.append("")

    parts.append(
        "Synthesize these options into a single coherent recommendation. "
        "If options conflict, acknowledge the tension and explain the tradeoff. "
        "Output JSON only."
    )

    return "\n".join(parts)
