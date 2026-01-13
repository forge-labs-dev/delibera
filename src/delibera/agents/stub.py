"""Stub agents for v0 testing without LLMs."""

import hashlib
from typing import TYPE_CHECKING, Any

from delibera.tools.spec import ToolDenied, ToolExecutionError

if TYPE_CHECKING:
    from delibera.tools import ToolCallback


class PlannerStub:
    """Stub planner that returns deterministic branch labels.

    For v0 testing, always returns 3 options.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate branch labels for expansion.

        Args:
            context: Must contain "question" key.

        Returns:
            Dict with "branches" key containing list of labels.
        """
        question = context.get("question", "")
        return {
            "branches": [
                f"Option A: Direct approach to '{question[:30]}...'",
                f"Option B: Alternative approach to '{question[:30]}...'",
                f"Option C: Conservative approach to '{question[:30]}...'",
            ],
            "role": "planner",
            "step": "PLAN",
        }


class ProposerStub:
    """Stub proposer that returns deterministic proposals with scores.

    Scores are deterministic based on option label for testing:
    - Option A: 0.8 (highest)
    - Option B: 0.6 (medium)
    - Option C: 0.4 (lowest, will be pruned)

    Optionally uses calculator tool to compute confidence when tool callback provided.

    Emits fact claims with keywords that match evidence excerpts for testing
    claim↔evidence linkage.
    """

    # Deterministic base scores for testing
    _BASE_SCORES: dict[str, str] = {
        "A": "0.75+0.05",  # = 0.8
        "B": "0.55+0.05",  # = 0.6
        "C": "0.35+0.05",  # = 0.4
    }

    # Fallback scores when tool is not available
    _FALLBACK_SCORES: dict[str, float] = {
        "A": 0.8,
        "B": 0.6,
        "C": 0.4,
    }

    # Fact claims that match ResearcherStub's search terms for evidence linkage
    _FACT_CLAIMS: dict[str, str] = {
        "A": "UV installation is extremely fast compared to pip.",
        "B": "UV is compatible with existing pip workflows.",
        "C": "UV generates reproducible lockfiles for builds.",
    }

    def execute(
        self,
        context: dict[str, Any],
        tool: "ToolCallback | None" = None,
    ) -> dict[str, Any]:
        """Generate a proposal for a branch.

        Args:
            context: Must contain "label" key with branch label,
                and "question" key with the original question.
            tool: Optional tool callback for computing confidence.

        Returns:
            Dict with proposal details and score.
        """
        label = context.get("label", "")
        question = context.get("question", "")

        # Extract option letter (A, B, or C) from label
        option_letter = "A"  # default
        for letter in ["A", "B", "C"]:
            if f"Option {letter}" in label:
                option_letter = letter
                break

        # Compute score using calculator if tool is available
        score: float
        used_tool = False
        if tool is not None:
            expression = self._BASE_SCORES.get(option_letter, "0.5")
            try:
                result = tool("calculator", {"expression": expression})
                score = float(result.get("result", 0.5))
                used_tool = True
            except Exception:
                # Fallback if tool fails
                score = self._FALLBACK_SCORES.get(option_letter, 0.5)
        else:
            score = self._FALLBACK_SCORES.get(option_letter, 0.5)

        # Get fact claim that matches evidence keywords
        fact_claim = self._FACT_CLAIMS.get(option_letter, "")

        output: dict[str, Any] = {
            "proposal": f"Proposal for {label}",
            "summary": f"This approach addresses '{question[:50]}...' by...",
            "pros": [f"Pro 1 for option {option_letter}", f"Pro 2 for option {option_letter}"],
            "cons": [f"Con 1 for option {option_letter}"],
            "facts": [fact_claim] if fact_claim else [],
            "score": score,
            "role": "proposer",
        }

        if used_tool:
            output["confidence_computed_by"] = "calculator"

        return output


class SubplannerStub:
    """Stub subplanner that returns deterministic sub-branch labels.

    For 2-level expansion, each option generates 2 subplans.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate sub-branch labels for second-level expansion.

        Args:
            context: Must contain "label" key with parent option label.

        Returns:
            Dict with "sub_branches" key containing list of subplan labels.
        """
        label = context.get("label", "Unknown")

        # Extract option letter from parent label
        option_letter = "X"
        for letter in ["A", "B", "C"]:
            if f"Option {letter}" in label:
                option_letter = letter
                break

        return {
            "sub_branches": [
                f"Subplan {option_letter}.1: Detailed implementation",
                f"Subplan {option_letter}.2: Alternative approach",
            ],
            "role": "subplanner",
            "step": "SUBPLAN",
        }


class ResearcherStub:
    """Stub researcher that collects evidence via docs.read tool.

    Deterministically reads from evidence/uv_notes.txt and extracts
    an excerpt based on the option label.

    Evidence is returned in a structured format for the engine to
    merge into the node ledger.
    """

    # Map option letters to search terms for deterministic excerpt selection
    _SEARCH_TERMS: dict[str, str] = {
        "A": "fast",  # Will find "10-100x faster"
        "B": "compatible",  # Will find "Compatible with"
        "C": "reproducible",  # Will find "reproducible uv.lock"
    }

    # Default evidence file
    _EVIDENCE_FILE = "evidence/uv_notes.txt"

    def execute(
        self,
        context: dict[str, Any],
        tool: "ToolCallback | None" = None,
    ) -> dict[str, Any]:
        """Collect evidence for a branch.

        Args:
            context: Must contain "label" key with branch label.
            tool: Tool callback for docs.read.

        Returns:
            Dict with evidence items and notes.
        """
        label = context.get("label", "")

        # Extract option letter
        option_letter = "A"  # default
        for letter in ["A", "B", "C"]:
            if f"Option {letter}" in label:
                option_letter = letter
                break

        # Get search term for this option
        search_term = self._SEARCH_TERMS.get(option_letter, "fast")

        evidence_items: list[dict[str, Any]] = []
        notes: list[str] = []

        if tool is not None:
            try:
                # Read the evidence file
                result = tool("docs.read", {"path": self._EVIDENCE_FILE})
                text = result.get("text", "")

                # Extract an excerpt containing the search term
                excerpt = self._extract_excerpt(text, search_term)

                if excerpt:
                    evidence_items.append(
                        {
                            "source": self._EVIDENCE_FILE,
                            "excerpt": excerpt,
                        }
                    )
                    notes.append(f"Found evidence for '{search_term}' in {self._EVIDENCE_FILE}")
                else:
                    notes.append(f"No excerpt found for '{search_term}'")

            except (KeyError, ValueError, OSError, ToolDenied, ToolExecutionError) as e:
                # KeyError: tool not registered
                # ValueError: invalid input
                # OSError: file read error
                # ToolDenied: policy denied tool access
                # ToolExecutionError: tool execution failed
                notes.append(f"Failed to read evidence: {e}")
        else:
            notes.append("No tool callback provided; skipping evidence collection")

        return {
            "evidence": evidence_items,
            "notes": notes,
            "role": "researcher",
            "step": "RESEARCH",
        }

    def _extract_excerpt(self, text: str, search_term: str) -> str:
        """Extract an excerpt containing the search term.

        Args:
            text: The full document text.
            search_term: The term to search for.

        Returns:
            A short excerpt (up to 200 chars) containing the term, or empty string.
        """
        # Case-insensitive search
        lower_text = text.lower()
        search_lower = search_term.lower()

        pos = lower_text.find(search_lower)
        if pos == -1:
            return ""

        # Extract context around the match
        start = max(0, pos - 50)
        end = min(len(text), pos + 150)

        # Find line boundaries for cleaner excerpt
        excerpt = text[start:end]

        # Trim to complete words at boundaries
        if start > 0:
            # Find first space and trim before it
            space_pos = excerpt.find(" ")
            if space_pos > 0 and space_pos < 20:
                excerpt = "..." + excerpt[space_pos + 1 :]

        if end < len(text):
            # Find last space and trim after it
            space_pos = excerpt.rfind(" ")
            if space_pos > len(excerpt) - 20:
                excerpt = excerpt[:space_pos] + "..."

        return excerpt.strip()


class RedTeamStub:
    """Stub red team agent that raises objections deterministically.

    Examines node artifact and ledger to identify issues:
    1. If any inference claims are weak after validation, create blocking objection
    2. If no evidence exists, create blocking objection targeting artifact
    3. Otherwise, create one nonblocking objection for tradeoff consideration

    Generates up to 2 objections per node in deterministic order.
    Objection IDs are sha1 hashes for determinism.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate objections based on artifact and claims.

        Args:
            context: Must contain:
                - "node_id": Node ID for deterministic objection IDs
                - "artifact": The node's artifact dict
                - "claims": List of claim dicts with "status", "claim_type", "claim_id"
                - "evidence_count": Number of evidence items in ledger

        Returns:
            Dict with "objections" list and "role" field.
        """
        node_id = context.get("node_id", "")
        claims = context.get("claims", [])
        evidence_count = context.get("evidence_count", 0)

        objections: list[dict[str, Any]] = []

        # Rule 1: Check for weak inference claims
        weak_inference_claims = [
            c for c in claims if c.get("claim_type") == "inference" and c.get("status") == "weak"
        ]

        if weak_inference_claims:
            # Create blocking objection targeting first weak inference claim
            first_weak = weak_inference_claims[0]
            target = first_weak.get("claim_id", "artifact")
            rationale = "Inference claim is weak; provide evidence or revise."
            severity = "blocking"

            obj_id = self._make_objection_id(node_id, target, severity, rationale)
            objections.append(
                {
                    "objection_id": obj_id,
                    "target": target,
                    "severity": severity,
                    "status": "open",
                    "rationale": rationale,
                }
            )

        # Rule 2: Check for missing evidence
        if evidence_count == 0:
            target = "artifact"
            rationale = "No evidence attached; decision is not supported."
            severity = "blocking"

            obj_id = self._make_objection_id(node_id, target, severity, rationale)
            # Avoid duplicate if this would be the same as rule 1 objection
            if not any(o["objection_id"] == obj_id for o in objections):
                objections.append(
                    {
                        "objection_id": obj_id,
                        "target": target,
                        "severity": severity,
                        "status": "open",
                        "rationale": rationale,
                    }
                )

        # Rule 3: If no blocking objections yet, create nonblocking tradeoff objection
        if not objections:
            target = "artifact"
            rationale = "Consider alternative tradeoffs."
            severity = "nonblocking"

            obj_id = self._make_objection_id(node_id, target, severity, rationale)
            objections.append(
                {
                    "objection_id": obj_id,
                    "target": target,
                    "severity": severity,
                    "status": "open",
                    "rationale": rationale,
                }
            )

        # Limit to 2 objections max (deterministic order)
        objections = objections[:2]

        return {
            "objections": objections,
            "role": "redteam",
            "step": "REDTEAM",
        }

    def _make_objection_id(
        self,
        node_id: str,
        target: str,
        severity: str,
        rationale: str,
    ) -> str:
        """Generate deterministic objection ID.

        Args:
            node_id: The node ID.
            target: Target claim_id or "artifact".
            severity: "blocking" or "nonblocking".
            rationale: The objection rationale.

        Returns:
            10-character hex hash as objection ID.
        """
        content = f"{node_id}:{target}:{severity}:{rationale}"
        return hashlib.sha1(content.encode()).hexdigest()[:10]
