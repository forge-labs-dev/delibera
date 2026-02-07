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
        question = context.get("question", "")  # noqa: F841
        return {
            "branches": [
                "Option A: Direct migration approach",
                "Option B: Incremental adoption approach",
                "Option C: Conservative wait-and-see approach",
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

        # Generate meaningful summaries based on option
        summaries = {
            "A": (
                "Migrate fully to GraphQL, replacing the REST API entirely. "
                "Provides a unified API experience but requires upfront investment."
            ),
            "B": (
                "Adopt GraphQL incrementally alongside REST. New features use GraphQL "
                "while existing endpoints remain on REST, allowing gradual migration."
            ),
            "C": (
                "Maintain the current REST API and evaluate GraphQL adoption later. "
                "Monitor industry trends and wait for the ecosystem to mature."
            ),
        }

        output: dict[str, Any] = {
            "proposal": f"Proposal for {label}",
            "summary": summaries.get(option_letter, f"Approach for addressing: {question}"),
            "pros": [f"Pro 1 for option {option_letter}", f"Pro 2 for option {option_letter}"],
            "cons": [f"Con 1 for option {option_letter}"],
            "facts": [fact_claim] if fact_claim else [],
            "score": score,
            "role": "proposer",
            "sub_branches": [
                f"Subplan {option_letter}.1: Detailed implementation",
                f"Subplan {option_letter}.2: Alternative approach",
            ],
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
    """Stub researcher that collects evidence via docs.search + docs.read tools.

    Uses docs.search to discover relevant files within the evidence pack,
    then docs.read to extract content from top results.

    Evidence is returned in a structured format for the engine to
    merge into the node ledger.
    """

    # Map option letters to search terms for deterministic excerpt selection
    _SEARCH_TERMS: dict[str, str] = {
        "A": "fast",  # Will find "10-100x faster"
        "B": "compatible",  # Will find "Compatible with"
        "C": "reproducible",  # Will find "reproducible uv.lock"
    }

    # Maximum number of search results to read
    _MAX_READ_RESULTS = 2

    def execute(
        self,
        context: dict[str, Any],
        tool: "ToolCallback | None" = None,
    ) -> dict[str, Any]:
        """Collect evidence for a branch.

        Uses docs.search to find relevant files, then docs.read to get content.

        Args:
            context: Must contain "label" key with branch label.
            tool: Tool callback for docs.search and docs.read.

        Returns:
            Dict with evidence items and notes.
        """
        label = context.get("label", "")
        question = context.get("question", "")

        # Extract option letter
        option_letter = "A"  # default
        for letter in ["A", "B", "C"]:
            if f"Option {letter}" in label:
                option_letter = letter
                break

        # Build search query from option-specific term + first keywords from label/question
        base_term = self._SEARCH_TERMS.get(option_letter, "fast")
        search_query = self._build_search_query(base_term, label, question)

        evidence_items: list[dict[str, Any]] = []
        notes: list[str] = []

        if tool is not None:
            # Step 1: Search for relevant files
            try:
                search_result = tool(
                    "docs.search",
                    {"query": search_query, "max_results": self._MAX_READ_RESULTS},
                )
                search_hits = search_result.get("results", [])
                notes.append(f"docs.search found {len(search_hits)} results for '{search_query}'")

            except (KeyError, ToolDenied, ToolExecutionError) as e:
                # docs.search not available, fall back to direct read
                notes.append(f"docs.search failed ({e}), falling back to direct read")
                search_hits = []

            # Step 2: Read top results and extract evidence
            if search_hits:
                for hit in search_hits[: self._MAX_READ_RESULTS]:
                    path = hit.get("path", "")
                    snippet = hit.get("snippet", "")

                    if not path:
                        continue

                    try:
                        read_result = tool("docs.read", {"path": path})
                        text = read_result.get("text", "")

                        # Extract excerpt around the search term
                        excerpt = self._extract_excerpt(text, base_term)
                        if not excerpt and snippet:
                            # Use snippet from search if no excerpt found
                            excerpt = snippet

                        if excerpt:
                            evidence_items.append(
                                {
                                    "source": path,
                                    "excerpt": excerpt,
                                }
                            )
                            notes.append(f"Found evidence in {path}")

                    except (KeyError, ToolDenied, ToolExecutionError) as e:
                        notes.append(f"Failed to read {path}: {e}")

            # Fallback: if no search hits, try reading a default file
            if not search_hits and not evidence_items:
                fallback_path = "uv_notes.txt"
                try:
                    read_result = tool("docs.read", {"path": fallback_path})
                    text = read_result.get("text", "")

                    excerpt = self._extract_excerpt(text, base_term)
                    if excerpt:
                        evidence_items.append(
                            {
                                "source": fallback_path,
                                "excerpt": excerpt,
                            }
                        )
                        notes.append(f"Fallback: found evidence in {fallback_path}")

                except (KeyError, ToolDenied, ToolExecutionError) as e:
                    notes.append(f"Fallback read failed: {e}")
        else:
            notes.append("No tool callback provided; skipping evidence collection")

        return {
            "evidence": evidence_items,
            "notes": notes,
            "role": "researcher",
            "step": "RESEARCH",
        }

    def _build_search_query(self, base_term: str, _label: str, question: str) -> str:
        """Build a search query from the base term and context.

        Args:
            base_term: The primary search term for this option.
            _label: The branch label (unused, reserved for future use).
            question: The original question.

        Returns:
            A search query string.
        """
        # Start with the base term
        terms = [base_term]

        # Add a keyword from question (first 3 words, skip common words)
        skip_words = {"should", "we", "the", "a", "an", "is", "are", "what", "how", "why"}
        words = question.lower().split()
        for word in words[:6]:
            clean = word.strip("?.,!\"'")
            if clean and clean not in skip_words and len(clean) > 2:
                terms.append(clean)
                break

        return " ".join(terms)

    def _extract_excerpt(self, text: str, search_term: str) -> str:
        """Extract an excerpt containing the search term.

        Args:
            text: The full document text.
            search_term: The term to search for.

        Returns:
            An excerpt (up to 400 chars) containing the term, or empty string.
        """
        # Case-insensitive search
        lower_text = text.lower()
        search_lower = search_term.lower()

        pos = lower_text.find(search_lower)
        if pos == -1:
            return ""

        # Extract more context around the match (400 chars total)
        start = max(0, pos - 100)
        end = min(len(text), pos + 300)

        # Find line boundaries for cleaner excerpt
        excerpt = text[start:end]

        # Trim to complete words at boundaries
        if start > 0:
            # Find first space and trim before it
            space_pos = excerpt.find(" ")
            if space_pos > 0 and space_pos < 30:
                excerpt = "..." + excerpt[space_pos + 1 :]

        if end < len(text):
            # Find last space and trim after it
            space_pos = excerpt.rfind(" ")
            if space_pos > len(excerpt) - 30:
                excerpt = excerpt[:space_pos] + "..."

        return excerpt.strip()


class RefinerStub:
    """Stub refiner agent that deterministically improves artifacts.

    Refines artifacts by:
    1. Marking weak claims as "refined" in the text
    2. Adding clarification notes for unsupported claims
    3. Producing a deterministically improved artifact

    This enables testing the refinement loop without LLM calls.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Refine an artifact based on validation feedback.

        Args:
            context: Must contain:
                - "node_id": Node ID for deterministic output
                - "artifact": The current artifact dict
                - "claims": List of claim dicts with "status", "claim_id", "text"
                - "round": Current refinement round number

        Returns:
            Dict with refined artifact and refinement notes.
        """
        node_id = context.get("node_id", "")  # noqa: F841
        artifact = context.get("artifact", {})
        claims = context.get("claims", [])
        round_num = context.get("round", 1)

        # Count claim categories
        weak_claims = [c for c in claims if c.get("status") == "weak"]
        unsupported_claims = [c for c in claims if c.get("status") == "unsupported"]

        # Build refinement notes
        refinement_notes: list[str] = []

        # Deterministically "fix" weak claims
        for claim in weak_claims[:2]:  # Fix up to 2 weak claims per round
            claim_id = claim.get("claim_id", "unknown")
            refinement_notes.append(f"Strengthened weak claim {claim_id} with additional context")

        # Add clarification for unsupported claims
        for claim in unsupported_claims[:1]:  # Address up to 1 unsupported claim per round
            claim_id = claim.get("claim_id", "unknown")
            refinement_notes.append(f"Added clarification for unsupported claim {claim_id}")

        # Create refined artifact (shallow copy with updates)
        refined_artifact = dict(artifact)

        # Mark as refined
        refined_artifact["refinement_round"] = round_num
        refined_artifact["refinement_notes"] = refinement_notes

        # Update summary to indicate refinement
        original_summary = artifact.get("summary", "")
        if original_summary:
            refined_artifact["summary"] = f"{original_summary} [Refined in round {round_num}]"

        # Deterministically improve score (diminishing returns)
        original_score = artifact.get("score", 0.5)
        improvement = 0.05 / round_num  # Smaller improvements each round
        refined_artifact["score"] = min(1.0, original_score + improvement)

        return {
            "artifact": refined_artifact,
            "refinement_notes": refinement_notes,
            "weak_addressed": len(weak_claims[:2]),
            "unsupported_addressed": len(unsupported_claims[:1]),
            "role": "refiner",
            "step": "REFINE",
        }


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
