"""LLM-backed Claim Validator agent for Delibera.

Uses an LLM to semantically assess whether evidence supports claims,
replacing the keyword-based heuristic with semantic understanding.
"""

from typing import TYPE_CHECKING, Any

from delibera.epistemics.models import Claim, ClaimStatus, ClaimType, Evidence
from delibera.epistemics.validate import ClaimCheckReport
from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_validator_system_prompt, build_validator_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient


class ClaimValidatorLLM:
    """LLM-backed claim validator.

    Replaces keyword-based evidence matching with LLM semantic assessment.
    The LLM evaluates whether evidence actually supports, weakens, or
    contradicts each claim, providing confidence scores and reasoning.

    Attributes:
        llm_client: The LLM client to use for generation.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.1 for consistency).
        max_output_tokens: Maximum output tokens (default 800).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1500,
    ) -> None:
        """Initialize the LLM claim validator.

        Args:
            llm_client: The LLM client for generation.
            model: Model name (None uses client default).
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens (default 1500 for multi-claim output).
        """
        self._llm_client = llm_client
        self._model = model or ""
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def validate(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> ClaimCheckReport:
        """Validate claims against evidence using LLM.

        Args:
            claims: List of claims to validate.
            evidence: List of available evidence from ledger.

        Returns:
            ClaimCheckReport with validation results and support_relations.
        """
        if not claims:
            return ClaimCheckReport(
                supported=0, weak=0, unsupported=0, details=[], support_relations={}
            )

        # Build claim and evidence dicts for the prompt
        claims_data = [
            {
                "claim_id": c.claim_id,
                "claim_type": c.claim_type.value,
                "text": c.text,
            }
            for c in claims
        ]

        evidence_data = [
            {
                "evidence_id": e.evidence_id,
                "source": e.source,
                "excerpt": e.excerpt[:500],
            }
            for e in evidence
        ]

        # Build LLM request
        system_prompt = build_validator_system_prompt()
        user_prompt = build_validator_user_prompt(
            claims=claims_data,
            evidence=evidence_data,
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
                "role": "validator",
                "step": "CLAIM_CHECK",
            },
        )

        # Generate response
        response = self._llm_client.generate(request)
        parsed = response.parsed_json or {}

        return self._normalize_output(parsed, claims, evidence)

    def _normalize_output(
        self,
        parsed: dict[str, Any],
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> ClaimCheckReport:
        """Normalize LLM output to ClaimCheckReport format.

        Maps LLM assessments back to claims, updates their status,
        and builds the report with support_relations.

        Args:
            parsed: The parsed JSON from LLM.
            claims: The original claims list.
            evidence: The original evidence list.

        Returns:
            ClaimCheckReport with normalized results.
        """
        # Build lookups
        claim_map = {c.claim_id: c for c in claims}
        evidence_ids_set = {e.evidence_id for e in evidence}

        # Parse assessments from LLM
        assessments_raw = parsed.get("assessments", [])
        if not isinstance(assessments_raw, list):
            assessments_raw = []

        # Build assessment lookup by claim_id
        assessment_map: dict[str, dict[str, Any]] = {}
        for a in assessments_raw:
            if not isinstance(a, dict):
                continue
            cid = str(a.get("claim_id", "")).strip()
            if cid and cid in claim_map:
                assessment_map[cid] = a

        # Apply assessments to claims
        supported = 0
        weak = 0
        unsupported = 0
        details: list[dict[str, Any]] = []
        support_relations: dict[str, list[str]] = {}

        for claim in claims:
            assessment = assessment_map.get(claim.claim_id)

            if assessment:
                status = self._parse_status(assessment.get("status", ""), claim.claim_type)
                supporting_ids = self._filter_evidence_ids(
                    assessment.get("supporting_evidence_ids", []),
                    evidence_ids_set,
                )
                contradicting_ids = self._filter_evidence_ids(
                    assessment.get("contradicting_evidence_ids", []),
                    evidence_ids_set,
                )

                # Update confidence from LLM
                try:
                    confidence = float(assessment.get("confidence", claim.confidence))
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = claim.confidence

                claim.status = status
                claim.confidence = confidence

                if supporting_ids:
                    support_relations[claim.claim_id] = supporting_ids

                # Build detail with extra LLM info
                detail: dict[str, Any] = {
                    "claim_id": claim.claim_id,
                    "status": status.value,
                    "type": claim.claim_type.value,
                    "supporting_evidence_ids": supporting_ids,
                    "contradicting_evidence_ids": contradicting_ids,
                    "reasoning": str(assessment.get("reasoning", ""))[:300],
                    "needs_more_evidence": bool(assessment.get("needs_more_evidence", False)),
                    "llm_confidence": confidence,
                }
            else:
                # No LLM assessment: fall back to type-based defaults
                status = self._default_status(claim.claim_type, bool(evidence))
                claim.status = status

                detail = {
                    "claim_id": claim.claim_id,
                    "status": status.value,
                    "type": claim.claim_type.value,
                    "supporting_evidence_ids": [],
                    "reasoning": "No LLM assessment available",
                    "needs_more_evidence": claim.claim_type == ClaimType.FACT,
                }

            # Count by status
            if claim.status == ClaimStatus.SUPPORTED:
                supported += 1
            elif claim.status == ClaimStatus.WEAK:
                weak += 1
            elif claim.status == ClaimStatus.UNSUPPORTED:
                unsupported += 1

            details.append(detail)

        return ClaimCheckReport(
            supported=supported,
            weak=weak,
            unsupported=unsupported,
            details=details,
            support_relations=support_relations,
        )

    @staticmethod
    def _parse_status(status_str: str, claim_type: ClaimType) -> ClaimStatus:
        """Parse and validate a status string from LLM output.

        Args:
            status_str: Raw status string from LLM.
            claim_type: The claim type for fallback logic.

        Returns:
            Validated ClaimStatus.
        """
        status_str = status_str.strip().lower()
        valid_statuses = {"supported", "weak", "unsupported", "contradicted"}

        if status_str not in valid_statuses:
            # Fall back to type-based default
            return ClaimValidatorLLM._default_status(claim_type, evidence_exists=True)

        # Map "contradicted" to "unsupported" (contradicted is worse than unsupported)
        if status_str == "contradicted":
            return ClaimStatus.UNSUPPORTED

        return ClaimStatus(status_str)

    @staticmethod
    def _default_status(claim_type: ClaimType, evidence_exists: bool) -> ClaimStatus:
        """Get default status for a claim type when no LLM assessment available.

        Args:
            claim_type: The type of claim.
            evidence_exists: Whether any evidence is available.

        Returns:
            Default ClaimStatus.
        """
        if claim_type in (ClaimType.PLAN, ClaimType.VALUE):
            return ClaimStatus.SUPPORTED
        if claim_type == ClaimType.FACT:
            return ClaimStatus.UNSUPPORTED
        # INFERENCE
        if evidence_exists:
            return ClaimStatus.WEAK
        return ClaimStatus.WEAK

    @staticmethod
    def _filter_evidence_ids(
        raw_ids: Any,
        valid_ids: set[str],
    ) -> list[str]:
        """Filter evidence IDs to only include valid ones.

        Args:
            raw_ids: Raw list from LLM (may contain invalid IDs).
            valid_ids: Set of valid evidence IDs.

        Returns:
            Filtered list of valid evidence IDs.
        """
        if not isinstance(raw_ids, list):
            return []
        return [str(eid).strip() for eid in raw_ids if str(eid).strip() in valid_ids]
