"""Unit tests for the epistemics package."""

from delibera.epistemics.extract import extract_claims
from delibera.epistemics.ledger import Ledger, merge_ledgers
from delibera.epistemics.models import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    Objection,
    ObjectionSeverity,
    ObjectionStatus,
)
from delibera.epistemics.validate import validate_claims


class TestClaimExtraction:
    """Tests for claim extraction from artifacts."""

    def test_extract_plan_claim_from_proposal(self) -> None:
        """Test that proposal text is extracted as a plan claim."""
        artifact = {"proposal": "We should adopt uv for package management."}
        claims = extract_claims(artifact, "proposer")

        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.PLAN
        assert claims[0].text == "We should adopt uv for package management."
        assert claims[0].confidence == 0.8
        assert claims[0].owner == "proposer"

    def test_extract_inference_claims_from_pros(self) -> None:
        """Test that pros list items are extracted as inference claims."""
        artifact = {
            "proposal": "Main recommendation",
            "pros": ["Faster than pip", "Better dependency resolution"],
        }
        claims = extract_claims(artifact, "proposer")

        # 1 plan + 2 inference claims
        assert len(claims) == 3
        inference_claims = [c for c in claims if c.claim_type == ClaimType.INFERENCE]
        assert len(inference_claims) == 2
        assert all(c.confidence == 0.6 for c in inference_claims)

    def test_extract_claims_skips_empty_strings(self) -> None:
        """Test that empty strings are not extracted as claims."""
        artifact = {
            "proposal": "Main recommendation",
            "pros": ["Valid pro", "", "  ", "Another pro"],
        }
        claims = extract_claims(artifact, "proposer")

        # 1 plan + 2 valid inference claims
        assert len(claims) == 3

    def test_claim_id_is_deterministic(self) -> None:
        """Test that claim IDs are deterministic for same input."""
        artifact = {"proposal": "Test proposal"}

        claims_1 = extract_claims(artifact, "proposer")
        claims_2 = extract_claims(artifact, "proposer")

        assert claims_1[0].claim_id == claims_2[0].claim_id

    def test_claim_id_same_text_same_id(self) -> None:
        """Test that identical text produces the same claim ID regardless of node."""
        artifact = {"proposal": "Test proposal"}

        claims_1 = extract_claims(artifact, "proposer")
        claims_2 = extract_claims(artifact, "proposer")

        assert claims_1[0].claim_id == claims_2[0].claim_id

    def test_extract_from_empty_artifact(self) -> None:
        """Test extraction from empty artifact returns empty list."""
        claims = extract_claims({}, "proposer")
        assert claims == []

    def test_extract_handles_none_values(self) -> None:
        """Test that None values in artifact are handled gracefully."""
        artifact = {
            "proposal": None,
            "pros": None,
            "rationale": None,
        }
        claims = extract_claims(artifact, "proposer")
        assert claims == []

    def test_extract_handles_mixed_none_values(self) -> None:
        """Test extraction with valid proposal but None in lists."""
        artifact = {
            "proposal": "Valid proposal",
            "pros": [None, "Valid pro", None],
        }
        claims = extract_claims(artifact, "proposer")

        # 1 plan claim + 1 valid inference claim
        assert len(claims) == 2
        plan_claims = [c for c in claims if c.claim_type == ClaimType.PLAN]
        inference_claims = [c for c in claims if c.claim_type == ClaimType.INFERENCE]
        assert len(plan_claims) == 1
        assert len(inference_claims) == 1
        assert inference_claims[0].text == "Valid pro"


class TestClaimValidation:
    """Tests for claim validation."""

    def test_plan_claims_are_supported(self) -> None:
        """Test that plan claims are automatically supported."""
        claims = [
            Claim(
                claim_id="c1",
                text="We should do X",
                claim_type=ClaimType.PLAN,
                confidence=0.8,
                owner="proposer",
            )
        ]
        report = validate_claims(claims, [])

        assert report.supported == 1
        assert report.weak == 0
        assert report.unsupported == 0
        assert claims[0].status == ClaimStatus.SUPPORTED

    def test_value_claims_are_supported(self) -> None:
        """Test that value claims are automatically supported."""
        claims = [
            Claim(
                claim_id="c1",
                text="Performance is important",
                claim_type=ClaimType.VALUE,
                confidence=0.7,
                owner="proposer",
            )
        ]
        report = validate_claims(claims, [])

        assert report.supported == 1
        assert claims[0].status == ClaimStatus.SUPPORTED

    def test_inference_claims_are_weak_without_evidence(self) -> None:
        """Test that inference claims are weak in v1 (no evidence)."""
        claims = [
            Claim(
                claim_id="c1",
                text="Therefore, uv is faster",
                claim_type=ClaimType.INFERENCE,
                confidence=0.6,
                owner="proposer",
            )
        ]
        report = validate_claims(claims, [])

        assert report.supported == 0
        assert report.weak == 1
        assert claims[0].status == ClaimStatus.WEAK

    def test_fact_claims_are_unsupported_without_evidence(self) -> None:
        """Test that fact claims are unsupported in v1 (no evidence)."""
        claims = [
            Claim(
                claim_id="c1",
                text="UV is 10x faster than pip",
                claim_type=ClaimType.FACT,
                confidence=0.5,
                owner="proposer",
            )
        ]
        report = validate_claims(claims, [])

        assert report.unsupported == 1
        assert claims[0].status == ClaimStatus.UNSUPPORTED

    def test_validation_report_details(self) -> None:
        """Test that validation report includes details for each claim."""
        claims = [
            Claim("c1", "Plan claim", ClaimType.PLAN, 0.8, "proposer"),
            Claim("c2", "Inference claim", ClaimType.INFERENCE, 0.6, "proposer"),
        ]
        report = validate_claims(claims, [])

        assert len(report.details) == 2
        assert report.details[0]["claim_id"] == "c1"
        assert report.details[0]["status"] == "supported"
        assert report.details[0]["type"] == "plan"

    def test_validation_with_empty_claims(self) -> None:
        """Test validation of empty claim list."""
        report = validate_claims([], [])

        assert report.supported == 0
        assert report.weak == 0
        assert report.unsupported == 0
        assert report.details == []


class TestLedger:
    """Tests for ledger operations."""

    def test_empty_ledger_creation(self) -> None:
        """Test that empty ledger is properly initialized."""
        ledger = Ledger()

        assert ledger.claims == []
        assert ledger.evidence == []
        assert ledger.objections == []
        assert ledger.support_relations == {}

    def test_merge_ledgers_combines_claims(self) -> None:
        """Test that merge combines claims from multiple ledgers."""
        claim1 = Claim("c1", "Claim 1", ClaimType.PLAN, 0.8, "proposer", ClaimStatus.SUPPORTED)
        claim2 = Claim("c2", "Claim 2", ClaimType.PLAN, 0.8, "proposer", ClaimStatus.SUPPORTED)

        ledger1 = Ledger(claims=[claim1])
        ledger2 = Ledger(claims=[claim2])

        merged = merge_ledgers([ledger1, ledger2])

        assert len(merged.claims) == 2
        assert any(c.claim_id == "c1" for c in merged.claims)
        assert any(c.claim_id == "c2" for c in merged.claims)

    def test_merge_ledgers_drops_unsupported_claims(self) -> None:
        """Test that merge drops unsupported claims."""
        supported = Claim("c1", "Supported", ClaimType.PLAN, 0.8, "proposer", ClaimStatus.SUPPORTED)
        unsupported = Claim(
            "c2", "Unsupported", ClaimType.FACT, 0.5, "proposer", ClaimStatus.UNSUPPORTED
        )

        ledger = Ledger(claims=[supported, unsupported])
        merged = merge_ledgers([ledger])

        assert len(merged.claims) == 1
        assert merged.claims[0].claim_id == "c1"

    def test_merge_ledgers_deduplicates_claims(self) -> None:
        """Test that merge doesn't duplicate claims with same ID."""
        claim = Claim("c1", "Claim", ClaimType.PLAN, 0.8, "proposer", ClaimStatus.SUPPORTED)

        ledger1 = Ledger(claims=[claim])
        ledger2 = Ledger(claims=[claim])

        merged = merge_ledgers([ledger1, ledger2])

        assert len(merged.claims) == 1

    def test_merge_ledgers_preserves_evidence(self) -> None:
        """Test that merge preserves all evidence."""
        evidence1 = Evidence("e1", "source1", "excerpt1", "prov1")
        evidence2 = Evidence("e2", "source2", "excerpt2", "prov2")

        ledger1 = Ledger(evidence=[evidence1])
        ledger2 = Ledger(evidence=[evidence2])

        merged = merge_ledgers([ledger1, ledger2])

        assert len(merged.evidence) == 2

    def test_merge_ledgers_merges_support_relations(self) -> None:
        """Test that merge combines support relations."""
        ledger1 = Ledger(support_relations={"c1": ["e1"]})
        ledger2 = Ledger(support_relations={"c1": ["e2"], "c2": ["e3"]})

        merged = merge_ledgers([ledger1, ledger2])

        assert "c1" in merged.support_relations
        assert "e1" in merged.support_relations["c1"]
        assert "e2" in merged.support_relations["c1"]
        assert "c2" in merged.support_relations

    def test_merge_empty_ledgers(self) -> None:
        """Test merging empty ledgers."""
        merged = merge_ledgers([Ledger(), Ledger()])

        assert merged.claims == []
        assert merged.evidence == []
        assert merged.objections == []


class TestModels:
    """Tests for epistemic models."""

    def test_claim_default_status(self) -> None:
        """Test that new claims have unvalidated status."""
        claim = Claim("c1", "Test", ClaimType.PLAN, 0.8, "proposer")
        assert claim.status == ClaimStatus.UNVALIDATED

    def test_claim_type_enum_values(self) -> None:
        """Test ClaimType enum values."""
        assert ClaimType.FACT.value == "fact"
        assert ClaimType.INFERENCE.value == "inference"
        assert ClaimType.VALUE.value == "value"
        assert ClaimType.PLAN.value == "plan"

    def test_claim_status_enum_values(self) -> None:
        """Test ClaimStatus enum values."""
        assert ClaimStatus.SUPPORTED.value == "supported"
        assert ClaimStatus.WEAK.value == "weak"
        assert ClaimStatus.UNSUPPORTED.value == "unsupported"
        assert ClaimStatus.UNVALIDATED.value == "unvalidated"

    def test_objection_model(self) -> None:
        """Test Objection model creation."""
        objection = Objection(
            objection_id="o1",
            target="c1",
            severity=ObjectionSeverity.BLOCKING,
            status=ObjectionStatus.OPEN,
            rationale="This claim is not justified",
            owner="redteam",
        )

        assert objection.objection_id == "o1"
        assert objection.severity == ObjectionSeverity.BLOCKING
        assert objection.status == ObjectionStatus.OPEN
        assert objection.owner == "redteam"

    def test_evidence_model(self) -> None:
        """Test Evidence model creation."""
        evidence = Evidence(
            evidence_id="e1",
            source="https://example.com",
            excerpt="Relevant text here",
            provenance="tool_call:123",
        )

        assert evidence.evidence_id == "e1"
        assert evidence.source == "https://example.com"

    def test_objection_to_dict(self) -> None:
        """Test Objection.to_dict() serialization."""
        objection = Objection(
            objection_id="o1",
            target="c1",
            severity=ObjectionSeverity.BLOCKING,
            status=ObjectionStatus.OPEN,
            rationale="This claim is not justified",
            owner="redteam",
        )
        d = objection.to_dict()

        assert d["objection_id"] == "o1"
        assert d["target"] == "c1"
        assert d["severity"] == "blocking"
        assert d["status"] == "open"
        assert d["rationale"] == "This claim is not justified"
        assert d["owner"] == "redteam"

    def test_objection_severity_enum_values(self) -> None:
        """Test ObjectionSeverity enum values."""
        assert ObjectionSeverity.BLOCKING.value == "blocking"
        assert ObjectionSeverity.NONBLOCKING.value == "nonblocking"

    def test_objection_status_enum_values(self) -> None:
        """Test ObjectionStatus enum values."""
        assert ObjectionStatus.OPEN.value == "open"
        assert ObjectionStatus.RESOLVED.value == "resolved"
        assert ObjectionStatus.ACCEPTED.value == "accepted"


class TestLedgerObjections:
    """Tests for ledger objection operations."""

    def test_add_objection_to_ledger(self) -> None:
        """Test adding objection to ledger."""
        ledger = Ledger()
        objection = Objection(
            objection_id="o1",
            target="c1",
            severity=ObjectionSeverity.BLOCKING,
            status=ObjectionStatus.OPEN,
            rationale="Test rationale",
            owner="redteam",
        )

        ledger.add_objection(objection)

        assert len(ledger.objections) == 1
        assert ledger.objections[0].objection_id == "o1"

    def test_add_duplicate_objection_is_idempotent(self) -> None:
        """Test that adding duplicate objection doesn't create copies."""
        ledger = Ledger()
        objection = Objection(
            objection_id="o1",
            target="c1",
            severity=ObjectionSeverity.BLOCKING,
            status=ObjectionStatus.OPEN,
            rationale="Test rationale",
            owner="redteam",
        )

        ledger.add_objection(objection)
        ledger.add_objection(objection)

        assert len(ledger.objections) == 1

    def test_merge_objections_deduplicates(self) -> None:
        """Test that merging objections deduplicates by ID."""
        from delibera.epistemics.ledger import merge_objections

        obj1 = Objection("o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.OPEN, "r1", "t1")
        obj2 = Objection("o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.OPEN, "r1", "t1")

        merged = merge_objections([[obj1], [obj2]])

        assert len(merged) == 1
        assert merged[0].objection_id == "o1"

    def test_merge_objections_status_priority(self) -> None:
        """Test that merge prefers resolved > accepted > open."""
        from delibera.epistemics.ledger import merge_objections

        obj_open = Objection("o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.OPEN, "r", "t")
        obj_accepted = Objection(
            "o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.ACCEPTED, "r", "t"
        )
        obj_resolved = Objection(
            "o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.RESOLVED, "r", "t"
        )

        # Resolved wins over accepted and open
        merged = merge_objections([[obj_open], [obj_resolved], [obj_accepted]])
        assert merged[0].status == ObjectionStatus.RESOLVED

        # Accepted wins over open
        merged2 = merge_objections([[obj_open], [obj_accepted]])
        assert merged2[0].status == ObjectionStatus.ACCEPTED

    def test_merge_ledgers_merges_objections(self) -> None:
        """Test that merge_ledgers properly merges objections."""
        obj1 = Objection("o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.OPEN, "r1", "t1")
        obj2 = Objection(
            "o2", "c2", ObjectionSeverity.NONBLOCKING, ObjectionStatus.OPEN, "r2", "t2"
        )

        ledger1 = Ledger(objections=[obj1])
        ledger2 = Ledger(objections=[obj2])

        merged = merge_ledgers([ledger1, ledger2])

        assert len(merged.objections) == 2
        ids = {o.objection_id for o in merged.objections}
        assert ids == {"o1", "o2"}
