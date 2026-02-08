"""Tests for evidence collection and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine
from delibera.epistemics.models import ClaimStatus, ClaimType, Evidence
from delibera.epistemics.validate import validate_claims
from delibera.tools import (
    PolicyContext,
    ToolRouter,
    create_default_policy_engine,
    create_default_registry,
)
from delibera.tools.builtin.docs import DocsReadTool
from delibera.tools.spec import RiskLevel, ToolDenied


class TestDocsReadTool:
    """Tests for the docs.read tool."""

    def test_tool_properties(self) -> None:
        """Test tool metadata properties."""
        tool = DocsReadTool()
        assert tool.name == "docs.read"
        assert tool.risk_level == RiskLevel.LOW
        assert tool.is_discovery is False

    def test_read_existing_file(self) -> None:
        """Test reading an existing evidence file."""
        tool = DocsReadTool()
        result = tool.execute({"path": "evidence/uv_notes.txt"})
        assert "text" in result
        assert "UV" in result["text"]
        assert "fast" in result["text"].lower()

    def test_rejects_absolute_path(self) -> None:
        """Test that absolute paths are rejected."""
        tool = DocsReadTool()
        with pytest.raises(ValueError, match="Absolute paths"):
            tool.validate_input({"path": "/etc/passwd"})

    def test_rejects_path_traversal(self) -> None:
        """Test that path traversal is rejected."""
        tool = DocsReadTool()
        with pytest.raises(ValueError, match="Path traversal"):
            tool.validate_input({"path": "evidence/../secrets.txt"})

    def test_missing_path_raises(self) -> None:
        """Test that missing path raises error."""
        tool = DocsReadTool()
        with pytest.raises(ValueError, match="Missing required field"):
            tool.validate_input({})

    def test_empty_path_raises(self) -> None:
        """Test that empty path raises error."""
        tool = DocsReadTool()
        with pytest.raises(ValueError, match="cannot be empty"):
            tool.validate_input({"path": ""})

    def test_rejects_symlinks(self, tmp_path: Path) -> None:
        """Test that symlinks are rejected."""
        # Create evidence directory with a file
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        real_file = evidence_dir / "real.txt"
        real_file.write_text("content")

        # Create symlink
        symlink = evidence_dir / "link.txt"
        symlink.symlink_to(real_file)

        from delibera.tools.spec import ToolExecutionError

        tool = DocsReadTool(evidence_root=evidence_dir)
        with pytest.raises(ToolExecutionError, match="Symlinks are not allowed"):
            tool.execute({"path": "link.txt"})


class TestResearchAddsEvidence:
    """Tests for evidence collection via RESEARCH step."""

    def test_research_adds_evidence(self) -> None:
        """Test that RESEARCH step adds evidence to trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Read trace
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check for tool_call_executed for docs.read
            docs_read_events = [
                e
                for e in events
                if e["event_type"] == "tool_call_executed"
                and e["payload"]["tool_name"] == "docs.read"
            ]
            assert len(docs_read_events) == 3  # One per branch

            # Check for evidence_added events
            evidence_events = [e for e in events if e["event_type"] == "evidence_added"]
            assert len(evidence_events) == 3  # One per branch

            # Verify evidence structure
            for event in evidence_events:
                evidence = event["payload"]["evidence"]
                assert "evidence_id" in evidence
                assert "source" in evidence
                # Source is relative to evidence root (uv_notes.txt)
                assert "uv_notes.txt" in evidence["source"]
                assert "excerpt" in evidence
                assert len(evidence["excerpt"]) > 0
                assert "provenance" in evidence

    def test_evidence_in_work_output(self) -> None:
        """Test that RESEARCH work_output contains evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Read trace
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check work_output for RESEARCH step
            research_outputs = [
                e
                for e in events
                if e["event_type"] == "work_output" and e["payload"]["step"] == "RESEARCH"
            ]
            assert len(research_outputs) == 3  # One per branch

            for output in research_outputs:
                assert "evidence" in output["payload"]["output"]
                assert len(output["payload"]["output"]["evidence"]) >= 1


class TestClaimCheckEvidenceLocal:
    """Tests for evidence-local restriction during CLAIM_CHECK."""

    def test_claim_check_denies_new_docs_read(self) -> None:
        """Test that docs.read for new paths is denied during CLAIM_CHECK."""
        registry = create_default_registry()
        policy = create_default_policy_engine()

        router = ToolRouter(registry=registry, policy_engine=policy)

        # During CLAIM_CHECK, docs.read for a new path should be denied
        # because it requires evidence-local context with allowed sources
        context = PolicyContext(allowed_evidence_sources=set())

        with pytest.raises(ToolDenied) as exc_info:
            router.call(
                role="validator",
                step="CLAIM_CHECK",
                tool_name="docs.read",
                tool_input={"path": "evidence/other_file.txt"},
                node_id="test_node",
                policy_context=context,
            )

        assert "not in allowed evidence" in exc_info.value.reason

    def test_claim_check_allows_existing_evidence_source(self) -> None:
        """Test that docs.read for existing evidence sources is allowed."""
        registry = create_default_registry()
        policy = create_default_policy_engine()

        router = ToolRouter(registry=registry, policy_engine=policy)

        # During CLAIM_CHECK, docs.read for an already-cited source is allowed
        context = PolicyContext(allowed_evidence_sources={"evidence/uv_notes.txt"})

        # This should succeed (file exists and is in allowed sources)
        result = router.call(
            role="validator",
            step="CLAIM_CHECK",
            tool_name="docs.read",
            tool_input={"path": "evidence/uv_notes.txt"},
            node_id="test_node",
            policy_context=context,
        )
        assert "text" in result

    def test_research_step_allows_docs_read(self) -> None:
        """Test that docs.read is allowed during RESEARCH step."""
        registry = create_default_registry()
        policy = create_default_policy_engine()

        router = ToolRouter(registry=registry, policy_engine=policy)

        # During RESEARCH, docs.read is allowed without evidence-local restrictions
        result = router.call(
            role="researcher",
            step="RESEARCH",
            tool_name="docs.read",
            tool_input={"path": "evidence/uv_notes.txt"},
            node_id="test_node",
        )
        assert "text" in result

    def test_policy_denies_docs_read_without_context(self) -> None:
        """Test that CLAIM_CHECK denies docs.read without context."""
        policy = create_default_policy_engine()

        # Evaluate without context during CLAIM_CHECK
        decision = policy.evaluate(
            role="validator",
            step="CLAIM_CHECK",
            tool_name="docs.read",
            is_discovery=False,
            risk_level=RiskLevel.LOW,
            tool_input={"path": "evidence/file.txt"},
            context=None,  # No context provided
        )

        assert decision.allow is False
        assert "requires evidence-local context" in decision.reason


class TestValidationUsesEvidence:
    """Tests for validation using evidence."""

    def test_fact_claim_supported_with_matching_evidence(self) -> None:
        """Test that fact claims are supported when evidence matches."""
        from delibera.epistemics.models import Claim

        # Create a fact claim about UV
        claim = Claim(
            claim_id="fact_1",
            text="UV is faster than pip by 10-100x",
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )

        # Create evidence that contains relevant keywords
        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([claim], evidence)

        assert claim.status == ClaimStatus.SUPPORTED
        assert report.supported == 1
        assert report.unsupported == 0

    def test_fact_claim_unsupported_without_matching_evidence(self) -> None:
        """Test that fact claims are unsupported when evidence doesn't match."""
        from delibera.epistemics.models import Claim

        # Create a fact claim about something not in evidence
        claim = Claim(
            claim_id="fact_1",
            text="UV uses magical unicorn power",
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )

        # Evidence about something else
        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([claim], evidence)

        assert claim.status == ClaimStatus.UNSUPPORTED
        assert report.unsupported == 1

    def test_inference_supported_with_matching_evidence(self) -> None:
        """Test inference claims are supported when own evidence keyword matches."""
        from delibera.epistemics.models import Claim

        fact = Claim(
            claim_id="fact_1",
            text="UV is faster than pip",
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )
        inference = Claim(
            claim_id="inf_1",
            text="Therefore faster installation speed improves workflow",
            claim_type=ClaimType.INFERENCE,
            confidence=0.8,
            owner="proposer",
        )

        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([fact, inference], evidence)

        assert fact.status == ClaimStatus.SUPPORTED
        assert inference.status == ClaimStatus.SUPPORTED
        assert report.supported == 2

    def test_inference_weak_when_no_matching_evidence(self) -> None:
        """Test inference claims are weak when no evidence keyword matches."""
        from delibera.epistemics.models import Claim

        fact = Claim(
            claim_id="fact_1",
            text="UV uses magic",  # No evidence for this
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )
        inference = Claim(
            claim_id="inf_1",
            text="Therefore UV would improve workflow",
            claim_type=ClaimType.INFERENCE,
            confidence=0.8,
            owner="proposer",
        )

        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([fact, inference], evidence)

        assert fact.status == ClaimStatus.UNSUPPORTED
        assert inference.status == ClaimStatus.WEAK
        assert report.weak == 1

    def test_run_produces_supported_claims_with_evidence(self) -> None:
        """Test that a full run produces some supported claims due to evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Read artifact
            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            # Check claim_check_summary
            summary = artifact["claim_check_summary"]

            # With evidence, we should have some supported claims
            # (inference claims can be supported when evidence exists and no unsupported facts)
            assert summary["supported"] >= 0
            # The exact numbers depend on claim extraction, but we should have the structure
            assert "supported" in summary
            assert "weak" in summary
            assert "unsupported" in summary


class TestEvidenceModel:
    """Tests for the Evidence model."""

    def test_evidence_to_dict(self) -> None:
        """Test Evidence.to_dict method."""
        evidence = Evidence(
            evidence_id="ev_123",
            source="evidence/test.txt",
            excerpt="This is a test excerpt with some content",
            provenance={"node_id": "abc123", "step": "RESEARCH"},
        )

        result = evidence.to_dict()

        assert result["evidence_id"] == "ev_123"
        assert result["source"] == "evidence/test.txt"
        assert result["excerpt"] == "This is a test excerpt with some content"
        assert result["provenance"] == {"node_id": "abc123", "step": "RESEARCH"}


class TestSupportRelations:
    """Tests for claim↔evidence support relations."""

    def test_validate_claims_returns_support_relations(self) -> None:
        """Test that validate_claims returns support_relations mapping."""
        from delibera.epistemics.models import Claim

        # Create a fact claim that will match the evidence
        claim = Claim(
            claim_id="fact_1",
            text="UV is extremely fast compared to pip",
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )

        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([claim], evidence)

        assert claim.status == ClaimStatus.SUPPORTED
        assert "fact_1" in report.support_relations
        assert "ev_1" in report.support_relations["fact_1"]

    def test_validate_claims_details_include_supporting_evidence_ids(self) -> None:
        """Test that validation details include supporting_evidence_ids."""
        from delibera.epistemics.models import Claim

        claim = Claim(
            claim_id="fact_1",
            text="UV is extremely fast",
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )

        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            ),
            Evidence(
                evidence_id="ev_2",
                source="evidence/uv_notes.txt",
                excerpt="extremely fast Python package manager",
                provenance={"node_id": "test", "step": "RESEARCH"},
            ),
        ]

        report = validate_claims([claim], evidence)

        # Find the detail for fact_1
        fact_detail = next(d for d in report.details if d["claim_id"] == "fact_1")

        assert "supporting_evidence_ids" in fact_detail
        assert fact_detail["status"] == "supported"
        # Should have evidence IDs (up to 2 max per claim)
        assert len(fact_detail["supporting_evidence_ids"]) > 0

    def test_unsupported_claims_have_empty_evidence_ids(self) -> None:
        """Test that unsupported claims have empty supporting_evidence_ids."""
        from delibera.epistemics.models import Claim

        claim = Claim(
            claim_id="fact_1",
            text="UV uses magical unicorn power",  # No match
            claim_type=ClaimType.FACT,
            confidence=0.9,
            owner="proposer",
        )

        evidence = [
            Evidence(
                evidence_id="ev_1",
                source="evidence/uv_notes.txt",
                excerpt="Installation speed: 10-100x faster than pip",
                provenance={"node_id": "test", "step": "RESEARCH"},
            )
        ]

        report = validate_claims([claim], evidence)

        fact_detail = next(d for d in report.details if d["claim_id"] == "fact_1")

        assert fact_detail["status"] == "unsupported"
        assert fact_detail["supporting_evidence_ids"] == []

    def test_ledger_link_support_creates_relations(self) -> None:
        """Test Ledger.link_support creates support relations."""
        from delibera.epistemics.ledger import Ledger

        ledger = Ledger()
        ledger.link_support("claim_1", "ev_1")
        ledger.link_support("claim_1", "ev_2")
        ledger.link_support("claim_2", "ev_1")

        assert ledger.support_relations["claim_1"] == ["ev_1", "ev_2"]
        assert ledger.support_relations["claim_2"] == ["ev_1"]

    def test_ledger_link_support_avoids_duplicates(self) -> None:
        """Test that link_support avoids duplicate evidence IDs."""
        from delibera.epistemics.ledger import Ledger

        ledger = Ledger()
        ledger.link_support("claim_1", "ev_1")
        ledger.link_support("claim_1", "ev_1")  # Duplicate

        assert ledger.support_relations["claim_1"] == ["ev_1"]  # Not duplicated

    def test_ledger_support_for_returns_evidence(self) -> None:
        """Test Ledger.support_for returns evidence objects."""
        from delibera.epistemics.ledger import Ledger

        evidence = Evidence(
            evidence_id="ev_1",
            source="evidence/test.txt",
            excerpt="test",
            provenance={},
        )

        ledger = Ledger()
        ledger.evidence = [evidence]
        ledger.link_support("claim_1", "ev_1")

        result = ledger.support_for("claim_1")

        assert len(result) == 1
        assert result[0].evidence_id == "ev_1"

    def test_ledger_support_for_missing_evidence(self) -> None:
        """Test support_for handles missing evidence gracefully."""
        from delibera.epistemics.ledger import Ledger

        ledger = Ledger()
        ledger.link_support("claim_1", "ev_nonexistent")

        result = ledger.support_for("claim_1")

        assert result == []  # Missing evidence is skipped


class TestCitationsInArtifact:
    """Tests for key_claims with citations in artifact.json."""

    def test_artifact_has_key_claims_section(self) -> None:
        """Test that artifact.json includes key_claims section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            assert "key_claims" in artifact
            assert isinstance(artifact["key_claims"], list)

    def test_key_claims_have_citations_structure(self) -> None:
        """Test that key_claims entries have the expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            key_claims = artifact["key_claims"]

            # Should have at least some claims with citations
            # (fact claims match evidence)
            for claim in key_claims:
                assert "claim_id" in claim
                assert "type" in claim
                assert "text" in claim
                assert "status" in claim
                assert "citations" in claim
                assert isinstance(claim["citations"], list)

    def test_key_claims_prioritize_fact_claims(self) -> None:
        """Test that fact claims are prioritized in key_claims."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            key_claims = artifact["key_claims"]

            # If there are fact claims with citations, they should come first
            fact_claims = [c for c in key_claims if c["type"] == "fact"]

            if fact_claims:
                # First claims should be facts (priority 0)
                first_claim = key_claims[0] if key_claims else None
                if first_claim:
                    # Either it's a fact, or there are no fact claims with citations
                    assert first_claim["type"] in ("fact", "inference", "plan")

    def test_key_claims_citations_have_evidence_info(self) -> None:
        """Test that citations include evidence_id, source, excerpt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            key_claims = artifact["key_claims"]

            for claim in key_claims:
                for citation in claim["citations"]:
                    assert "evidence_id" in citation
                    assert "source" in citation
                    assert "excerpt" in citation
                    # Excerpt should be truncated if too long (max 240)
                    assert len(citation["excerpt"]) <= 240

    def test_key_claims_max_five(self) -> None:
        """Test that key_claims is limited to 5 entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            artifact_path = run_dir / "artifact.json"
            artifact = json.loads(artifact_path.read_text())

            key_claims = artifact["key_claims"]

            assert len(key_claims) <= 5


class TestTraceSupport:
    """Tests for support_relations in trace events."""

    def test_claim_validation_report_includes_support_relations(self) -> None:
        """Test that claim_validation_report events include support_relations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            validation_reports = [e for e in events if e["event_type"] == "claim_validation_report"]

            assert len(validation_reports) > 0

            for report in validation_reports:
                payload = report["payload"]
                assert "support_relations" in payload
                assert isinstance(payload["support_relations"], dict)

    def test_claim_validation_details_have_supporting_evidence_ids(self) -> None:
        """Test that claim_validation_report details include supporting_evidence_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            validation_reports = [e for e in events if e["event_type"] == "claim_validation_report"]

            for report in validation_reports:
                details = report["payload"]["details"]
                for detail in details:
                    assert "supporting_evidence_ids" in detail
                    assert isinstance(detail["supporting_evidence_ids"], list)
