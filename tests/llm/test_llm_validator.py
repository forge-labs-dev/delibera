"""Tests for LLM claim validator agent.

These tests use a FakeLLMClient to avoid network calls and API key requirements.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from delibera.epistemics.models import Claim, ClaimStatus, ClaimType, Evidence
from delibera.llm.base import LLMError, LLMRequest, LLMResponse, LLMUsage


class FakeLLMClient:
    """Fake LLM client for testing that returns configurable responses."""

    def __init__(
        self,
        response_json: dict[str, Any] | None = None,
        response_text: str | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.response_json = response_json
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)

        if self.raise_error:
            raise self.raise_error

        if self.response_text is not None:
            text = self.response_text
        elif self.response_json is not None:
            text = json.dumps(self.response_json)
        else:
            text = '{"assessments": []}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


def _make_claim(
    claim_id: str,
    text: str,
    claim_type: ClaimType = ClaimType.FACT,
    confidence: float = 0.9,
) -> Claim:
    """Helper to create a Claim for testing."""
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=claim_type,
        confidence=confidence,
        owner="proposer",
    )


def _make_evidence(evidence_id: str, source: str, excerpt: str) -> Evidence:
    """Helper to create Evidence for testing."""
    return Evidence(
        evidence_id=evidence_id,
        source=source,
        excerpt=excerpt,
        provenance={"node_id": "test", "step": "RESEARCH"},
    )


class TestClaimValidatorLLMParsesJson:
    """Test that ClaimValidatorLLM correctly parses LLM JSON responses."""

    def test_parses_valid_assessments(self) -> None:
        """ClaimValidatorLLM should parse valid assessments."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 0.9,
                    "supporting_evidence_ids": ["ev1"],
                    "reasoning": "Evidence directly confirms this claim",
                    "needs_more_evidence": False,
                },
                {
                    "claim_id": "c2",
                    "status": "unsupported",
                    "confidence": 0.1,
                    "supporting_evidence_ids": [],
                    "reasoning": "No evidence found",
                    "needs_more_evidence": True,
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [
            _make_claim("c1", "GraphQL reduces bandwidth"),
            _make_claim("c2", "Migration takes 3 months"),
        ]
        evidence = [_make_evidence("ev1", "benchmark.txt", "GraphQL reduces payload")]

        report = validator.validate(claims, evidence)

        assert report.supported == 1
        assert report.unsupported == 1
        assert report.weak == 0
        assert len(report.details) == 2
        assert report.support_relations == {"c1": ["ev1"]}

    def test_updates_claim_status_in_place(self) -> None:
        """ClaimValidatorLLM should update claim.status in-place."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "weak",
                    "confidence": 0.4,
                    "supporting_evidence_ids": [],
                    "reasoning": "Partial support",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claim = _make_claim("c1", "Some claim")
        assert claim.status == ClaimStatus.UNVALIDATED

        validator.validate([claim], [])

        assert claim.status == ClaimStatus.WEAK

    def test_updates_claim_confidence(self) -> None:
        """ClaimValidatorLLM should update claim.confidence from LLM."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 0.95,
                    "supporting_evidence_ids": [],
                    "reasoning": "Strong support",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claim = _make_claim("c1", "Some claim", confidence=0.5)
        validator.validate([claim], [])

        assert claim.confidence == pytest.approx(0.95)

    def test_clamps_confidence_to_unit_range(self) -> None:
        """ClaimValidatorLLM should clamp confidence to [0, 1]."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 1.5,
                    "supporting_evidence_ids": [],
                    "reasoning": "Test",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claim = _make_claim("c1", "Some claim")
        validator.validate([claim], [])

        assert claim.confidence <= 1.0

    def test_handles_contradicted_status(self) -> None:
        """ClaimValidatorLLM should map 'contradicted' to unsupported."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "contradicted",
                    "confidence": 0.1,
                    "supporting_evidence_ids": [],
                    "contradicting_evidence_ids": ["ev1"],
                    "reasoning": "Evidence directly contradicts",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Migration is free")]
        evidence = [_make_evidence("ev1", "report.txt", "Migration costs $100k")]

        report = validator.validate(claims, evidence)

        assert report.unsupported == 1
        assert claims[0].status == ClaimStatus.UNSUPPORTED
        # Contradicting evidence should appear in details
        detail = report.details[0]
        assert detail["contradicting_evidence_ids"] == ["ev1"]

    def test_filters_invalid_evidence_ids(self) -> None:
        """ClaimValidatorLLM should filter out invalid evidence IDs."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 0.8,
                    "supporting_evidence_ids": ["ev1", "fake_id", "ev2"],
                    "reasoning": "Supported",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Test claim")]
        evidence = [
            _make_evidence("ev1", "a.txt", "Evidence 1"),
            _make_evidence("ev2", "b.txt", "Evidence 2"),
        ]

        report = validator.validate(claims, evidence)

        # Only valid IDs should be in support_relations
        assert report.support_relations["c1"] == ["ev1", "ev2"]

    def test_handles_empty_claims(self) -> None:
        """ClaimValidatorLLM should return empty report for no claims."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": []})
        validator = ClaimValidatorLLM(llm_client=client)

        report = validator.validate([], [])

        assert report.supported == 0
        assert report.weak == 0
        assert report.unsupported == 0
        assert report.details == []
        # Should not call LLM for empty claims
        assert len(client.calls) == 0

    def test_handles_missing_assessments(self) -> None:
        """ClaimValidatorLLM should use defaults for missing assessments."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        # LLM only returns assessment for c1, not c2
        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 0.9,
                    "supporting_evidence_ids": [],
                    "reasoning": "OK",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [
            _make_claim("c1", "Assessed claim"),
            _make_claim("c2", "Unassessed fact claim"),
        ]

        report = validator.validate(claims, [])

        assert report.supported == 1  # c1
        assert report.unsupported == 1  # c2 (fact with no assessment defaults to unsupported)

    def test_plan_claims_default_supported(self) -> None:
        """ClaimValidatorLLM should default plan claims to supported."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        # LLM doesn't return assessment for the plan claim
        response_json = {"assessments": []}

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Migrate to GraphQL", ClaimType.PLAN)]

        report = validator.validate(claims, [])

        assert report.supported == 1
        assert claims[0].status == ClaimStatus.SUPPORTED

    def test_value_claims_default_supported(self) -> None:
        """ClaimValidatorLLM should default value claims to supported."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {"assessments": []}

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Performance is important", ClaimType.VALUE)]

        report = validator.validate(claims, [])

        assert report.supported == 1

    def test_handles_invalid_status_string(self) -> None:
        """ClaimValidatorLLM should fallback for invalid status strings."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "definitely_true",
                    "confidence": 0.9,
                    "supporting_evidence_ids": [],
                    "reasoning": "Invalid status",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Test", ClaimType.FACT)]

        report = validator.validate(claims, [_make_evidence("ev1", "a.txt", "e")])

        # Invalid status should fall back to type-based default
        assert report.details[0]["status"] in ("supported", "weak", "unsupported")

    def test_handles_non_list_assessments(self) -> None:
        """ClaimValidatorLLM should handle non-list assessments value."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": "not a list"})
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Test", ClaimType.PLAN)]

        report = validator.validate(claims, [])

        # Should fall back to defaults
        assert report.supported == 1  # PLAN default

    def test_truncates_reasoning(self) -> None:
        """ClaimValidatorLLM should truncate long reasoning strings."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        long_reasoning = "A" * 500
        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "supported",
                    "confidence": 0.8,
                    "supporting_evidence_ids": [],
                    "reasoning": long_reasoning,
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Test")]

        report = validator.validate(claims, [])

        assert len(report.details[0]["reasoning"]) <= 300

    def test_includes_needs_more_evidence(self) -> None:
        """ClaimValidatorLLM should include needs_more_evidence in details."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        response_json = {
            "assessments": [
                {
                    "claim_id": "c1",
                    "status": "weak",
                    "confidence": 0.3,
                    "supporting_evidence_ids": [],
                    "reasoning": "Partial support",
                    "needs_more_evidence": True,
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        validator = ClaimValidatorLLM(llm_client=client)

        claims = [_make_claim("c1", "Test")]
        report = validator.validate(claims, [])

        assert report.details[0]["needs_more_evidence"] is True


class TestClaimValidatorLLMRequestFormat:
    """Test that ClaimValidatorLLM creates correct LLM requests."""

    def test_uses_json_response_format(self) -> None:
        """ClaimValidatorLLM should request JSON response format."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": []})
        validator = ClaimValidatorLLM(llm_client=client)

        validator.validate(
            [_make_claim("c1", "Test")],
            [_make_evidence("ev1", "a.txt", "evidence")],
        )

        assert len(client.calls) == 1
        assert client.calls[0].response_format == "json"

    def test_includes_metadata(self) -> None:
        """ClaimValidatorLLM should include role and step in request metadata."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": []})
        validator = ClaimValidatorLLM(llm_client=client)

        validator.validate(
            [_make_claim("c1", "Test")],
            [],
        )

        metadata = client.calls[0].metadata
        assert metadata["role"] == "validator"
        assert metadata["step"] == "CLAIM_CHECK"

    def test_includes_claims_in_prompt(self) -> None:
        """ClaimValidatorLLM should include claim details in the prompt."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": []})
        validator = ClaimValidatorLLM(llm_client=client)

        validator.validate(
            [_make_claim("c1", "GraphQL reduces bandwidth", ClaimType.FACT)],
            [],
        )

        user_messages = [m for m in client.calls[0].messages if m.role == "user"]
        assert len(user_messages) == 1
        assert "c1" in user_messages[0].content
        assert "GraphQL reduces bandwidth" in user_messages[0].content
        assert "fact" in user_messages[0].content

    def test_includes_evidence_in_prompt(self) -> None:
        """ClaimValidatorLLM should include evidence details in the prompt."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(response_json={"assessments": []})
        validator = ClaimValidatorLLM(llm_client=client)

        validator.validate(
            [_make_claim("c1", "Test")],
            [_make_evidence("ev1", "benchmark.txt", "Performance data")],
        )

        user_messages = [m for m in client.calls[0].messages if m.role == "user"]
        assert "ev1" in user_messages[0].content
        assert "benchmark.txt" in user_messages[0].content
        assert "Performance data" in user_messages[0].content


class TestClaimValidatorLLMErrors:
    """Test error handling in ClaimValidatorLLM."""

    def test_raises_on_llm_error(self) -> None:
        """ClaimValidatorLLM should propagate LLM errors."""
        from delibera.agents.llm_validator import ClaimValidatorLLM

        client = FakeLLMClient(raise_error=LLMError("API error"))
        validator = ClaimValidatorLLM(llm_client=client)

        with pytest.raises(LLMError, match="API error"):
            validator.validate(
                [_make_claim("c1", "Test")],
                [],
            )


class TestClaimValidatorLLMTraceEvents:
    """Test that LLM validator trace events are properly emitted."""

    def test_validator_llm_events_emitted(self) -> None:
        """Run with LLM validator should emit llm_call events."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "assessments": [
                {
                    "claim_id": "test",
                    "status": "supported",
                    "confidence": 0.9,
                    "supporting_evidence_ids": [],
                    "reasoning": "OK",
                },
            ],
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_validator=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question for validator")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            validator_requested = [
                e
                for e in events
                if e["event_type"] == "llm_call_requested" and e["payload"]["role"] == "validator"
            ]
            validator_succeeded = [
                e
                for e in events
                if e["event_type"] == "llm_call_succeeded" and e["payload"]["role"] == "validator"
            ]

            assert len(validator_requested) >= 1
            assert validator_requested[0]["payload"]["step"] == "CLAIM_CHECK"
            assert len(validator_succeeded) >= 1

    def test_validator_llm_fallback_on_error(self) -> None:
        """LLM validator error should fall back to heuristic validation."""
        from delibera.engine.orchestrator import Engine

        fake_client = FakeLLMClient(raise_error=LLMError("Test error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_validator=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question with failing validator")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            validator_failed = [
                e
                for e in events
                if e["event_type"] == "llm_call_failed" and e["payload"]["role"] == "validator"
            ]
            assert len(validator_failed) >= 1

            # Run should still complete (fell back to heuristic)
            assert (run_dir / "artifact.json").exists()

            event_types = [e["event_type"] for e in events]
            assert "run_end" in event_types


class TestValidatorPrompts:
    """Test validator prompt helpers."""

    def test_system_prompt_is_non_empty(self) -> None:
        """Validator system prompt should be a non-empty string."""
        from delibera.llm.prompts import build_validator_system_prompt

        prompt = build_validator_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_user_prompt_contains_claims(self) -> None:
        """Validator user prompt should contain claim details."""
        from delibera.llm.prompts import build_validator_user_prompt

        claims = [
            {"claim_id": "c1", "claim_type": "fact", "text": "GraphQL is fast"},
        ]
        prompt = build_validator_user_prompt(claims=claims, evidence=[])

        assert "c1" in prompt
        assert "GraphQL is fast" in prompt
        assert "fact" in prompt

    def test_user_prompt_contains_evidence(self) -> None:
        """Validator user prompt should contain evidence details."""
        from delibera.llm.prompts import build_validator_user_prompt

        evidence = [
            {
                "evidence_id": "ev1",
                "source": "benchmark.txt",
                "excerpt": "Performance benchmark data",
            },
        ]
        prompt = build_validator_user_prompt(claims=[], evidence=evidence)

        assert "ev1" in prompt
        assert "benchmark.txt" in prompt
        assert "Performance benchmark data" in prompt

    def test_user_prompt_handles_no_evidence(self) -> None:
        """Validator user prompt should handle empty evidence."""
        from delibera.llm.prompts import build_validator_user_prompt

        prompt = build_validator_user_prompt(
            claims=[{"claim_id": "c1", "claim_type": "fact", "text": "Test"}],
            evidence=[],
        )
        assert "None" in prompt

    def test_validator_json_schema_structure(self) -> None:
        """Validator JSON schema should have required fields."""
        from delibera.llm.prompts import VALIDATOR_JSON_SCHEMA

        assert VALIDATOR_JSON_SCHEMA["type"] == "object"
        assert "assessments" in VALIDATOR_JSON_SCHEMA["properties"]
        assert "assessments" in VALIDATOR_JSON_SCHEMA["required"]

        # Check assessment item schema
        item_schema = VALIDATOR_JSON_SCHEMA["properties"]["assessments"]["items"]
        assert "claim_id" in item_schema["properties"]
        assert "status" in item_schema["properties"]
        assert "confidence" in item_schema["properties"]
        assert "supporting_evidence_ids" in item_schema["properties"]
        assert "reasoning" in item_schema["properties"]
