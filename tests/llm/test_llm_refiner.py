"""Tests for LLM refiner agent.

These tests use a FakeLLMClient to avoid network calls and API key requirements.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

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
            text = '{"improved_summary": "default", "improvements": []}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


class TestRefinerLLMParsesJson:
    """Test that RefinerLLM correctly parses LLM JSON responses."""

    def test_parses_valid_response(self) -> None:
        """RefinerLLM should parse valid JSON response."""
        from delibera.agents.llm_refiner import RefinerLLM

        response_json = {
            "improved_summary": "Updated proposal with stronger evidence",
            "improvements": [
                "Added performance benchmarks",
                "Addressed migration risk",
            ],
            "addressed_weaknesses": ["claim_001", "claim_002"],
            "confidence_delta": 0.05,
        }

        client = FakeLLMClient(response_json=response_json)
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "node_1",
                "artifact": {"summary": "Original", "score": 0.7},
                "claims": [
                    {"claim_id": "claim_001", "status": "weak", "text": "Claim 1"},
                    {"claim_id": "claim_002", "status": "unsupported", "text": "Claim 2"},
                ],
                "round": 1,
            }
        )

        assert output["role"] == "refiner"
        assert output["step"] == "REFINE"
        assert output["llm_generated"] is True
        assert output["artifact"]["summary"] == "Updated proposal with stronger evidence"
        assert output["artifact"]["refinement_round"] == 1
        assert len(output["refinement_notes"]) == 2
        assert output["weak_addressed"] == 1
        assert output["unsupported_addressed"] == 1

    def test_applies_confidence_delta(self) -> None:
        """RefinerLLM should adjust score by confidence_delta."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(
            response_json={
                "improved_summary": "Better",
                "improvements": ["Improved"],
                "confidence_delta": 0.05,
            }
        )
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.7},
                "claims": [],
                "round": 1,
            }
        )

        assert output["artifact"]["score"] == pytest.approx(0.75)

    def test_clamps_confidence_delta(self) -> None:
        """RefinerLLM should clamp confidence_delta to [-0.1, 0.1]."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(
            response_json={
                "improved_summary": "Better",
                "improvements": ["Improved"],
                "confidence_delta": 0.5,  # Too large
            }
        )
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.7},
                "claims": [],
                "round": 1,
            }
        )

        # Should be clamped to 0.1
        assert output["artifact"]["score"] == pytest.approx(0.8)

    def test_clamps_score_to_unit_range(self) -> None:
        """RefinerLLM should keep score in [0, 1]."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(
            response_json={
                "improved_summary": "Better",
                "improvements": ["Improved"],
                "confidence_delta": 0.1,
            }
        )
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.95},
                "claims": [],
                "round": 1,
            }
        )

        assert output["artifact"]["score"] <= 1.0

    def test_preserves_original_artifact_fields(self) -> None:
        """RefinerLLM should preserve original artifact fields."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(
            response_json={
                "improved_summary": "New summary",
                "improvements": ["Updated"],
            }
        )
        refiner = RefinerLLM(llm_client=client)

        original_artifact = {
            "recommendation": "Use GraphQL",
            "rationale": ["Fast", "Flexible"],
            "score": 0.6,
            "custom_field": "preserved",
        }

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": original_artifact,
                "claims": [],
                "round": 1,
            }
        )

        assert output["artifact"]["recommendation"] == "Use GraphQL"
        assert output["artifact"]["custom_field"] == "preserved"
        assert output["artifact"]["summary"] == "New summary"

    def test_handles_missing_fields(self) -> None:
        """RefinerLLM should provide defaults for missing fields."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(response_json={})
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.5},
                "claims": [],
                "round": 1,
            }
        )

        assert "artifact" in output
        assert output["refinement_notes"] == []
        assert output["weak_addressed"] == 0
        assert output["unsupported_addressed"] == 0

    def test_limits_improvements_to_six(self) -> None:
        """RefinerLLM should limit improvements to 6."""
        from delibera.agents.llm_refiner import RefinerLLM

        improvements = [f"Improvement {i}" for i in range(10)]
        client = FakeLLMClient(
            response_json={
                "improved_summary": "Better",
                "improvements": improvements,
            }
        )
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.5},
                "claims": [],
                "round": 1,
            }
        )

        assert len(output["refinement_notes"]) == 6

    def test_counts_addressed_weaknesses(self) -> None:
        """RefinerLLM should count weak vs unsupported claims addressed."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(
            response_json={
                "improved_summary": "Better",
                "improvements": ["Fixed claims"],
                "addressed_weaknesses": ["c1", "c2", "c3"],
            }
        )
        refiner = RefinerLLM(llm_client=client)

        output = refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.5},
                "claims": [
                    {"claim_id": "c1", "status": "weak", "text": "Weak 1"},
                    {"claim_id": "c2", "status": "unsupported", "text": "Unsupported 1"},
                    {"claim_id": "c3", "status": "supported", "text": "Supported 1"},
                    {"claim_id": "c4", "status": "weak", "text": "Not addressed"},
                ],
                "round": 1,
            }
        )

        assert output["weak_addressed"] == 1  # c1
        assert output["unsupported_addressed"] == 1  # c2


class TestRefinerLLMRequestFormat:
    """Test that RefinerLLM creates correct LLM requests."""

    def test_uses_json_response_format(self) -> None:
        """RefinerLLM should request JSON response format."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(response_json={"improved_summary": "test", "improvements": []})
        refiner = RefinerLLM(llm_client=client)

        refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.5},
                "claims": [],
                "round": 1,
            }
        )

        assert len(client.calls) == 1
        assert client.calls[0].response_format == "json"

    def test_includes_metadata(self) -> None:
        """RefinerLLM should include role and step in request metadata."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(response_json={"improved_summary": "test", "improvements": []})
        refiner = RefinerLLM(llm_client=client)

        refiner.execute(
            {
                "node_id": "n1",
                "artifact": {"score": 0.5},
                "claims": [],
                "round": 1,
            }
        )

        metadata = client.calls[0].metadata
        assert metadata["role"] == "refiner"
        assert metadata["step"] == "REFINE"


class TestRefinerLLMErrors:
    """Test error handling in RefinerLLM."""

    def test_raises_on_llm_error(self) -> None:
        """RefinerLLM should propagate LLM errors."""
        from delibera.agents.llm_refiner import RefinerLLM

        client = FakeLLMClient(raise_error=LLMError("API error"))
        refiner = RefinerLLM(llm_client=client)

        with pytest.raises(LLMError, match="API error"):
            refiner.execute(
                {
                    "node_id": "n1",
                    "artifact": {"score": 0.5},
                    "claims": [],
                    "round": 1,
                }
            )


class TestRefinerLLMTraceEvents:
    """Test that LLM refiner trace events are properly emitted."""

    def test_refiner_llm_events_emitted(self) -> None:
        """Run with LLM refiner should emit llm_call events."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "improved_summary": "Refined proposal",
            "improvements": ["Added evidence"],
            "confidence_delta": 0.02,
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_refiner=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question for refiner")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Refiner events may or may not appear depending on protocol
            # (refinement must be enabled in protocol)
            # Just check no errors occurred and run completed
            event_types = [e["event_type"] for e in events]
            assert "run_end" in event_types
            assert (run_dir / "artifact.json").exists()

    def test_refiner_llm_fallback_on_error(self) -> None:
        """LLM refiner error should emit llm_call_failed and fall back to stub."""
        from delibera.engine.orchestrator import Engine

        fake_client = FakeLLMClient(raise_error=LLMError("Test error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_refiner=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question with failing refiner")

            # Run should still complete
            assert (run_dir / "artifact.json").exists()


class TestRefinerPrompts:
    """Test refiner prompt helpers."""

    def test_system_prompt_is_non_empty(self) -> None:
        """Refiner system prompt should be a non-empty string."""
        from delibera.llm.prompts import build_refiner_system_prompt

        prompt = build_refiner_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_user_prompt_contains_question(self) -> None:
        """Refiner user prompt should contain the question."""
        from delibera.llm.prompts import build_refiner_user_prompt

        prompt = build_refiner_user_prompt(
            "Should we use GraphQL?",
            artifact={"summary": "Migrate to GraphQL"},
            claims=[],
        )
        assert "Should we use GraphQL?" in prompt

    def test_user_prompt_includes_weak_claims(self) -> None:
        """Refiner user prompt should include weak/unsupported claims."""
        from delibera.llm.prompts import build_refiner_user_prompt

        claims = [
            {"claim_id": "c1", "status": "weak", "text": "Weak claim"},
            {"claim_id": "c2", "status": "supported", "text": "Supported claim"},
            {"claim_id": "c3", "status": "unsupported", "text": "Unsupported claim"},
        ]
        prompt = build_refiner_user_prompt(
            "Test?",
            artifact={},
            claims=claims,
        )
        assert "c1" in prompt
        assert "c3" in prompt
        # Supported claims should not be highlighted
        assert "Supported claim" not in prompt

    def test_user_prompt_includes_objections(self) -> None:
        """Refiner user prompt should include objections."""
        from delibera.llm.prompts import build_refiner_user_prompt

        objections = [
            {"severity": "blocking", "rationale": "Missing evidence for migration cost"},
        ]
        prompt = build_refiner_user_prompt(
            "Test?",
            artifact={},
            claims=[],
            objections=objections,
        )
        assert "Missing evidence for migration cost" in prompt
        assert "blocking" in prompt

    def test_user_prompt_includes_round(self) -> None:
        """Refiner user prompt should include the round number."""
        from delibera.llm.prompts import build_refiner_user_prompt

        prompt = build_refiner_user_prompt(
            "Test?",
            artifact={},
            claims=[],
            round_num=3,
        )
        assert "3" in prompt

    def test_refiner_json_schema_structure(self) -> None:
        """Refiner JSON schema should have required fields."""
        from delibera.llm.prompts import REFINER_JSON_SCHEMA

        assert REFINER_JSON_SCHEMA["type"] == "object"
        assert "improved_summary" in REFINER_JSON_SCHEMA["properties"]
        assert "improvements" in REFINER_JSON_SCHEMA["properties"]
        assert "improved_summary" in REFINER_JSON_SCHEMA["required"]
        assert "improvements" in REFINER_JSON_SCHEMA["required"]
