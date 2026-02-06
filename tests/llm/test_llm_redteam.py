"""Tests for LLM red-teamer agent.

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
            text = '{"objections": [], "analysis": "default"}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


class TestRedTeamLLMParsesJson:
    """Test that RedTeamLLM correctly parses LLM JSON responses."""

    def test_parses_valid_objections(self) -> None:
        """RedTeamLLM should parse valid objections from LLM response."""
        from delibera.agents.llm_redteam import RedTeamLLM

        response_json = {
            "objections": [
                {
                    "target": "claim_001",
                    "severity": "blocking",
                    "rationale": "No evidence supports this claim",
                },
                {
                    "target": "artifact",
                    "severity": "nonblocking",
                    "rationale": "Alternative approach not considered",
                },
            ],
            "analysis": "Proposal has significant gaps",
        }

        client = FakeLLMClient(response_json=response_json)
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "node_1",
                "artifact": {"recommendation": "Test"},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output["role"] == "redteam"
        assert output["step"] == "REDTEAM"
        assert output["llm_generated"] is True
        assert len(output["objections"]) == 2
        assert output["objections"][0]["severity"] == "blocking"
        assert output["objections"][0]["target"] == "claim_001"
        assert output["analysis"] == "Proposal has significant gaps"

    def test_generates_deterministic_objection_ids(self) -> None:
        """RedTeamLLM should generate deterministic IDs for objections."""
        from delibera.agents.llm_redteam import RedTeamLLM

        response_json = {
            "objections": [
                {
                    "target": "artifact",
                    "severity": "blocking",
                    "rationale": "Critical flaw",
                },
            ],
            "analysis": "test",
        }

        client = FakeLLMClient(response_json=response_json)
        redteam = RedTeamLLM(llm_client=client)

        output1 = redteam.execute(
            {
                "node_id": "node_1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        # Same input should produce same ID
        client2 = FakeLLMClient(response_json=response_json)
        redteam2 = RedTeamLLM(llm_client=client2)
        output2 = redteam2.execute(
            {
                "node_id": "node_1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output1["objections"][0]["objection_id"] == output2["objections"][0]["objection_id"]

    def test_normalizes_invalid_severity(self) -> None:
        """RedTeamLLM should default invalid severity to nonblocking."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(
            response_json={
                "objections": [
                    {"target": "artifact", "severity": "critical", "rationale": "Test"},
                ],
                "analysis": "test",
            }
        )
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output["objections"][0]["severity"] == "nonblocking"

    def test_limits_objections_to_four(self) -> None:
        """RedTeamLLM should cap at 4 objections."""
        from delibera.agents.llm_redteam import RedTeamLLM

        objections = [
            {"target": "artifact", "severity": "nonblocking", "rationale": f"Concern {i}"}
            for i in range(6)
        ]
        client = FakeLLMClient(
            response_json={"objections": objections, "analysis": "many concerns"}
        )
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert len(output["objections"]) == 4

    def test_truncates_long_rationale(self) -> None:
        """RedTeamLLM should truncate rationale longer than 300 characters."""
        from delibera.agents.llm_redteam import RedTeamLLM

        long_rationale = "A" * 350
        client = FakeLLMClient(
            response_json={
                "objections": [
                    {"target": "artifact", "severity": "blocking", "rationale": long_rationale},
                ],
                "analysis": "test",
            }
        )
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert len(output["objections"][0]["rationale"]) == 300  # 297 + "..."
        assert output["objections"][0]["rationale"].endswith("...")

    def test_filters_empty_rationale(self) -> None:
        """RedTeamLLM should filter out objections with empty rationale."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(
            response_json={
                "objections": [
                    {"target": "artifact", "severity": "blocking", "rationale": ""},
                    {"target": "artifact", "severity": "blocking", "rationale": "Valid concern"},
                ],
                "analysis": "test",
            }
        )
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert len(output["objections"]) == 1
        assert output["objections"][0]["rationale"] == "Valid concern"

    def test_defaults_empty_target_to_artifact(self) -> None:
        """RedTeamLLM should default empty target to 'artifact'."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(
            response_json={
                "objections": [
                    {"target": "", "severity": "nonblocking", "rationale": "Test"},
                ],
                "analysis": "test",
            }
        )
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output["objections"][0]["target"] == "artifact"

    def test_handles_missing_fields(self) -> None:
        """RedTeamLLM should handle empty JSON gracefully."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(response_json={})
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output["objections"] == []
        assert output["analysis"] == ""

    def test_handles_non_list_objections(self) -> None:
        """RedTeamLLM should handle non-list objections value."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(response_json={"objections": "not a list", "analysis": "test"})
        redteam = RedTeamLLM(llm_client=client)

        output = redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert output["objections"] == []


class TestRedTeamLLMRequestFormat:
    """Test that RedTeamLLM creates correct LLM requests."""

    def test_uses_json_response_format(self) -> None:
        """RedTeamLLM should request JSON response format."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(response_json={"objections": [], "analysis": "test"})
        redteam = RedTeamLLM(llm_client=client)

        redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        assert len(client.calls) == 1
        assert client.calls[0].response_format == "json"

    def test_includes_metadata(self) -> None:
        """RedTeamLLM should include role and step in request metadata."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(response_json={"objections": [], "analysis": "test"})
        redteam = RedTeamLLM(llm_client=client)

        redteam.execute(
            {
                "node_id": "n1",
                "artifact": {},
                "claims": [],
                "evidence_count": 0,
            }
        )

        metadata = client.calls[0].metadata
        assert metadata["role"] == "redteam"
        assert metadata["step"] == "REDTEAM"


class TestRedTeamLLMErrors:
    """Test error handling in RedTeamLLM."""

    def test_raises_on_llm_error(self) -> None:
        """RedTeamLLM should propagate LLM errors."""
        from delibera.agents.llm_redteam import RedTeamLLM

        client = FakeLLMClient(raise_error=LLMError("API error"))
        redteam = RedTeamLLM(llm_client=client)

        with pytest.raises(LLMError, match="API error"):
            redteam.execute(
                {
                    "node_id": "n1",
                    "artifact": {},
                    "claims": [],
                    "evidence_count": 0,
                }
            )


class TestRedTeamLLMTraceEvents:
    """Test that LLM red-teamer trace events are properly emitted."""

    def test_redteam_llm_events_emitted(self) -> None:
        """Run with LLM red-teamer should emit llm_call events."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "objections": [
                {"target": "artifact", "severity": "nonblocking", "rationale": "Test concern"},
            ],
            "analysis": "Needs improvement",
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_redteam=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question for red-teamer")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            redteam_requested = [
                e
                for e in events
                if e["event_type"] == "llm_call_requested" and e["payload"]["role"] == "redteam"
            ]
            redteam_succeeded = [
                e
                for e in events
                if e["event_type"] == "llm_call_succeeded" and e["payload"]["role"] == "redteam"
            ]

            assert len(redteam_requested) >= 1
            assert redteam_requested[0]["payload"]["step"] == "REDTEAM"
            assert len(redteam_succeeded) >= 1

    def test_redteam_llm_fallback_on_error(self) -> None:
        """LLM red-teamer error should emit llm_call_failed and fall back to stub."""
        from delibera.engine.orchestrator import Engine

        fake_client = FakeLLMClient(raise_error=LLMError("Test error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_redteam=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question with failing red-teamer")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            redteam_failed = [
                e
                for e in events
                if e["event_type"] == "llm_call_failed" and e["payload"]["role"] == "redteam"
            ]
            assert len(redteam_failed) >= 1

            # Run should still complete
            assert (run_dir / "artifact.json").exists()


class TestRedTeamPrompts:
    """Test red-teamer prompt helpers."""

    def test_system_prompt_is_non_empty(self) -> None:
        """Red-teamer system prompt should be a non-empty string."""
        from delibera.llm.prompts import build_redteam_system_prompt

        prompt = build_redteam_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_user_prompt_contains_question(self) -> None:
        """Red-teamer user prompt should contain the question."""
        from delibera.llm.prompts import build_redteam_user_prompt

        prompt = build_redteam_user_prompt(
            "Should we use GraphQL?",
            artifact={"recommendation": "Migrate to GraphQL"},
            claims=[],
            evidence_count=3,
        )
        assert "Should we use GraphQL?" in prompt
        assert "Migrate to GraphQL" in prompt
        assert "3" in prompt

    def test_user_prompt_includes_claims(self) -> None:
        """Red-teamer user prompt should include claims."""
        from delibera.llm.prompts import build_redteam_user_prompt

        claims = [
            {"claim_id": "c1", "claim_type": "fact", "status": "supported"},
            {"claim_id": "c2", "claim_type": "inference", "status": "weak"},
        ]
        prompt = build_redteam_user_prompt(
            "Test?",
            artifact={},
            claims=claims,
            evidence_count=0,
        )
        assert "c1" in prompt
        assert "c2" in prompt

    def test_redteam_json_schema_structure(self) -> None:
        """Red-teamer JSON schema should have required fields."""
        from delibera.llm.prompts import REDTEAM_JSON_SCHEMA

        assert REDTEAM_JSON_SCHEMA["type"] == "object"
        assert "objections" in REDTEAM_JSON_SCHEMA["properties"]
        assert "analysis" in REDTEAM_JSON_SCHEMA["properties"]
        assert "objections" in REDTEAM_JSON_SCHEMA["required"]
        assert "analysis" in REDTEAM_JSON_SCHEMA["required"]
