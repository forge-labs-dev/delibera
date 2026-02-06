"""Tests for LLM planner agent.

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
            text = '{"branches": ["Option A", "Option B", "Option C"], "reasoning": "default"}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


class TestPlannerLLMParsesJson:
    """Test that PlannerLLM correctly parses LLM JSON responses."""

    def test_parses_valid_branches(self) -> None:
        """PlannerLLM should parse valid branch labels from LLM response."""
        from delibera.agents.llm_planner import PlannerLLM

        response_json = {
            "branches": [
                "Full migration to GraphQL",
                "Hybrid REST + GraphQL gateway",
                "Keep REST with better documentation",
            ],
            "reasoning": "Covers the spectrum from full adoption to status quo",
        }

        client = FakeLLMClient(response_json=response_json)
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Should we migrate from REST to GraphQL?"})

        assert output["branches"] == [
            "Full migration to GraphQL",
            "Hybrid REST + GraphQL gateway",
            "Keep REST with better documentation",
        ]
        assert output["role"] == "planner"
        assert output["step"] == "PLAN"
        assert output["llm_generated"] is True
        assert output["reasoning"] == "Covers the spectrum from full adoption to status quo"

    def test_normalizes_empty_branches(self) -> None:
        """PlannerLLM should fall back to default labels when branches are empty."""
        from delibera.agents.llm_planner import _DEFAULT_BRANCHES, PlannerLLM

        client = FakeLLMClient(response_json={"branches": [], "reasoning": ""})
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert output["branches"] == list(_DEFAULT_BRANCHES)

    def test_limits_max_branches(self) -> None:
        """PlannerLLM should cap branches at 5."""
        from delibera.agents.llm_planner import PlannerLLM

        branches = [f"Option {i}" for i in range(8)]
        client = FakeLLMClient(response_json={"branches": branches, "reasoning": "many options"})
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert len(output["branches"]) == 5

    def test_strips_whitespace_from_branches(self) -> None:
        """PlannerLLM should strip leading/trailing whitespace from labels."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(
            response_json={
                "branches": ["  Option A  ", "\tOption B\n", "Option C"],
                "reasoning": "test",
            }
        )
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert output["branches"] == ["Option A", "Option B", "Option C"]

    def test_handles_missing_fields(self) -> None:
        """PlannerLLM should provide defaults for missing fields."""
        from delibera.agents.llm_planner import _DEFAULT_BRANCHES, PlannerLLM

        client = FakeLLMClient(response_json={})
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert "branches" in output
        assert output["branches"] == list(_DEFAULT_BRANCHES)
        assert output["reasoning"] == ""


class TestPlannerLLMRequestFormat:
    """Test that PlannerLLM creates correct LLM requests."""

    def test_uses_json_response_format(self) -> None:
        """PlannerLLM should request JSON response format."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(response_json={"branches": ["A", "B"], "reasoning": "test"})
        planner = PlannerLLM(llm_client=client)

        planner.execute({"question": "Test?"})

        assert len(client.calls) == 1
        assert client.calls[0].response_format == "json"

    def test_includes_metadata(self) -> None:
        """PlannerLLM should include role and step in request metadata."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(response_json={"branches": ["A", "B"], "reasoning": "test"})
        planner = PlannerLLM(llm_client=client)

        planner.execute({"question": "Test?"})

        metadata = client.calls[0].metadata
        assert metadata["role"] == "planner"
        assert metadata["step"] == "PLAN"

    def test_passes_question_in_prompt(self) -> None:
        """PlannerLLM should include the question in the user message."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(response_json={"branches": ["A", "B"], "reasoning": "test"})
        planner = PlannerLLM(llm_client=client)

        planner.execute({"question": "Should we migrate to GraphQL?"})

        # Check user message contains the question
        user_messages = [m for m in client.calls[0].messages if m.role == "user"]
        assert len(user_messages) == 1
        assert "Should we migrate to GraphQL?" in user_messages[0].content


class TestPlannerLLMNormalization:
    """Test output normalization edge cases."""

    def test_truncates_long_labels(self) -> None:
        """PlannerLLM should truncate labels longer than 100 characters."""
        from delibera.agents.llm_planner import PlannerLLM

        long_label = "A" * 150
        client = FakeLLMClient(
            response_json={
                "branches": [long_label, "Short label"],
                "reasoning": "test",
            }
        )
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert len(output["branches"][0]) == 100  # 97 + "..."
        assert output["branches"][0].endswith("...")

    def test_removes_empty_labels(self) -> None:
        """PlannerLLM should filter out empty string labels."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(
            response_json={
                "branches": ["Option A", "", "  ", "Option B"],
                "reasoning": "test",
            }
        )
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert output["branches"] == ["Option A", "Option B"]

    def test_minimum_two_branches(self) -> None:
        """PlannerLLM should fall back to defaults if fewer than 2 valid branches."""
        from delibera.agents.llm_planner import _DEFAULT_BRANCHES, PlannerLLM

        client = FakeLLMClient(
            response_json={
                "branches": ["Only one option"],
                "reasoning": "test",
            }
        )
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert output["branches"] == list(_DEFAULT_BRANCHES)

    def test_non_list_branches_falls_back(self) -> None:
        """PlannerLLM should handle non-list branches value."""
        from delibera.agents.llm_planner import _DEFAULT_BRANCHES, PlannerLLM

        client = FakeLLMClient(
            response_json={
                "branches": "not a list",
                "reasoning": "test",
            }
        )
        planner = PlannerLLM(llm_client=client)

        output = planner.execute({"question": "Test?"})

        assert output["branches"] == list(_DEFAULT_BRANCHES)


class TestPlannerLLMErrors:
    """Test error handling in PlannerLLM."""

    def test_raises_on_llm_error(self) -> None:
        """PlannerLLM should propagate LLM errors (fallback is in orchestrator)."""
        from delibera.agents.llm_planner import PlannerLLM

        client = FakeLLMClient(raise_error=LLMError("API error"))
        planner = PlannerLLM(llm_client=client)

        with pytest.raises(LLMError, match="API error"):
            planner.execute({"question": "Test?"})


class TestPlannerLLMTraceEvents:
    """Test that LLM planner trace events are properly emitted during runs."""

    def test_planner_llm_events_emitted(self) -> None:
        """Run with LLM planner should emit llm_call_requested and llm_call_succeeded."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "branches": [
                "Full migration to GraphQL",
                "Hybrid REST + GraphQL",
                "Keep REST",
            ],
            "reasoning": "Covers the spectrum",
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_planner=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Should we migrate to GraphQL?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Find planner LLM events
            planner_requested = [
                e
                for e in events
                if e["event_type"] == "llm_call_requested" and e["payload"]["role"] == "planner"
            ]
            planner_succeeded = [
                e
                for e in events
                if e["event_type"] == "llm_call_succeeded" and e["payload"]["role"] == "planner"
            ]

            assert len(planner_requested) == 1
            assert planner_requested[0]["payload"]["step"] == "PLAN"
            assert len(planner_succeeded) == 1

    def test_planner_llm_fallback_on_error(self) -> None:
        """LLM planner error should emit llm_call_failed and fall back to stub."""
        from delibera.engine.orchestrator import Engine

        fake_client = FakeLLMClient(raise_error=LLMError("Test error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_planner=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question with failing planner")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check for failure event
            planner_failed = [
                e
                for e in events
                if e["event_type"] == "llm_call_failed" and e["payload"]["role"] == "planner"
            ]
            assert len(planner_failed) == 1

            # Run should still complete (fell back to stub)
            assert (run_dir / "artifact.json").exists()

            # Check that stub branches were used
            plan_output = [
                e
                for e in events
                if e["event_type"] == "work_output" and e["payload"]["step"] == "PLAN"
            ]
            assert len(plan_output) == 1
            branches = plan_output[0]["payload"]["output"]["branches"]
            assert "Option A: Direct migration approach" in branches[0]

    def test_planner_uses_llm_branches_in_expansion(self) -> None:
        """LLM planner branches should be used for tree expansion."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "branches": [
                "Full GraphQL migration",
                "Hybrid approach with gateway",
                "REST with OpenAPI improvements",
            ],
            "reasoning": "Three distinct strategies",
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_planner=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Should we migrate to GraphQL?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check expand event uses LLM branches
            expand_events = [e for e in events if e["event_type"] == "expand"]
            assert len(expand_events) >= 1
            labels = expand_events[0]["payload"]["labels"]
            assert "Full GraphQL migration" in labels
            assert "Hybrid approach with gateway" in labels
            assert "REST with OpenAPI improvements" in labels


class TestPlannerPrompts:
    """Test planner prompt helpers."""

    def test_system_prompt_is_non_empty(self) -> None:
        """Planner system prompt should be a non-empty string."""
        from delibera.llm.prompts import build_planner_system_prompt

        prompt = build_planner_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_user_prompt_contains_question(self) -> None:
        """Planner user prompt should contain the question."""
        from delibera.llm.prompts import build_planner_user_prompt

        prompt = build_planner_user_prompt("Should we use GraphQL?")
        assert "Should we use GraphQL?" in prompt

    def test_user_prompt_includes_constraints(self) -> None:
        """Planner user prompt should include constraints when provided."""
        from delibera.llm.prompts import build_planner_user_prompt

        prompt = build_planner_user_prompt(
            "Should we use GraphQL?",
            constraints=["Must support legacy clients", "Budget under $50k"],
        )
        assert "Must support legacy clients" in prompt
        assert "Budget under $50k" in prompt

    def test_user_prompt_without_constraints(self) -> None:
        """Planner user prompt should work without constraints."""
        from delibera.llm.prompts import build_planner_user_prompt

        prompt = build_planner_user_prompt("Test question?")
        assert "CONSTRAINTS" not in prompt

    def test_planner_json_schema_structure(self) -> None:
        """Planner JSON schema should have required fields."""
        from delibera.llm.prompts import PLANNER_JSON_SCHEMA

        assert PLANNER_JSON_SCHEMA["type"] == "object"
        assert "branches" in PLANNER_JSON_SCHEMA["properties"]
        assert "reasoning" in PLANNER_JSON_SCHEMA["properties"]
        assert "branches" in PLANNER_JSON_SCHEMA["required"]
        assert "reasoning" in PLANNER_JSON_SCHEMA["required"]
