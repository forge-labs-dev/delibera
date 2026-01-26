"""Tests for LLM proposer agent.

These tests use a FakeLLMClient to avoid network calls and API key requirements.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from delibera.llm.base import LLMRequest, LLMResponse, LLMUsage


class FakeLLMClient:
    """Fake LLM client for testing that returns configurable responses."""

    def __init__(
        self,
        response_json: dict[str, Any] | None = None,
        response_text: str | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        """Initialize fake client.

        Args:
            response_json: JSON to return (will be serialized to text).
            response_text: Raw text to return (overrides response_json).
            raise_error: Exception to raise on generate.
        """
        self.response_json = response_json
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a fake response."""
        self.calls.append(request)

        if self.raise_error:
            raise self.raise_error

        if self.response_text is not None:
            text = self.response_text
        elif self.response_json is not None:
            text = json.dumps(self.response_json)
        else:
            text = '{"recommendation": "default", "rationale": [], "claims": [], "confidence": 0.5}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


class TestProposerLLMParsesJson:
    """Test that ProposerLLM correctly parses LLM JSON responses."""

    def test_parses_valid_json_response(self) -> None:
        """ProposerLLM should parse valid JSON from LLM response."""
        from delibera.agents.llm_proposer import ProposerLLM

        response_json = {
            "recommendation": "Use uv for faster dependency management",
            "rationale": [
                "10-100x faster than pip",
                "Drop-in replacement for pip commands",
                "Generates reproducible lockfiles",
            ],
            "claims": [
                {"type": "fact", "text": "uv is 10-100x faster than pip"},
                {"type": "inference", "text": "Migration will be smooth"},
            ],
            "confidence": 0.85,
        }

        client = FakeLLMClient(response_json=response_json)
        proposer = ProposerLLM(llm_client=client)

        context = {
            "label": "Option A: Adopt uv",
            "question": "Should we adopt uv for dependency management?",
            "node_id": "node_123",
        }

        output = proposer.execute(context)

        # Check output structure
        assert output["proposal"] == "Use uv for faster dependency management"
        assert output["recommendation"] == "Use uv for faster dependency management"
        assert len(output["rationale"]) == 3
        assert output["confidence"] == 0.85
        assert output["llm_generated"] is True
        assert "score" in output  # Should have score from confidence

    def test_normalizes_invalid_confidence(self) -> None:
        """ProposerLLM should clamp confidence to [0, 1]."""
        from delibera.agents.llm_proposer import ProposerLLM

        # Test confidence > 1
        client = FakeLLMClient(
            response_json={
                "recommendation": "Test",
                "rationale": [],
                "claims": [],
                "confidence": 1.5,  # Invalid - should be clamped
            }
        )
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Test", "question": "Q", "node_id": "n1"})
        assert output["confidence"] == 1.0

        # Test confidence < 0
        client = FakeLLMClient(
            response_json={
                "recommendation": "Test",
                "rationale": [],
                "claims": [],
                "confidence": -0.5,  # Invalid - should be clamped
            }
        )
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Test", "question": "Q", "node_id": "n2"})
        assert output["confidence"] == 0.0

    def test_limits_rationale_to_max(self) -> None:
        """ProposerLLM should limit rationale to 6 items."""
        from delibera.agents.llm_proposer import ProposerLLM

        client = FakeLLMClient(
            response_json={
                "recommendation": "Test",
                "rationale": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],  # 8 items
                "claims": [],
                "confidence": 0.5,
            }
        )
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Test", "question": "Q", "node_id": "n1"})
        assert len(output["rationale"]) == 6

    def test_limits_claims_to_max(self) -> None:
        """ProposerLLM should limit claims to 8 items."""
        from delibera.agents.llm_proposer import ProposerLLM

        claims = [{"type": "fact", "text": f"Claim {i}"} for i in range(10)]
        client = FakeLLMClient(
            response_json={
                "recommendation": "Test",
                "rationale": [],
                "claims": claims,  # 10 claims
                "confidence": 0.5,
            }
        )
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Test", "question": "Q", "node_id": "n1"})
        assert len(output["claims"]) == 8

    def test_handles_missing_fields_gracefully(self) -> None:
        """ProposerLLM should provide defaults for missing fields."""
        from delibera.agents.llm_proposer import ProposerLLM

        # Empty response
        client = FakeLLMClient(response_json={})
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Option X", "question": "Q", "node_id": "n1"})

        assert "recommendation" in output
        assert "rationale" in output
        assert "claims" in output
        assert "confidence" in output
        assert output["confidence"] == 0.5  # Default

    def test_extracts_facts_for_facts_field(self) -> None:
        """ProposerLLM should extract fact claims for the 'facts' field."""
        from delibera.agents.llm_proposer import ProposerLLM

        client = FakeLLMClient(
            response_json={
                "recommendation": "Test",
                "rationale": [],
                "claims": [
                    {"type": "fact", "text": "Fact 1"},
                    {"type": "inference", "text": "Inference 1"},
                    {"type": "fact", "text": "Fact 2"},
                ],
                "confidence": 0.5,
            }
        )
        proposer = ProposerLLM(llm_client=client)
        output = proposer.execute({"label": "Test", "question": "Q", "node_id": "n1"})

        assert output["facts"] == ["Fact 1", "Fact 2"]


class TestLLMTraceEventsEmitted:
    """Test that LLM trace events are properly emitted during runs."""

    def test_llm_call_events_emitted_with_fake_client(self) -> None:
        """Run with LLM proposer should emit llm_call_requested and llm_call_succeeded."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "recommendation": "Test recommendation",
            "rationale": ["Reason 1"],
            "claims": [{"type": "fact", "text": "Test fact"}],
            "confidence": 0.7,
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_proposer=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question for LLM")

            # Read trace
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check for LLM events
            event_types = [e["event_type"] for e in events]
            assert "llm_call_requested" in event_types
            assert "llm_call_succeeded" in event_types

            # Check llm_call_requested content
            llm_requested = [e for e in events if e["event_type"] == "llm_call_requested"]
            assert len(llm_requested) >= 1
            assert llm_requested[0]["payload"]["role"] == "proposer"
            assert llm_requested[0]["payload"]["step"] == "PROPOSE"

    def test_llm_call_failed_emitted_on_error(self) -> None:
        """LLM errors should emit llm_call_failed and fall back to stub."""
        from delibera.engine.orchestrator import Engine
        from delibera.llm.base import LLMError

        fake_client = FakeLLMClient(raise_error=LLMError("Test error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_proposer=True,
                gates_enabled=False,
            )

            # Should still complete (falls back to stub)
            run_dir = engine.run("Test question with failing LLM")

            # Read trace
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check for failure event
            event_types = [e["event_type"] for e in events]
            assert "llm_call_failed" in event_types

            # Verify run still completed
            assert (run_dir / "artifact.json").exists()


class TestLLMDisabledInClaimCheck:
    """Test that LLM is not used during CLAIM_CHECK validation steps."""

    def test_llm_not_called_in_claim_check(self) -> None:
        """CLAIM_CHECK step should not invoke LLM."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "recommendation": "Test",
            "rationale": ["R1"],
            "claims": [{"type": "fact", "text": "F1"}],
            "confidence": 0.5,
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_proposer=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question")

            # Read trace
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Check that LLM was only called in PROPOSE, not CLAIM_CHECK
            llm_calls = [
                e for e in events if e["event_type"] in ("llm_call_requested", "llm_call_succeeded")
            ]
            for call in llm_calls:
                assert call["payload"]["step"] == "PROPOSE"
                # Should never see CLAIM_CHECK or validate step
                assert call["payload"]["step"] != "CLAIM_CHECK"

    def test_check_llm_allowed_raises_in_validate(self) -> None:
        """check_llm_allowed_in_step should raise for validate steps."""
        from delibera.agents.llm_proposer import LLMNotAllowedError, check_llm_allowed_in_step

        # Work steps are allowed
        check_llm_allowed_in_step("work")  # Should not raise

        # Validate steps are not allowed
        with pytest.raises(LLMNotAllowedError):
            check_llm_allowed_in_step("validate")

        # Score steps are not allowed
        with pytest.raises(LLMNotAllowedError):
            check_llm_allowed_in_step("score")


class TestCLIRequiresApiKey:
    """Test that CLI properly requires API key for LLM mode."""

    def test_cli_fails_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI should fail with clear error when API key is missing."""
        import os
        import subprocess
        import sys

        # Ensure GEMINI_API_KEY is not set
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Build environment without GEMINI_API_KEY
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "delibera.cli",
                "run",
                "--question",
                "Test",
                "--use-llm-proposer",
                "--no-gates",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Should fail
        assert result.returncode != 0
        # Should mention API key
        assert "GEMINI_API_KEY" in result.stderr or "api" in result.stderr.lower()


class TestRedaction:
    """Test redaction helpers for trace logging."""

    def test_redact_email(self) -> None:
        """redact_text should redact email addresses."""
        from delibera.llm.redaction import redact_text

        text = "Contact me at user@example.com for details"
        redacted = redact_text(text)
        assert "user@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_api_key(self) -> None:
        """redact_text should redact API keys."""
        from delibera.llm.redaction import redact_text

        text = "api_key=sk-1234567890abcdef1234567890abcdef"
        redacted = redact_text(text)
        assert "sk-1234567890" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_long_numbers(self) -> None:
        """redact_text should redact long numbers (10+ digits)."""
        from delibera.llm.redaction import redact_text

        text = "Account number: 1234567890123456"
        redacted = redact_text(text)
        assert "1234567890123456" not in redacted
        assert "[REDACTED]" in redacted

    def test_summarize_messages(self) -> None:
        """summarize_messages should create safe summary without full content."""
        from delibera.llm.base import LLMMessage
        from delibera.llm.redaction import summarize_messages

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Tell me about user@email.com"),
        ]

        summary = summarize_messages(messages)

        assert summary["message_count"] == 2
        assert "system" in summary["chars_by_role"]
        assert "user" in summary["chars_by_role"]
        # User preview should be redacted
        assert "user@email.com" not in summary["user_preview"]


class TestReplayIgnoresLLMEvents:
    """Test that replay works correctly with LLM trace events."""

    def test_replay_works_without_llm(self) -> None:
        """Replay should succeed even with LLM trace events present."""
        from delibera.engine.orchestrator import Engine
        from delibera.trace.replay import replay_from_directory

        response_json = {
            "recommendation": "Test",
            "rationale": ["R1"],
            "claims": [{"type": "fact", "text": "F1"}],
            "confidence": 0.5,
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run with LLM
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_proposer=True,
                gates_enabled=False,
            )
            run_dir = engine.run("Test question")

            # Replay should work without LLM client
            replayed = replay_from_directory(run_dir)

            # Should complete without errors
            assert replayed.errors == []
            assert replayed.question == "Test question"
            assert "recommendation" in replayed.final_artifact
