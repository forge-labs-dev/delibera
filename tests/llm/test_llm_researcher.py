"""Tests for LLM researcher agent.

These tests use a FakeLLMClient and FakeRetriever to avoid network calls.
"""

import json
import tempfile
from dataclasses import dataclass
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
            text = '{"queries": [{"query": "default", "intent": "test"}]}'

        return LLMResponse(
            text=text,
            parsed_json=self.response_json,
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            provider="fake",
            model="fake-model",
            latency_ms=10,
        )


@dataclass
class FakeRetrievalResult:
    """Fake retrieval result for testing."""

    source: str
    excerpt: str
    score: float = 1.0
    source_type: str = "local"
    method: str = "keyword"
    metadata: dict[str, Any] | None = None


class FakeRetriever:
    """Fake retriever for testing."""

    def __init__(
        self,
        results: list[FakeRetrievalResult] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.raise_error = raise_error
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, max_results: int = 5) -> list[FakeRetrievalResult]:
        self.calls.append((query, max_results))
        if self.raise_error:
            raise self.raise_error
        return self.results[:max_results]


class TestResearcherLLMParsesJson:
    """Test that ResearcherLLM correctly parses LLM JSON responses."""

    def test_parses_valid_queries(self) -> None:
        """ResearcherLLM should parse valid queries from LLM response."""
        from delibera.agents.llm_researcher import ResearcherLLM

        response_json = {
            "queries": [
                {"query": "GraphQL migration best practices", "intent": "Find migration guides"},
                {
                    "query": "GraphQL vs REST performance benchmarks",
                    "intent": "Compare performance",
                },
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        researcher = ResearcherLLM(llm_client=client)

        output = researcher.execute({"question": "Should we migrate?", "label": "Full migration"})

        assert output["role"] == "researcher"
        assert output["step"] == "RESEARCH"
        assert output["llm_generated"] is True
        assert len(output["queries_generated"]) == 2
        assert output["queries_generated"][0]["query"] == "GraphQL migration best practices"

    def test_executes_queries_with_retriever(self) -> None:
        """ResearcherLLM should use retriever to execute generated queries."""
        from delibera.agents.llm_researcher import ResearcherLLM

        response_json = {
            "queries": [
                {"query": "query 1", "intent": "test"},
                {"query": "query 2", "intent": "test"},
            ],
        }

        results = [
            FakeRetrievalResult(source="file1.txt", excerpt="Evidence 1"),
            FakeRetrievalResult(source="file2.txt", excerpt="Evidence 2"),
        ]

        client = FakeLLMClient(response_json=response_json)
        retriever = FakeRetriever(results=results)
        researcher = ResearcherLLM(llm_client=client, retriever=retriever)

        output = researcher.execute({"question": "Q", "label": "L"})

        assert len(output["evidence"]) == 2
        assert output["evidence"][0]["source"] == "file1.txt"
        assert len(retriever.calls) == 2

    def test_deduplicates_evidence_by_source(self) -> None:
        """ResearcherLLM should not return duplicate evidence items."""
        from delibera.agents.llm_researcher import ResearcherLLM

        response_json = {
            "queries": [
                {"query": "query 1", "intent": "test"},
                {"query": "query 2", "intent": "test"},
            ],
        }

        # Same source returned for both queries
        results = [
            FakeRetrievalResult(source="same_file.txt", excerpt="Evidence"),
        ]

        client = FakeLLMClient(response_json=response_json)
        retriever = FakeRetriever(results=results)
        researcher = ResearcherLLM(llm_client=client, retriever=retriever)

        output = researcher.execute({"question": "Q", "label": "L"})

        assert len(output["evidence"]) == 1

    def test_works_without_retriever(self) -> None:
        """ResearcherLLM should return queries even without a retriever."""
        from delibera.agents.llm_researcher import ResearcherLLM

        response_json = {
            "queries": [
                {"query": "test query", "intent": "test"},
            ],
        }

        client = FakeLLMClient(response_json=response_json)
        researcher = ResearcherLLM(llm_client=client)  # No retriever

        output = researcher.execute({"question": "Q", "label": "L"})

        assert len(output["evidence"]) == 0
        assert len(output["queries_generated"]) == 1
        assert any("not executed" in n for n in output["notes"])

    def test_handles_empty_queries(self) -> None:
        """ResearcherLLM should handle empty query list."""
        from delibera.agents.llm_researcher import ResearcherLLM

        client = FakeLLMClient(response_json={"queries": []})
        researcher = ResearcherLLM(llm_client=client)

        output = researcher.execute({"question": "Q", "label": "L"})

        assert output["queries_generated"] == []
        assert output["evidence"] == []

    def test_limits_queries_to_five(self) -> None:
        """ResearcherLLM should cap at 5 queries."""
        from delibera.agents.llm_researcher import ResearcherLLM

        queries = [{"query": f"q{i}", "intent": f"i{i}"} for i in range(8)]
        client = FakeLLMClient(response_json={"queries": queries})
        researcher = ResearcherLLM(llm_client=client)

        output = researcher.execute({"question": "Q", "label": "L"})

        assert len(output["queries_generated"]) == 5

    def test_truncates_long_queries(self) -> None:
        """ResearcherLLM should truncate queries longer than 200 chars."""
        from delibera.agents.llm_researcher import ResearcherLLM

        long_query = "A" * 250
        client = FakeLLMClient(response_json={"queries": [{"query": long_query, "intent": "test"}]})
        researcher = ResearcherLLM(llm_client=client)

        output = researcher.execute({"question": "Q", "label": "L"})

        assert len(output["queries_generated"][0]["query"]) == 200


class TestResearcherLLMRequestFormat:
    """Test that ResearcherLLM creates correct LLM requests."""

    def test_uses_json_response_format(self) -> None:
        """ResearcherLLM should request JSON response format."""
        from delibera.agents.llm_researcher import ResearcherLLM

        client = FakeLLMClient(response_json={"queries": [{"query": "q", "intent": "i"}]})
        researcher = ResearcherLLM(llm_client=client)

        researcher.execute({"question": "Q", "label": "L"})

        assert len(client.calls) == 1
        assert client.calls[0].response_format == "json"

    def test_includes_metadata(self) -> None:
        """ResearcherLLM should include role and step in request metadata."""
        from delibera.agents.llm_researcher import ResearcherLLM

        client = FakeLLMClient(response_json={"queries": [{"query": "q", "intent": "i"}]})
        researcher = ResearcherLLM(llm_client=client)

        researcher.execute({"question": "Q", "label": "L"})

        metadata = client.calls[0].metadata
        assert metadata["role"] == "researcher"
        assert metadata["step"] == "RESEARCH"

    def test_passes_question_in_prompt(self) -> None:
        """ResearcherLLM should include the question in the user message."""
        from delibera.agents.llm_researcher import ResearcherLLM

        client = FakeLLMClient(response_json={"queries": [{"query": "q", "intent": "i"}]})
        researcher = ResearcherLLM(llm_client=client)

        researcher.execute({"question": "Should we use GraphQL?", "label": "Option A"})

        user_messages = [m for m in client.calls[0].messages if m.role == "user"]
        assert len(user_messages) == 1
        assert "Should we use GraphQL?" in user_messages[0].content
        assert "Option A" in user_messages[0].content


class TestResearcherLLMErrors:
    """Test error handling in ResearcherLLM."""

    def test_raises_on_llm_error(self) -> None:
        """ResearcherLLM should propagate LLM errors."""
        from delibera.agents.llm_researcher import ResearcherLLM

        client = FakeLLMClient(raise_error=LLMError("API error"))
        researcher = ResearcherLLM(llm_client=client)

        with pytest.raises(LLMError, match="API error"):
            researcher.execute({"question": "Q", "label": "L"})

    def test_handles_retriever_error_gracefully(self) -> None:
        """ResearcherLLM should handle retriever errors without failing."""
        from delibera.agents.llm_researcher import ResearcherLLM

        response_json = {
            "queries": [{"query": "test", "intent": "test"}],
        }
        client = FakeLLMClient(response_json=response_json)
        retriever = FakeRetriever(raise_error=RuntimeError("Connection failed"))
        researcher = ResearcherLLM(llm_client=client, retriever=retriever)

        output = researcher.execute({"question": "Q", "label": "L"})

        # Should not fail, just note the error
        assert any("failed" in n.lower() for n in output["notes"])


class TestResearcherLLMTraceEvents:
    """Test that LLM researcher trace events are properly emitted."""

    def test_researcher_llm_events_emitted(self) -> None:
        """Run with LLM researcher should emit llm_call events."""
        from delibera.engine.orchestrator import Engine

        response_json = {
            "queries": [
                {"query": "test query", "intent": "find evidence"},
            ],
        }
        fake_client = FakeLLMClient(response_json=response_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                llm_client=fake_client,
                use_llm_researcher=True,
                gates_enabled=False,
            )

            run_dir = engine.run("Test question for researcher")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            researcher_requested = [
                e
                for e in events
                if e["event_type"] == "llm_call_requested" and e["payload"]["role"] == "researcher"
            ]
            researcher_succeeded = [
                e
                for e in events
                if e["event_type"] == "llm_call_succeeded" and e["payload"]["role"] == "researcher"
            ]

            assert len(researcher_requested) >= 1
            assert researcher_requested[0]["payload"]["step"] == "RESEARCH"
            assert len(researcher_succeeded) >= 1


class TestResearcherPrompts:
    """Test researcher prompt helpers."""

    def test_system_prompt_is_non_empty(self) -> None:
        """Researcher system prompt should be a non-empty string."""
        from delibera.llm.prompts import build_researcher_system_prompt

        prompt = build_researcher_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_user_prompt_contains_question_and_label(self) -> None:
        """Researcher user prompt should contain the question and label."""
        from delibera.llm.prompts import build_researcher_user_prompt

        prompt = build_researcher_user_prompt("Should we use GraphQL?", "Full migration")
        assert "Should we use GraphQL?" in prompt
        assert "Full migration" in prompt

    def test_user_prompt_includes_proposal(self) -> None:
        """Researcher user prompt should include proposal when provided."""
        from delibera.llm.prompts import build_researcher_user_prompt

        prompt = build_researcher_user_prompt(
            "Test?", "Option A", proposal="Use GraphQL for all new APIs"
        )
        assert "Use GraphQL for all new APIs" in prompt

    def test_user_prompt_without_proposal(self) -> None:
        """Researcher user prompt should work without proposal."""
        from delibera.llm.prompts import build_researcher_user_prompt

        prompt = build_researcher_user_prompt("Test?", "Option A")
        assert "PROPOSAL" not in prompt

    def test_researcher_json_schema_structure(self) -> None:
        """Researcher JSON schema should have required fields."""
        from delibera.llm.prompts import RESEARCHER_JSON_SCHEMA

        assert RESEARCHER_JSON_SCHEMA["type"] == "object"
        assert "queries" in RESEARCHER_JSON_SCHEMA["properties"]
        assert "queries" in RESEARCHER_JSON_SCHEMA["required"]
