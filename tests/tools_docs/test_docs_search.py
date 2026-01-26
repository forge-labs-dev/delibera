"""Tests for the docs.search tool and evidence pack functionality."""

from pathlib import Path

import pytest

from delibera.tools.builtin.docs import DocsReadTool, DocsSearchTool
from delibera.tools.policy import (
    GlobalPolicy,
    PolicyContext,
    PolicyEngine,
    StepPolicyOverride,
)
from delibera.tools.registry import ToolRegistry, create_default_registry
from delibera.tools.router import ToolRouter
from delibera.tools.spec import ToolCapability, ToolDenied


class TestDocsSearchTool:
    """Tests for DocsSearchTool."""

    def test_capability_is_discovery(self) -> None:
        """Test that docs.search is classified as a discovery tool."""
        tool = DocsSearchTool()
        assert tool.capability == ToolCapability.DISCOVERY
        assert tool.is_discovery is True

    def test_docs_read_capability_is_inspect(self) -> None:
        """Test that docs.read is classified as an inspect tool."""
        tool = DocsReadTool()
        assert tool.capability == ToolCapability.INSPECT
        assert tool.is_discovery is False

    def test_search_empty_directory(self, tmp_path: Path) -> None:
        """Test searching an empty evidence directory."""
        tool = DocsSearchTool(evidence_root=tmp_path)
        result = tool.execute({"query": "test", "max_results": 5})

        assert result["results"] == []

    def test_search_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test searching a non-existent evidence directory."""
        nonexistent = tmp_path / "does_not_exist"
        tool = DocsSearchTool(evidence_root=nonexistent)
        result = tool.execute({"query": "test", "max_results": 5})

        assert result["results"] == []
        assert "warning" in result

    def test_search_returns_deterministic_results(self, tmp_path: Path) -> None:
        """Test that search results are deterministic."""
        # Create evidence files
        (tmp_path / "alpha.txt").write_text("This file contains fast performance data.")
        (tmp_path / "beta.txt").write_text("Another file with fast execution metrics.")
        (tmp_path / "gamma.txt").write_text("No matching content here.")

        tool = DocsSearchTool(evidence_root=tmp_path)

        # Run search multiple times
        result1 = tool.execute({"query": "fast", "max_results": 5})
        result2 = tool.execute({"query": "fast", "max_results": 5})
        result3 = tool.execute({"query": "fast", "max_results": 5})

        # Results should be identical across runs
        assert result1 == result2 == result3

        # Should find two files with "fast"
        assert len(result1["results"]) == 2

        # Results should be sorted by score desc, then path asc
        paths = [r["path"] for r in result1["results"]]
        assert paths == ["alpha.txt", "beta.txt"]  # Same score, sorted by path

    def test_search_score_ordering(self, tmp_path: Path) -> None:
        """Test that results are ordered by score (token count)."""
        # Create files with different match counts
        (tmp_path / "one_match.txt").write_text("uv is a tool.")
        (tmp_path / "three_matches.txt").write_text("uv uv uv is the best tool.")
        (tmp_path / "two_matches.txt").write_text("uv is great uv")

        tool = DocsSearchTool(evidence_root=tmp_path)
        result = tool.execute({"query": "uv", "max_results": 5})

        # Should be ordered by score (match count) descending
        paths = [r["path"] for r in result["results"]]
        assert paths[0] == "three_matches.txt"  # 3 matches
        assert paths[1] == "two_matches.txt"  # 2 matches
        assert paths[2] == "one_match.txt"  # 1 match

    def test_search_snippet_extraction(self, tmp_path: Path) -> None:
        """Test that snippets are extracted around matches."""
        content = "This is some prefix text. UV is extremely fast. This is suffix text."
        (tmp_path / "test.txt").write_text(content)

        tool = DocsSearchTool(evidence_root=tmp_path)
        result = tool.execute({"query": "fast", "max_results": 5})

        assert len(result["results"]) == 1
        snippet = result["results"][0]["snippet"]
        assert "fast" in snippet.lower()
        # Snippet should be non-empty and reasonable length
        assert 10 < len(snippet) <= 250

    def test_search_max_results_limit(self, tmp_path: Path) -> None:
        """Test that max_results is respected."""
        # Create many files
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(f"This is test file {i} with search term.")

        tool = DocsSearchTool(evidence_root=tmp_path)
        result = tool.execute({"query": "test", "max_results": 3})

        assert len(result["results"]) == 3

    def test_search_validates_query(self) -> None:
        """Test that invalid queries are rejected."""
        tool = DocsSearchTool()

        with pytest.raises(ValueError, match="query"):
            tool.validate_input({})

        with pytest.raises(ValueError, match="query"):
            tool.validate_input({"query": ""})

        with pytest.raises(ValueError, match="max_results"):
            tool.validate_input({"query": "test", "max_results": 0})

    def test_search_text_files_only(self, tmp_path: Path) -> None:
        """Test that only text files are searched."""
        (tmp_path / "text.txt").write_text("search term here")
        (tmp_path / "data.json").write_text('{"key": "search term"}')
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01search term")

        tool = DocsSearchTool(evidence_root=tmp_path)
        result = tool.execute({"query": "search", "max_results": 10})

        # Should find txt and json (text-like), but not bin
        paths = {r["path"] for r in result["results"]}
        assert "text.txt" in paths
        assert "data.json" in paths
        assert "binary.bin" not in paths


class TestDocsSearchPolicy:
    """Tests for docs.search policy enforcement."""

    def test_docs_search_denied_in_claim_check(self, tmp_path: Path) -> None:
        """Test that docs.search is denied during CLAIM_CHECK step."""
        # Create evidence pack
        (tmp_path / "test.txt").write_text("test content")

        # Create registry with docs.search
        registry = create_default_registry(evidence_root=tmp_path)

        # Create policy engine with default step overrides
        policy = PolicyEngine(
            global_policy=GlobalPolicy(
                enabled_tools={"docs.search", "docs.read", "calculator"},
            ),
            step_override=StepPolicyOverride(
                evidence_local_steps={"CLAIM_CHECK"},
            ),
        )

        # Create router
        router = ToolRouter(registry=registry, policy_engine=policy)

        # docs.search should be denied during CLAIM_CHECK
        with pytest.raises(ToolDenied) as exc_info:
            router.call(
                role="validator",
                step="CLAIM_CHECK",
                tool_name="docs.search",
                tool_input={"query": "test"},
            )

        assert "discovery" in exc_info.value.reason.lower()

    def test_docs_search_allowed_in_research(self, tmp_path: Path) -> None:
        """Test that docs.search is allowed during RESEARCH step."""
        # Create evidence pack
        (tmp_path / "test.txt").write_text("searchable content")

        registry = create_default_registry(evidence_root=tmp_path)
        policy = PolicyEngine(
            global_policy=GlobalPolicy(
                enabled_tools={"docs.search", "docs.read", "calculator"},
            ),
        )
        router = ToolRouter(registry=registry, policy_engine=policy)

        # Should succeed during RESEARCH
        result = router.call(
            role="researcher",
            step="RESEARCH",
            tool_name="docs.search",
            tool_input={"query": "searchable"},
        )

        assert "results" in result
        assert len(result["results"]) == 1

    def test_docs_read_denied_for_uncited_source_in_claim_check(self, tmp_path: Path) -> None:
        """Test that docs.read is denied for non-cited sources during CLAIM_CHECK."""
        # Create evidence file
        (tmp_path / "uncited.txt").write_text("uncited content")

        registry = create_default_registry(evidence_root=tmp_path)
        policy = PolicyEngine(
            global_policy=GlobalPolicy(
                enabled_tools={"docs.read", "docs.search"},
            ),
            step_override=StepPolicyOverride(
                evidence_local_steps={"CLAIM_CHECK"},
                evidence_local_tools={"docs.read"},
            ),
        )
        router = ToolRouter(registry=registry, policy_engine=policy)

        # Context with no cited sources
        context = PolicyContext(allowed_evidence_sources=set())

        with pytest.raises(ToolDenied) as exc_info:
            router.call(
                role="validator",
                step="CLAIM_CHECK",
                tool_name="docs.read",
                tool_input={"path": "uncited.txt"},
                policy_context=context,
            )

        assert "not in allowed evidence" in exc_info.value.reason

    def test_docs_read_allowed_for_cited_source_in_claim_check(self, tmp_path: Path) -> None:
        """Test that docs.read is allowed for cited sources during CLAIM_CHECK."""
        # Create evidence file
        (tmp_path / "cited.txt").write_text("cited content for validation")

        registry = create_default_registry(evidence_root=tmp_path)
        policy = PolicyEngine(
            global_policy=GlobalPolicy(
                enabled_tools={"docs.read", "docs.search"},
            ),
            step_override=StepPolicyOverride(
                evidence_local_steps={"CLAIM_CHECK"},
                evidence_local_tools={"docs.read"},
            ),
        )
        router = ToolRouter(registry=registry, policy_engine=policy)

        # Context with cited source
        context = PolicyContext(allowed_evidence_sources={"cited.txt"})

        # Should succeed for cited source
        result = router.call(
            role="validator",
            step="CLAIM_CHECK",
            tool_name="docs.read",
            tool_input={"path": "cited.txt"},
            policy_context=context,
        )

        assert "text" in result
        assert "cited content" in result["text"]


class TestResearchWithDocsSearch:
    """Tests for ResearcherStub using docs.search + docs.read."""

    def test_research_uses_docs_search_then_read(self, tmp_path: Path) -> None:
        """Test that research calls docs.search then docs.read."""
        from delibera.agents.stub import ResearcherStub

        # Create evidence pack
        (tmp_path / "uv_notes.txt").write_text(
            "UV is 10-100x faster than pip for dependency resolution."
        )

        registry = create_default_registry(evidence_root=tmp_path)

        # Track tool calls
        tool_calls: list[tuple[str, dict]] = []

        def tool_callback(name: str, input: dict) -> dict:
            tool_calls.append((name, input))
            tool = registry.get(name)
            return tool.execute(input)

        researcher = ResearcherStub()
        result = researcher.execute(
            context={
                "label": "Option A: Direct approach",
                "question": "Should we adopt uv?",
            },
            tool=tool_callback,
        )

        # Should have called docs.search first
        search_calls = [(n, i) for n, i in tool_calls if n == "docs.search"]
        assert len(search_calls) >= 1

        # Should have called docs.read after search
        read_calls = [(n, i) for n, i in tool_calls if n == "docs.read"]
        assert len(read_calls) >= 1

        # Should have found evidence
        assert len(result["evidence"]) > 0
        assert "fast" in result["evidence"][0]["excerpt"].lower()

    def test_research_fallback_when_search_unavailable(self, tmp_path: Path) -> None:
        """Test that research falls back to direct read if search fails."""
        from delibera.agents.stub import ResearcherStub

        # Create evidence file
        (tmp_path / "uv_notes.txt").write_text("UV is 10-100x faster than pip.")

        # Create registry without docs.search
        registry = ToolRegistry()
        from delibera.tools.builtin.docs import DocsReadTool

        registry.register(DocsReadTool(evidence_root=tmp_path))

        def tool_callback(name: str, input: dict) -> dict:
            tool = registry.get(name)
            return tool.execute(input)

        researcher = ResearcherStub()
        result = researcher.execute(
            context={
                "label": "Option A: Direct approach",
                "question": "Should we adopt uv?",
            },
            tool=tool_callback,
        )

        # Should still find evidence via fallback
        assert "notes" in result
        # Notes should indicate fallback was used
        notes_text = " ".join(result["notes"])
        assert "fallback" in notes_text.lower() or "failed" in notes_text.lower()


class TestToolCapabilityClassification:
    """Tests for tool capability classification."""

    def test_calculator_is_compute(self) -> None:
        """Test that calculator is classified as compute."""
        from delibera.tools.builtin.calculator import CalculatorTool

        tool = CalculatorTool()
        assert tool.capability == ToolCapability.COMPUTE
        assert tool.is_discovery is False

    def test_docs_read_is_inspect(self) -> None:
        """Test that docs.read is classified as inspect."""
        tool = DocsReadTool()
        assert tool.capability == ToolCapability.INSPECT
        assert tool.is_discovery is False

    def test_docs_search_is_discovery(self) -> None:
        """Test that docs.search is classified as discovery."""
        tool = DocsSearchTool()
        assert tool.capability == ToolCapability.DISCOVERY
        assert tool.is_discovery is True

    def test_default_registry_has_all_tools(self, tmp_path: Path) -> None:
        """Test that default registry includes docs.search."""
        registry = create_default_registry(evidence_root=tmp_path)

        assert registry.has("calculator")
        assert registry.has("docs.read")
        assert registry.has("docs.search")

        # Verify capabilities
        assert registry.get("calculator").capability == ToolCapability.COMPUTE
        assert registry.get("docs.read").capability == ToolCapability.INSPECT
        assert registry.get("docs.search").capability == ToolCapability.DISCOVERY
