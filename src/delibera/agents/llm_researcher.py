"""LLM-backed Researcher agent for Delibera.

Uses an LLM to generate targeted search queries, then retrieves
evidence using the configured retriever.
"""

from typing import TYPE_CHECKING, Any

from delibera.llm.base import LLMMessage, LLMRequest
from delibera.llm.prompts import build_researcher_system_prompt, build_researcher_user_prompt

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient
    from delibera.retrieval.base import EvidenceRetriever
    from delibera.tools import ToolCallback


class ResearcherLLM:
    """LLM-backed researcher agent.

    Generates targeted search queries using an LLM, then uses a retriever
    to find relevant evidence for each query.

    Attributes:
        llm_client: The LLM client to use for query generation.
        retriever: Optional evidence retriever to execute queries.
        model: Model name to use (None uses client default).
        temperature: Sampling temperature (default 0.3 for variety).
        max_output_tokens: Maximum output tokens (default 400).
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        retriever: "EvidenceRetriever | None" = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 400,
    ) -> None:
        """Initialize the LLM researcher.

        Args:
            llm_client: The LLM client for query generation.
            retriever: Optional retriever for executing queries.
            model: Model name (None uses client default).
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens.
        """
        self._llm_client = llm_client
        self._retriever = retriever
        self._model = model or ""
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def execute(
        self,
        context: dict[str, Any],
        tool: "ToolCallback | None" = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Generate search queries and retrieve evidence.

        Args:
            context: Must contain:
                - "label": Branch label
                - "question": Original question
                Optional:
                - "node_id": Node ID for tracing
                - "proposal": Proposal summary for context
            tool: Optional tool callback (not used by LLM researcher).

        Returns:
            Dict with evidence list and metadata, matching ResearcherStub format.
        """
        question = context.get("question", "")
        label = context.get("label", "")
        node_id = context.get("node_id", "")
        proposal = context.get("proposal")

        # Build LLM request for query generation
        system_prompt = build_researcher_system_prompt()
        user_prompt = build_researcher_user_prompt(
            question=question,
            label=label,
            proposal=proposal,
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
                "node_id": node_id,
                "role": "researcher",
                "step": "RESEARCH",
            },
        )

        # Generate queries via LLM
        response = self._llm_client.generate(request)
        parsed = response.parsed_json or {}

        # Extract and normalize queries
        queries = self._extract_queries(parsed)

        # Use retriever to execute queries if available
        evidence: list[dict[str, str]] = []
        notes: list[str] = [f"LLM generated {len(queries)} search queries"]

        if self._retriever is not None:
            for q in queries:
                query_text = q["query"]
                try:
                    results = self._retriever.retrieve(query_text, max_results=3)
                    for r in results:
                        ev: dict[str, str] = {
                            "source": r.source,
                            "excerpt": r.excerpt,
                            "query": query_text,
                        }
                        title = (r.metadata or {}).get("title", "")
                        if title:
                            ev["title"] = title
                        # Deduplicate by source
                        if not any(e["source"] == ev["source"] for e in evidence):
                            evidence.append(ev)
                except Exception as e:
                    notes.append(f"Query failed: {type(e).__name__}")

            notes.append(f"Retrieved {len(evidence)} unique evidence items")

            # Generate one-line summaries for all evidence items in a single LLM call
            if evidence:
                self._summarize_evidence(evidence, node_id)
        else:
            notes.append("No retriever available; queries generated but not executed")

        return {
            "evidence": evidence,
            "notes": notes,
            "queries_generated": queries,
            "role": "researcher",
            "step": "RESEARCH",
            "llm_generated": True,
        }

    def _summarize_evidence(
        self,
        evidence: list[dict[str, str]],
        node_id: str,
    ) -> None:
        """Generate one-line summaries for all evidence items in a single LLM call.

        Mutates evidence dicts in-place, adding a "summary" key.
        Silently skips on any error.
        """
        # Build numbered list of excerpts for the LLM
        lines = [
            "For each evidence excerpt below, write a concise one-line summary "
            "(max 15 words). Return a JSON object mapping the index to the summary.",
            'Example: {"0": "Summary of first item", "1": "Summary of second item"}',
            "",
        ]
        for i, ev in enumerate(evidence):
            excerpt = ev.get("excerpt", "")[:300]
            lines.append(f"[{i}]: {excerpt}")

        try:
            request = LLMRequest(
                model=self._model,
                messages=[
                    LLMMessage(role="user", content="\n".join(lines)),
                ],
                temperature=0.0,
                max_output_tokens=300,
                response_format="json",
                metadata={
                    "node_id": node_id,
                    "role": "researcher",
                    "step": "SUMMARIZE_EVIDENCE",
                },
            )
            response = self._llm_client.generate(request)
            summaries = response.parsed_json or {}

            for i, ev in enumerate(evidence):
                summary = summaries.get(str(i), "")
                if summary:
                    ev["summary"] = str(summary)[:150]
        except Exception:
            pass  # Non-critical; UI falls back to excerpt

    def _extract_queries(
        self,
        parsed: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Extract and normalize search queries from LLM output.

        Args:
            parsed: The parsed JSON from LLM.

        Returns:
            List of query dicts with "query" and "intent" keys.
        """
        queries_raw = parsed.get("queries", [])
        if not isinstance(queries_raw, list):
            return []

        queries: list[dict[str, str]] = []
        for item in queries_raw[:5]:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).strip()
            intent = str(item.get("intent", "")).strip()
            if query:
                # Truncate long queries
                if len(query) > 200:
                    query = query[:200]
                queries.append({"query": query, "intent": intent})

        return queries
