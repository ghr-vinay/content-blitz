from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import ResearchResult
from src.integrations.base_tool import BaseLLMClient, BaseSearchTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Search query extraction ────────────────────────────────────────────────────
_SEARCH_QUERY_PROMPT = """You are a search query specialist.

Convert the user's research request into 2-3 concise, targeted Google search queries.
Each query should target a distinct angle of the request so together they cover the full intent.

User request: "{user_request}"

Rules:
- Each query should be 3-7 words, keyword-style (not a sentence)
- Together the queries must cover ALL aspects of the request (topic + any specific requirements like examples, comparisons, statistics, names of people, etc.)
- Return ONLY the queries, one per line, no numbering, no extra text
"""

_SYNTHESIS_PROMPT = """You are a professional research analyst.

The user's original request was: "{user_request}"

Using the raw search results below, produce a structured research summary that fully satisfies the user's request.

Raw search results:
{raw_results}

Your output must be valid JSON with this exact structure:
{{
  "summary": "<2-3 paragraph narrative summary>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>", ...],
  "sources": ["<url1>", "<url2>", ...]
}}

Rules:
- summary: MUST address ALL aspects of the user's request, not just the general topic — if the user asked for examples, include named examples; if they asked for comparisons, include comparisons; etc.
- key_findings: 4-6 distinct, specific bullet-point insights that directly answer the user's request
- sources: all URL strings from the search results
- Respond with ONLY the JSON object, no extra text
"""

_REFINEMENT_SYNTHESIS_PROMPT = """You are a professional research analyst.

The user's original request was: "{user_request}"

You previously produced the research summary below. The user now wants to EXTEND it with additional information.

Existing research summary:
{existing_summary}

Existing key findings:
{existing_findings}

New search results for the extension topic:
{raw_results}

Produce an UPDATED research summary that merges the original with the new information and fully satisfies the user's request.
Your output must be valid JSON with this exact structure:
{{
  "summary": "<updated 2-3 paragraph summary covering both original and new topic>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "sources": ["<url1>", "<url2>", ...]
}}

Rules:
- summary: MUST address ALL aspects of the user's request — preserve what was already covered and add what's new
- Respond with ONLY the JSON object, no extra text
"""


class ResearchAgent(BaseAgent):
    """
    Conducts web research on a topic and synthesises findings.

    SOLID:
    - SRP: Responsible only for research and synthesis.
    - DIP: Depends on BaseSearchTool and BaseLLMClient abstractions —
           works with SERP, Perplexity, or any BaseSearchTool.
    """

    def __init__(self, search_tool: BaseSearchTool, llm: BaseLLMClient) -> None:
        self._search = search_tool
        self._llm = llm

    @property
    def name(self) -> str:
        return "research_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_request: str = (state.get("clarified_user_query") or state.get("user_query", "")).strip()
        is_refinement: bool = state.get("is_refinement", False)
        existing_research: ResearchResult | None = state.get("research") if is_refinement else None

        if not user_request:
            return {"error": "ResearchAgent: no topic provided."}

        logger.info("ResearchAgent: researching | is_refinement=%s | request=%r", is_refinement, user_request[:80])

        try:
            # Step 1: Generate targeted search queries from the full user request
            search_queries = self._extract_search_queries(user_request)
            logger.debug("ResearchAgent: search queries=%r", search_queries)

            # Step 2: Run all queries and merge results
            all_results: list[dict[str, Any]] = []
            seen_urls: set[str] = set()
            for query in search_queries:
                for r in self._search.search(query):
                    url = r.get("url", "")
                    if url not in seen_urls:
                        all_results.append(r)
                        seen_urls.add(url)

            raw_results = self._format_results(all_results)
            sources = [r.get("url", "") for r in all_results if r.get("url")]
            snippets = [r.get("snippet", "") for r in all_results if r.get("snippet")]

            # Step 3: Synthesise — merge with existing if refinement
            if is_refinement and existing_research:
                existing_findings_str = "\n".join(f"- {f}" for f in existing_research.key_findings)
                prompt = _REFINEMENT_SYNTHESIS_PROMPT.format(
                    user_request=user_request,
                    raw_results=raw_results,
                    existing_summary=existing_research.summary,
                    existing_findings=existing_findings_str,
                )
                all_sources = list(dict.fromkeys(existing_research.sources + sources))
                all_snippets = list(dict.fromkeys(existing_research.source_snippets + snippets))
            else:
                prompt = _SYNTHESIS_PROMPT.format(user_request=user_request, raw_results=raw_results)
                all_sources = sources
                all_snippets = snippets

            raw = self._llm.generate(prompt, config={"run_name": "research_agent~run"})
            parsed = self._parse_json(raw)

            research = ResearchResult(
                topic=user_request,
                summary=parsed.get("summary", raw),
                key_findings=parsed.get("key_findings", []),
                sources=parsed.get("sources", all_sources),
                source_snippets=all_snippets,
            )

            logger.info(
                "ResearchAgent: done | queries=%d | findings=%d | sources=%d",
                len(search_queries),
                len(research.key_findings),
                len(research.sources),
            )
            return {"research": research, "error": None}

        except Exception as exc:
            logger.exception("ResearchAgent failed: %s", exc)
            return {"error": str(exc)}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_search_queries(self, user_request: str) -> list[str]:
        """Use a small LLM call to convert the user request into 2-3 targeted search queries."""
        prompt = _SEARCH_QUERY_PROMPT.format(user_request=user_request)
        try:
            raw = self._llm.generate(prompt, config={"run_name": "research_agent~extract_queries"})
            queries = [q.strip() for q in raw.strip().splitlines() if q.strip()]
            if queries:
                return queries[:3]  # cap at 3
        except Exception as exc:
            logger.warning("ResearchAgent: query extraction failed, falling back to raw request: %s", exc)
        return [user_request]  # fallback: use full request as single query

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] Title: {r.get('title', '')}\n"
                f"    Snippet: {r.get('snippet', '')}\n"
                f"    URL: {r.get('url', '')}"
            )
        return "\n\n".join(lines)

    def _parse_json(self, raw: str) -> dict:
        import json
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            return {"summary": raw, "key_findings": [], "sources": []}
