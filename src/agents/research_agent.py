from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import ResearchResult
from src.integrations.base_tool import BaseLLMClient, BaseSearchTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYNTHESIS_PROMPT = """You are a professional research analyst.

Using the raw search results below, produce a structured research summary on the topic: "{topic}"

Raw search results:
{raw_results}

Your output must be valid JSON with this exact structure:
{{
  "summary": "<2-3 paragraph narrative summary of the topic>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>", ...],
  "sources": ["<url1>", "<url2>", ...]
}}

Rules:
- summary: concise, factual, well-written prose (2-3 paragraphs)
- key_findings: 4-6 distinct, specific bullet-point insights
- sources: all URL strings from the search results
- Respond with ONLY the JSON object, no extra text
"""

_REFINEMENT_SYNTHESIS_PROMPT = """You are a professional research analyst.

You previously produced the research summary below. The user now wants to EXTEND it with additional information.

Existing research summary:
{existing_summary}

Existing key findings:
{existing_findings}

New search results for the extension topic: "{topic}"
{raw_results}

Produce an UPDATED research summary that merges the original with the new information.
Your output must be valid JSON with this exact structure:
{{
  "summary": "<updated 2-3 paragraph summary covering both original and new topic>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "sources": ["<url1>", "<url2>", ...]
}}

Respond with ONLY the JSON object, no extra text.
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
        topic: str = (state.get("clarified_user_query") or state.get("user_query", "")).strip()
        is_refinement: bool = state.get("is_refinement", False)
        existing_research: ResearchResult | None = state.get("research") if is_refinement else None

        if not topic:
            return {"error": "ResearchAgent: no topic provided."}

        logger.info("ResearchAgent: researching topic=%r | is_refinement=%s", topic[:80], is_refinement)

        try:
            # Step 1: Search
            results = self._search.search(topic)
            raw_results = self._format_results(results)
            sources = [r.get("url", "") for r in results if r.get("url")]
            snippets = [r.get("snippet", "") for r in results if r.get("snippet")]

            # Step 2: Synthesise — merge with existing if refinement
            if is_refinement and existing_research:
                existing_findings_str = "\n".join(f"- {f}" for f in existing_research.key_findings)
                prompt = _REFINEMENT_SYNTHESIS_PROMPT.format(
                    topic=topic,
                    raw_results=raw_results,
                    existing_summary=existing_research.summary,
                    existing_findings=existing_findings_str,
                )
                all_sources = list(dict.fromkeys(existing_research.sources + sources))
                all_snippets = list(dict.fromkeys(existing_research.source_snippets + snippets))
            else:
                prompt = _SYNTHESIS_PROMPT.format(topic=topic, raw_results=raw_results)
                all_sources = sources
                all_snippets = snippets

            raw = self._llm.generate(prompt, config={"run_name": "research_agent~run"})
            parsed = self._parse_json(raw)

            research = ResearchResult(
                topic=topic,
                summary=parsed.get("summary", raw),
                key_findings=parsed.get("key_findings", []),
                sources=parsed.get("sources", all_sources),
                source_snippets=all_snippets,
            )

            logger.info(
                "ResearchAgent: done | findings=%d | sources=%d",
                len(research.key_findings),
                len(research.sources),
            )
            return {"research": research, "error": None}

        except Exception as exc:
            logger.exception("ResearchAgent failed: %s", exc)
            return {"error": str(exc)}

    # ── Helpers ────────────────────────────────────────────────────────────────

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
