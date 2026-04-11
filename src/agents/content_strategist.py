import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import ResearchResult
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STRATEGY_PROMPT = """You are a senior content strategist.

Based on the topic and optional research below, produce a structured content strategy.

Topic: {topic}

{research_context}

Your response must be valid JSON with this exact structure:
{{
  "executive_summary": "<2-3 sentence overview of the content opportunity>",
  "target_audience": "<who this content is for>",
  "content_pillars": ["<pillar 1>", "<pillar 2>", "<pillar 3>"],
  "recommended_formats": ["<format 1>", "<format 2>", ...],
  "content_series": [
    {{"title": "<piece 1 title>", "format": "<blog|linkedin|image>", "angle": "<unique angle>"}},
    {{"title": "<piece 2 title>", "format": "<blog|linkedin|image>", "angle": "<unique angle>"}},
    {{"title": "<piece 3 title>", "format": "<blog|linkedin|image>", "angle": "<unique angle>"}}
  ],
  "key_messages": ["<message 1>", "<message 2>", "<message 3>"],
  "cta": "<primary call-to-action for this content campaign>"
}}

Respond with ONLY the JSON object, no extra text.
"""

_RESEARCH_CONTEXT_TMPL = """
Research context:
Summary: {summary}

Key findings:
{findings}
"""


class ContentStrategistAgent(BaseAgent):
    """
    Formats and organises research into a structured content strategy.

    SOLID:
    - SRP: Responsible only for strategy synthesis and formatting.
    - DIP: Depends on BaseLLMClient abstraction.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "content_strategist"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        topic: str = state.get("user_query", "").strip()
        research: ResearchResult | None = state.get("research")

        if not topic:
            return {"error": "ContentStrategistAgent: no topic provided."}

        logger.info("ContentStrategistAgent: building strategy for topic=%r", topic[:80])

        research_context = ""
        if research:
            findings_str = "\n".join(f"- {f}" for f in research.key_findings)
            research_context = _RESEARCH_CONTEXT_TMPL.format(
                summary=research.summary,
                findings=findings_str,
            )

        prompt = _STRATEGY_PROMPT.format(
            topic=topic, research_context=research_context
        )

        try:
            raw = self._llm.generate(
                prompt, config={"run_name": "content_strategist~run"}
            )
            parsed = self._parse_json(raw)

            # Store formatted strategy as a readable markdown block in state
            formatted = self._to_markdown(topic, parsed)

            logger.info(
                "ContentStrategistAgent: done | pillars=%d | series=%d",
                len(parsed.get("content_pillars", [])),
                len(parsed.get("content_series", [])),
            )
            return {"content_strategy": formatted, "error": None}

        except Exception as exc:
            logger.exception("ContentStrategistAgent failed: %s", exc)
            return {"error": str(exc)}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            return {"executive_summary": raw}

    def _to_markdown(self, topic: str, data: dict) -> str:
        lines = [f"# Content Strategy: {topic}\n"]

        if summary := data.get("executive_summary"):
            lines += ["## Executive Summary", summary, ""]

        if audience := data.get("target_audience"):
            lines += ["## Target Audience", audience, ""]

        if pillars := data.get("content_pillars"):
            lines += ["## Content Pillars"]
            lines += [f"- {p}" for p in pillars]
            lines.append("")

        if formats := data.get("recommended_formats"):
            lines += ["## Recommended Formats"]
            lines += [f"- {f}" for f in formats]
            lines.append("")

        if series := data.get("content_series"):
            lines += ["## Content Series"]
            for item in series:
                lines.append(
                    f"- **{item.get('title', '')}** ({item.get('format', '')}) — {item.get('angle', '')}"
                )
            lines.append("")

        if messages := data.get("key_messages"):
            lines += ["## Key Messages"]
            lines += [f"- {m}" for m in messages]
            lines.append("")

        if cta := data.get("cta"):
            lines += ["## Call to Action", cta, ""]

        return "\n".join(lines)
