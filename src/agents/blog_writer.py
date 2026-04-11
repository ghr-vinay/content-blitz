import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import BlogPost, ResearchResult
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BLOG_PROMPT = """You are an expert SEO content writer.

Write a comprehensive, SEO-optimised blog post on the topic below.

Topic: {topic}

{research_context}

Your response must be valid JSON with this exact structure:
{{
  "title": "<compelling, SEO-friendly H1 title>",
  "meta_description": "<155-160 character meta description with primary keyword>",
  "keywords": ["<primary keyword>", "<secondary keyword 1>", "<secondary keyword 2>", ...],
  "headers": ["<H2 section title 1>", "<H2 section title 2>", ...],
  "content": "<full blog post in Markdown, using the headers above as H2 sections>"
}}

Writing guidelines:
- Length: 800-1200 words
- Naturally weave keywords into the content (avoid keyword stuffing)
- Start with a strong hook paragraph
- Use H2 headers to break up sections
- End with a clear call-to-action
- Tone: professional yet accessible
- Respond with ONLY the JSON object, no extra text
"""

_RESEARCH_CONTEXT_TMPL = """
Research context (use this as factual foundation):
Summary: {summary}

Key findings:
{findings}

Sources: {sources}
"""


class BlogWriterAgent(BaseAgent):
    """
    Generates long-form SEO-optimised blog posts.

    SOLID:
    - SRP: Single responsibility — blog post generation only.
    - DIP: Depends on BaseLLMClient abstraction.
    - OCP: Research context is optional — works standalone or research-first.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "blog_writer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        topic: str = state.get("user_query", "").strip()
        research: ResearchResult | None = state.get("research")

        if not topic:
            return {"error": "BlogWriterAgent: no topic provided."}

        logger.info("BlogWriterAgent: writing blog for topic=%r", topic[:80])

        research_context = ""
        if research:
            findings_str = "\n".join(f"- {f}" for f in research.key_findings)
            research_context = _RESEARCH_CONTEXT_TMPL.format(
                summary=research.summary,
                findings=findings_str,
                sources=", ".join(research.sources),
            )

        prompt = _BLOG_PROMPT.format(topic=topic, research_context=research_context)

        try:
            raw = self._llm.generate(prompt, config={"run_name": "blog_writer"})
            parsed = self._parse_json(raw)

            blog = BlogPost(
                title=parsed.get("title", f"The Complete Guide to {topic}"),
                content=parsed.get("content", raw),
                meta_description=parsed.get("meta_description", ""),
                keywords=parsed.get("keywords", []),
                headers=parsed.get("headers", []),
            )

            logger.info(
                "BlogWriterAgent: done | words=%d | keywords=%d",
                blog.word_count,
                len(blog.keywords),
            )
            return {"blog_post": blog, "error": None}

        except Exception as exc:
            logger.exception("BlogWriterAgent failed: %s", exc)
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
            return {"content": raw}
