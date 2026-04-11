import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import LinkedInPost, ResearchResult
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LINKEDIN_PROMPT = """You are an expert LinkedIn content strategist.

Write an engaging LinkedIn post on the topic below.

Topic: {topic}
Post type: {post_type}

{research_context}

Your response must be valid JSON with this exact structure:
{{
  "content": "<full LinkedIn post text>",
  "hashtags": ["#hashtag1", "#hashtag2", ...],
  "post_type": "{post_type}"
}}

LinkedIn writing guidelines:
- Character limit: 700-1200 characters for the content field (sweet spot for engagement)
- Open with a strong hook — first line must make people stop scrolling
- Use short paragraphs (1-2 lines) with blank lines between them for readability
- Add 3-5 relevant hashtags (in the hashtags array, NOT in the content)
- Close with a question or call-to-action to drive comments
- Tone options by post_type:
    thought-leadership → authoritative, insight-driven, first-person
    announcement       → excited, clear, benefit-focused
    story              → narrative, personal, emotionally engaging
    listicle           → structured, punchy, scannable
    general            → professional, conversational
- Respond with ONLY the JSON object, no extra text
"""

_RESEARCH_CONTEXT_TMPL = """
Research context:
{summary}

Key insights:
{findings}
"""


class LinkedInWriterAgent(BaseAgent):
    """
    Generates engaging LinkedIn posts with hashtag strategy.

    SOLID:
    - SRP: Responsible only for LinkedIn post generation.
    - DIP: Depends on BaseLLMClient abstraction.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "linkedin_writer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        topic: str = state.get("user_query", "").strip()
        research: ResearchResult | None = state.get("research")
        post_type: str = state.get("linkedin_post_type", "general")

        if not topic:
            return {"error": "LinkedInWriterAgent: no topic provided."}

        logger.info(
            "LinkedInWriterAgent: writing post | topic=%r | type=%s",
            topic[:60], post_type,
        )

        research_context = ""
        if research:
            findings_str = "\n".join(f"- {f}" for f in research.key_findings)
            research_context = _RESEARCH_CONTEXT_TMPL.format(
                summary=research.summary,
                findings=findings_str,
            )

        prompt = _LINKEDIN_PROMPT.format(
            topic=topic,
            post_type=post_type,
            research_context=research_context,
        )

        try:
            raw = self._llm.generate(prompt, config={"run_name": "linkedin_writer"})
            parsed = self._parse_json(raw)

            post = LinkedInPost(
                content=parsed.get("content", raw),
                hashtags=parsed.get("hashtags", []),
                post_type=parsed.get("post_type", post_type),
            )

            logger.info(
                "LinkedInWriterAgent: done | chars=%d | hashtags=%d",
                post.character_count,
                len(post.hashtags),
            )
            return {"linkedin_post": post, "error": None}

        except Exception as exc:
            logger.exception("LinkedInWriterAgent failed: %s", exc)
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
