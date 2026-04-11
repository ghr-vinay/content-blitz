import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Valid single-intent values the classifier may emit
_VALID_INTENTS = {"research", "blog", "linkedin", "image", "strategy", "multi"}

_SYSTEM_PROMPT = """You are a routing assistant for ContentBlitz, an AI content marketing system.

Classify the user's request into EXACTLY ONE of these intents:
- research   → user wants web research / fact-finding on a topic
- blog       → user wants an SEO blog post written
- linkedin   → user wants a LinkedIn post written
- image      → user wants an image generated
- strategy   → user wants a content strategy or plan
- multi      → user wants research AND one or more content formats (blog, linkedin, image)

Rules:
1. Reply with ONLY a JSON object: {{"intent": "<value>", "clarified_query": "<cleaned user query>"}}
2. If the request is ambiguous, default to "research".
3. "multi" applies when the user explicitly asks for multiple output formats or says things like
   "research X and write a blog about it".

Examples:
- "Write me a blog about LangGraph" → {{"intent": "blog", "clarified_query": "Write a blog post about LangGraph"}}
- "Research the latest AI trends and make a LinkedIn post" → {{"intent": "multi", "clarified_query": "AI trends 2025"}}
- "Generate an image of a futuristic city" → {{"intent": "image", "clarified_query": "futuristic city"}}
"""


class QueryHandlerAgent(BaseAgent):
    """
    Routes user requests to the appropriate downstream agent(s).

    SOLID:
    - SRP: Single responsibility — intent classification and routing only.
    - DIP: Depends on BaseLLMClient, not a concrete LLM provider.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "query_handler"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_query: str = state.get("user_query", "").strip()
        if not user_query:
            logger.warning("QueryHandlerAgent: empty user_query")
            return {"error": "No query provided.", "intent": None}

        logger.info("QueryHandlerAgent: classifying query=%r", user_query[:80])

        prompt = f"{_SYSTEM_PROMPT}\n\nUser request: {user_query}"

        try:
            raw = self._llm.generate(
                prompt,
                config={"run_name": "query_handler~run"},
            )
            parsed = self._parse_response(raw)
            intent: str = parsed.get("intent", "research")
            clarified: str = parsed.get("clarified_query", user_query)

            if intent not in _VALID_INTENTS:
                logger.warning(
                    "QueryHandlerAgent: unknown intent '%s', defaulting to research",
                    intent,
                )
                intent = "research"

            logger.info("QueryHandlerAgent: intent=%s", intent)
            return {
                "intent": intent,
                "user_query": clarified,
                "error": None,
            }

        except Exception as exc:
            logger.exception("QueryHandlerAgent failed: %s", exc)
            return {"error": str(exc), "intent": None}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict[str, str]:
        """
        Extract JSON from LLM response, which may contain extra prose.
        Tries strict JSON parse first, then scans for the first {...} block.
        """
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
            raise ValueError(f"Could not parse JSON from LLM response: {raw!r}")
