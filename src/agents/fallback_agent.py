from typing import Any

from src.agents.base_agent import BaseAgent
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger
from src.utils.progress import report_progress

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant for ContentBlitz, an AI content marketing tool.

A user has sent a message that is outside the scope of what ContentBlitz can do.

ContentBlitz specialises in:
- Researching topics using live web search
- Writing SEO-optimised blog posts
- Writing LinkedIn posts (thought-leadership, announcements, stories, listicles)
- Generating images via DALL-E 3
- Creating content strategies and marketing plans

Your job is to politely tell the user that either their request is outside ContentBlitz's
scope, or their intention isn't clear enough to act on — and guide them back by briefly
listing what you CAN help with.

If the request seems unclear, also gently ask them to rephrase.

Keep your response concise (2-4 sentences), friendly, and helpful.
Do NOT make up capabilities. Do NOT answer the off-topic question.
"""


class FallbackAgent(BaseAgent):
    """
    Handles off-topic or unsupported user requests.

    Returns a polite message redirecting the user to ContentBlitz's
    supported capabilities instead of attempting to answer.

    SOLID:
    - SRP: Solely responsible for generating out-of-scope redirection messages.
    - DIP: Depends on BaseLLMClient abstraction, not a concrete provider.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "fallback_agent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_query: str = state.get("user_query", "").strip()
        report_progress("💬 Thinking…")
        logger.info("FallbackAgent: handling off-topic query=%r", user_query[:80])

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"User's off-topic message: {user_query}\n\n"
            "Write a brief, friendly redirection message:"
        )

        try:
            message = self._llm.generate(
                prompt,
                config={"run_name": "fallback_agent~run"},
            )
            logger.info("FallbackAgent: response generated (%d chars)", len(message))
            return {"fallback_message": message.strip(), "error": None}

        except Exception as exc:
            logger.exception("FallbackAgent failed: %s", exc)
            return {
                "fallback_message": (
                    "I'm ContentBlitz, an AI content marketing assistant. "
                    "I can help you research topics, write blog posts and LinkedIn posts, "
                    "generate images, and build content strategies. "
                    "How can I help you with your content today?"
                ),
                "error": None,
            }
