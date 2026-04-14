import json
from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

# Number of prior messages (turns) to include as conversation context.
# Each turn = 1 message, so 6 ≈ 3 back-and-forth exchanges.
_HISTORY_WINDOW = 6

logger = get_logger(__name__)

# Valid single-intent values the classifier may emit
_VALID_INTENTS = {"research", "blog", "linkedin", "image", "strategy", "multi"}

_REFINEMENT_SIGNALS = (
    "also", "additionally", "include", "add", "update", "change", "modify",
    "make it", "now write", "rewrite", "expand", "extend", "improve", "revise",
)

_SYSTEM_PROMPT = """You are a routing assistant for ContentBlitz, an AI content marketing system.

Classify the user's request into EXACTLY ONE of these intents:
- research   → user wants web research / fact-finding on a topic
- blog       → user wants an SEO blog post written
- linkedin   → user wants a LinkedIn post written
- image      → user wants an image generated
- strategy   → user wants a content strategy or plan
- multi      → user wants research AND one or more content formats IN THE SAME REQUEST

Rules:
1. Reply with ONLY a JSON object:
   {"intent": "<value>", "clarified_query": "<full cumulative query>", "is_refinement": <true|false>}
2. If the request is ambiguous, default to "research".
3. "multi" applies ONLY when the CURRENT message explicitly asks for multiple output formats
   in a single request (e.g. "research X and write a blog about it").
   A research request that follows a prior LinkedIn post is still just "research" — do NOT
   inherit prior output types from history as the current intent.
4. Use conversation history ONLY to resolve topic references and pronouns
   (e.g. "research about this" after a LinkedIn post about video editing → clarified_query: "video editing software programs").
   Do NOT use history to infer that the user also wants the same output types they asked for before.
5. "clarified_query" must contain the FULL topic — merge the prior topic with the current
   request when needed to resolve pronouns (e.g. "this", "that", "it").
   But NEVER add output format types (blog/linkedin/image) to clarified_query unless the
   current message explicitly requests them.
6. Set "is_refinement": true when the user is extending or updating prior output
   (keywords: also, add, include, update, change, make it, expand, revise, etc.).
   Set "is_refinement": false for fresh requests or topic changes.

Examples:
- "Write me a blog about LangGraph" → {"intent": "blog", "clarified_query": "LangGraph deep dive", "is_refinement": false}
- "research what software is available for this" (after LinkedIn post about video editing) → {"intent": "research", "clarified_query": "video editing software programs", "is_refinement": false}
- "research AI trends and also write a blog about it" → {"intent": "multi", "clarified_query": "AI trends 2024", "is_refinement": false}
- "also include healthcare AI trends" (after prior research on AI) → {"intent": "research", "clarified_query": "AI trends 2024, including healthcare AI", "is_refinement": true}
- "Generate an image of a robot doctor" → {"intent": "image", "clarified_query": "robot doctor, futuristic medical setting", "is_refinement": false}
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

        history_block = self._format_history(state.get("messages", []))
        history_section = f"\n\nConversation history (last {_HISTORY_WINDOW} messages):\n{history_block}" if history_block else ""
        prompt = f"{_SYSTEM_PROMPT}{history_section}\n\nUser request: {user_query}"

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

            is_refinement: bool = bool(parsed.get("is_refinement", False))

            # Fast-path heuristic: override to True if message contains refinement signals
            # (guards against LLM missing the flag on obvious follow-ups)
            if not is_refinement and history_block:
                lower = user_query.lower()
                if any(signal in lower for signal in _REFINEMENT_SIGNALS):
                    is_refinement = True

            logger.info("QueryHandlerAgent: intent=%s | is_refinement=%s", intent, is_refinement)
            return {
                "intent": intent,
                "user_query": user_query,          # preserve original
                "clarified_user_query": clarified, # enriched cumulative query
                "is_refinement": is_refinement,
                "error": None,
            }

        except Exception as exc:
            logger.exception("QueryHandlerAgent failed: %s", exc)
            return {"error": str(exc), "intent": None}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_history(self, messages: list) -> str:
        """
        Format the last _HISTORY_WINDOW messages into a readable string.
        Excludes the final message (current query, already in 'User request').
        """
        # Exclude the last message — it's the current query already in the prompt
        prior = messages[:-1] if messages else []
        window = prior[-_HISTORY_WINDOW:]
        if not window:
            return ""
        lines = []
        for msg in window:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = str(getattr(msg, "content", msg))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

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
