"""
src/core/workflow.py

High-level public API for running ContentBlitz workflows.

This is the only module the UI (Streamlit) and tests need to import.
It hides all LangGraph internals behind a clean, typed interface.

SOLID:
- SRP: Responsible only for orchestrating a single user request end-to-end.
- OCP: New workflow modes can be added without changing existing method logic.
- DIP: Depends on build_app() (router abstraction), not on concrete agents.
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.core.config import Config
from src.core.models import AgentState
from src.core.router import build_app
from src.utils.logger import get_logger
from src.workflow.state_management import GraphState

logger = get_logger(__name__)

# Module-level cached app — built once, reused across all requests (singleton)
_app: Any = None


def _get_app(config: Config | None = None) -> Any:
    global _app
    if _app is None:
        _app = build_app(config)
    return _app


def run(
    user_query: str,
    *,
    image_style: str = "photorealistic",
    image_size: str = "1024x1024",
    linkedin_post_type: str = "general",
    conversation_history: list[dict] | None = None,
    config: Config | None = None,
) -> AgentState:
    """
    Run a full ContentBlitz workflow for a user query.

    This is the primary entrypoint used by the Streamlit UI and tests.

    Args:
        user_query:           The natural-language request from the user.
        image_style:          Style hint for image generation (e.g. "photorealistic").
        image_size:           DALL-E image size (e.g. "1024x1024").
        linkedin_post_type:   LinkedIn post style hint.
        conversation_history: Prior turns for multi-turn context.
        config:               Optional Config override (useful in tests).

    Returns:
        AgentState populated with all outputs produced by the workflow.
    """
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    logger.info("Workflow.run | query=%r", user_query[:80])

    app = _get_app(config)

    # Reconstruct prior history as properly typed messages, then append current query
    history_messages: list[BaseMessage] = []
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "human")
            content = str(turn.get("content", ""))
            history_messages.append(
                HumanMessage(content=content) if role == "human" else AIMessage(content=content)
            )

    # Build the initial LangGraph state
    initial_state: GraphState = {
        "user_query": user_query,
        "clarified_user_query": None,
        "intent": None,
        "is_refinement": False,
        "research": None,
        "blog_post": None,
        "linkedin_post": None,
        "image_result": None,
        "content_strategy": None,
        "fallback_message": None,
        "messages": [*history_messages, HumanMessage(content=user_query)],
        "error": None,
        # Pass configuration hints as declared GraphState fields
        "image_style": image_style,
        "image_size": image_size,
        "linkedin_post_type": linkedin_post_type,
    }

    try:
        final_state: GraphState = app.invoke(
            initial_state,
            config={"run_name": f"contentblitz/{user_query[:40]}"},
        )
    except Exception as exc:
        logger.exception("Workflow.run failed: %s", exc)
        final_state = {**initial_state, "error": str(exc)}  # type: ignore[assignment]

    result = _graph_state_to_agent_state(final_state)
    _log_result_summary(result)
    return result


def _graph_state_to_agent_state(state: GraphState) -> AgentState:
    """Convert LangGraph's TypedDict state into the typed AgentState Pydantic model."""
    return AgentState(
        user_query=state.get("user_query", ""),
        clarified_user_query=state.get("clarified_user_query"),
        intent=state.get("intent"),
        is_refinement=state.get("is_refinement", False),
        linkedin_post_type=state.get("linkedin_post_type", "general"),
        image_style=state.get("image_style", "photorealistic"),
        image_size=state.get("image_size", "1024x1024"),
        research=state.get("research"),
        blog_post=state.get("blog_post"),
        linkedin_post=state.get("linkedin_post"),
        image_result=state.get("image_result"),
        content_strategy=state.get("content_strategy"),
        fallback_message=state.get("fallback_message"),
        conversation_history=[
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            }
            for m in state.get("messages", [])
            if hasattr(m, "content")
        ],
        error=state.get("error"),
    )


def _log_result_summary(result: AgentState) -> None:
    outputs = [
        k for k in ("research", "blog_post", "linkedin_post", "image_result", "content_strategy")
        if getattr(result, k) is not None
    ]
    if result.error:
        logger.warning("Workflow completed with error: %s", result.error)
    else:
        logger.info("Workflow completed | intent=%s | outputs=%s", result.intent, outputs)


def reset_app() -> None:
    """Force rebuilding the graph on next call (useful in tests)."""
    global _app
    _app = None
