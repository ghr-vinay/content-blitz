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

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from src.core.config import Config
from src.core.models import AgentState, EvalScore
from src.core.router import build_app
from src.eval.llm_eval import evaluate_agent_state
from src.utils.logger import get_logger
from src.utils.progress import progress_context
from src.workflow.state_management import GraphState

logger = get_logger(__name__)

# Module-level cached app — built once, reused across all requests (singleton)
_app: Any = None

# Shared thread pool for background eval runs (daemon threads — exit with process)
_eval_executor_thread = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cb-eval")


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
    on_progress=None,
) -> AgentState:
    """
    Run a full ContentBlitz workflow for a user query (blocking eval).

    Primary entrypoint for the CLI and tests. Eval runs synchronously so the
    returned AgentState always contains populated eval_scores.
    """
    result, run_id = _invoke_workflow(
        user_query=user_query,
        image_style=image_style,
        image_size=image_size,
        linkedin_post_type=linkedin_post_type,
        conversation_history=conversation_history,
        config=config,
        on_progress=on_progress,
    )

    cfg = config or Config.get_instance()
    if cfg.eval_enabled and not result.error and not result.fallback_message:
        try:
            scores = evaluate_agent_state(
                result,
                run_id=run_id,
                model=cfg.eval_model,
                threshold=cfg.eval_threshold,
            )
            result = result.model_copy(update={"eval_scores": scores})
        except Exception as exc:
            logger.warning("Eval run failed (non-fatal): %s", exc)

    _log_result_summary(result)
    return result


def run_non_blocking_eval(
    user_query: str,
    *,
    image_style: str = "photorealistic",
    image_size: str = "1024x1024",
    linkedin_post_type: str = "general",
    conversation_history: list[dict] | None = None,
    config: Config | None = None,
    on_progress=None,
) -> tuple[AgentState, "Future[list[EvalScore]] | None"]:
    """
    Run workflow and return content immediately; eval runs in a background thread.

    Returns:
        (result, eval_future) — result has empty eval_scores; caller should
        resolve eval_future later and merge scores via result.model_copy().
    """
    result, run_id = _invoke_workflow(
        user_query=user_query,
        image_style=image_style,
        image_size=image_size,
        linkedin_post_type=linkedin_post_type,
        conversation_history=conversation_history,
        config=config,
        on_progress=on_progress,
    )
    _log_result_summary(result)

    eval_future: "Future[list[EvalScore]] | None" = None
    cfg = config or Config.get_instance()
    if cfg.eval_enabled and not result.error and not result.fallback_message:
        eval_future = _eval_executor_thread.submit(
            evaluate_agent_state,
            result,
            run_id,
            cfg.eval_model,
            cfg.eval_threshold,
        )

    return result, eval_future


def _invoke_workflow(
    user_query: str,
    *,
    image_style: str,
    image_size: str,
    linkedin_post_type: str,
    conversation_history: list[dict] | None,
    config: Config | None,
    on_progress=None,
) -> tuple[AgentState, str | None]:
    """Invoke the LangGraph app and return (AgentState, run_id). No eval."""
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    logger.info("Workflow._invoke | query=%r", user_query[:80])

    app = _get_app(config)

    history_messages: list[BaseMessage] = []
    if conversation_history:
        for turn in conversation_history:
            role = turn.get("role", "human")
            content = str(turn.get("content", ""))
            history_messages.append(
                HumanMessage(content=content) if role == "human" else AIMessage(content=content)
            )

    initial_state: GraphState = {
        "user_query": user_query,
        "clarified_user_query": None,
        "intent": None,
        "is_refinement": False,
        "remaining_nodes": [],  # populated by QueryHandlerAgent from _INTENT_TO_NODE
        "research": None,
        "blog_post": None,
        "linkedin_post": None,
        "image_result": None,
        "content_strategy": None,
        "fallback_message": None,
        "messages": [*history_messages, HumanMessage(content=user_query)],
        "error": None,
        "image_style": image_style,
        "image_size": image_size,
        "linkedin_post_type": linkedin_post_type,
    }

    try:
        run_collector = RunCollectorCallbackHandler()
        with progress_context(on_progress):
            final_state: GraphState = app.invoke(
                initial_state,
                config={
                    "run_name": f"contentblitz/{user_query[:40]}",
                    "callbacks": [run_collector],
                },
            )
        run_id = str(run_collector.traced_runs[0].id) if run_collector.traced_runs else None
    except Exception as exc:
        logger.exception("Workflow._invoke failed: %s", exc)
        final_state = {**initial_state, "error": str(exc)}  # type: ignore[assignment]
        run_id = None

    return _graph_state_to_agent_state(final_state), run_id


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
        eval_scores=[],
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
