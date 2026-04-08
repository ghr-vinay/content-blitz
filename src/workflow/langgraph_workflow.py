from typing import Any

from langgraph.graph import StateGraph, END

from src.workflow.state_management import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Intent → node name mapping ─────────────────────────────────────────────────
# Used by the router edge to decide which agent runs after query_handler.
_INTENT_TO_NODE: dict[str, list[str]] = {
    "research":  ["research_agent"],
    "blog":      ["blog_writer"],
    "linkedin":  ["linkedin_writer"],
    "image":     ["image_generator"],
    "strategy":  ["content_strategist"],
    # "multi" triggers research first, then all content agents
    "multi":     ["research_agent"],
}


# ── Conditional edge functions ─────────────────────────────────────────────────

def _route_after_query_handler(state: GraphState) -> str:
    """
    Route to the correct agent node based on classified intent.
    Falls back to END if intent is unrecognised or an error occurred.
    """
    if state.get("error"):
        logger.warning("Routing to END due to error: %s", state["error"])
        return END

    intent: str = state.get("intent", "")
    nodes = _INTENT_TO_NODE.get(intent, [])

    if not nodes:
        logger.warning("Unknown intent '%s' — routing to END", intent)
        return END

    next_node = nodes[0]
    logger.info("Routing intent='%s' → node='%s'", intent, next_node)
    return next_node


def _route_after_research(state: GraphState) -> str:
    """
    After research, decide whether to continue to content agents (multi intent)
    or finish.
    """
    if state.get("error"):
        return END

    intent = state.get("intent", "")
    if intent == "multi":
        # After research, run all content agents in sequence.
        # For simplicity, we route to blog_writer first; it will chain on.
        logger.info("Multi-intent: routing research → blog_writer")
        return "blog_writer"

    return END


def _route_after_blog(state: GraphState) -> str:
    """After blog, continue to linkedin_writer if multi-intent."""
    if state.get("error"):
        return END
    if state.get("intent") == "multi":
        logger.info("Multi-intent: routing blog_writer → linkedin_writer")
        return "linkedin_writer"
    return END


def _route_after_linkedin(state: GraphState) -> str:
    """After linkedin, continue to image_generator if multi-intent."""
    if state.get("error"):
        return END
    if state.get("intent") == "multi":
        logger.info("Multi-intent: routing linkedin_writer → image_generator")
        return "image_generator"
    return END


# ── Placeholder node — replaced by real agent in Phase 5 ──────────────────────

def _placeholder_node(name: str):
    """
    Returns a no-op node function used until the real agent is wired in Phase 5.
    Logs clearly so traces in LangSmith show the node was reached.
    """
    def _node(state: GraphState) -> dict[str, Any]:
        logger.info("Placeholder node reached: %s", name)
        return {}
    _node.__name__ = name
    return _node


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph(
    query_handler_fn=None,
    research_agent_fn=None,
    blog_writer_fn=None,
    linkedin_writer_fn=None,
    image_generator_fn=None,
    content_strategist_fn=None,
) -> Any:
    """
    Build and compile the ContentBlitz LangGraph StateGraph.

    Each agent function is injected (DIP) — the graph depends on callables,
    not concrete agent classes. Placeholder no-ops are used for any agent
    not yet wired in so the graph can be compiled and tested incrementally.

    Args:
        *_fn: Callable(state: GraphState) -> dict  for each agent node.
              Pass None to use a placeholder until the agent is implemented.

    Returns:
        A compiled LangGraph runnable ready to invoke.
    """
    graph = StateGraph(GraphState)

    # Register nodes — fall back to placeholders for anything not yet wired
    graph.add_node("query_handler",     query_handler_fn     or _placeholder_node("query_handler"))
    graph.add_node("research_agent",    research_agent_fn    or _placeholder_node("research_agent"))
    graph.add_node("blog_writer",       blog_writer_fn       or _placeholder_node("blog_writer"))
    graph.add_node("linkedin_writer",   linkedin_writer_fn   or _placeholder_node("linkedin_writer"))
    graph.add_node("image_generator",   image_generator_fn   or _placeholder_node("image_generator"))
    graph.add_node("content_strategist",content_strategist_fn or _placeholder_node("content_strategist"))

    # Entry point
    graph.set_entry_point("query_handler")

    # Conditional edges
    graph.add_conditional_edges(
        "query_handler",
        _route_after_query_handler,
        {
            "research_agent":     "research_agent",
            "blog_writer":        "blog_writer",
            "linkedin_writer":    "linkedin_writer",
            "image_generator":    "image_generator",
            "content_strategist": "content_strategist",
            END:                   END,
        },
    )

    graph.add_conditional_edges(
        "research_agent",
        _route_after_research,
        {
            "blog_writer": "blog_writer", 
            END: END
         },
    )

    graph.add_conditional_edges(
        "blog_writer",
        _route_after_blog,
        {
            "linkedin_writer": "linkedin_writer", 
            END: END
        },
    )

    graph.add_conditional_edges(
        "linkedin_writer",
        _route_after_linkedin,
        {
            "image_generator": "image_generator", 
            END: END
        },
    )

    # Terminal nodes always go to END
    graph.add_edge("image_generator",    END)
    graph.add_edge("content_strategist", END)

    compiled = graph.compile()
    logger.info("ContentBlitz graph compiled successfully")
    return compiled
