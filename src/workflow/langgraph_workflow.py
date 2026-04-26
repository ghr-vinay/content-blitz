from typing import Any

from langgraph.graph import StateGraph, END

from src.workflow.state_management import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Intent → agent chain mapping ───────────────────────────────────────────────
# QueryHandlerAgent writes _INTENT_TO_NODE[intent] into state["remaining_nodes"].
# Each agent pops the first entry; _route_next reads the new head.
# To add a new intent chain, add a line here — zero routing code changes needed.
_INTENT_TO_NODE: dict[str, list[str]] = {
    "research":               ["research_agent"],
    "blog":                   ["research_agent", "blog_writer"],
    "linkedin":               ["research_agent", "linkedin_writer"],
    "image":                  ["image_generator"],
    "strategy":               ["content_strategist"],
    "blog_with_image":        ["research_agent", "blog_writer", "image_generator"],
    "linkedin_with_image":    ["research_agent", "linkedin_writer", "image_generator"],
    "off_topic":              ["fallback_agent"],
}


# ── Generalised conditional edge routing ────────────────────────────────────────────────────────

def _route_next(state: GraphState) -> str:
    """
    Single routing function used after every agent node.

    Reads state["remaining_nodes"]:
    - If the list is non-empty, routes to the first entry.
    - If empty (or error), routes to END.
    """
    if state.get("error"):
        logger.warning("Routing to END due to error: %s", state["error"])
        return END

    remaining: list[str] = state.get("remaining_nodes", [])
    if remaining:
        next_node = remaining[0]
        logger.info("_route_next → '%s' (%d remaining)", next_node, len(remaining))
        return next_node

    logger.info("_route_next → END (chain complete)")
    return END


# ── Routing queue helper ───────────────────────────────────────────────────────

def _advance_queue(agent_fn):
    """
    Wrap an agent callable so it pops the first entry from remaining_nodes
    after the agent runs. Keeps agents fully unaware of routing concerns.
    """
    def _wrapped(state: GraphState) -> dict[str, Any]:
        result = agent_fn(state)
        # Advance the queue: drop the node that just ran
        remaining = state.get("remaining_nodes", [])
        result["remaining_nodes"] = remaining[1:]
        return result
    _wrapped.__name__ = getattr(agent_fn, "__name__", "agent")
    return _wrapped


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
    fallback_agent_fn=None,
) -> Any:
    """
    Build and compile the ContentBlitz LangGraph StateGraph.

    Each agent function is injected (DIP) — the graph depends on callables,
    not concrete agent classes. Placeholder no-ops are used for any agent
    not yet wired in so the graph can be compiled and tested incrementally.

    Routing is fully generalised: every node uses _route_next which follows
    the remaining_nodes queue written by QueryHandlerAgent. Adding a new intent
    chain requires only a new _INTENT_TO_NODE entry — no routing changes.

    Args:
        *_fn: Callable(state: GraphState) -> dict  for each agent node.
              Pass None to use a placeholder until the agent is implemented.

    Returns:
        A compiled LangGraph runnable ready to invoke.
    """
    graph = StateGraph(GraphState)

    # All possible destination nodes — used in every conditional edge map
    _all_destinations = {
        "research_agent":     "research_agent",
        "blog_writer":        "blog_writer",
        "linkedin_writer":    "linkedin_writer",
        "image_generator":    "image_generator",
        "content_strategist": "content_strategist",
        "fallback_agent":     "fallback_agent",
        END:                   END,
    }

    # Register nodes — fall back to placeholders for anything not yet wired.
    # Every real agent is wrapped with _advance_queue to pop itself off remaining_nodes.
    graph.add_node("query_handler",      query_handler_fn      or _placeholder_node("query_handler"))
    graph.add_node("research_agent",     _advance_queue(research_agent_fn     or _placeholder_node("research_agent")))
    graph.add_node("blog_writer",        _advance_queue(blog_writer_fn        or _placeholder_node("blog_writer")))
    graph.add_node("linkedin_writer",    _advance_queue(linkedin_writer_fn    or _placeholder_node("linkedin_writer")))
    graph.add_node("image_generator",    _advance_queue(image_generator_fn    or _placeholder_node("image_generator")))
    graph.add_node("content_strategist", _advance_queue(content_strategist_fn or _placeholder_node("content_strategist")))
    graph.add_node("fallback_agent",     _advance_queue(fallback_agent_fn     or _placeholder_node("fallback_agent")))

    # Entry point
    graph.set_entry_point("query_handler")

    # Every node after query_handler uses the same generalised router
    for node in ("query_handler", "research_agent", "blog_writer", "linkedin_writer",
                 "image_generator", "content_strategist", "fallback_agent"):
        graph.add_conditional_edges(node, _route_next, _all_destinations)

    compiled = graph.compile()
    logger.info("ContentBlitz graph compiled successfully")
    return compiled


def visualise_graph(output_path: str = "graph.png") -> str:
    """
    Render the ContentBlitz graph as a PNG and return the file path.

    Uses LangGraph's built-in Mermaid→PNG renderer (calls Mermaid.js public API).
    Falls back to ASCII in the terminal if the PNG render fails (e.g. no internet).

    Args:
        output_path: Where to save the PNG file. Defaults to 'graph.png'.

    Returns:
        Absolute path to the saved PNG, or an empty string on failure.
    """
    compiled = build_graph()
    try:
        png_bytes: bytes = compiled.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.info("Graph PNG saved to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("PNG render failed (%s) — falling back to ASCII", exc)
        compiled.get_graph().print_ascii()
        return ""
