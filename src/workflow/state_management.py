from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from src.core.models import BlogPost, ImageResult, LinkedInPost, ResearchResult


class GraphState(TypedDict):
    """
    LangGraph state shared across all agent nodes.

    LangGraph requires a TypedDict (not Pydantic) for its state schema.
    The Pydantic models in src/core/models.py are used as the typed values
    within the state, giving us both LangGraph compatibility and rich
    data contracts.

    Field notes:
    - messages: uses LangGraph's add_messages reducer — appends rather than
                replaces, preserving the full conversation history.
    - All other fields use the default 'replace' reducer (last writer wins).
    - error: set by any node that catches an exception; workflow checks this
             in conditional edges to route to a fallback/end node.
    """

    # ── Core inputs ────────────────────────────────────────────────────────────
    # Original user input — preserved for logging and reference.
    user_query: str
    # Enriched query produced by QueryHandlerAgent — merges current input with
    # relevant history context. Downstream agents use this instead of user_query.
    clarified_user_query: Optional[str]

    # ── Routing ────────────────────────────────────────────────────────────────
    # Classified intent from QueryHandlerAgent.
    # Values: "research" | "blog" | "linkedin" | "image" | "strategy" | "multi"
    intent: Optional[str]
    # True when QueryHandlerAgent detects the user is refining/extending prior output.
    # Downstream agents use this to patch existing output rather than regenerate.
    is_refinement: bool

    # ── Agent outputs ──────────────────────────────────────────────────────────
    research: Optional[ResearchResult]
    blog_post: Optional[BlogPost]
    linkedin_post: Optional[LinkedInPost]
    image_result: Optional[ImageResult]

    # ── Conversation memory ────────────────────────────────────────────────────
    # add_messages reducer appends new messages instead of replacing the list,
    # preserving full multi-turn history across graph invocations.
    messages: Annotated[list, add_messages]

    # ── Content strategy output ───────────────────────────────────────────────
    content_strategy: Optional[str]

    # ── Agent configuration hints ───────────────────────────────────────────────
    # Passed in at workflow entry and preserved throughout the graph so every
    # node can read them without being re-injected at each step.
    linkedin_post_type: str
    image_style: str
    image_size: str

    # ── Fallback output ────────────────────────────────────────────────────────
    # Set by FallbackAgent when the user's request is out of scope.
    fallback_message: Optional[str]

    # ── Error handling ─────────────────────────────────────────────────────────
    error: Optional[str]
