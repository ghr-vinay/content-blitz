"""
src/core/router.py

Dependency-injection container that constructs every agent with its concrete
dependencies and registers them as node callables in the LangGraph graph.

SOLID:
- DIP: The graph (langgraph_workflow.build_graph) depends only on callables;
       this is the single place where concrete classes are wired together.
- OCP: Adding a new agent = add one block here + one node in build_graph.
       Nothing else changes.
- SRP: Router's only job is agent construction and graph assembly.
"""

from typing import Any

from src.agents.blog_writer import BlogWriterAgent
from src.agents.content_strategist import ContentStrategistAgent
from src.agents.image_generator import ImageGeneratorAgent
from src.agents.linkedin_writer import LinkedInWriterAgent
from src.agents.query_handler import QueryHandlerAgent
from src.agents.research_agent import ResearchAgent
from src.core.config import Config
from src.integrations.image_clients import DallE3Client
from src.integrations.openai_client import OpenAIClient
from src.integrations.serp_client import SerpClient
from src.workflow.langgraph_workflow import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_app(config: Config | None = None) -> Any:
    """
    Wire all agents with their dependencies and return a compiled LangGraph app.

    Args:
        config: Optional Config instance. Uses the singleton if not provided.

    Returns:
        Compiled LangGraph runnable (invoke-able with a GraphState dict).
    """
    cfg = config or Config.get_instance()

    # ── Shared infrastructure (injected into agents) ───────────────────────────
    llm = OpenAIClient(
        api_key=cfg.openai_api_key,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )

    search_tool = SerpClient(
        api_key=cfg.serp_api_key,
        num_results=cfg.search_num_results,
    )

    image_tool = DallE3Client(
        api_key=cfg.openai_api_key,
        size=cfg.image_size,
        quality=cfg.image_quality,
    )

    # ── Agent construction (DIP: agents receive abstractions, not concretes) ───
    query_handler     = QueryHandlerAgent(llm=llm)
    research_agent    = ResearchAgent(search_tool=search_tool, llm=llm)
    blog_writer       = BlogWriterAgent(llm=llm)
    linkedin_writer   = LinkedInWriterAgent(llm=llm)
    image_generator   = ImageGeneratorAgent(image_tool=image_tool, llm=llm)
    content_strategist = ContentStrategistAgent(llm=llm)

    logger.info("Router: all agents constructed — wiring graph")

    # ── Assemble graph (agents registered by their .run method) ───────────────
    app = build_graph(
        query_handler_fn      = query_handler.run,
        research_agent_fn     = research_agent.run,
        blog_writer_fn        = blog_writer.run,
        linkedin_writer_fn    = linkedin_writer.run,
        image_generator_fn    = image_generator.run,
        content_strategist_fn = content_strategist.run,
    )

    logger.info("Router: graph compiled and ready")
    return app
