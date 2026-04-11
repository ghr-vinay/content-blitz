from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """Output contract for the Deep Research Agent."""

    topic: str
    summary: str
    key_findings: list[str]
    sources: list[str] = Field(default_factory=list)
    researched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class BlogPost(BaseModel):
    """Output contract for the SEO Blog Writer Agent."""

    title: str
    content: str
    meta_description: str
    keywords: list[str] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    word_count: int = 0

    def model_post_init(self, __context: object) -> None:
        if not self.word_count:
            self.word_count = len(self.content.split())


class LinkedInPost(BaseModel):
    """Output contract for the LinkedIn Post Writer Agent."""

    content: str
    hashtags: list[str] = Field(default_factory=list)
    character_count: int = 0
    post_type: str = "general"  # thought-leadership | announcement | story | listicle

    def model_post_init(self, __context: object) -> None:
        if not self.character_count:
            self.character_count = len(self.content)


class ImageResult(BaseModel):
    """Output contract for the Image Generation Agent."""

    url: str
    prompt_used: str
    style: str = "photorealistic"
    size: str = "1024x1024"


class AgentState(BaseModel):
    """
    Shared LangGraph state passed between all agent nodes.

    Each agent reads from and writes back to this state.
    Using a typed model enforces the shared data contract across the graph.
    """

    user_query: str = ""
    intent: Optional[str] = None          # classified intent from QueryHandlerAgent
    research: Optional[ResearchResult] = None
    blog_post: Optional[BlogPost] = None
    linkedin_post: Optional[LinkedInPost] = None
    image_result: Optional[ImageResult] = None
    conversation_history: list[dict] = Field(default_factory=list)
    content_strategy: Optional[str] = None
    error: Optional[str] = None
