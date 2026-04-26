"""
tests/unit/test_linkedin_writer.py

Unit tests for LinkedInWriterAgent.

Strategy: mock the LLM client (BaseLLMClient abstraction) so tests are fast
and deterministic — no API calls, no network, no LangChain setup.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.linkedin_writer import LinkedInWriterAgent
from src.core.models import LinkedInPost, ResearchResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_llm(response: str) -> MagicMock:
    """Return a mock BaseLLMClient whose generate() returns *response*."""
    llm = MagicMock()
    llm.generate.return_value = response
    return llm


_VALID_JSON_RESPONSE = json.dumps({
    "content": "AI is transforming how we work. Here's what you need to know.",
    "hashtags": ["#AI", "#FutureOfWork"],
    "post_type": "thought-leadership",
})


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_run_returns_linkedin_post_on_valid_llm_response():
    """Agent returns a LinkedInPost when the LLM produces valid JSON."""
    agent = LinkedInWriterAgent(llm=_make_llm(_VALID_JSON_RESPONSE))
    state = {"user_query": "AI in the workplace", "clarified_user_query": "AI in the workplace"}

    result = agent.run(state)

    assert result.get("error") is None
    post = result.get("linkedin_post")
    assert isinstance(post, LinkedInPost)
    assert "AI" in post.content
    assert "#AI" in post.hashtags


def test_run_returns_error_when_no_topic_provided():
    """Agent returns an error dict when the state carries no query."""
    agent = LinkedInWriterAgent(llm=_make_llm(_VALID_JSON_RESPONSE))

    result = agent.run({"user_query": "", "clarified_user_query": ""})

    assert result.get("error") is not None
    assert result.get("linkedin_post") is None


def test_run_uses_refinement_prompt_when_is_refinement_true():
    """Refinement flag causes the agent to pass an existing post into the prompt."""
    captured_prompts: list[str] = []

    def capture_and_respond(prompt: str, **_) -> str:
        captured_prompts.append(prompt)
        return _VALID_JSON_RESPONSE

    llm = MagicMock()
    llm.generate.side_effect = capture_and_respond

    existing_post = LinkedInPost(content="Old post content", hashtags=["#Old"])
    agent = LinkedInWriterAgent(llm=llm)
    state = {
        "user_query": "make it shorter",
        "clarified_user_query": "make it shorter",
        "is_refinement": True,
        "linkedin_post": existing_post,
    }

    agent.run(state)

    assert len(captured_prompts) == 1
    # Refinement prompt contains the existing content
    assert "Old post content" in captured_prompts[0]
    assert "previously wrote" in captured_prompts[0].lower() or "existing post" in captured_prompts[0].lower()
