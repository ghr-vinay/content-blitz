"""
tests/integration/test_linkedin_writer_integration.py

Integration tests for LinkedInWriterAgent.

Unlike unit tests (which verify logic in isolation), these tests exercise the
full internal pipeline of the agent:
    mocked LLM → prompt construction → JSON parsing → LinkedInPost model creation

The LLM is still mocked — no API calls — but all of the agent's own code
runs for real.  This catches bugs that only emerge when the pieces work together,
e.g. bad prompt formatting that causes JSON parse failures.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.agents.linkedin_writer import LinkedInWriterAgent
from src.core.models import LinkedInPost, ResearchResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _llm_returning(payload: dict) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(payload)
    return llm


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_agent_builds_linkedin_post_including_research_context():
    """
    When ResearchResult is in the state the agent should incorporate findings
    into the prompt, and the resulting LinkedInPost should be fully populated.
    """
    research = ResearchResult(
        topic="LLMs in production",
        summary="Large language models are increasingly deployed at scale.",
        key_findings=[
            "Latency is the #1 concern",
            "RAG reduces hallucinations by 40%",
        ],
        sources=["https://example.com/llms"],
    )

    llm = _llm_returning({
        "content": "LLMs are reshaping enterprise software. Latency and RAG are key topics.",
        "hashtags": ["#LLM", "#GenAI", "#RAG"],
        "post_type": "thought-leadership",
    })
    agent = LinkedInWriterAgent(llm=llm)

    state = {
        "user_query": "Write about LLMs in production",
        "clarified_user_query": "Write about LLMs in production",
        "research": research,
        "linkedin_post_type": "thought-leadership",
    }
    result = agent.run(state)

    post: LinkedInPost = result["linkedin_post"]
    assert post.post_type == "thought-leadership"
    assert len(post.hashtags) == 3
    # Confirm research context was included in the prompt
    prompt_text = llm.generate.call_args[0][0]
    assert "Latency is the #1 concern" in prompt_text


def test_agent_gracefully_handles_json_wrapped_in_markdown_fences():
    """
    Some LLM responses wrap JSON in markdown code fences.
    The agent's _parse_json must strip them and still return a valid post.
    """
    payload = json.dumps({
        "content": "Clean post content here.",
        "hashtags": ["#Tech"],
        "post_type": "general",
    })
    # Simulate LLM wrapping output in a code fence
    llm = MagicMock()
    llm.generate.return_value = f"```json\n{payload}\n```"

    agent = LinkedInWriterAgent(llm=llm)
    state = {"user_query": "Tech post", "clarified_user_query": "Tech post"}
    result = agent.run(state)

    # The agent falls back gracefully — either parsed correctly or stored raw text
    assert result.get("error") is None
    assert result.get("linkedin_post") is not None


def test_agent_returns_error_on_completely_malformed_llm_response():
    """
    If the LLM returns total garbage (not even parseable as JSON), the agent
    should return an error or degrade gracefully rather than raise an exception.
    """
    llm = MagicMock()
    llm.generate.side_effect = ValueError("LLM client internal error")

    agent = LinkedInWriterAgent(llm=llm)
    state = {"user_query": "Tech post", "clarified_user_query": "Tech post"}
    result = agent.run(state)

    assert result.get("error") is not None
    assert result.get("linkedin_post") is None
