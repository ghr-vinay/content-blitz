"""
tests/e2e/test_workflow_e2e.py

End-to-end smoke tests for the ContentBlitz workflow.

Scope: the full LangGraph graph runs (real graph compilation, real routing,
real agent code), but all external I/O is stubbed:
  - OpenAIClient.generate → canned JSON responses keyed by prompt keyword
  - evaluate_agent_state → patched out entirely (not what we're testing here)

This catches graph-wiring bugs (missing edges, wrong state keys, silent drops)
that unit/integration tests would miss.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import AgentState


# ── Canned LLM responses ───────────────────────────────────────────────────────
# The workflow calls OpenAIClient.generate() multiple times (once per agent).
# We inspect the prompt text to decide which canned payload to return.

_QUERY_HANDLER_RESPONSE = json.dumps({
    "clarified_query": "Write a LinkedIn post about AI agents",
    "intent": "linkedin",
    "is_refinement": False,
})

_LINKEDIN_RESPONSE = json.dumps({
    "content": "AI agents are changing how we build software. Here is why that matters.",
    "hashtags": ["#AI", "#Agents", "#SoftwareDev"],
    "post_type": "thought-leadership",
})


def _stub_generate(prompt: str, **_kwargs: Any) -> str:
    """Route canned responses based on keywords in the prompt."""
    if "intent" in prompt.lower() or "clarif" in prompt.lower():
        return _QUERY_HANDLER_RESPONSE
    if "linkedin" in prompt.lower():
        return _LINKEDIN_RESPONSE
    # Fallback for any other agent (shouldn't be reached in this test)
    return json.dumps({"content": "fallback", "hashtags": [], "post_type": "general"})


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("src.eval.llm_eval.evaluate_agent_state", return_value=[])
@patch("src.integrations.openai_client.OpenAIClient.generate", side_effect=_stub_generate)
def test_linkedin_workflow_produces_post(mock_generate, mock_eval):
    """
    Full graph run for a LinkedIn intent should populate linkedin_post and set
    no error.  eval is skipped (stubbed) so the test stays free and fast.
    """
    # Import here so the module-level _app singleton is not shared across tests
    import importlib
    import src.core.workflow as wf_module
    wf_module._app = None  # reset cached app so this test builds a fresh graph

    from src.core.workflow import run
    from src.core.config import Config

    # Disable eval so we don't need real DeepEval credentials
    cfg = MagicMock(spec=Config)
    cfg.openai_api_key = "test-key"
    cfg.llm_model = "gpt-4o"
    cfg.llm_temperature = 0.7
    cfg.serp_api_key = "test-serp-key"
    cfg.image_api_key = "test-image-key"
    cfg.eval_enabled = False
    cfg.eval_model = "gpt-4o"
    cfg.eval_threshold = 0.5

    result: AgentState = run(
        "Write a LinkedIn post about AI agents",
        config=cfg,
    )

    assert result.error is None, f"Workflow returned an error: {result.error}"
    assert result.linkedin_post is not None, "Expected linkedin_post to be populated"
    assert result.linkedin_post.content != ""


@patch("src.eval.llm_eval.evaluate_agent_state", return_value=[])
@patch("src.integrations.openai_client.OpenAIClient.generate")
def test_off_topic_query_produces_fallback_message(mock_generate, mock_eval):
    """
    An off-topic query should trigger the FallbackAgent and set fallback_message,
    not raise an exception or set a hard error.
    """
    import src.core.workflow as wf_module
    wf_module._app = None

    mock_generate.return_value = json.dumps({
        "clarified_query": "How do I bake a cake?",
        "intent": "off_topic",
        "is_refinement": False,
    })

    from src.core.workflow import run
    from src.core.config import Config

    cfg = MagicMock(spec=Config)
    cfg.openai_api_key = "test-key"
    cfg.llm_model = "gpt-4o"
    cfg.llm_temperature = 0.7
    cfg.serp_api_key = "test-serp-key"
    cfg.image_api_key = "test-image-key"
    cfg.eval_enabled = False
    cfg.eval_model = "gpt-4o"
    cfg.eval_threshold = 0.5

    result: AgentState = run("How do I bake a cake?", config=cfg)

    # Fallback agent sets fallback_message; no blog/linkedin/research output
    assert result.fallback_message is not None
    assert result.blog_post is None
    assert result.linkedin_post is None
