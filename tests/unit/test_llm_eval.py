"""
tests/unit/test_llm_eval.py

Unit tests for src/eval/llm_eval.py

Strategy: mock DeepEval's GEval.measure() so tests are fast and free —
no LLM calls, no API keys needed.  We verify that _run_metrics correctly
maps metric results to EvalScore objects, and that evaluate_agent_state
skips evaluation when the AgentState carries no content.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.models import AgentState, EvalScore, LinkedInPost
from src.eval.llm_eval import _run_metrics, evaluate_agent_state
from deepeval.test_case import LLMTestCase


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_metric(name: str, score: float, successful: bool) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.score = score
    m.reason = "mocked reason"
    m.is_successful.return_value = successful
    return m


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_run_metrics_returns_eval_scores():
    """_run_metrics should return one EvalScore per metric with correct fields."""
    metric = _make_mock_metric("Relevance", score=0.8, successful=True)
    test_case = LLMTestCase(input="topic", actual_output="some LinkedIn post")

    scores = _run_metrics([metric], test_case, content_type="linkedin")

    assert len(scores) == 1
    s = scores[0]
    assert isinstance(s, EvalScore)
    assert s.metric == "Relevance"
    assert s.content_type == "linkedin"
    assert s.score == 0.8
    assert s.passed is True
    metric.measure.assert_called_once_with(test_case)


def test_run_metrics_marks_failed_when_metric_not_successful():
    """A metric scoring below threshold should produce passed=False."""
    metric = _make_mock_metric("Coherence", score=0.3, successful=False)
    test_case = LLMTestCase(input="topic", actual_output="poor content")

    scores = _run_metrics([metric], test_case, content_type="blog")

    assert scores[0].passed is False
    assert scores[0].score == 0.3


def test_evaluate_agent_state_returns_empty_when_no_content():
    """evaluate_agent_state should return [] when there is nothing to evaluate."""
    empty_state = AgentState(user_query="Write something")

    # No blog_post, no linkedin_post, no research → nothing to evaluate
    scores = evaluate_agent_state(empty_state, run_id=None)

    assert scores == []
