"""
src/eval/llm_eval.py

LLM-as-Judge evaluation using DeepEval's GEval metric.

GEval defines criteria in plain English and uses an LLM (GPT-4o by default)
to score generated content on a 0–1 scale with a natural-language reason.

Usage:
    from src.eval.llm_eval import evaluate_agent_state
    scores = evaluate_agent_state(result, run_id="abc123")

SOLID:
- SRP: This module's only job is content quality evaluation.
- OCP: New metrics / content types can be added without changing existing ones.
- DIP: Depends on DeepEval abstractions (GEval, LLMTestCase), not raw LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.core.models import AgentState, EvalScore

logger = get_logger(__name__)

# ── Metric threshold ───────────────────────────────────────────────────────────
# Defaults used when no Config is available (e.g. standalone scripts / tests).
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_MODEL = "gpt-4o"


# ── Metric factories (fresh instance per evaluation run) ───────────────────────
# GEval stores .score / .reason as mutable state after .measure(), so we must
# create a new instance each time rather than reusing module-level singletons.

def _blog_metrics(model: str, threshold: float) -> list[GEval]:
    return [
        GEval(
            name="Relevance",
            criteria=(
                "The blog post directly addresses the requested topic. "
                "All sections are on-topic and do not drift into unrelated content."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Coherence",
            criteria=(
                "The blog post is well-structured with a clear introduction, body, and conclusion. "
                "Ideas flow logically between paragraphs and the writing reads smoothly."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="SEO Quality",
            criteria=(
                "The blog post includes relevant keywords naturally integrated into the text, "
                "a clear H1 title, H2/H3 subheadings, and a meta description that accurately "
                "summarises the content."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
    ]


def _linkedin_metrics(model: str, threshold: float) -> list[GEval]:
    return [
        GEval(
            name="Relevance",
            criteria=(
                "The LinkedIn post directly addresses the requested topic and "
                "delivers value relevant to the subject matter."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Engagement Quality",
            criteria=(
                "The LinkedIn post opens with a strong hook that makes readers want to continue. "
                "It ends with a clear call-to-action or thought-provoking question. "
                "Hashtags are relevant and appropriately used."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Conciseness",
            criteria=(
                "The post delivers its message without unnecessary filler. "
                "Each sentence adds value. Short paragraphs and line breaks are used "
                "appropriately for LinkedIn readability."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
    ]


def _research_metrics(model: str, threshold: float) -> list[GEval]:
    return [
        GEval(
            name="Relevance",
            criteria=(
                "The research summary directly addresses the requested topic. "
                "Key findings are on-topic and contribute meaningfully to understanding the subject."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Comprehensiveness",
            criteria=(
                "The research covers the topic with sufficient depth — key aspects, "
                "recent developments, and important context are included. "
                "It does not leave obvious important angles unexplored."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            threshold=threshold,
        ),
        GEval(
            name="Source Quality",
            criteria=(
                "The research cites credible, relevant sources. "
                "Sources are attributed clearly and appear trustworthy for the subject matter."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=model,
            threshold=threshold,
        ),
    ]


# ── Per-content evaluators ─────────────────────────────────────────────────────

def _run_metrics(
    metrics: list[GEval],
    test_case: LLMTestCase,
    content_type: str,
) -> list["EvalScore"]:
    """Run a list of metrics against a test case, returning EvalScore objects."""
    # Import here to avoid circular import at module load time
    from src.core.models import EvalScore

    scores: list[EvalScore] = []
    for metric in metrics:
        try:
            metric.measure(test_case)
            score = round(float(metric.score), 3)
            # .is_successful() is the stable API; fall back to score >= threshold
            try:
                passed = bool(metric.is_successful())
            except AttributeError:
                passed = score >= _DEFAULT_THRESHOLD
            scores.append(
                EvalScore(
                    metric=metric.name,
                    content_type=content_type,
                    score=score,
                    reason=metric.reason or "",
                    passed=passed,
                )
            )
            logger.debug(
                "Eval [%s/%s]: score=%.3f passed=%s",
                content_type, metric.name, score, passed,
            )
        except Exception as exc:
            logger.warning("Metric '%s' failed for %s: %s", metric.name, content_type, exc)

    return scores


def _evaluate_blog(result: "AgentState", model: str, threshold: float) -> list["EvalScore"]:
    b = result.blog_post
    if not b:
        return []
    full_output = f"Title: {b.title}\n\nMeta: {b.meta_description}\n\nKeywords: {', '.join(b.keywords)}\n\n{b.content}"
    test_case = LLMTestCase(
        input=result.clarified_user_query or result.user_query,
        actual_output=full_output,
    )
    return _run_metrics(_blog_metrics(model, threshold), test_case, "blog")


def _evaluate_linkedin(result: "AgentState", model: str, threshold: float) -> list["EvalScore"]:
    lp = result.linkedin_post
    if not lp:
        return []
    full_output = lp.content + "\n\n" + "  ".join(lp.hashtags)
    test_case = LLMTestCase(
        input=result.clarified_user_query or result.user_query,
        actual_output=full_output,
    )
    return _run_metrics(_linkedin_metrics(model, threshold), test_case, "linkedin")


def _evaluate_research(result: "AgentState", model: str, threshold: float) -> list["EvalScore"]:
    r = result.research
    if not r:
        return []
    findings_str = "\n".join(f"- {f}" for f in r.key_findings)
    full_output = f"Summary: {r.summary}\n\nKey Findings:\n{findings_str}"
    test_case = LLMTestCase(
        input=result.clarified_user_query or result.user_query,
        actual_output=full_output,
        retrieval_context=r.sources,
    )
    return _run_metrics(_research_metrics(model, threshold), test_case, "research")


# ── LangSmith feedback bridge ──────────────────────────────────────────────────

def _log_to_langsmith(scores: list["EvalScore"], run_id: str) -> None:
    """
    Post DeepEval scores to LangSmith as feedback on the parent trace.

    DeepEval has no native LangSmith integration, so we bridge manually
    via langsmith.Client.create_feedback().
    """
    try:
        from langsmith import Client as LangSmithClient
        ls = LangSmithClient()
        for s in scores:
            key = f"eval_{s.content_type}_{s.metric.lower().replace(' ', '_')}"
            ls.create_feedback(
                run_id=run_id,
                key=key,
                score=s.score,
                comment=s.reason,
            )
        logger.info("Logged %d eval scores to LangSmith (run_id=%s)", len(scores), run_id)
    except Exception as exc:
        logger.warning("Could not log eval scores to LangSmith: %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_agent_state(
    result: "AgentState",
    run_id: str | None = None,
    model: str = _DEFAULT_MODEL,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list["EvalScore"]:
    """
    Run GEval metrics on all generated content in an AgentState.

    Evaluates blog posts, LinkedIn posts, and research summaries independently.
    Scores are optionally logged to LangSmith if a run_id is provided.

    Args:
        result:    The completed AgentState from the workflow.
        run_id:    LangSmith trace run ID for feedback logging. Pass None to skip.
        model:     Judge model (read from Config.eval_model; default: gpt-4o).
        threshold: Pass/fail threshold (read from Config.eval_threshold; default: 0.5).

    Returns:
        List of EvalScore — one per (content_type, metric) pair.
    """
    scores: list[EvalScore] = []

    if result.blog_post:
        scores.extend(_evaluate_blog(result, model, threshold))
    if result.linkedin_post:
        scores.extend(_evaluate_linkedin(result, model, threshold))
    if result.research:
        scores.extend(_evaluate_research(result, model, threshold))

    if not scores:
        logger.info("No evaluable content in result — skipping eval")
        return scores

    logger.info(
        "Eval complete: %d scores across %d content types",
        len(scores),
        len({s.content_type for s in scores}),
    )

    if run_id:
        _log_to_langsmith(scores, run_id)

    return scores
