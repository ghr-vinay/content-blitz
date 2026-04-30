"""
src/web_app/streamlit_app.py

ContentBlitz Streamlit Web Interface.

Usage:
    streamlit run src/web_app/streamlit_app.py

Features:
- Chat-based UI with multi-turn conversation history
- Renders research summaries, blog posts, LinkedIn posts, images, content strategy
- Sidebar with content type hint, image style, LinkedIn post type, clear/export controls
- Spinner for long-running agent tasks
- Download buttons for all generated content
"""

import json
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when
# Streamlit runs this file directly (i.e. not as `python -m ...`).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.core.config import Config
from src.core.models import AgentState
from src.core.workflow import reset_app
from src.core.workflow import run as workflow_run
from src.core.workflow import run_non_blocking_eval as workflow_run_async

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContentBlitz",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, result?}]
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "config_ok" not in st.session_state:
    st.session_state.config_ok = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "eval_future" not in st.session_state:
    st.session_state.eval_future = None     # Future[list[EvalScore]] | None
if "eval_msg_idx" not in st.session_state:
    st.session_state.eval_msg_idx = None    # index into messages list to back-fill scores


# ── Config / API key validation ────────────────────────────────────────────────
@st.cache_resource
def _init_config() -> bool:
    try:
        Config.get_instance()
        return True
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _render_sidebar() -> dict:
    with st.sidebar:
        st.title("⚡ ContentBlitz")
        st.caption("AI Content Marketing Assistant")
        st.divider()

        st.subheader("⚙️ Settings")

        image_style = st.selectbox(
            "Image Style",
            ["photorealistic", "illustration", "digital art", "watercolor", "minimalist"],
            index=0,
        )

        linkedin_post_type = st.selectbox(
            "LinkedIn Post Type",
            ["general", "thought-leadership", "announcement", "story", "listicle"],
            index=0,
        )

        image_size = st.selectbox(
            "Image Size",
            ["1024x1024", "1792x1024", "1024x1792"],
            index=0,
        )

        st.divider()
        st.subheader("🗂️ Session")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.last_result = None
            st.session_state.pop("suggestion_samples", None)  # re-sample on next zero state
            reset_app()
            st.rerun()

        st.divider()
        st.caption("Built with LangGraph · OpenAI · DALL-E 3 · LangSmith")
        st.caption("Contact at ghr247@gmail.com")

    return {
        "image_style": image_style,
        "linkedin_post_type": linkedin_post_type,
        "image_size": image_size,
    }


# ── Content renderers ──────────────────────────────────────────────────────────
def _render_research(result: AgentState, msg_idx: int = 0) -> None:
    r = result.research
    if not r:
        return
    # Collapse research if a writing output is also present — it's supporting context
    expanded = not (result.blog_post or result.linkedin_post)
    with st.expander("🔍 Research Summary", expanded=expanded):
        st.markdown(f"**Topic:** {r.topic}")
        st.markdown(r.summary)
        if r.key_findings:
            st.markdown("**Key Findings:**")
            for finding in r.key_findings:
                st.markdown(f"- {finding}")
        if r.sources:
            st.markdown("**Sources:**")
            for src in r.sources:
                st.markdown(f"- {src}")
        _download_button(
            label="⬇️ Download Research",
            data=_format_research_md(r),
            filename="research.md",
            mime="text/markdown",
            key=f"dl_research_{msg_idx}_{hash(r.topic)}",
        )


def _render_blog(result: AgentState, msg_idx: int = 0) -> None:
    b = result.blog_post
    if not b:
        return
    with st.expander("📝 Blog Post", expanded=True):
        st.markdown(f"# {b.title}")
        st.caption(f"Meta: {b.meta_description}")
        if b.keywords:
            st.caption(f"Keywords: {', '.join(b.keywords)}")
        st.divider()
        st.markdown(b.content)
        _download_button(
            label="⬇️ Download Blog Post",
            data=_format_blog_md(b),
            filename="blog_post.md",
            mime="text/markdown",
            key=f"dl_blog_{msg_idx}_{hash(b.title)}",
        )


def _render_linkedin(result: AgentState, msg_idx: int = 0) -> None:
    lp = result.linkedin_post
    if not lp:
        return
    with st.expander("💼 LinkedIn Post", expanded=True):
        st.caption(f"Type: {lp.post_type}  |  {lp.character_count} chars")
        st.markdown(lp.content)
        if lp.hashtags:
            st.markdown(" ".join(f"`{h}`" for h in lp.hashtags))
        col1, col2 = st.columns(2)
        with col1:
            _download_button(
                label="⬇️ Download Post",
                data=lp.content + "\n\n" + " ".join(lp.hashtags),
                filename="linkedin_post.txt",
                mime="text/plain",
                key=f"dl_li_{msg_idx}_{hash(lp.content[:40])}",
            )
        with col2:
            if st.button("📋 Copy to Clipboard", key=f"copy_li_{msg_idx}_{hash(lp.content[:40])}"):
                st.write("Paste-ready content copied below:")
                st.code(lp.content + "\n\n" + " ".join(lp.hashtags))


def _render_image(result: AgentState) -> None:
    img = result.image_result
    if not img:
        return
    with st.expander("🖼️ Generated Image", expanded=True):
        st.image(img.url, use_container_width=True)
        st.caption(f"**Prompt used:** {img.prompt_used}")
        st.caption(f"Style: {img.style}  |  Size: {img.size}")
        st.markdown(f"🔗 [Open full image]({img.url})")


def _render_strategy(result: AgentState, msg_idx: int = 0) -> None:
    if not result.content_strategy:
        return
    with st.expander("🗺️ Content Strategy", expanded=True):
        st.markdown(result.content_strategy)
        _download_button(
            label="⬇️ Download Strategy",
            data=result.content_strategy,
            filename="content_strategy.md",
            mime="text/markdown",
            key=f"dl_strat_{msg_idx}_{hash(result.content_strategy[:40])}",
        )


def _render_eval_scores(result: AgentState) -> None:
    if not result.eval_scores:
        return
    with st.expander("📊 Eval Scores (GEval / LLM-as-Judge)", expanded=False):
        by_type: dict = {}
        for s in result.eval_scores:
            by_type.setdefault(s.content_type, []).append(s)

        for ctype, scores in by_type.items():
            st.markdown(f"**{ctype.upper()}**")
            for s in scores:
                col1, col2, col3 = st.columns([2, 1, 5])
                with col1:
                    st.markdown(s.metric)
                with col2:
                    colour = "green" if s.passed else "orange"
                    st.markdown(f":{colour}[{s.score:.2f}]")
                with col3:
                    st.caption(s.reason)
            st.divider()


def _render_result(result: AgentState, msg_idx: int = 0) -> None:
    if result.error:
        st.error(f"❌ {result.error}")
        return

    if result.fallback_message:
        st.info(f"💬 {result.fallback_message}")
        return

    intent_badge = {
        "research": "🔍 Research",
        "blog": "📝 Blog",
        "linkedin": "💼 LinkedIn",
        "image": "🖼️ Image",
        "strategy": "🗺️ Strategy",
        "linkedin_with_image": "🔀 Multi-format",
        "blog_with_image": "🔀 Multi-format",
    }.get(result.intent or "", f"({result.intent})")

    refinement_tag = " *(refinement)*" if result.is_refinement else ""
    st.markdown(f"**Intent:** {intent_badge}{refinement_tag}")

    _render_research(result, msg_idx)
    _render_blog(result, msg_idx)
    _render_linkedin(result, msg_idx)
    _render_image(result)
    _render_strategy(result, msg_idx)
    _render_eval_scores(result)


# ── Download helper ────────────────────────────────────────────────────────────
def _download_button(
    label: str,
    data: str,
    filename: str,
    mime: str,
    key: str,
) -> None:
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
    )


# ── Markdown formatters ────────────────────────────────────────────────────────
def _format_research_md(r) -> str:
    lines = [f"# Research: {r.topic}\n", r.summary, "\n## Key Findings\n"]
    lines += [f"- {f}" for f in r.key_findings]
    lines += ["\n## Sources\n"] + [f"- {s}" for s in r.sources]
    return "\n".join(lines)


def _format_blog_md(b) -> str:
    return (
        f"# {b.title}\n\n"
        f"**Meta Description:** {b.meta_description}\n\n"
        f"**Keywords:** {', '.join(b.keywords)}\n\n"
        f"---\n\n{b.content}"
    )


# ── Chat history renderer ──────────────────────────────────────────────────────
def _render_chat_history() -> None:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                _render_result(msg["result"], msg_idx=idx)


# ── Background eval polling ────────────────────────────────────────────────────
def _poll_eval_future() -> str:
    """
    Check whether the background eval thread has finished.

    Returns:
        "done"     — future just completed; caller should st.rerun() to show scores.
        "running"  — still in progress; caller should sleep + st.rerun() to poll again.
        "idle"     — no active future; nothing to do.
    """
    future = st.session_state.get("eval_future")
    if future is None:
        return "idle"

    if future.done():
        try:
            scores = future.result()
            idx = st.session_state.eval_msg_idx
            if idx is not None and idx < len(st.session_state.messages):
                msg = st.session_state.messages[idx]
                if msg.get("result"):
                    msg["result"] = msg["result"].model_copy(update={"eval_scores": scores})
                    if st.session_state.last_result and not st.session_state.last_result.eval_scores:
                        st.session_state.last_result = msg["result"]
        except Exception:
            pass  # eval failure is non-fatal; scores simply won't appear
        finally:
            st.session_state.eval_future = None
            st.session_state.eval_msg_idx = None
        return "done"   # scores merged — rerun to render them
    else:
        st.status("⏳ Evaluating content quality in background…", state="running")
        return "running"   # still running — rerun after a pause


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    settings = _render_sidebar()

    st.title("⚡ ContentBlitz")
    st.caption("Your AI-powered content marketing assistant")
    st.caption("_**Web research runs automatically for grounded, factual results_")

    config_ok = _init_config()
    if not config_ok:
        st.error(
            "❌ Configuration error — missing API keys. "
            "Copy `.env.example` → `.env` and fill in your keys, then restart."
        )
        st.stop()

    _render_chat_history()

    # ── Zero-state suggestions ───────────────────────────────────────────────
    # Shown only when the conversation is empty AND no prompt is pending.
    # Samples are stored in session state so the button→text mapping is stable
    # across reruns (avoids picking up the wrong suggestion on rerun).
    _SUGGESTIONS = [
        ("📝", "Blog post", "Write a blog post about the rise of AI agents in 2025"),
        ("💼", "LinkedIn post", "Write a thought-leadership LinkedIn post about RAG in enterprise AI"),
        ("🔍", "Research", "Research the latest trends in large language models"),
        ("🎨", "Blog + image", "Write a blog post about vector databases and generate a cover image"),
        ("🗺️", "Content strategy", "Give me a content strategy for a SaaS company launching a developer tool"),
        ("🔄", "LinkedIn + image", "Write a LinkedIn announcement post about an AI product launch and generate an image"),
    ]

    if not st.session_state.messages and "_pending_prompt" not in st.session_state:
        # Sample once and freeze — cleared when the conversation is cleared
        if "suggestion_samples" not in st.session_state:
            st.session_state.suggestion_samples = random.sample(_SUGGESTIONS, 3)
        shown = st.session_state.suggestion_samples

        st.markdown('<div style="height:12vh"></div>', unsafe_allow_html=True)
        _, center, _ = st.columns([1, 1.2, 1])
        with center:
            st.markdown("#### What would you like to create today?")
            for i, (icon, label, suggestion_text) in enumerate(shown):
                if st.button(
                    f"{icon} **{label}**\n\n{suggestion_text}",
                    use_container_width=True,
                    key=f"suggestion_{i}",
                ):
                    st.session_state["_pending_prompt"] = suggestion_text
                    st.rerun()
        st.markdown('<div style="height:12vh"></div>', unsafe_allow_html=True)

    # Apply a suggestion that was clicked in the previous run.
    # Chat input also routes through _pending_prompt so the suggestions guard
    # (`"_pending_prompt" not in st.session_state`) fires on the very same rerun
    # the user submits — identical behaviour to clicking a suggestion card.
    _chat_input = st.chat_input("Ask me to research, write a blog, LinkedIn post, generate an image…")
    if _chat_input and "_pending_prompt" not in st.session_state:
        st.session_state["_pending_prompt"] = _chat_input
        st.rerun()

    if "_pending_prompt" in st.session_state:
        prompt = st.session_state.pop("_pending_prompt")
    else:
        prompt = None

    if prompt:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the workflow with a spinner (eval runs in background thread)
        with st.chat_message("assistant"):
            st.caption("_May take ~30-50 seconds more for blog posts and multi-format intents_")
            with st.spinner("Thinking…"):
                try:
                    result, eval_future = workflow_run_async(
                        user_query=prompt,
                        image_style=settings["image_style"],
                        image_size=settings["image_size"],
                        linkedin_post_type=settings["linkedin_post_type"],
                        conversation_history=st.session_state.conversation_history,
                    )
                except Exception as exc:
                    result = AgentState(user_query=prompt, error=str(exc))
                    eval_future = None

            # Build assistant message text
            if result.error:
                assistant_text = f"❌ {result.error}"
            else:
                parts = []
                if result.research:
                    parts.append("research summary")
                if result.blog_post:
                    parts.append(f"blog post *{result.blog_post.title}*")
                if result.linkedin_post:
                    parts.append("LinkedIn post")
                if result.image_result:
                    parts.append("generated image")
                if result.content_strategy:
                    parts.append("content strategy")
                assistant_text = "Here's what I produced: " + ", ".join(parts) + "." if parts else "Done."

            st.markdown(assistant_text)
            _render_result(result, msg_idx=len(st.session_state.messages))

        # Persist to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text,
            "result": result,
        })
        st.session_state.conversation_history.append({"role": "human", "content": prompt})
        st.session_state.conversation_history.append({"role": "ai", "content": assistant_text})
        st.session_state.last_result = result

        # Register background eval future so the polling loop can surface scores
        if eval_future is not None:
            st.session_state.eval_future = eval_future
            st.session_state.eval_msg_idx = len(st.session_state.messages) - 1
            st.rerun()  # first rerun: re-render from history and start polling

    # ── Poll AFTER all content is rendered ──────────────────────────────────
    # st.rerun() is only triggered here so the full page (chat history +
    # generated content) is committed to the browser before refreshing.
    eval_status = _poll_eval_future()
    if eval_status == "running":
        time.sleep(2)   # throttle polling to ~1 rerun per 2 s
        st.rerun()
    elif eval_status == "done":
        st.rerun()      # re-render immediately to show the merged scores


if __name__ == "__main__":
    main()
