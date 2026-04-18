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
import sys
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
            reset_app()
            st.rerun()

        st.divider()
        st.caption("Phases complete: 1–7 ✅")
        st.caption("Model: GPT-4o + DALL-E 3")

    return {
        "image_style": image_style,
        "linkedin_post_type": linkedin_post_type,
        "image_size": image_size,
    }


# ── Content renderers ──────────────────────────────────────────────────────────
def _render_research(result: AgentState) -> None:
    r = result.research
    if not r:
        return
    with st.expander("🔍 Research Summary", expanded=True):
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
            key=f"dl_research_{hash(r.topic)}",
        )


def _render_blog(result: AgentState) -> None:
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
            key=f"dl_blog_{hash(b.title)}",
        )


def _render_linkedin(result: AgentState) -> None:
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
                key=f"dl_li_{hash(lp.content[:40])}",
            )
        with col2:
            if st.button("📋 Copy to Clipboard", key=f"copy_li_{hash(lp.content[:40])}"):
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


def _render_strategy(result: AgentState) -> None:
    if not result.content_strategy:
        return
    with st.expander("🗺️ Content Strategy", expanded=True):
        st.markdown(result.content_strategy)
        _download_button(
            label="⬇️ Download Strategy",
            data=result.content_strategy,
            filename="content_strategy.md",
            mime="text/markdown",
            key=f"dl_strat_{hash(result.content_strategy[:40])}",
        )


def _render_result(result: AgentState) -> None:
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
        "multi": "🔀 Multi-format",
    }.get(result.intent or "", f"({result.intent})")

    refinement_tag = " *(refinement)*" if result.is_refinement else ""
    st.markdown(f"**Intent:** {intent_badge}{refinement_tag}")

    _render_research(result)
    _render_blog(result)
    _render_linkedin(result)
    _render_image(result)
    _render_strategy(result)


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
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                _render_result(msg["result"])


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    settings = _render_sidebar()

    st.title("⚡ ContentBlitz")
    st.caption("Your AI-powered content marketing assistant")

    config_ok = _init_config()
    if not config_ok:
        st.error(
            "❌ Configuration error — missing API keys. "
            "Copy `.env.example` → `.env` and fill in your keys, then restart."
        )
        st.stop()

    _render_chat_history()

    if prompt := st.chat_input("Ask me to research, write a blog, LinkedIn post, generate an image…"):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the workflow with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result: AgentState = workflow_run(
                        user_query=prompt,
                        image_style=settings["image_style"],
                        image_size=settings["image_size"],
                        linkedin_post_type=settings["linkedin_post_type"],
                        conversation_history=st.session_state.conversation_history,
                    )
                except Exception as exc:
                    result = AgentState(user_query=prompt, error=str(exc))

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
            _render_result(result)

        # Persist to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text,
            "result": result,
        })
        st.session_state.conversation_history.append({"role": "human", "content": prompt})
        st.session_state.conversation_history.append({"role": "ai", "content": assistant_text})
        st.session_state.last_result = result


if __name__ == "__main__":
    main()
