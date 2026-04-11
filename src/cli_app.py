"""
src/cli_app.py

Interactive CLI for ContentBlitz — run multi-turn conversations with the
agent system directly from the terminal.

Usage:
    python -m src.cli_app
"""

from src.core.config import Config
from src.core.workflow import run, reset_app
from src.core.models import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BANNER = """
╔══════════════════════════════════════════════╗
║         ContentBlitz  —  CLI Mode            ║
║  AI Content Marketing Assistant              ║
╠══════════════════════════════════════════════╣
║  Commands:                                   ║
║    /quit  or  /exit  — exit                  ║
║    /clear            — clear history         ║
║    /help             — show this message     ║
╚══════════════════════════════════════════════╝
"""

_INTENT_LABELS = {
    "research":  "🔍 Research",
    "blog":      "📝 Blog Post",
    "linkedin":  "💼 LinkedIn Post",
    "image":     "🖼️  Image",
    "strategy":  "🗺️  Content Strategy",
    "multi":     "🔀 Multi-format",
}


def _print_result(result: AgentState) -> None:
    print()

    if result.error:
        print(f"❌  Error: {result.error}")
        return

    intent_label = _INTENT_LABELS.get(result.intent or "", f"({result.intent})")
    print(f"━━━  {intent_label}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if result.research:
        r = result.research
        print(f"\n📚  RESEARCH: {r.topic}")
        print(f"\n{r.summary}")
        if r.key_findings:
            print("\nKey Findings:")
            for f in r.key_findings:
                print(f"  • {f}")
        if r.sources:
            print("\nSources:")
            for s in r.sources:
                print(f"  → {s}")

    if result.blog_post:
        b = result.blog_post
        print(f"\n📝  BLOG POST: {b.title}")
        print(f"Meta: {b.meta_description}")
        print(f"Keywords: {', '.join(b.keywords)}")
        print(f"\n{b.content}")

    if result.linkedin_post:
        lp = result.linkedin_post
        print(f"\n💼  LINKEDIN POST  [{lp.post_type}]")
        print(f"\n{lp.content}")
        if lp.hashtags:
            print("\n" + "  ".join(lp.hashtags))

    if result.image_result:
        img = result.image_result
        print(f"\n🖼️   IMAGE GENERATED")
        print(f"Prompt used: {img.prompt_used}")
        print(f"URL: {img.url}")

    if result.content_strategy:
        print(f"\n🗺️   CONTENT STRATEGY\n")
        print(result.content_strategy)

    print("\n" + "─" * 50)


def main() -> None:
    print(_BANNER)

    # Initialise config (validates API keys, sets up LangSmith)
    try:
        Config.get_instance()
    except EnvironmentError as e:
        print(f"❌  Config error: {e}")
        print("    Copy .env.example → .env and fill in your API keys.")
        return

    conversation_history: list[dict] = []

    print("Ready! Type your content request below.\n")

    while True:
        try:
            user_input = input("You › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            conversation_history.clear()
            reset_app()
            print("✓  Conversation history cleared.\n")
            continue

        if user_input.lower() == "/help":
            print(_BANNER)
            continue

        # ── Run workflow ──────────────────────────────────────────────────────
        print("\n⏳  Thinking...\n")
        try:
            result = run(
                user_query=user_input,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            print(f"❌  Unexpected error: {exc}\n")
            logger.exception("CLI run error: %s", exc)
            continue

        _print_result(result)

        # Append turn to history for multi-turn context
        conversation_history.append({"role": "user", "content": user_input})
        if result.error:
            conversation_history.append({"role": "assistant", "content": f"Error: {result.error}"})
        else:
            conversation_history.append({"role": "assistant", "content": f"Completed: intent={result.intent}"})

        print()


if __name__ == "__main__":
    main()
