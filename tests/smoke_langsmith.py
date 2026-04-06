"""
Smoke test for Phase 2.3 — verify LangSmith tracing is wired correctly.

Run with:
    python -m tests.smoke_langsmith

Then open https://smith.langchain.com and check the project
configured in config/development.yaml for a trace named "smoke-test".
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.core.config import Config


def run_smoke_test() -> None:
    config = Config.get_instance()

    print(f"Environment  : {config.environment}")
    print(f"LangSmith key: {'set' if config.langsmith_api_key else 'NOT SET ⚠️'}")
    print(f"LLM model    : {config.llm_model}")
    print()

    llm = ChatOpenAI(
        model=config.llm_model,
        api_key=config.openai_api_key,
        temperature=0,
    )

    print("Sending a test message to the LLM...")
    response = llm.invoke(
        [HumanMessage(content="Reply with exactly: ContentBlitz smoke test OK")],
        config={"run_name": "smoke-test"},
    )

    print(f"Response: {response.content}")
    print()
    print("✓ Smoke test passed.")
    print("  → Open https://smith.langchain.com to verify the trace appears.")


if __name__ == "__main__":
    run_smoke_test()
