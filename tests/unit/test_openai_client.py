"""
tests/unit/test_openai_client.py

Unit tests for OpenAIClient.

Strategy: patch LangChain's ChatOpenAI.invoke so we never touch the network.
We verify that generate() extracts .content from the response, and that the
tenacity retry decorator fires on transient errors.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.integrations.openai_client import OpenAIClient


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_generate_returns_llm_content(monkeypatch):
    """generate() should return the .content string from the LangChain response."""
    fake_response = MagicMock()
    fake_response.content = "Hello from GPT"

    with patch("src.integrations.openai_client.ChatOpenAI") as MockChatOpenAI:
        instance = MockChatOpenAI.return_value
        instance.invoke.return_value = fake_response

        client = OpenAIClient(api_key="test-key", model="gpt-4o")
        result = client.generate("Say hello")

    assert result == "Hello from GPT"
    instance.invoke.assert_called_once()


def test_generate_retries_on_transient_error():
    """generate() should retry up to 3 times before re-raising the exception."""
    fake_response = MagicMock()
    fake_response.content = "recovered"

    call_count = 0

    def flaky_invoke(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient network error")
        return fake_response

    with patch("src.integrations.openai_client.ChatOpenAI") as MockChatOpenAI:
        instance = MockChatOpenAI.return_value
        instance.invoke.side_effect = flaky_invoke

        client = OpenAIClient(api_key="test-key", model="gpt-4o")
        # Override wait to make retry instant in tests
        client.generate.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
        result = client.generate("Say hello")

    assert result == "recovered"
    assert call_count == 3


def test_generate_raises_after_max_retries():
    """generate() raises after exhausting all 3 retry attempts."""
    with patch("src.integrations.openai_client.ChatOpenAI") as MockChatOpenAI:
        instance = MockChatOpenAI.return_value
        instance.invoke.side_effect = ConnectionError("always fails")

        client = OpenAIClient(api_key="test-key", model="gpt-4o")
        client.generate.retry.wait = lambda *_: 0  # type: ignore[attr-defined]

        with pytest.raises(ConnectionError):
            client.generate("Fail me")
