from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.integrations.base_tool import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    LangChain-backed OpenAI GPT-4 implementation of BaseLLMClient.

    SOLID:
    - DIP: Agents depend on BaseLLMClient, not this class directly.
    - SRP: Handles only LLM text generation via OpenAI.
    - LSP: Can be swapped with any other BaseLLMClient implementation.

    Retry policy (3.4): exponential backoff on rate limit / transient errors.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7) -> None:
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        logger.info("OpenAIClient initialised | model=%s", model)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a prompt to GPT-4 and return the response text.

        Args:
            prompt: The full prompt string to send.
            **kwargs: Optional LangChain invoke config (e.g., run_name for LangSmith).

        Returns:
            The model's response as a plain string.
        """
        logger.debug("OpenAIClient.generate | prompt_len=%d", len(prompt))
        response = self._llm.invoke(
            [HumanMessage(content=prompt)],
            config=kwargs.get("config"),
        )
        return response.content
