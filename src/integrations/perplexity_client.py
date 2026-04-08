from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.integrations.base_tool import BaseSearchTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient(BaseSearchTool):
    """
    Perplexity Sonar implementation of BaseSearchTool (fallback research provider).

    SOLID:
    - OCP: Demonstrates provider swappability — same BaseSearchTool interface
           as SerpClient, zero changes needed in agents (Open/Closed).
    - LSP: Fully substitutable for SerpClient wherever BaseSearchTool is used.
    - DIP: Agents depend only on the BaseSearchTool abstraction.

    Retry policy (3.4): exponential backoff on transient errors.
    """

    def __init__(self, api_key: str, model: str = "sonar") -> None:
        self._api_key = api_key
        self._model = model
        logger.info("PerplexityClient initialised | model=%s", model)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """
        Query Perplexity Sonar and return structured results.

        Args:
            query: The research query or topic.
            num_results: Unused by Perplexity (single synthesised response);
                         kept for interface compatibility.

        Returns:
            List with a single dict containing: title, snippet, url.
        """
        logger.debug("PerplexityClient.search | query=%r", query)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Provide a concise, factual summary with sources.",
                },
                {"role": "user", "content": query},
            ],
        }

        with httpx.Client(timeout=30) as client:
            response = client.post(_PERPLEXITY_URL, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        content: str = data["choices"][0]["message"]["content"]

        return [
            {
                "title": query,
                "snippet": content,
                "url": "https://www.perplexity.ai",
            }
        ]
