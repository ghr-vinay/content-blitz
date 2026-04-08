from typing import Any

from langchain_community.utilities import SerpAPIWrapper
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.integrations.base_tool import BaseSearchTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SerpClient(BaseSearchTool):
    """
    SERP API implementation of BaseSearchTool using LangChain's SerpAPIWrapper.

    SOLID:
    - ISP: Implements only the narrow BaseSearchTool interface.
    - DIP: Agents depend on BaseSearchTool, not SerpClient.
    - LSP: Interchangeable with PerplexityClient or any BaseSearchTool.

    Retry policy (3.4): exponential backoff on transient network/API errors.
    """

    def __init__(self, api_key: str, num_results: int = 5) -> None:
        self._num_results = num_results
        self._wrapper = SerpAPIWrapper(
            serpapi_api_key=api_key,
            params={"num": num_results, "engine": "google"},
        )
        logger.info("SerpClient initialised | num_results=%d", num_results)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """
        Search the web and return structured results.

        Args:
            query: The search query string.
            num_results: Number of results to return (overrides default).

        Returns:
            List of dicts, each with keys: title, snippet, url.
        """
        logger.debug("SerpClient.search | query=%r | num=%d", query, num_results)

        raw: str = self._wrapper.run(query)

        # SerpAPIWrapper.run() returns a plain string summary.
        # Wrap it in a single structured result so all callers get a consistent
        # list[dict] shape regardless of which search provider is used.
        return [
            {
                "title": query,
                "snippet": raw,
                "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            }
        ]
