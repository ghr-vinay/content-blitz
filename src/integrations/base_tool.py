from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """
    Interface for LLM text generation providers.

    SOLID — DIP: Agents depend on this abstraction, not on OpenAI or any
    specific provider. Swapping providers requires zero agent code changes.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt and return the response string."""
        ...


class BaseSearchTool(ABC):
    """
    Interface for web search / research tools.

    SOLID — ISP: Deliberately narrow interface. Agents that only need search
    don't inherit image-generation concerns or vice versa.
    """

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """
        Execute a web search and return a list of result dicts.

        Each result dict should contain at minimum:
            - "title": str
            - "snippet": str
            - "url": str
        """
        ...


class BaseImageTool(ABC):
    """
    Interface for image generation providers.

    SOLID — LSP: Any implementation (DALL-E 3, Stability AI, etc.) can replace
    another wherever a BaseImageTool is expected.
    """

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate an image from a prompt.

        Returns:
            URL or local file path of the generated image.
        """
        ...
