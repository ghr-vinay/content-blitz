from typing import Any, Literal

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.integrations.base_tool import BaseImageTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

ImageSize = Literal["1024x1024", "1792x1024", "1024x1792"]
ImageQuality = Literal["standard", "hd"]


class DallE3Client(BaseImageTool):
    """
    DALL-E 3 implementation of BaseImageTool.

    SOLID:
    - LSP: Any BaseImageTool implementation (Stability AI, Google Imagen, etc.)
           can replace this without changing agent code.
    - DIP: ImageGenerationAgent depends on BaseImageTool, not DallE3Client.
    - SRP: Handles only image generation via DALL-E 3.

    Retry policy (3.4): exponential backoff on rate limit / transient errors.
    """

    def __init__(
        self,
        api_key: str,
        size: ImageSize = "1024x1024",
        quality: ImageQuality = "standard",
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._size = size
        self._quality = quality
        logger.info(
            "DallE3Client initialised | size=%s | quality=%s", size, quality
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate_image(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate an image using DALL-E 3 and return the image URL.

        Args:
            prompt: Detailed image description (already optimised by the agent).
            **kwargs: Optional overrides — size, quality.

        Returns:
            Temporary URL of the generated image (valid ~60 minutes).

        Raises:
            openai.BadRequestError: If the prompt violates content policy.
            Exception: On transient errors after all retry attempts.
        """
        size: ImageSize = kwargs.get("size", self._size)
        quality: ImageQuality = kwargs.get("quality", self._quality)

        logger.debug(
            "DallE3Client.generate_image | size=%s | quality=%s | prompt_len=%d",
            size, quality, len(prompt),
        )

        response = self._client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

        url: str = response.data[0].url
        logger.info("DallE3Client: image generated | url=%s", url[:60])
        return url
