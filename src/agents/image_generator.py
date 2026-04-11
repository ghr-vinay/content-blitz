from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import ImageResult
from src.integrations.base_tool import BaseImageTool, BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_OPTIMIZER = """You are a DALL-E 3 prompt engineer.

Transform the user's image request into a detailed, optimised DALL-E 3 prompt.

User request: {user_request}
Style: {style}

Rules:
- Be specific about subject, environment, lighting, mood, and composition
- Include the style: {style}
- Keep the prompt under 400 characters
- Respond with ONLY the optimised prompt text — no explanation, no JSON
"""


class ImageGeneratorAgent(BaseAgent):
    """
    Generates images via an injected BaseImageTool with prompt optimisation.

    SOLID:
    - SRP: Responsible only for prompt optimisation + image generation.
    - DIP: Depends on BaseImageTool and BaseLLMClient — not DALL-E directly.
    - LSP: Any BaseImageTool (DALL-E, Stability AI, etc.) slots in seamlessly.
    """

    def __init__(self, image_tool: BaseImageTool, llm: BaseLLMClient) -> None:
        self._image_tool = image_tool
        self._llm = llm

    @property
    def name(self) -> str:
        return "image_generator"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        user_request: str = state.get("user_query", "").strip()
        style: str = state.get("image_style", "photorealistic")
        size: str = state.get("image_size", "1024x1024")

        if not user_request:
            return {"error": "ImageGeneratorAgent: no image request provided."}

        logger.info("ImageGeneratorAgent: request=%r | style=%s", user_request[:60], style)

        try:
            # Step 1: Optimise the prompt with the LLM
            optimiser_prompt = _PROMPT_OPTIMIZER.format(
                user_request=user_request, style=style
            )
            optimised_prompt: str = self._llm.generate(
                optimiser_prompt,
                config={"run_name": "image_prompt_optimizer"},
            ).strip()

            logger.debug("ImageGeneratorAgent: optimised prompt=%r", optimised_prompt[:120])

            # Step 2: Generate the image
            url: str = self._image_tool.generate_image(
                optimised_prompt, size=size
            )

            result = ImageResult(
                url=url,
                prompt_used=optimised_prompt,
                style=style,
                size=size,
            )

            logger.info("ImageGeneratorAgent: done | url=%s", url[:60])
            return {"image_result": result, "error": None}

        except Exception as exc:
            logger.exception("ImageGeneratorAgent failed: %s", exc)
            return {"error": str(exc)}
