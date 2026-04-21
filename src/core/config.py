import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger, configure_logging

logger = get_logger(__name__)


class Config:
    """
    Centralized configuration loader for ContentBlitz.

    Responsibilities (SRP):
    - Load environment variables from .env
    - Load YAML config for the current environment
    - Set up LangSmith tracing environment variables
    - Validate required API keys at startup

    DIP: All other modules depend on this Config abstraction for settings,
    not on os.getenv() calls scattered throughout the codebase.
    """

    _instance: Optional["Config"] = None

    def __init__(self) -> None:
        load_dotenv()
        self._env: str = os.getenv("ENV", "development")
        self._data: dict[str, Any] = {}
        self._load_yaml_configs()
        self._configure_langsmith()
        self._configure_logging()
        self._validate_required_keys()
        logger.info("Config loaded for environment: %s", self._env)

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "Config":
        """Return the singleton Config instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Internal setup ─────────────────────────────────────────────────────────

    def _load_yaml_configs(self) -> None:
        config_dir = Path(__file__).parent.parent.parent / "config"

        env_file = config_dir / f"{self._env}.yaml"
        if env_file.exists():
            with open(env_file) as f:
                self._data.update(yaml.safe_load(f) or {})
        else:
            logger.warning("Config file not found: %s", env_file)

        services_file = config_dir / "services.yaml"
        if services_file.exists():
            with open(services_file) as f:
                self._data["services"] = yaml.safe_load(f) or {}

    def _configure_langsmith(self) -> None:
        """
        Wire LangSmith tracing via environment variables.
        LangChain automatically picks these up — no extra code needed in agents.
        """
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

        project = self._data.get("langsmith", {}).get("project", "contentblitz")
        os.environ.setdefault("LANGCHAIN_PROJECT", project)

        api_key = os.getenv("LANGSMITH_API_KEY")
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
        else:
            logger.warning(
                "LANGSMITH_API_KEY not set — LangSmith tracing will be disabled."
            )

    def _configure_logging(self) -> None:
        log_cfg = self._data.get("logging", {})
        level_str: str = log_cfg.get("level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger("src").setLevel(level)

        fmt: str = log_cfg.get(
            "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        configure_logging(fmt=fmt)

    def _validate_required_keys(self) -> None:
        required = ["OPENAI_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}. "
                "Copy .env.example to .env and fill in the values."
            )

    # ── Public accessors ───────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the loaded YAML config."""
        return self._data.get(key, default)

    @property
    def environment(self) -> str:
        return self._env

    @property
    def debug(self) -> bool:
        return bool(self._data.get("debug", False))

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def serp_api_key(self) -> str:
        return os.getenv("SERP_API_KEY", "")

    @property
    def langsmith_api_key(self) -> str:
        return os.getenv("LANGSMITH_API_KEY", "")

    @property
    def llm_model(self) -> str:
        return self._data.get("services", {}).get("llm", {}).get("model", "gpt-4o")

    @property
    def llm_temperature(self) -> float:
        return float(
            self._data.get("services", {}).get("llm", {}).get("temperature", 0.7)
        )

    @property
    def search_num_results(self) -> int:
        return int(
            self._data.get("services", {}).get("search", {}).get("num_results", 5)
        )

    @property
    def image_size(self) -> str:
        return self._data.get("services", {}).get("image", {}).get("size", "1024x1024")

    @property
    def image_quality(self) -> str:
        return self._data.get("services", {}).get("image", {}).get("quality", "standard")

    @property
    def eval_model(self) -> str:
        return self._data.get("services", {}).get("eval", {}).get("model", "gpt-4o")

    @property
    def eval_threshold(self) -> float:
        return float(
            self._data.get("services", {}).get("eval", {}).get("threshold", 0.5)
        )
