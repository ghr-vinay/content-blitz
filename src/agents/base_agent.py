from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class that all ContentBlitz agents must extend.

    SOLID:
    - SRP: Subclasses have exactly one content responsibility (research, blog, etc.)
    - OCP: New agents are added by subclassing — no existing code changes.
    - LSP: Any BaseAgent subclass can be used interchangeably in the LangGraph graph.
    - DIP: The LangGraph workflow depends on this abstraction, not concrete agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent (used in LangSmith traces and routing)."""
        ...

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the agent's task and return an updated LangGraph state.

        Args:
            state: Current LangGraph state dict shared across the graph.

        Returns:
            Updated state dict with this agent's output merged in.
        """
        ...
