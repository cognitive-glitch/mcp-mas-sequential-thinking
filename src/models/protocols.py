"""
Protocol definitions for type safety across the codebase.
Replaces generic Any type hints with specific protocols.
"""

from typing import Protocol, Any, Optional, Union, Dict, List
from abc import abstractmethod


class ModelProtocol(Protocol):
    """Protocol for LLM model instances used by teams and agents."""

    @abstractmethod
    async def aresponse(
        self,
        messages: List[Any],
        response_format: Any = None,
        tools: Any = None,
        functions: Any = None,
        tool_choice: Any = None,
        tool_call_limit: Any = None,
    ) -> Any:
        """Async response method for Agno models with message list."""
        ...

    @abstractmethod
    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke method for Agno models with message list."""
        ...


class AgentModelProtocol(Protocol):
    """Protocol specifically for agent model instances."""

    def invoke(self, prompt: Union[str, Dict[str, Any]]) -> Any:
        """Synchronous invoke for agent models."""
        ...

    async def aresponse(
        self,
        messages: List[Any],
        response_format: Any = None,
        tools: Any = None,
        functions: Any = None,
        tool_choice: Any = None,
        tool_call_limit: Any = None,
    ) -> Any:
        """Async response for agent models."""
        ...

    async def ainvoke(self, messages: List[Any]) -> Any:
        """Async invoke for agent models."""
        ...


class AppContextProtocol(Protocol):
    """Protocol for application context used in handlers."""

    teams_initialized: bool
    primary_team: Optional[Any]
    reflection_team: Optional[Any]
    shared_context: Any
    error_handler: Any
    thought_history: List[Any]

    @abstractmethod
    async def initialize_teams(self) -> None:
        """Initialize both primary and reflection teams."""
        ...

    @abstractmethod
    async def add_thought(self, thought_data: Any) -> None:
        """Add a thought to the context."""
        ...

    @abstractmethod
    async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
        """Get context relevant to the current thought."""
        ...

    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the session."""
        ...
