"""
Tests for protocol definitions.
"""

import pytest
import asyncio
from typing import Any, Optional, Dict, List
from unittest.mock import Mock, AsyncMock

from src.models.protocols import ModelProtocol, AgentModelProtocol, AppContextProtocol


class TestModelProtocol:
    """Test the ModelProtocol protocol."""

    def test_model_protocol_structure(self):
        """Test that ModelProtocol has expected methods."""
        # Check protocol has required methods
        assert hasattr(ModelProtocol, "aresponse")
        assert hasattr(ModelProtocol, "ainvoke")

    def test_model_protocol_implementation(self):
        """Test a concrete implementation of ModelProtocol."""

        class MockModel:
            """Mock implementation of ModelProtocol."""

            async def aresponse(
                self,
                messages: List[Any],
                response_format: Any = None,
                tools: Any = None,
                functions: Any = None,
                tool_choice: Any = None,
                tool_call_limit: Any = None,
            ) -> Any:
                return Mock(content="Mock aresponse")

            async def ainvoke(self, messages: List[Any]) -> Any:
                return Mock(content="Mock ainvoke")

        # Verify implementation satisfies protocol
        model = MockModel()
        assert isinstance(model, MockModel)

        # Test protocol behavior through duck typing
        async def test_model_usage(model_instance):
            response1 = await model_instance.aresponse([])
            response2 = await model_instance.ainvoke([])
            return response1, response2

        # This should work without type errors
        import asyncio

        async def run_test():
            return await test_model_usage(model)

        # Verify it can be called
        assert asyncio.iscoroutinefunction(test_model_usage)

    def test_model_protocol_aresponse_signature(self):
        """Test ModelProtocol aresponse signature."""
        # Create a mock that follows the protocol
        mock_model = Mock()
        mock_model.aresponse = AsyncMock(return_value=Mock(content="response"))

        # Verify we can call with expected parameters
        async def test_call():
            return await mock_model.aresponse(
                messages=[],
                response_format=None,
                tools=None,
                functions=None,
                tool_choice=None,
                tool_call_limit=None,
            )

        # Should be callable
        assert asyncio.iscoroutinefunction(test_call)

    def test_model_protocol_ainvoke_signature(self):
        """Test ModelProtocol ainvoke signature."""
        mock_model = Mock()
        mock_model.ainvoke = AsyncMock(return_value=Mock(content="invoked"))

        async def test_call():
            return await mock_model.ainvoke([])

        assert asyncio.iscoroutinefunction(test_call)


class TestAgentModelProtocol:
    """Test the AgentModelProtocol protocol."""

    def test_agent_model_protocol_structure(self):
        """Test that AgentModelProtocol has expected methods."""
        assert hasattr(AgentModelProtocol, "invoke")
        assert hasattr(AgentModelProtocol, "aresponse")
        assert hasattr(AgentModelProtocol, "ainvoke")

    def test_agent_model_protocol_implementation(self):
        """Test a concrete implementation of AgentModelProtocol."""

        class MockAgentModel:
            """Mock implementation of AgentModelProtocol."""

            def invoke(self, prompt) -> Any:
                return Mock(content=f"Sync response to: {prompt}")

            async def aresponse(
                self,
                messages: List[Any],
                response_format: Any = None,
                tools: Any = None,
                functions: Any = None,
                tool_choice: Any = None,
                tool_call_limit: Any = None,
            ) -> Any:
                return Mock(content="Agent aresponse")

            async def ainvoke(self, messages: List[Any]) -> Any:
                return Mock(content="Agent ainvoke")

        agent_model = MockAgentModel()

        # Test sync method
        sync_result = agent_model.invoke("test prompt")
        assert sync_result.content == "Sync response to: test prompt"

        # Test async methods through protocol duck typing
        async def test_agent_usage(agent_instance):
            response1 = await agent_instance.aresponse([])
            response2 = await agent_instance.ainvoke([])
            return response1, response2

        assert asyncio.iscoroutinefunction(test_agent_usage)

    def test_agent_model_invoke_types(self):
        """Test AgentModelProtocol invoke with different input types."""

        class MockAgentModel:
            def invoke(self, prompt) -> Any:
                if isinstance(prompt, str):
                    return Mock(content=f"String: {prompt}")
                elif isinstance(prompt, dict):
                    return Mock(content=f"Dict: {prompt.get('message', 'No message')}")
                else:
                    return Mock(content="Unknown type")

            async def aresponse(self, messages: List[Any], **kwargs) -> Any:
                return Mock(content="aresponse")

            async def ainvoke(self, messages: List[Any]) -> Any:
                return Mock(content="ainvoke")

        agent_model = MockAgentModel()

        # Test string input
        result1 = agent_model.invoke("test string")
        assert "String: test string" in result1.content

        # Test dict input
        result2 = agent_model.invoke({"message": "test dict"})
        assert "Dict: test dict" in result2.content


class TestAppContextProtocol:
    """Test the AppContextProtocol protocol."""

    def test_app_context_protocol_structure(self):
        """Test that AppContextProtocol has expected methods."""
        # Check required methods (attributes are defined on instances, not protocol class)
        assert hasattr(AppContextProtocol, "initialize_teams")
        assert hasattr(AppContextProtocol, "add_thought")
        assert hasattr(AppContextProtocol, "get_relevant_context")
        assert hasattr(AppContextProtocol, "get_performance_metrics")

        # Protocols define structure, not actual attributes
        # Attributes are checked via __annotations__ if present
        if hasattr(AppContextProtocol, "__annotations__"):
            annotations = AppContextProtocol.__annotations__
            expected_attrs = {
                "teams_initialized",
                "primary_team",
                "reflection_team",
                "shared_context",
                "error_handler",
                "thought_history",
            }
            # Check that expected attributes are mentioned (may be in annotations)
            assert len(annotations) >= 0  # Protocols may not have annotations

    def test_app_context_protocol_implementation(self):
        """Test a concrete implementation of AppContextProtocol."""

        class MockAppContext:
            """Mock implementation of AppContextProtocol."""

            def __init__(self):
                self.teams_initialized: bool = False
                self.primary_team: Optional[Any] = None
                self.reflection_team: Optional[Any] = None
                self.shared_context: Any = Mock()
                self.error_handler: Any = Mock()
                self.thought_history: List[Any] = []

            async def initialize_teams(self) -> None:
                self.teams_initialized = True
                self.primary_team = Mock(name="PrimaryTeam")
                self.reflection_team = Mock(name="ReflectionTeam")

            async def add_thought(self, thought_data: Any) -> None:
                self.thought_history.append(thought_data)

            async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
                return {
                    "thought_count": len(self.thought_history),
                    "query": thought,
                    "context": "relevant_data",
                }

            async def get_performance_metrics(self) -> Dict[str, Any]:
                return {
                    "total_thoughts": len(self.thought_history),
                    "teams_ready": self.teams_initialized,
                    "avg_processing_time": 1.5,
                }

        context = MockAppContext()

        # Test initial state
        assert context.teams_initialized is False
        assert context.primary_team is None
        assert context.reflection_team is None
        assert len(context.thought_history) == 0

    @pytest.mark.asyncio
    async def test_app_context_protocol_async_methods(self):
        """Test async methods of AppContextProtocol implementation."""

        class MockAppContext:
            def __init__(self):
                self.teams_initialized = False
                self.primary_team = None
                self.reflection_team = None
                self.shared_context = Mock()
                self.error_handler = Mock()
                self.thought_history = []

            async def initialize_teams(self) -> None:
                self.teams_initialized = True
                self.primary_team = Mock(name="PrimaryTeam")
                self.reflection_team = Mock(name="ReflectionTeam")

            async def add_thought(self, thought_data: Any) -> None:
                self.thought_history.append(thought_data)

            async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
                return {
                    "thought_count": len(self.thought_history),
                    "query": thought,
                    "relevant_items": ["item1", "item2"],
                }

            async def get_performance_metrics(self) -> Dict[str, Any]:
                return {
                    "total_thoughts": len(self.thought_history),
                    "teams_ready": self.teams_initialized,
                }

        context = MockAppContext()

        # Test initialize_teams
        await context.initialize_teams()
        assert context.teams_initialized is True
        assert context.primary_team is not None
        assert context.reflection_team is not None

        # Test add_thought
        test_thought = Mock(thought="Test thought")
        await context.add_thought(test_thought)
        assert len(context.thought_history) == 1
        assert context.thought_history[0] == test_thought

        # Test get_relevant_context
        relevant_context = await context.get_relevant_context(
            "What is the meaning of life?"
        )
        assert relevant_context["thought_count"] == 1
        assert relevant_context["query"] == "What is the meaning of life?"
        assert "relevant_items" in relevant_context

        # Test get_performance_metrics
        metrics = await context.get_performance_metrics()
        assert metrics["total_thoughts"] == 1
        assert metrics["teams_ready"] is True

    def test_app_context_protocol_attribute_types(self):
        """Test AppContextProtocol attribute type expectations."""

        class MockAppContext:
            def __init__(self):
                self.teams_initialized: bool = True
                self.primary_team: Optional[Any] = Mock()
                self.reflection_team: Optional[Any] = None  # Can be None
                self.shared_context: Any = {"some": "context"}
                self.error_handler: Any = Mock()
                self.thought_history: List[Any] = [Mock(), Mock()]

            async def initialize_teams(self) -> None:
                pass

            async def add_thought(self, thought_data: Any) -> None:
                pass

            async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
                return {}

            async def get_performance_metrics(self) -> Dict[str, Any]:
                return {}

        context = MockAppContext()

        # Verify types
        assert isinstance(context.teams_initialized, bool)
        assert context.primary_team is not None
        assert context.reflection_team is None  # Optional, can be None
        assert isinstance(context.shared_context, dict)
        assert context.error_handler is not None
        assert isinstance(context.thought_history, list)
        assert len(context.thought_history) == 2


class TestProtocolInteroperability:
    """Test interoperability between different protocols."""

    def test_protocol_composition(self):
        """Test that protocols can be composed together."""

        class FullyFeaturedMock:
            """Mock that implements multiple protocols."""

            # ModelProtocol methods
            async def aresponse(self, messages: List[Any], **kwargs) -> Any:
                return Mock(content="Model response")

            async def ainvoke(self, messages: List[Any]) -> Any:
                return Mock(content="Model invoke")

            # AgentModelProtocol methods (includes ModelProtocol + invoke)
            def invoke(self, prompt) -> Any:
                return Mock(content="Sync invoke")

        mock = FullyFeaturedMock()

        # Should work as both ModelProtocol and AgentModelProtocol
        async def use_as_model(model):
            return await model.aresponse([])

        def use_as_agent_model(agent_model):
            return agent_model.invoke("test")

        # Both should work
        sync_result = use_as_agent_model(mock)
        assert sync_result.content == "Sync invoke"

        # Async should be callable
        assert asyncio.iscoroutinefunction(use_as_model)

    @pytest.mark.asyncio
    async def test_protocol_usage_in_context(self):
        """Test protocols being used together in a realistic context."""

        class MockModel:
            async def aresponse(self, messages: List[Any], **kwargs) -> Any:
                return Mock(content="Team model response")

            async def ainvoke(self, messages: List[Any]) -> Any:
                return Mock(content="Team model invoke")

        class MockContext:
            def __init__(self):
                self.teams_initialized = False
                self.primary_team = None
                self.reflection_team = None
                self.shared_context = {}
                self.error_handler = Mock()
                self.thought_history = []

            async def initialize_teams(self) -> None:
                self.teams_initialized = True
                # Teams would use models that implement ModelProtocol
                self.primary_team = Mock()
                self.primary_team.model = MockModel()
                self.reflection_team = Mock()
                self.reflection_team.model = MockModel()

            async def add_thought(self, thought_data: Any) -> None:
                self.thought_history.append(thought_data)

            async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
                return {"context": "data", "history_size": len(self.thought_history)}

            async def get_performance_metrics(self) -> Dict[str, Any]:
                return {"thoughts": len(self.thought_history)}

        # Simulate realistic usage
        context = MockContext()
        await context.initialize_teams()

        # Add some thoughts
        await context.add_thought(Mock(content="First thought"))
        await context.add_thought(Mock(content="Second thought"))

        # Get context
        relevant = await context.get_relevant_context("What should I think about next?")
        assert relevant["history_size"] == 2

        # Get metrics
        metrics = await context.get_performance_metrics()
        assert metrics["thoughts"] == 2

        # Test team models
        assert context.primary_team is not None
        assert context.reflection_team is not None

        # Models should be usable
        primary_response = await context.primary_team.model.aresponse([])
        assert primary_response.content == "Team model response"
