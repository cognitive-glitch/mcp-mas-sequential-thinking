"""
Tests for AsyncTeam class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.team.async_team import (
    AsyncTeam,
    MockResponse,
    AgentExecutionError,
    TeamExecutionError,
)


class MockAgent:  # type: ignore
    """Mock agent for testing."""

    def __init__(self, name: str, model=None):
        self.name = name
        self.model = model
        self.role = f"test-{name}"


class MockModel:
    """Mock model for testing."""

    def __init__(self, response_content: str = "Mock response"):
        self.response_content = response_content

    async def aresponse(self, messages):
        return Mock(content=self.response_content)

    async def ainvoke(self, messages):
        return Mock(content=self.response_content)


class TestMockResponse:
    """Test the MockResponse helper class."""

    def test_mock_response_creation(self):
        """Test MockResponse object creation."""
        content = "Test response content"
        response = MockResponse(content)

        assert response.content == content
        assert hasattr(response, "content")


class TestAsyncTeam:
    """Test the AsyncTeam class."""

    def test_async_team_initialization(self):
        """Test AsyncTeam initialization with all parameters."""
        name = "TestTeam"
        members = [MockAgent("agent1"), MockAgent("agent2")]
        instructions = ["instruction1", "instruction2"]
        model = MockModel()
        max_concurrency = 3
        timeout = 30.0

        team = AsyncTeam(
            name=name,
            members=members,  # type: ignore
            instructions=instructions,
            model=model,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )

        assert team.name == name
        assert team.members == members
        assert team.instructions == instructions
        assert team.model == model
        assert team.max_concurrency == max_concurrency
        assert team.timeout == timeout

    def test_async_team_default_parameters(self):
        """Test AsyncTeam initialization with default parameters."""
        team = AsyncTeam(
            name="DefaultTeam",
            members=[MockAgent("agent1")],  # type: ignore
            instructions=["default instruction"],
            model=MockModel(),
        )

        # Should use default values from constants
        assert team.max_concurrency == 3  # AGENT_CONCURRENCY_LIMIT
        assert team.timeout == 30.0  # DEFAULT_AGENT_TIMEOUT

    @pytest.mark.asyncio
    async def test_arun_with_string_input(self):
        """Test arun method with string input."""
        model = MockModel("Team synthesis response")
        agents = [
            MockAgent("agent1", MockModel("Agent 1 response")),
            MockAgent("agent2", MockModel("Agent 2 response")),
        ]

        team = AsyncTeam(
            name="TestTeam",
            members=agents,  # type: ignore
            instructions=["Analyze the input"],
            model=model,
        )

        result = await team.arun("Test input prompt")

        assert isinstance(result, MockResponse)
        assert "Team synthesis response" in result.content

    @pytest.mark.asyncio
    async def test_arun_with_dict_input(self):
        """Test arun method with dictionary input."""
        model = MockModel("Team dict response")
        agents = [MockAgent("agent1", MockModel("Dict agent response"))]

        team = AsyncTeam(
            name="DictTeam",
            members=agents,  # type: ignore
            instructions=["Process dict input"],
            model=model,
        )

        dict_input = {"task": "analyze", "data": "test data"}
        result = await team.arun(dict_input)

        assert isinstance(result, MockResponse)
        assert "Team dict response" in result.content

    @pytest.mark.asyncio
    async def test_run_agent_safe_with_aresponse(self):
        """Test _run_agent_safe with model that has aresponse method."""
        model = MockModel("Agent aresponse result")
        agent = MockAgent("test_agent", model)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
        )

        # Access the private method for testing
        result = await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert result == "Agent aresponse result"

    @pytest.mark.asyncio
    async def test_run_agent_safe_with_ainvoke(self):
        """Test _run_agent_safe with model that only has ainvoke method."""
        # Create a model with only ainvoke
        model = Mock()
        model.ainvoke = AsyncMock(return_value=Mock(content="Agent ainvoke result"))
        # Remove aresponse to ensure only ainvoke is available
        if hasattr(model, "aresponse"):
            delattr(model, "aresponse")
        agent = MockAgent("test_agent", model)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
        )

        result = await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert result == "Agent ainvoke result"

    @pytest.mark.asyncio
    async def test_run_agent_safe_no_model(self):
        """Test _run_agent_safe with agent that has no model."""
        agent = MockAgent("no_model_agent", None)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert "has no model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_agent_safe_no_async_method(self):
        """Test _run_agent_safe with model that has no async methods."""
        model = Mock()  # No aresponse or ainvoke methods
        agent = MockAgent("no_async_agent", model)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert "object Mock can't be used in 'await' expression" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_agent_safe_timeout(self):
        """Test _run_agent_safe with timeout."""
        # Create a model that takes too long
        model = Mock()

        async def slow_response(messages):
            await asyncio.sleep(1)  # Longer than our timeout
            return Mock(content="Slow response")

        model.aresponse = slow_response
        agent = MockAgent("slow_agent", model)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
            timeout=0.1,  # Very short timeout
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert "Agent slow_agent timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_agent_safe_with_exception(self):
        """Test _run_agent_safe when agent raises an exception."""
        model = Mock()

        async def failing_response(messages):
            raise RuntimeError("Model failure")

        model.aresponse = failing_response
        agent = MockAgent("failing_agent", model)

        team = AsyncTeam(
            name="TestTeam",
            members=[agent],  # type: ignore
            instructions=["test"],
            model=MockModel(),
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            await team._run_agent_safe(agent, "test prompt")  # type: ignore

        assert "Agent failing_agent failed: Model failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_synthesize_responses_with_aresponse(self):
        """Test _synthesize_responses with model that has aresponse."""
        model = MockModel("Synthesized final response")
        team = AsyncTeam(
            name="TestTeam",
            members=[MockAgent("agent1")],  # type: ignore
            instructions=["Synthesize responses"],
            model=model,
        )

        agent_responses = [
            {"agent": "agent1", "content": "Response 1"},
            {"agent": "agent2", "content": "Response 2"},
            {"agent": "agent3", "content": "Response 3"},
        ]
        result = await team._synthesize_responses(agent_responses, "original prompt")

        assert result == "Synthesized final response"

    @pytest.mark.asyncio
    async def test_synthesize_responses_with_ainvoke(self):
        """Test _synthesize_responses with model that only has ainvoke."""
        model = Mock()
        model.ainvoke = AsyncMock(return_value=Mock(content="Ainvoke synthesis"))
        # Ensure aresponse is not available
        if hasattr(model, "aresponse"):
            delattr(model, "aresponse")

        team = AsyncTeam(
            name="TestTeam",
            members=[MockAgent("agent1")],  # type: ignore
            instructions=["Synthesize"],
            model=model,
        )

        agent_responses = [{"agent": "agent1", "content": "Response A"}]
        result = await team._synthesize_responses(agent_responses, "test prompt")

        # AsyncMock works correctly
        assert result == "Ainvoke synthesis"

    @pytest.mark.asyncio
    async def test_synthesize_responses_no_async_method(self):
        """Test _synthesize_responses with model that has no async methods."""
        model = Mock()  # No async methods

        team = AsyncTeam(
            name="TestTeam",
            members=[MockAgent("agent1")],  # type: ignore
            instructions=["Synthesize"],
            model=model,
        )

        agent_responses = [{"agent": "agent1", "content": "Response"}]

        # This should use the fallback - no exception expected
        result = await team._synthesize_responses(agent_responses, "test prompt")

        # Should return fallback formatting
        assert "Team TestTeam Analysis:" in result

    @pytest.mark.asyncio
    async def test_arun_with_concurrency_limit(self):
        """Test arun with concurrency limiting."""
        # Create more agents than the concurrency limit
        agents = [MockAgent(f"agent{i}", MockModel(f"Response {i}")) for i in range(5)]

        team = AsyncTeam(
            name="ConcurrentTeam",
            members=agents,  # type: ignore
            instructions=["Process concurrently"],
            model=MockModel("Final synthesis"),
            max_concurrency=2,  # Limit to 2 concurrent agents
        )

        result = await team.arun("Test concurrent processing")

        assert isinstance(result, MockResponse)
        assert "Final synthesis" in result.content

    @pytest.mark.asyncio
    async def test_arun_with_empty_members(self):
        """Test arun with no team members."""
        team = AsyncTeam(
            name="EmptyTeam",
            members=[],  # type: ignore  # No agents
            instructions=["Process with no agents"],
            model=MockModel("No agent responses to synthesize"),
        )

        # Should raise TeamExecutionError when no agents
        with pytest.raises(TeamExecutionError) as exc_info:
            await team.arun("Test empty team")

        assert "EmptyTeam: All agents failed to process the request" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_arun_with_mixed_agent_failures(self):
        """Test arun when some agents fail and others succeed."""
        successful_agent = MockAgent("good_agent", MockModel("Good response"))
        failing_agent = MockAgent("bad_agent", None)  # No model - will fail

        team = AsyncTeam(
            name="MixedTeam",
            members=[successful_agent, failing_agent],  # type: ignore
            instructions=["Handle mixed results"],
            model=MockModel("Synthesis of available responses"),
        )

        result = await team.arun("Test mixed failures")

        # Should still work with partial responses
        assert isinstance(result, MockResponse)
        assert "Synthesis of available responses" in result.content

    @pytest.mark.asyncio
    async def test_arun_response_content_extraction(self):
        """Test different response content extraction scenarios."""
        # Test with response that has content attribute
        model_with_content = Mock()
        model_with_content.aresponse = AsyncMock(
            return_value=Mock(content="Response with content attr")
        )

        # Test with response that doesn't have content (fallback to str)
        model_without_content = Mock()
        response_obj = Mock()
        del response_obj.content  # Remove content attribute
        response_obj.__str__ = lambda: "String response"
        model_without_content.aresponse = AsyncMock(return_value=response_obj)

        agents = [
            MockAgent("content_agent", model_with_content),
            MockAgent("string_agent", model_without_content),
        ]

        team = AsyncTeam(
            name="ResponseTeam",
            members=agents,  # type: ignore
            instructions=["Test response extraction"],
            model=MockModel("Combined responses"),
        )

        result = await team.arun("Test response types")

        assert isinstance(result, MockResponse)
        assert "Combined responses" in result.content

    def test_agent_execution_error_creation(self):
        """Test AgentExecutionError exception creation."""
        error_msg = "Agent test_agent failed: Test error message"

        exc = AgentExecutionError(error_msg)

        assert error_msg in str(exc)

    def test_agent_execution_error_with_cause(self):
        """Test AgentExecutionError with underlying cause."""
        error_msg = "Agent failing_agent failed: Task failed due to RuntimeError('Underlying error')"

        exc = AgentExecutionError(error_msg)

        error_str = str(exc)
        assert error_msg in error_str

    @pytest.mark.asyncio
    async def test_integration_with_real_workflow(self):
        """Test AsyncTeam integration in a realistic workflow."""
        # Create agents with different response patterns
        planner = MockAgent("planner", MockModel("Planning analysis complete"))
        analyzer = MockAgent("analyzer", MockModel("Technical analysis done"))
        critic = MockAgent("critic", MockModel("Quality review finished"))

        team = AsyncTeam(
            name="IntegrationTeam",
            members=[planner, analyzer, critic],  # type: ignore
            instructions=[
                "Coordinate team analysis",
                "Ensure comprehensive coverage",
                "Maintain quality standards",
            ],
            model=MockModel("Integration team synthesis: All analysis complete"),
            max_concurrency=2,
            timeout=10.0,
        )

        workflow_input = {
            "task": "System architecture review",
            "requirements": ["scalability", "security", "maintainability"],
            "context": "Enterprise application",
        }

        result = await team.arun(workflow_input)

        assert isinstance(result, MockResponse)
        assert "Integration team synthesis" in result.content
        assert len(result.content) > 0
