"""
Comprehensive integration tests for MCP tool endpoints.
Tests the full workflow of reflective thinking, tool selection, and review.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from main import (
    reflectivethinking,
    toolselectthinking,
    reflectivereview,
    EnhancedAppContext as AppContext,
)


class TestMCPEndpoints:
    """Test the MCP tool endpoints for full integration."""

    @pytest.fixture
    async def mock_team_response(self):
        """Mock team response for testing."""
        response = Mock()
        response.content = """
        Based on my analysis:
        
        **Key Insights:**
        1. The problem requires multi-step analysis
        2. We should consider performance implications
        3. Testing strategy is crucial
        
        **Recommended Approach:**
        - Start with requirement analysis
        - Design the architecture
        - Implement core functionality
        - Add comprehensive tests
        
        **Tool Recommendations:**
        - Use ThinkingTools for deep analysis
        - Apply code_analysis for reviewing implementation
        - Leverage test_runner for validation
        
        RECOMMENDATION: Consider branching to explore alternative architectures.
        """
        return response

    @pytest.fixture
    def mock_app_context(self, mock_team_response):
        """Create a mock app context for testing."""
        # AppContext is already imported as EnhancedAppContext

        context = AppContext()

        # Mock the team
        mock_team = AsyncMock()
        mock_team.arun = AsyncMock(return_value=mock_team_response)

        # Patch the team initialization
        with patch.object(context, "initialize_teams", new_callable=AsyncMock):
            context.primary_team = mock_team
            context.reflection_team = mock_team
            context.teams_initialized = True

        return context

    @pytest.mark.asyncio
    async def test_reflectivethinking_basic_flow(self, mock_app_context):
        """Test basic reflective thinking flow."""
        with patch("main.app_context", mock_app_context):
            # First thought
            result = await reflectivethinking(
                thought="Analyze the requirements for a distributed cache system",
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="System Design",
                domain="technical",
            )

            assert "Based on my analysis" in result
            assert "Key Insights" in result
            assert "distributed cache" in result.lower()

            # Verify session was created
            assert mock_app_context.session_id is not None
            assert mock_app_context.session_context is not None

    @pytest.mark.asyncio
    async def test_reflectivethinking_with_revision(self, mock_app_context):
        """Test reflective thinking with revision."""
        with patch("main.app_context", mock_app_context):
            # Initial thought
            await reflectivethinking(
                thought="Design a simple key-value store",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Revision
            result = await reflectivethinking(
                thought="Revise design to include distributed consensus",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
                isRevision=True,
                revisesThought=1,
            )

            assert "Based on my analysis" in result
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() >= 2

    @pytest.mark.asyncio
    async def test_reflectivethinking_with_branching(self, mock_app_context):
        """Test reflective thinking with branching."""
        with patch("main.app_context", mock_app_context):
            # Initial thought
            await reflectivethinking(
                thought="Implement user authentication",
                thoughtNumber=1,
                totalThoughts=4,
                nextThoughtNeeded=True,
            )

            # Branch to explore JWT
            jwt_result = await reflectivethinking(
                thought="Explore JWT-based authentication approach",
                thoughtNumber=2,
                totalThoughts=4,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="jwt-approach",
            )

            # Branch to explore OAuth
            oauth_result = await reflectivethinking(
                thought="Explore OAuth2 authentication approach",
                thoughtNumber=3,
                totalThoughts=4,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="oauth-approach",
            )

            assert "Based on my analysis" in jwt_result
            assert "Based on my analysis" in oauth_result

            # Check graph structure
            graph = mock_app_context.shared_context.thought_graph
            assert graph.number_of_nodes() >= 3
            assert any(
                edge[2].get("branch_id") == "jwt-approach"
                for edge in graph.edges(data=True)
            )

    @pytest.mark.asyncio
    async def test_reflectivethinking_error_handling(self, mock_app_context):
        """Test error handling in reflective thinking."""
        # Test with invalid thought number
        with patch("main.app_context", mock_app_context):
            result = await reflectivethinking(
                thought="Test thought",
                thoughtNumber=0,  # Invalid
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            assert "error" in result.lower()
            assert "thought number must be >= 1" in result.lower()

        # Test with revision without target
        with patch("main.app_context", mock_app_context):
            result = await reflectivethinking(
                thought="Revise something",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
                isRevision=True,
                # Missing revisesThought
            )

            assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_toolselectthinking_basic(self, mock_app_context):
        """Test basic tool selection functionality."""
        with patch("main.app_context", mock_app_context):
            result = await toolselectthinking(
                thought="I need to analyze code performance and find bottlenecks",
                available_tools=[
                    "code_analysis",
                    "profiler",
                    "benchmark",
                    "test_runner",
                ],
                domain="technical",
            )

            assert "Tool Selection Analysis" in result
            assert "Recommended tools" in result
            assert "code_analysis" in result or "profiler" in result

            # Should include confidence scores
            assert "confidence:" in result.lower()

    @pytest.mark.asyncio
    async def test_toolselectthinking_with_context(self, mock_app_context):
        """Test tool selection with context from previous thoughts."""
        with patch("main.app_context", mock_app_context):
            # First, create some context
            await reflectivethinking(
                thought="Optimize database queries",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="Performance",
            )

            # Now select tools with context
            result = await toolselectthinking(
                thought="Select tools to profile and optimize SQL queries",
                domain="technical",
                context={"previous_topic": "Performance", "focus": "database"},
            )

            assert "Tool Selection Analysis" in result
            # Should recommend database/SQL related tools if available
            assert "sql" in result.lower() or "database" in result.lower()

    @pytest.mark.asyncio
    async def test_reflectivereview_basic(self, mock_app_context):
        """Test basic review functionality."""
        with patch("main.app_context", mock_app_context):
            # Create a thinking session
            for i in range(1, 4):
                await reflectivethinking(
                    thought=f"Step {i}: Implement feature component {i}",
                    thoughtNumber=i,
                    totalThoughts=3,
                    nextThoughtNeeded=(i < 3),
                )

            # Review the session
            result = await reflectivereview()

            assert "Thought Sequence Review" in result
            assert "Total thoughts: 3" in result
            assert "Key insights" in result
            assert "Final confidence" in result

    @pytest.mark.asyncio
    async def test_reflectivereview_with_branches(self, mock_app_context):
        """Test review with branched thinking paths."""
        with patch("main.app_context", mock_app_context):
            # Create main path
            await reflectivethinking(
                thought="Design API architecture",
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # Create branches
            await reflectivethinking(
                thought="REST API approach",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="rest-api",
            )

            await reflectivethinking(
                thought="GraphQL API approach",
                thoughtNumber=3,
                totalThoughts=5,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="graphql-api",
            )

            # Review should mention branches
            result = await reflectivereview()

            assert "Branches explored: 2" in result
            assert "rest-api" in result or "graphql-api" in result

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, mock_app_context):
        """Test complete workflow: think -> select tools -> implement -> review."""
        with patch("main.app_context", mock_app_context):
            # 1. Initial analysis
            thought1 = await reflectivethinking(
                thought="Plan implementation of a real-time notification system",
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="System Design",
                domain="technical",
            )

            # 2. Select tools for implementation
            tools = await toolselectthinking(
                thought="Select tools for implementing WebSocket server and message queue",
                available_tools=[
                    "websocket_lib",
                    "redis",
                    "rabbitmq",
                    "kafka",
                    "test_framework",
                ],
                domain="technical",
            )

            # 3. Implementation thoughts
            thought2 = await reflectivethinking(
                thought="Implement WebSocket server with selected tools",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
                keywords=["websocket", "real-time", "implementation"],
            )

            # 4. Testing approach
            thought3 = await reflectivethinking(
                thought="Design comprehensive testing strategy",
                thoughtNumber=3,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # 5. Performance considerations
            thought4 = await reflectivethinking(
                thought="Optimize for scale and latency",
                thoughtNumber=4,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # 6. Final integration
            thought5 = await reflectivethinking(
                thought="Complete integration and deployment plan",
                thoughtNumber=5,
                totalThoughts=5,
                nextThoughtNeeded=False,
            )

            # 7. Review entire session
            review = await reflectivereview()

            # Verify workflow completion
            assert all(
                [thought1, tools, thought2, thought3, thought4, thought5, review]
            )
            assert "Total thoughts: 5" in review
            assert "notification system" in review.lower()
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() == 5

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, mock_app_context):
        """Test handling of concurrent thinking sessions."""
        with patch("main.app_context", mock_app_context):
            # Run multiple thoughts concurrently
            tasks = []
            for i in range(1, 4):
                task = reflectivethinking(
                    thought=f"Concurrent thought {i}",
                    thoughtNumber=i,
                    totalThoughts=3,
                    nextThoughtNeeded=(i < 3),
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert all(isinstance(r, str) and "error" not in r.lower() for r in results)
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() == 3

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Test that different sessions are properly isolated."""
        # Create two separate contexts
        context1 = AppContext()
        context2 = AppContext()

        mock_team = AsyncMock()
        mock_team.arun = AsyncMock(return_value=Mock(content="Test response"))

        with patch.object(context1, "initialize_teams", new_callable=AsyncMock):
            context1.primary_team = mock_team
            context1.reflection_team = mock_team
            context1.teams_initialized = True

        with patch.object(context2, "initialize_teams", new_callable=AsyncMock):
            context2.primary_team = mock_team
            context2.reflection_team = mock_team
            context2.teams_initialized = True

        # Use context1
        with patch("main.app_context", context1):
            await reflectivethinking(
                thought="Session 1 thought",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
            )

        # Use context2
        with patch("main.app_context", context2):
            await reflectivethinking(
                thought="Session 2 thought",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
            )

        # Sessions should have different IDs
        assert context1.session_id != context2.session_id

        # Each should have exactly 1 thought
        assert context1.shared_context.thought_graph.number_of_nodes() == 1
        assert context2.shared_context.thought_graph.number_of_nodes() == 1


class TestMCPPrompts:
    """Test the MCP prompt generation."""

    def test_sequential_thinking_prompt(self):
        """Test sequential thinking prompt generation."""
        from main import sequential_thinking_prompt

        result = sequential_thinking_prompt(
            problem="Design a distributed task queue",
            context="High throughput, fault tolerance required",
        )

        assert len(result) == 1
        assert "messages" in result[0]
        messages = result[0]["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # Check content
        user_content = messages[0]["content"]["text"]
        assert "distributed task queue" in user_content
        assert "High throughput" in user_content

        assistant_content = messages[1]["content"]["text"]
        assert "Sequential Thinking Goals" in assistant_content
        assert "reflectivethinking" in assistant_content

    def test_tool_selection_prompt(self):
        """Test tool selection prompt generation."""
        from main import tool_selection_prompt

        result = tool_selection_prompt(
            task="Analyze Python code for security vulnerabilities",
            available_tools="bandit, pylint, mypy, black",
        )

        assert len(result) == 1
        messages = result[0]["messages"]

        user_content = messages[0]["content"]["text"]
        assert "security vulnerabilities" in user_content
        assert "bandit" in user_content

    def test_thought_review_prompt(self):
        """Test thought review prompt generation."""
        from main import thought_review_prompt

        result = thought_review_prompt(session_id="test-session-123")

        messages = result[0]["messages"]
        user_content = messages[0]["content"]["text"]

        assert "test-session-123" in user_content
        assert "Key insights" in user_content

    def test_complex_problem_prompt(self):
        """Test complex problem prompt generation."""
        from main import complex_problem_prompt

        result = complex_problem_prompt(
            problem="Migrate monolith to microservices",
            constraints="6 month timeline, limited budget",
            goals="Improve scalability, maintain reliability",
        )

        messages = result[0]["messages"]
        user_content = messages[0]["content"]["text"]

        assert "monolith to microservices" in user_content
        assert "6 month timeline" in user_content
        assert "Improve scalability" in user_content


# TODO: Create various scenario tests
# - Test different domain types (technical, creative, analytical, etc.)
# - Test long-running sessions with many thoughts
# - Test complex revision chains
# - Test multiple interleaved branches
# - Test tool selection with no available tools
# - Test review of failed/incomplete sessions

# TODO: Create various edge case tests
# - Test with empty thoughts
# - Test with very long thoughts (>10k characters)
# - Test with special characters and unicode
# - Test with malformed input data
# - Test recovery from team initialization failures
# - Test handling of network timeouts
# - Test memory limits and cleanup
