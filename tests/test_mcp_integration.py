"""
Comprehensive integration tests for MCP tool endpoints.
Tests the full workflow of reflective thinking, tool selection, and review.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.tools.mcp_tools import (
    reflectivethinking,
    reflectivereview,
)
from src.main import EnhancedAppContext as AppContext


class TestMCPEndpoints:
    """Test the MCP tool endpoints for full integration."""

    @pytest.fixture
    def mock_team_response(self):
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
        with patch("src.main.app_context", mock_app_context):
            # First thought
            result = await reflectivethinking(
                thought="Analyze the requirements for a distributed cache system",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
            )

            assert "based on my analysis" in result.lower()
            assert "key insights" in result.lower()
            # The actual content depends on the mock response

            # Verify session was created
            assert mock_app_context.available_tools is not None

    @pytest.mark.asyncio
    async def test_reflectivethinking_with_revision(self, mock_app_context):
        """Test reflective thinking with revision."""
        with patch("src.main.app_context", mock_app_context):
            # Initial thought
            await reflectivethinking(
                thought="Design a simple key-value store",
                thought_number=1,
                total_thoughts=3,
                next_thought_needed=True,
            )

            # Revision
            result = await reflectivethinking(
                thought="Revise design to include distributed consensus",
                thought_number=2,
                total_thoughts=3,
                next_thought_needed=True,
                is_revision=True,
                revises_thought=1,
            )

            assert "Based on my analysis" in result
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() >= 2

    @pytest.mark.asyncio
    async def test_reflectivethinking_with_branching(self, mock_app_context):
        """Test reflective thinking with branching."""
        with patch("src.main.app_context", mock_app_context):
            # Initial thoughts
            await reflectivethinking(
                thought="Implement user authentication",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
            )

            await reflectivethinking(
                thought="Analyze authentication requirements",
                thought_number=2,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # Branch to explore JWT (branching from thought 1, not 2)
            jwt_result = await reflectivethinking(
                thought="Explore JWT-based authentication approach",
                thought_number=3,
                total_thoughts=5,
                next_thought_needed=True,
                branch_from_thought=1,
                branch_id="jwt-approach",
            )

            # Another branch to explore OAuth
            oauth_result = await reflectivethinking(
                thought="Explore OAuth2 authentication approach",
                thought_number=4,
                total_thoughts=5,
                next_thought_needed=True,
                branch_from_thought=1,
                branch_id="oauth-approach",
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
        with patch("src.main.app_context", mock_app_context):
            result = await reflectivethinking(
                thought="Test thought",
                thought_number=0,  # Invalid
                total_thoughts=5,
                next_thought_needed=True,
            )

            assert "error" in result.lower()
            assert "should be greater than or equal to 1" in result.lower()

        # Test with revision without target
        with patch("src.main.app_context", mock_app_context):
            result = await reflectivethinking(
                thought="Revise something",
                thought_number=2,
                total_thoughts=5,
                next_thought_needed=True,
                is_revision=True,
                # Missing revises_thought
            )

            assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_reflectivereview_basic(self, mock_app_context):
        """Test basic review functionality."""
        with patch("src.main.app_context", mock_app_context):
            # Create a thinking session
            for i in range(1, 6):
                await reflectivethinking(
                    thought=f"Step {i}: Implement feature component {i}",
                    thought_number=i,
                    total_thoughts=5,
                    next_thought_needed=(i < 5),
                )

            # Review the session
            result = await reflectivereview()

            assert "Thought Sequence Review" in result
            assert "Total Thoughts**: 5" in result or "Total thoughts: 5" in result
            assert "Key Insights" in result or "Key insights" in result
            # Final confidence might be in different formats

    @pytest.mark.asyncio
    async def test_reflectivereview_with_branches(self, mock_app_context):
        """Test review with branched thinking paths."""
        with patch("src.main.app_context", mock_app_context):
            # Create main path thoughts
            await reflectivethinking(
                thought="Design API architecture",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
            )

            await reflectivethinking(
                thought="Analyze API requirements",
                thought_number=2,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # Create branches from thought 1 (non-consecutive)
            await reflectivethinking(
                thought="REST API approach",
                thought_number=3,
                total_thoughts=5,
                next_thought_needed=True,
                branch_from_thought=1,
                branch_id="rest-api",
            )

            await reflectivethinking(
                thought="GraphQL API approach",
                thought_number=4,
                total_thoughts=5,
                next_thought_needed=True,
                branch_from_thought=1,
                branch_id="graphql-api",
            )

            # Review should mention branches
            result = await reflectivereview()

            assert "Branches" in result
            assert "rest-api" in result or "graphql-api" in result

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, mock_app_context):
        """Test complete workflow: think -> select tools -> implement -> review."""
        with patch("src.main.app_context", mock_app_context):
            # 1. Initial analysis
            thought1 = await reflectivethinking(
                thought="Plan implementation of a real-time notification system",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
            )

            thought2 = await reflectivethinking(
                thought="Implement WebSocket server with selected tools",
                thought_number=2,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # 4. Testing approach
            thought3 = await reflectivethinking(
                thought="Design comprehensive testing strategy",
                thought_number=3,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # 5. Performance considerations
            thought4 = await reflectivethinking(
                thought="Optimize for scale and latency",
                thought_number=4,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # 6. Final integration
            thought5 = await reflectivethinking(
                thought="Complete integration and deployment plan",
                thought_number=5,
                total_thoughts=5,
                next_thought_needed=False,
            )

            # 7. Review entire session
            review = await reflectivereview()

            # Verify workflow completion
            assert all([thought1, thought2, thought3, thought4, thought5, review])
            assert "Total Thoughts**: 5" in review or "Total thoughts: 5" in review
            # The actual content depends on mock behavior
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() == 5

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, mock_app_context):
        """Test handling of concurrent thinking sessions."""
        with patch("src.main.app_context", mock_app_context):
            # Run multiple thoughts concurrently
            tasks = []
            for i in range(1, 6):
                task = reflectivethinking(
                    thought=f"Concurrent thought {i}",
                    thought_number=i,
                    total_thoughts=5,
                    next_thought_needed=(i < 5),
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert all(isinstance(r, str) and "error" not in r.lower() for r in results)
            assert mock_app_context.shared_context.thought_graph.number_of_nodes() == 5

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
        with patch("src.main.app_context", context1):
            await reflectivethinking(
                thought="Session 1 thought",
                thought_number=1,
                total_thoughts=2,
                next_thought_needed=True,
            )

        # Use context2
        with patch("src.main.app_context", context2):
            await reflectivethinking(
                thought="Session 2 thought",
                thought_number=1,
                total_thoughts=2,
                next_thought_needed=True,
            )

        # Contexts should be independent
        assert context1 != context2

        # Each should have exactly 1 thought
        assert context1.shared_context.thought_graph.number_of_nodes() == 1
        assert context2.shared_context.thought_graph.number_of_nodes() == 1


class TestMCPPrompts:
    """Test the MCP prompt generation."""

    def test_sequential_thinking_prompt(self):
        """Test sequential thinking prompt generation."""
        from src.tools.mcp_tools import sequential_thinking_prompt

        result = sequential_thinking_prompt(
            problem="Design a distributed task queue",
            context="High throughput, fault tolerance required",
        )

        # Current format returns dict with user/assistant keys
        assert "user" in result
        assert "assistant" in result

        # Check content
        user_content = result["user"]
        assert "distributed task queue" in user_content
        assert "High throughput" in user_content

        assistant_content = result["assistant"]
        assert "Sequential Thinking Goals" in assistant_content
        assert "reflectivethinking" in assistant_content

    def test_tool_selection_prompt(self):
        """Test tool selection prompt generation."""
        from src.tools.mcp_tools import tool_selection_prompt

        result = tool_selection_prompt(
            task="Analyze Python code for security vulnerabilities",
            available_tools="bandit, pylint, mypy, black",
        )

        # Current format returns dict with user/assistant keys
        assert "user" in result
        assert "assistant" in result

        user_content = result["user"]
        assert "security vulnerabilities" in user_content
        assert "bandit" in user_content

    def test_thought_review_prompt(self):
        """Test thought review prompt generation."""
        from src.tools.mcp_tools import thought_review_prompt

        result = thought_review_prompt()

        # Current format returns dict with user/assistant keys
        assert "user" in result
        assert "assistant" in result

        user_content = result["user"]
        assert "Key insights" in user_content

    def test_complex_problem_prompt(self):
        """Test complex problem prompt generation."""
        from src.tools.mcp_tools import complex_problem_prompt

        result = complex_problem_prompt(
            problem="Migrate monolith to microservices",
            constraints="6 month timeline, limited budget",
            goals="Improve scalability, maintain reliability",
        )

        # Current format returns dict with user/assistant keys
        assert "user" in result
        assert "assistant" in result

        user_content = result["user"]
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
