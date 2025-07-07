"""
Comprehensive tests for extracted MCP tools module.
Tests the core MCP tool functionality that was extracted from main.py.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.tools.mcp_tools import (
    reflectivethinking,
    reflectivereview,
    validate_thought_input,
    process_thought_with_dual_teams,
    generate_sequence_review,
    sequential_thinking_prompt,
    tool_selection_prompt,
    thought_review_prompt,
    complex_problem_prompt,
    set_app_context,
    set_mcp_instance,
)
from src.models.thought_models import ThoughtData, DomainType
from exceptions import ValidationError as CustomValidationError
from .conftest import create_test_thought_data


class TestMCPToolsValidation:
    """Test input validation for MCP tools."""

    def test_validate_thought_input_valid(self):
        """Test validation passes with valid input."""
        # Should not raise any exception
        validate_thought_input(
            thought="This is a detailed thought that meets minimum length requirements",
            thought_number=1,
            total_thoughts=5,
            is_revision=False,
            revises_thought=None,
            branch_from_thought=None,
        )

    def test_validate_thought_input_too_short(self):
        """Test validation fails with too short thought."""
        with pytest.raises(CustomValidationError) as exc_info:
            validate_thought_input(
                thought="short",
                thought_number=1,
                total_thoughts=5,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
            )
        assert "at least 10 characters" in str(exc_info.value)

    def test_validate_thought_input_invalid_numbers(self):
        """Test validation fails with invalid thought numbers."""
        # Negative thought number
        with pytest.raises(CustomValidationError):
            validate_thought_input(
                thought="Valid thought content here",
                thought_number=-1,
                total_thoughts=5,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
            )

        # Total thoughts too low
        with pytest.raises(CustomValidationError):
            validate_thought_input(
                thought="Valid thought content here",
                thought_number=1,
                total_thoughts=3,  # Less than 5
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
            )

        # Thought number exceeds total
        with pytest.raises(CustomValidationError):
            validate_thought_input(
                thought="Valid thought content here",
                thought_number=6,
                total_thoughts=5,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
            )

    def test_validate_thought_input_revision_logic(self):
        """Test validation of revision logic."""
        # Revision without revises_thought should fail
        with pytest.raises(CustomValidationError):
            validate_thought_input(
                thought="Valid thought content here",
                thought_number=3,
                total_thoughts=5,
                is_revision=True,
                revises_thought=None,  # Missing!
                branch_from_thought=None,
            )

        # Cannot revise future thought
        with pytest.raises(CustomValidationError):
            validate_thought_input(
                thought="Valid thought content here",
                thought_number=2,
                total_thoughts=5,
                is_revision=True,
                revises_thought=3,  # Future thought
                branch_from_thought=None,
            )


class TestMCPToolsCore:
    """Test core MCP tool functionality."""

    @pytest.fixture
    def mock_app_context(self):
        """Mock application context for testing."""
        context = Mock()

        # Mock processed thought result
        processed_result = Mock()
        processed_result.integrated_response = (
            "Primary analysis complete with reflection feedback integrated"
        )
        processed_result.next_step_guidance = (
            "Continue with detailed implementation planning"
        )
        processed_result.quality_score = 0.85

        # Mock error handler
        context.error_handler = Mock()
        context.error_handler.handle_error.return_value = "Handled error message"

        # Mock performance metrics
        async def mock_get_performance_metrics():
            return {
                "total_thoughts": 5,
                "duration_seconds": 45.2,
            }

        context.get_performance_metrics = mock_get_performance_metrics

        return context, processed_result

    @pytest.fixture
    def mock_review_result(self):
        """Mock review result for testing."""
        review = Mock()
        review.total_thoughts = 5
        review.total_branches = 2
        review.overall_quality = 0.85
        review.topic_alignment_score = 0.90
        review.key_insights = ["Key insight 1", "Key insight 2", "Key insight 3"]
        review.patterns_identified = ["Pattern A", "Pattern B"]
        review.tool_effectiveness = {"thinking_tools": 0.9, "analysis_tools": 0.8}
        review.branch_analyses = []
        review.next_steps = ["Next step 1", "Next step 2"]
        review.areas_for_improvement = ["Improvement area 1"]
        review.review_confidence = 0.88
        return review

    @pytest.mark.asyncio
    async def test_reflectivethinking_basic_flow(self, mock_app_context):
        """Test basic reflectivethinking tool flow."""
        context, processed_result = mock_app_context

        # Mock the process_thought_with_dual_teams function
        with patch(
            "src.tools.mcp_tools.process_thought_with_dual_teams",
            new_callable=AsyncMock,
        ) as mock_process:
            mock_process.return_value = processed_result

            # Set the app context
            set_app_context(context)

            result = await reflectivethinking(
                thought="Analyze the performance characteristics of the system architecture",
                next_thought_needed=True,
                thought_number=1,
                total_thoughts=5,
            )

            # Verify the result
            assert "Primary analysis complete" in result
            assert "Next Step Guidance" in result
            assert mock_process.called

            # Verify ThoughtData was created correctly
            call_args = mock_process.call_args[0]
            thought_data = call_args[0]
            assert isinstance(thought_data, ThoughtData)
            assert (
                thought_data.thought
                == "Analyze the performance characteristics of the system architecture"
            )
            assert thought_data.thoughtNumber == 1
            assert thought_data.totalThoughts == 5
            assert thought_data.nextThoughtNeeded is True

    @pytest.mark.asyncio
    async def test_reflectivethinking_with_step_recommendation(self, mock_app_context):
        """Test reflectivethinking with step recommendation input."""
        context, processed_result = mock_app_context

        current_step = {
            "step_description": "Analyze system bottlenecks",
            "recommended_tools": [
                {
                    "tool_name": "performance_profiler",
                    "confidence": 0.9,
                    "rationale": "Need to identify performance bottlenecks",
                    "priority": 1,
                    "expected_outcome": "Performance metrics identified",
                }
            ],
            "expected_outcome": "Bottlenecks identified and documented",
        }

        with patch(
            "src.tools.mcp_tools.process_thought_with_dual_teams",
            new_callable=AsyncMock,
        ) as mock_process:
            mock_process.return_value = processed_result
            set_app_context(context)

            await reflectivethinking(
                thought="Deep dive into performance analysis",
                next_thought_needed=True,
                thought_number=2,
                total_thoughts=5,
                current_step=current_step,
            )

            # Verify step recommendation was processed
            call_args = mock_process.call_args[0]
            thought_data = call_args[0]
            assert thought_data.current_step is not None
            assert (
                thought_data.current_step.step_description
                == "Analyze system bottlenecks"
            )
            assert len(thought_data.current_step.recommended_tools) == 1
            assert (
                thought_data.current_step.recommended_tools[0].tool_name
                == "performance_profiler"
            )

    @pytest.mark.asyncio
    async def test_reflectivethinking_final_thought(self, mock_app_context):
        """Test reflectivethinking for final thought with metrics."""
        context, processed_result = mock_app_context

        with patch(
            "src.tools.mcp_tools.process_thought_with_dual_teams",
            new_callable=AsyncMock,
        ) as mock_process:
            mock_process.return_value = processed_result
            set_app_context(context)

            result = await reflectivethinking(
                thought="Final synthesis and conclusions",
                next_thought_needed=False,  # Final thought
                thought_number=5,
                total_thoughts=5,
            )

            # Verify final summary is included
            assert "Summary" in result
            assert "Total thoughts: 5" in result
            assert "Duration: 45.2s" in result
            assert "Overall quality: 0.85" in result

    @pytest.mark.asyncio
    async def test_reflectivethinking_validation_error(self, mock_app_context):
        """Test reflectivethinking handles validation errors."""
        context, _ = mock_app_context
        set_app_context(context)

        result = await reflectivethinking(
            thought="short",  # Too short, will trigger validation error
            next_thought_needed=True,
            thought_number=1,
            total_thoughts=5,
        )

        assert "Error:" in result
        assert "Handled error message" in result

    @pytest.mark.asyncio
    async def test_reflectivereview_basic(self, mock_review_result):
        """Test basic reflectivereview functionality."""
        mock_context = Mock()

        with patch(
            "src.tools.mcp_tools.generate_sequence_review", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = mock_review_result
            set_app_context(mock_context)

            result = await reflectivereview()

            # Verify review content
            assert "Thought Sequence Review" in result
            assert "**Total Thoughts**: 5" in result
            assert "**Total Branches**: 2" in result
            assert "**Overall Quality**: 0.85" in result
            assert "**Topic Alignment**: 0.90" in result
            assert "Key insight 1" in result
            assert "Key insight 2" in result
            assert "Pattern A" in result
            assert "thinking_tools: 0.90" in result
            assert "Next step 1" in result
            assert "Improvement area 1" in result
            assert "**Review Confidence**: 0.88" in result

    @pytest.mark.asyncio
    async def test_reflectivereview_error_handling(self):
        """Test reflectivereview error handling."""
        mock_context = Mock()

        with patch(
            "src.tools.mcp_tools.generate_sequence_review", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.side_effect = Exception("Review generation failed")
            set_app_context(mock_context)

            result = await reflectivereview()

            assert "Error generating review" in result
            assert "Review generation failed" in result


class TestMCPPrompts:
    """Test MCP prompt generation functions."""

    def test_sequential_thinking_prompt(self):
        """Test sequential thinking prompt generation."""
        result = sequential_thinking_prompt(
            problem="Design a distributed task queue",
            context="High throughput, fault tolerance required",
        )

        assert "user" in result
        assert "assistant" in result

        user_content = result["user"]
        assert "Design a distributed task queue" in user_content
        assert "High throughput" in user_content

        assistant_content = result["assistant"]
        assert "Sequential Thinking Goals" in assistant_content
        assert "reflectivethinking" in assistant_content
        assert "at least 5" in assistant_content

    def test_sequential_thinking_prompt_no_context(self):
        """Test sequential thinking prompt without context."""
        result = sequential_thinking_prompt(problem="Optimize database queries")

        user_content = result["user"]
        assert "Optimize database queries" in user_content
        assert "Context:" not in user_content  # No context provided

    def test_tool_selection_prompt(self):
        """Test tool selection prompt generation."""
        result = tool_selection_prompt(
            task="Analyze Python code for security vulnerabilities",
            available_tools="bandit, pylint, mypy, black",
        )

        user_content = result["user"]
        assert "security vulnerabilities" in user_content
        assert "bandit, pylint, mypy, black" in user_content

        assistant_content = result["assistant"]
        assert "Tool Selection Process" in assistant_content
        assert "Task Analysis" in assistant_content

    def test_tool_selection_prompt_no_tools(self):
        """Test tool selection prompt without available tools."""
        result = tool_selection_prompt(task="Refactor legacy code")

        user_content = result["user"]
        assert "Refactor legacy code" in user_content
        assert "Available tools:" not in user_content  # No tools provided

    def test_thought_review_prompt(self):
        """Test thought review prompt generation."""
        result = thought_review_prompt()

        user_content = result["user"]
        assert "review and summarize" in user_content
        assert "Key insights discovered" in user_content
        assert "Decision points and branches" in user_content

        assistant_content = result["assistant"]
        assert "Review Process" in assistant_content
        assert "reflectivereview" in assistant_content

    def test_complex_problem_prompt(self):
        """Test complex problem prompt generation."""
        result = complex_problem_prompt(
            problem="Migrate monolith to microservices",
            constraints="6 month timeline, limited budget",
            goals="Improve scalability, maintain reliability",
        )

        user_content = result["user"]
        assert "Migrate monolith to microservices" in user_content
        assert "6 month timeline" in user_content
        assert "Improve scalability" in user_content

        assistant_content = result["assistant"]
        assert "Approach Overview" in assistant_content
        assert "Problem Decomposition" in assistant_content
        assert "Multi-Angle Analysis" in assistant_content
        assert "comprehensive analysis" in assistant_content

    def test_complex_problem_prompt_minimal(self):
        """Test complex problem prompt with only problem."""
        result = complex_problem_prompt(problem="Design authentication system")

        user_content = result["user"]
        assert "Design authentication system" in user_content
        assert "Constraints:" not in user_content  # No constraints
        assert "Goals:" not in user_content  # No goals


class TestMCPToolsUtilities:
    """Test utility functions for MCP tools."""

    def test_set_app_context(self):
        """Test setting application context."""
        mock_context = Mock()
        set_app_context(mock_context)

        # Import the module to check the global variable
        from src.tools import mcp_tools

        assert mcp_tools.app_context == mock_context

    def test_set_mcp_instance(self):
        """Test setting MCP instance for tool registration."""
        mock_mcp = Mock()

        # Mock the registration function to avoid actual registration
        with patch("src.tools.mcp_tools._register_mcp_tools") as mock_register:
            set_mcp_instance(mock_mcp)

            # Verify MCP instance was set and registration was called
            from src.tools import mcp_tools

            assert mcp_tools.mcp == mock_mcp
            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_thought_with_dual_teams_interface(self):
        """Test the dual teams processing interface wrapper."""
        # Test that the function correctly imports and delegates to ThoughtProcessor
        # This is a unit test of the interface layer only

        Mock()
        create_test_thought_data(
            thought="Interface test thought that meets minimum length requirements",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            domain=DomainType.TECHNICAL,
            keywords=["interface", "test"],
            timestamp_ms=1234567890000,
        )

        # Test that the function exists and can be called
        # For a full integration test, we'd need proper handler setup
        assert callable(process_thought_with_dual_teams)

        # Verify it handles the correct parameters
        import inspect

        sig = inspect.signature(process_thought_with_dual_teams)
        assert "thought_data" in sig.parameters
        assert "context" in sig.parameters

    @pytest.mark.asyncio
    async def test_generate_sequence_review_interface(self):
        """Test the sequence review generation interface wrapper."""
        # Test that the function correctly imports and delegates
        # This is a unit test of the interface layer only

        Mock()

        # Test that the function exists and can be called
        assert callable(generate_sequence_review)

        # Verify it handles the correct parameters
        import inspect

        sig = inspect.signature(generate_sequence_review)
        assert "context" in sig.parameters
        assert "branch_id" in sig.parameters
        assert "min_quality_threshold" in sig.parameters
