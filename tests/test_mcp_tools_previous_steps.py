"""
Tests for previous_steps conversion in MCP tools.
Verifies that the TODO task for converting previous_steps has been properly implemented.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.mcp_tools import reflectivethinking
from src.models.tool_models import StepRecommendation, ToolRecommendation


# Mock app_context for all tests
@pytest.fixture
def mock_app_context():
    """Mock application context for testing MCP tools."""
    mock_context = Mock()
    mock_context.error_handler = Mock()
    mock_context.error_handler.handle_error = Mock(return_value="Mock error message")
    mock_context.get_performance_metrics = AsyncMock(
        return_value={
            "total_thoughts": 3,
            "duration_seconds": 5.2,
        }
    )
    return mock_context


class TestPreviousStepsConversion:
    """Test that previous_steps are properly converted from dicts to StepRecommendation objects."""

    @pytest.mark.asyncio
    async def test_previous_steps_conversion_empty(self, mock_app_context):
        """Test that empty previous_steps works correctly."""

        with (
            patch("tools.mcp_tools.process_thought_with_dual_teams") as mock_process,
            patch("tools.mcp_tools.app_context", mock_app_context),
        ):
            # Mock the processing function
            mock_result = Mock()
            mock_result.success = True
            mock_result.integrated_response = "Test response"
            mock_result.next_step_guidance = "Continue"
            mock_result.execution_time_ms = 100
            mock_result.quality_score = 0.8
            mock_process.return_value = mock_result

            await reflectivethinking(
                thought="Test thought with no previous steps",
                next_thought_needed=True,
                thought_number=1,
                total_thoughts=3,
                previous_steps=None,  # No previous steps
            )

            # Verify the function was called and returned a result
            assert mock_process.called
            thought_data = mock_process.call_args[0][
                0
            ]  # First argument should be ThoughtData

            # Verify previous_steps is an empty list when None is passed
            assert thought_data.previous_steps == []

    @pytest.mark.asyncio
    async def test_previous_steps_conversion_with_data(self, mock_app_context):
        """Test that previous_steps are correctly converted from dicts to StepRecommendation objects."""

        # Sample previous steps as dictionaries (as they would come from MCP)
        previous_steps_dicts = [
            {
                "step_description": "Analyze the problem domain",
                "recommended_tools": [
                    {
                        "tool_name": "ThinkingTools",
                        "confidence": 0.9,
                        "rationale": "Deep analysis needed",
                        "priority": 1,
                        "expected_outcome": "Clear understanding",
                    }
                ],
                "expected_outcome": "Domain analysis complete",
                "next_step_conditions": ["If successful, proceed to design"],
                "estimated_complexity": 0.7,
                "dependencies": ["requirements gathering"],
            },
            {
                "step_description": "Design the solution architecture",
                "recommended_tools": [
                    {
                        "tool_name": "ExaTools",
                        "confidence": 0.8,
                        "rationale": "Research architectural patterns",
                        "priority": 1,
                        "expected_outcome": "Pattern recommendations",
                    }
                ],
                "expected_outcome": "Architecture design ready",
                "next_step_conditions": ["Review with team"],
                "estimated_complexity": 0.8,
                "dependencies": ["problem analysis"],
            },
        ]

        with (
            patch("tools.mcp_tools.process_thought_with_dual_teams") as mock_process,
            patch("tools.mcp_tools.app_context", mock_app_context),
        ):
            # Mock the processing function
            mock_result = Mock()
            mock_result.success = True
            mock_result.integrated_response = "Test response with previous steps"
            mock_result.next_step_guidance = "Continue with implementation"
            mock_result.execution_time_ms = 150
            mock_result.quality_score = 0.85
            mock_process.return_value = mock_result

            await reflectivethinking(
                thought="Test thought building on previous work",
                next_thought_needed=True,
                thought_number=3,
                total_thoughts=5,
                previous_steps=previous_steps_dicts,
            )

            # Verify the function was called
            assert mock_process.called
            thought_data = mock_process.call_args[0][
                0
            ]  # First argument should be ThoughtData

            # Verify previous_steps were converted correctly
            assert len(thought_data.previous_steps) == 2

            # Check first previous step
            step1 = thought_data.previous_steps[0]
            assert isinstance(step1, StepRecommendation)
            assert step1.step_description == "Analyze the problem domain"
            assert step1.expected_outcome == "Domain analysis complete"
            assert step1.estimated_complexity == 0.7
            assert step1.dependencies == ["requirements gathering"]
            assert len(step1.next_step_conditions) == 1
            assert step1.next_step_conditions[0] == "If successful, proceed to design"

            # Check tool recommendations in first step
            assert len(step1.recommended_tools) == 1
            tool1 = step1.recommended_tools[0]
            assert isinstance(tool1, ToolRecommendation)
            assert tool1.tool_name == "ThinkingTools"
            assert tool1.confidence == 0.9
            assert tool1.rationale == "Deep analysis needed"
            assert tool1.priority == 1

            # Check second previous step
            step2 = thought_data.previous_steps[1]
            assert isinstance(step2, StepRecommendation)
            assert step2.step_description == "Design the solution architecture"
            assert step2.expected_outcome == "Architecture design ready"
            assert step2.estimated_complexity == 0.8
            assert step2.dependencies == ["problem analysis"]

            # Check tool recommendations in second step
            assert len(step2.recommended_tools) == 1
            tool2 = step2.recommended_tools[0]
            assert isinstance(tool2, ToolRecommendation)
            assert tool2.tool_name == "ExaTools"
            assert tool2.confidence == 0.8
            assert tool2.rationale == "Research architectural patterns"

    @pytest.mark.asyncio
    async def test_previous_steps_conversion_with_invalid_data(self, mock_app_context):
        """Test that invalid previous_steps are handled gracefully."""

        # Mix of valid and invalid previous steps
        previous_steps_mixed = [
            {
                "step_description": "Valid step",
                "recommended_tools": [
                    {
                        "tool_name": "ThinkingTools",
                        "confidence": 0.9,
                        "rationale": "Analysis needed",
                        "priority": 1,
                        "expected_outcome": "Understanding",
                    }
                ],
                "expected_outcome": "Step complete",
            },
            {
                # Invalid step - missing required fields
                "step_description": "Invalid step - missing expected_outcome"
                # Missing expected_outcome and recommended_tools
            },
            {
                "step_description": "Another valid step",
                "recommended_tools": [],  # Empty tools list is valid
                "expected_outcome": "Second step complete",
                "estimated_complexity": 0.5,
            },
        ]

        with (
            patch("tools.mcp_tools.process_thought_with_dual_teams") as mock_process,
            patch("tools.mcp_tools.app_context", mock_app_context),
        ):
            # Mock the processing function
            mock_result = Mock()
            mock_result.success = True
            mock_result.integrated_response = "Processed with partial previous steps"
            mock_result.next_step_guidance = "Continue despite some invalid steps"
            mock_result.execution_time_ms = 120
            mock_result.quality_score = 0.75
            mock_process.return_value = mock_result

            # Mock logger to capture warnings
            with patch("tools.mcp_tools.logger") as mock_logger:
                await reflectivethinking(
                    thought="Test thought with mixed valid/invalid previous steps",
                    next_thought_needed=True,
                    thought_number=2,
                    total_thoughts=4,
                    previous_steps=previous_steps_mixed,
                )

                # Verify the function was called
                assert mock_process.called
                thought_data = mock_process.call_args[0][0]

                # Should only have converted the valid steps (1st and 3rd)
                assert len(thought_data.previous_steps) == 2

                # Check first valid step
                step1 = thought_data.previous_steps[0]
                assert step1.step_description == "Valid step"
                assert step1.expected_outcome == "Step complete"

                # Check second valid step (originally 3rd in input)
                step2 = thought_data.previous_steps[1]
                assert step2.step_description == "Another valid step"
                assert step2.expected_outcome == "Second step complete"
                assert step2.estimated_complexity == 0.5

                # Verify warning was logged for invalid step
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "Failed to convert previous step" in warning_call

    @pytest.mark.asyncio
    async def test_previous_steps_conversion_empty_tools(self, mock_app_context):
        """Test that previous steps with empty tool lists are handled correctly."""

        previous_steps_no_tools = [
            {
                "step_description": "Step with no tools",
                "recommended_tools": [],  # Empty tools list
                "expected_outcome": "Step completed without specific tools",
                "next_step_conditions": ["Manual review required"],
                "estimated_complexity": 0.3,
            }
        ]

        with (
            patch("tools.mcp_tools.process_thought_with_dual_teams") as mock_process,
            patch("tools.mcp_tools.app_context", mock_app_context),
        ):
            mock_result = Mock()
            mock_result.success = True
            mock_result.integrated_response = "Processed step with no tools"
            mock_result.next_step_guidance = "Continue to next phase"
            mock_result.execution_time_ms = 80
            mock_result.quality_score = 0.7
            mock_process.return_value = mock_result

            await reflectivethinking(
                thought="Test thought with tool-less previous step",
                next_thought_needed=True,
                thought_number=2,
                total_thoughts=3,
                previous_steps=previous_steps_no_tools,
            )

            # Verify conversion worked
            assert mock_process.called
            thought_data = mock_process.call_args[0][0]

            assert len(thought_data.previous_steps) == 1
            step = thought_data.previous_steps[0]
            assert isinstance(step, StepRecommendation)
            assert step.step_description == "Step with no tools"
            assert step.expected_outcome == "Step completed without specific tools"
            assert len(step.recommended_tools) == 0  # Empty tools list
            assert step.estimated_complexity == 0.3

    @pytest.mark.asyncio
    async def test_previous_steps_conversion_minimal_data(self, mock_app_context):
        """Test conversion with minimal required data in previous steps."""

        # Only provide required fields
        previous_steps_minimal = [
            {
                "step_description": "Minimal step data",
                "expected_outcome": "Basic outcome",
                # No recommended_tools, next_step_conditions, etc.
            }
        ]

        with (
            patch("tools.mcp_tools.process_thought_with_dual_teams") as mock_process,
            patch("tools.mcp_tools.app_context", mock_app_context),
        ):
            mock_result = Mock()
            mock_result.success = True
            mock_result.integrated_response = "Processed minimal step"
            mock_result.next_step_guidance = "Proceed with caution"
            mock_result.execution_time_ms = 60
            mock_result.quality_score = 0.6
            mock_process.return_value = mock_result

            await reflectivethinking(
                thought="Test thought with minimal previous step data",
                next_thought_needed=False,
                thought_number=5,  # Must be at least 80% of total_thoughts
                total_thoughts=5,
                previous_steps=previous_steps_minimal,
            )

            # Verify conversion worked with defaults
            assert mock_process.called
            thought_data = mock_process.call_args[0][0]

            assert len(thought_data.previous_steps) == 1
            step = thought_data.previous_steps[0]
            assert isinstance(step, StepRecommendation)
            assert step.step_description == "Minimal step data"
            assert step.expected_outcome == "Basic outcome"

            # Check defaults were applied
            assert len(step.recommended_tools) == 0
            assert len(step.next_step_conditions) == 0
            assert step.estimated_complexity == 0.5  # Default value
            assert len(step.dependencies) == 0
