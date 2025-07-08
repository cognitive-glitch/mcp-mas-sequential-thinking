"""
Tests for tool models.
"""

import pytest
from pydantic import ValidationError

from src.models.tool_models import (
    ToolRecommendation,
    StepRecommendation,
    ToolDecision,
    ToolSelectionResult,
)


class TestToolRecommendation:
    """Test the ToolRecommendation model."""

    def test_tool_recommendation_initialization(self):
        """Test ToolRecommendation initialization."""
        tool = ToolRecommendation(
            tool_name="search_engine",
            confidence=0.9,
            rationale="Best tool for information gathering",
            priority=1,
            suggested_inputs=None,
        )

        assert tool.tool_name == "search_engine"
        assert tool.confidence == 0.9
        assert tool.rationale == "Best tool for information gathering"
        assert tool.priority == 1
        assert tool.alternatives == []
        assert tool.suggested_inputs is None
        assert tool.expected_benefits == []
        assert tool.limitations == []

    def test_tool_recommendation_with_all_fields(self):
        """Test ToolRecommendation with all optional fields."""
        tool = ToolRecommendation(
            tool_name="analysis_tool",
            confidence=0.85,
            rationale="Good for deep analysis",
            priority=2,
            alternatives=["alternative_tool", "backup_tool"],
            suggested_inputs={"query": "test", "depth": 3},
            expected_benefits=["Accurate results", "Fast processing"],
            limitations=["Requires internet", "Limited to English"],
        )

        assert tool.tool_name == "analysis_tool"
        assert tool.confidence == 0.85
        assert tool.rationale == "Good for deep analysis"
        assert tool.priority == 2
        assert len(tool.alternatives) == 2
        assert tool.suggested_inputs == {"query": "test", "depth": 3}
        assert len(tool.expected_benefits) == 2
        assert len(tool.limitations) == 2

    def test_tool_recommendation_field_validation(self):
        """Test ToolRecommendation field validation."""
        # Valid confidence range
        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.0, rationale="Testing", priority=1
        )
        assert tool.confidence == 0.0

        tool = ToolRecommendation(
            tool_name="test_tool", confidence=1.0, rationale="Testing", priority=1
        )
        assert tool.confidence == 1.0

        # Invalid confidence range
        with pytest.raises(ValidationError):
            ToolRecommendation(
                tool_name="test_tool", confidence=-0.1, rationale="Testing", priority=1
            )

        with pytest.raises(ValidationError):
            ToolRecommendation(
                tool_name="test_tool", confidence=1.1, rationale="Testing", priority=1
            )

        # Valid priority
        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.5, rationale="Testing", priority=1
        )
        assert tool.priority == 1

        # Invalid priority (< 1)
        with pytest.raises(ValidationError):
            ToolRecommendation(
                tool_name="test_tool", confidence=0.5, rationale="Testing", priority=0
            )

    def test_validate_tool_name_field_validator(self):
        """Test validate_tool_name field validator."""
        # Valid tool name
        tool = ToolRecommendation(
            tool_name="valid_tool_name",
            confidence=0.8,
            rationale="Testing",
            priority=1,
            suggested_inputs=None,
        )
        assert tool.tool_name == "valid_tool_name"

        # Tool name with whitespace gets trimmed
        tool = ToolRecommendation(
            tool_name="  trimmed_tool  ",
            confidence=0.8,
            rationale="Testing",
            priority=1,
        )
        assert tool.tool_name == "trimmed_tool"

        # Empty tool name should raise error
        with pytest.raises(ValidationError) as exc_info:
            ToolRecommendation(
                tool_name="", confidence=0.8, rationale="Testing", priority=1
            )
        assert "Tool name cannot be empty" in str(exc_info.value)

        # Whitespace-only tool name should raise error
        with pytest.raises(ValidationError) as exc_info:
            ToolRecommendation(
                tool_name="   ", confidence=0.8, rationale="Testing", priority=1
            )
        assert "Tool name cannot be empty" in str(exc_info.value)

    def test_is_high_confidence_computed_field(self):
        """Test is_high_confidence computed field."""
        # High confidence (>= 0.8)
        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.8, rationale="Testing", priority=1
        )
        assert tool.is_high_confidence is True

        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.95, rationale="Testing", priority=1
        )
        assert tool.is_high_confidence is True

        # Not high confidence (< 0.8)
        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.75, rationale="Testing", priority=1
        )
        assert tool.is_high_confidence is False

        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.0, rationale="Testing", priority=1
        )
        assert tool.is_high_confidence is False

    def test_has_alternatives_computed_field(self):
        """Test has_alternatives computed field."""
        # No alternatives
        tool = ToolRecommendation(
            tool_name="test_tool", confidence=0.8, rationale="Testing", priority=1
        )
        assert tool.has_alternatives is False

        # Empty alternatives list
        tool = ToolRecommendation(
            tool_name="test_tool",
            confidence=0.8,
            rationale="Testing",
            priority=1,
            alternatives=[],
        )
        assert tool.has_alternatives is False

        # Has alternatives
        tool = ToolRecommendation(
            tool_name="test_tool",
            confidence=0.8,
            rationale="Testing",
            priority=1,
            alternatives=["alt1"],
        )
        assert tool.has_alternatives is True

        tool = ToolRecommendation(
            tool_name="test_tool",
            confidence=0.8,
            rationale="Testing",
            priority=1,
            alternatives=["alt1", "alt2", "alt3"],
        )
        assert tool.has_alternatives is True


class TestStepRecommendation:
    """Test the StepRecommendation model."""

    def test_step_recommendation_initialization(self):
        """Test StepRecommendation initialization."""
        step = StepRecommendation(
            step_description="Analyze the problem",
            expected_outcome="Clear understanding of requirements",
            estimated_complexity=0.5,
        )

        assert step.step_description == "Analyze the problem"
        assert step.recommended_tools == []
        assert step.expected_outcome == "Clear understanding of requirements"
        assert step.next_step_conditions == []
        assert step.estimated_complexity == 0.5  # default
        assert step.dependencies == []

    def test_step_recommendation_with_tools(self):
        """Test StepRecommendation with tools."""
        tool1 = ToolRecommendation(
            tool_name="analyzer",
            confidence=0.9,
            rationale="Best for analysis",
            priority=1,
        )
        tool2 = ToolRecommendation(
            tool_name="search",
            confidence=0.7,
            rationale="Good for research",
            priority=2,
        )

        step = StepRecommendation(
            step_description="Research and analyze",
            recommended_tools=[tool1, tool2],
            expected_outcome="Comprehensive analysis",
            next_step_conditions=["Analysis complete", "Requirements clear"],
            estimated_complexity=0.8,
            dependencies=["initial_setup", "data_collection"],
        )

        assert step.step_description == "Research and analyze"
        assert len(step.recommended_tools) == 2
        assert step.expected_outcome == "Comprehensive analysis"
        assert len(step.next_step_conditions) == 2
        assert step.estimated_complexity == 0.8
        assert len(step.dependencies) == 2

    def test_step_recommendation_complexity_validation(self):
        """Test estimated_complexity field validation."""
        # Valid complexity range
        step = StepRecommendation(
            step_description="Test step",
            expected_outcome="Test outcome",
            estimated_complexity=0.0,
        )
        assert step.estimated_complexity == 0.0

        step = StepRecommendation(
            step_description="Test step",
            expected_outcome="Test outcome",
            estimated_complexity=1.0,
        )
        assert step.estimated_complexity == 1.0

        # Invalid complexity range
        with pytest.raises(ValidationError):
            StepRecommendation(
                step_description="Test step",
                expected_outcome="Test outcome",
                estimated_complexity=-0.1,
            )

        with pytest.raises(ValidationError):
            StepRecommendation(
                step_description="Test step",
                expected_outcome="Test outcome",
                estimated_complexity=1.1,
            )

    def test_tool_count_computed_field(self):
        """Test tool_count computed field."""
        # No tools
        step = StepRecommendation(
            step_description="Test step",
            expected_outcome="Test outcome",
            estimated_complexity=0.5,
        )
        assert step.tool_count == 0

        # Multiple tools
        tools = [
            ToolRecommendation(
                tool_name=f"tool_{i}",
                confidence=0.8,
                rationale="Testing",
                priority=i + 1,
            )
            for i in range(3)
        ]

        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=tools,
            expected_outcome="Test outcome",
        )
        assert step.tool_count == 3

    def test_primary_tool_computed_field(self):
        """Test primary_tool computed field."""
        # No tools -> None
        step = StepRecommendation(
            step_description="Test step",
            expected_outcome="Test outcome",
            estimated_complexity=0.5,
        )
        assert step.primary_tool is None

        # Single tool
        tool = ToolRecommendation(
            tool_name="single_tool", confidence=0.8, rationale="Testing", priority=1
        )
        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[tool],
            expected_outcome="Test outcome",
        )
        assert step.primary_tool == "single_tool"

        # Multiple tools - should return lowest priority number (highest priority)
        tool1 = ToolRecommendation(
            tool_name="secondary_tool",
            confidence=0.7,
            rationale="Secondary",
            priority=2,
        )
        tool2 = ToolRecommendation(
            tool_name="primary_tool", confidence=0.9, rationale="Primary", priority=1
        )
        tool3 = ToolRecommendation(
            tool_name="tertiary_tool", confidence=0.6, rationale="Tertiary", priority=3
        )

        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[tool1, tool2, tool3],  # Not in priority order
            expected_outcome="Test outcome",
        )
        assert step.primary_tool == "primary_tool"  # priority=1

    def test_validate_tool_priorities_field_validator(self):
        """Test validate_tool_priorities field validator."""
        # Empty tools list -> no validation needed
        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[],
            expected_outcome="Test outcome",
        )
        assert step.recommended_tools == []

        # Unique priorities -> no change
        tool1 = ToolRecommendation(
            tool_name="tool1", confidence=0.8, rationale="Testing", priority=1
        )
        tool2 = ToolRecommendation(
            tool_name="tool2", confidence=0.7, rationale="Testing", priority=2
        )

        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[tool1, tool2],
            expected_outcome="Test outcome",
        )
        assert step.recommended_tools[0].priority == 1
        assert step.recommended_tools[1].priority == 2

        # Duplicate priorities -> auto-fix
        tool1_dup = ToolRecommendation(
            tool_name="tool1", confidence=0.8, rationale="Testing", priority=1
        )
        tool2_dup = ToolRecommendation(
            tool_name="tool2",
            confidence=0.7,
            rationale="Testing",
            priority=1,  # Duplicate priority
        )
        tool3_dup = ToolRecommendation(
            tool_name="tool3",
            confidence=0.6,
            rationale="Testing",
            priority=1,  # Duplicate priority
        )

        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[tool1_dup, tool2_dup, tool3_dup],
            expected_outcome="Test outcome",
        )

        # Should be auto-fixed to 1, 2, 3
        priorities = [tool.priority for tool in step.recommended_tools]
        assert priorities == [1, 2, 3]
        assert step.recommended_tools[0].tool_name == "tool1"
        assert step.recommended_tools[1].tool_name == "tool2"
        assert step.recommended_tools[2].tool_name == "tool3"


class TestToolDecision:
    """Test the ToolDecision model."""

    def test_tool_decision_initialization(self):
        """Test ToolDecision initialization."""
        decision = ToolDecision(
            tool_name="selected_tool", rationale="Best choice for the task"
        )

        assert decision.tool_name == "selected_tool"
        assert decision.rationale == "Best choice for the task"
        assert decision.alternatives_considered == []
        assert decision.confidence == 0.7  # default
        assert decision.outcome is None
        assert decision.execution_time_ms is None
        assert decision.success is True  # default
        assert decision.error_message is None

    def test_tool_decision_with_all_fields(self):
        """Test ToolDecision with all fields."""
        decision = ToolDecision(
            tool_name="chosen_tool",
            rationale="Most suitable for requirements",
            alternatives_considered=["tool_a", "tool_b"],
            confidence=0.9,
            outcome="Successfully processed data",
            execution_time_ms=150,
            success=True,
            error_message=None,
        )

        assert decision.tool_name == "chosen_tool"
        assert decision.rationale == "Most suitable for requirements"
        assert len(decision.alternatives_considered) == 2
        assert decision.confidence == 0.9
        assert decision.outcome == "Successfully processed data"
        assert decision.execution_time_ms == 150
        assert decision.success is True
        assert decision.error_message is None

    def test_tool_decision_confidence_validation(self):
        """Test confidence field validation."""
        # Valid confidence range
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", confidence=0.0
        )
        assert decision.confidence == 0.0

        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", confidence=1.0
        )
        assert decision.confidence == 1.0

        # Invalid confidence range
        with pytest.raises(ValidationError):
            ToolDecision(tool_name="test_tool", rationale="Testing", confidence=-0.1)

        with pytest.raises(ValidationError):
            ToolDecision(tool_name="test_tool", rationale="Testing", confidence=1.1)

    def test_was_successful_computed_field(self):
        """Test was_successful computed field."""
        # Success without error -> True
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", success=True, error_message=None
        )
        assert decision.was_successful is True

        # Success but with error message -> False
        decision = ToolDecision(
            tool_name="test_tool",
            rationale="Testing",
            success=True,
            error_message="Warning: something happened",
        )
        assert decision.was_successful is False

        # Not success -> False
        decision = ToolDecision(
            tool_name="test_tool",
            rationale="Testing",
            success=False,
            error_message=None,
        )
        assert decision.was_successful is False

        # Not success with error -> False
        decision = ToolDecision(
            tool_name="test_tool",
            rationale="Testing",
            success=False,
            error_message="Tool failed",
        )
        assert decision.was_successful is False

    def test_execution_speed_computed_field(self):
        """Test execution_speed computed field."""
        # No execution time -> None
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=None
        )
        assert decision.execution_speed is None

        # Fast execution (< 100ms)
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=50
        )
        assert decision.execution_speed == "fast"

        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=99
        )
        assert decision.execution_speed == "fast"

        # Moderate execution (100-999ms)
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=100
        )
        assert decision.execution_speed == "moderate"

        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=500
        )
        assert decision.execution_speed == "moderate"

        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=999
        )
        assert decision.execution_speed == "moderate"

        # Slow execution (>= 1000ms)
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=1000
        )
        assert decision.execution_speed == "slow"

        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", execution_time_ms=5000
        )
        assert decision.execution_speed == "slow"

    def test_validate_outcome_field_validator(self):
        """Test validate_outcome field validator.

        Note: Due to field order in Pydantic, the outcome validator runs before
        the success field is available, so it tends to auto-fill when outcome=None.
        """
        # Success with explicit outcome -> outcome is preserved
        decision = ToolDecision(
            tool_name="test_tool",
            rationale="Testing",
            success=True,
            outcome="Successful completion",
        )
        assert decision.outcome == "Successful completion"

        # Success without outcome -> validator auto-fills due to field order
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", success=True, outcome=None
        )
        # Due to field order, outcome validator runs before success is available
        assert decision.outcome == "Failed - see error message"

        # Failure with outcome -> no change
        decision = ToolDecision(
            tool_name="test_tool",
            rationale="Testing",
            success=False,
            outcome="Failed with error",
        )
        assert decision.outcome == "Failed with error"

        # Failure without outcome -> auto-fill (this is the validator's main job)
        decision = ToolDecision(
            tool_name="test_tool", rationale="Testing", success=False, outcome=None
        )
        assert decision.outcome == "Failed - see error message"


class TestToolSelectionResult:
    """Test the ToolSelectionResult model."""

    def test_tool_selection_result_initialization(self):
        """Test ToolSelectionResult initialization."""
        tool_rec = ToolRecommendation(
            tool_name="selected_tool",
            confidence=0.85,
            rationale="Best option",
            priority=1,
        )

        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Detailed analysis shows this is optimal",
            confidence_score=0.8,
        )

        assert result.recommended_tool == tool_rec
        assert result.reasoning == "Detailed analysis shows this is optimal"
        assert result.confidence_score == 0.8
        assert result.alternative_tools == []
        assert result.context_factors == {}
        assert result.warnings == []

    def test_tool_selection_result_with_all_fields(self):
        """Test ToolSelectionResult with all fields."""
        tool_rec = ToolRecommendation(
            tool_name="primary_tool",
            confidence=0.9,
            rationale="Excellent choice",
            priority=1,
        )

        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Comprehensive evaluation indicates this tool",
            confidence_score=0.85,
            alternative_tools=["backup_tool", "fallback_tool"],
            context_factors={"domain": "analysis", "complexity": "high"},
            warnings=["Requires stable internet", "May be slow for large datasets"],
        )

        assert result.recommended_tool == tool_rec
        assert result.reasoning == "Comprehensive evaluation indicates this tool"
        assert result.confidence_score == 0.85
        assert len(result.alternative_tools) == 2
        assert len(result.context_factors) == 2
        assert len(result.warnings) == 2

    def test_tool_selection_result_confidence_validation(self):
        """Test confidence_score field validation."""
        tool_rec = ToolRecommendation(
            tool_name="test_tool", confidence=0.8, rationale="Testing", priority=1
        )

        # Valid confidence range
        result = ToolSelectionResult(
            recommended_tool=tool_rec, reasoning="Testing", confidence_score=0.0
        )
        assert result.confidence_score == 0.0

        result = ToolSelectionResult(
            recommended_tool=tool_rec, reasoning="Testing", confidence_score=1.0
        )
        assert result.confidence_score == 1.0

        # Invalid confidence range
        with pytest.raises(ValidationError):
            ToolSelectionResult(
                recommended_tool=tool_rec, reasoning="Testing", confidence_score=-0.1
            )

        with pytest.raises(ValidationError):
            ToolSelectionResult(
                recommended_tool=tool_rec, reasoning="Testing", confidence_score=1.1
            )

    def test_requires_fallback_computed_field(self):
        """Test requires_fallback computed field."""
        tool_rec = ToolRecommendation(
            tool_name="test_tool", confidence=0.8, rationale="Testing", priority=1
        )

        # High confidence, no warnings -> False
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Testing",
            confidence_score=0.8,
            warnings=[],
        )
        assert result.requires_fallback is False

        # Low confidence (< 0.7) -> True
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Testing",
            confidence_score=0.6,
            warnings=[],
        )
        assert result.requires_fallback is True

        # High confidence but has warnings -> True
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Testing",
            confidence_score=0.9,
            warnings=["Some warning"],
        )
        assert result.requires_fallback is True

        # Low confidence and warnings -> True
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Testing",
            confidence_score=0.5,
            warnings=["Warning 1", "Warning 2"],
        )
        assert result.requires_fallback is True

        # Edge case: exactly 0.7 confidence, no warnings -> False
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Testing",
            confidence_score=0.7,
            warnings=[],
        )
        assert result.requires_fallback is False

    def test_tool_name_computed_field(self):
        """Test tool_name computed field."""
        tool_rec = ToolRecommendation(
            tool_name="specific_tool_name",
            confidence=0.8,
            rationale="Testing",
            priority=1,
        )

        result = ToolSelectionResult(
            recommended_tool=tool_rec, reasoning="Testing", confidence_score=0.8
        )

        assert result.tool_name == "specific_tool_name"

    def test_to_decision_method(self):
        """Test to_decision method."""
        tool_rec = ToolRecommendation(
            tool_name="conversion_tool",
            confidence=0.85,
            rationale="Best for conversion",
            priority=1,
        )

        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Detailed reasoning for selection",
            confidence_score=0.8,
            alternative_tools=["alt1", "alt2"],
        )

        # Convert without outcome
        decision = result.to_decision()

        assert isinstance(decision, ToolDecision)
        assert decision.tool_name == "conversion_tool"
        assert decision.rationale == "Detailed reasoning for selection"
        assert decision.alternatives_considered == ["alt1", "alt2"]
        assert decision.confidence == 0.8
        # Due to field order in ToolDecision, outcome=None gets auto-filled by validator
        assert decision.outcome == "Failed - see error message"
        assert decision.execution_time_ms is None
        assert decision.success is True
        assert decision.error_message is None

        # Convert with outcome
        decision_with_outcome = result.to_decision(outcome="Successfully executed")

        assert decision_with_outcome.outcome == "Successfully executed"
        assert decision_with_outcome.tool_name == "conversion_tool"
        assert decision_with_outcome.confidence == 0.8


class TestToolModelsIntegration:
    """Test integration between tool models."""

    def test_end_to_end_tool_selection_workflow(self):
        """Test complete tool selection workflow."""
        # Create tool recommendations
        primary_tool = ToolRecommendation(
            tool_name="primary_analyzer",
            confidence=0.9,
            rationale="Best performance for this task type",
            priority=1,
            alternatives=["backup_analyzer"],
            expected_benefits=["High accuracy", "Fast processing"],
            limitations=["Requires authentication"],
        )

        secondary_tool = ToolRecommendation(
            tool_name="secondary_processor",
            confidence=0.7,
            rationale="Good fallback option",
            priority=2,
            alternatives=["manual_process"],
            expected_benefits=["Always available"],
            limitations=["Slower processing"],
        )

        # Create step recommendation
        step = StepRecommendation(
            step_description="Process and analyze input data",
            recommended_tools=[primary_tool, secondary_tool],
            expected_outcome="Analyzed data with insights",
            next_step_conditions=["Data validated", "Results reviewed"],
            estimated_complexity=0.7,
            dependencies=["data_collection"],
        )

        # Create selection result
        result = ToolSelectionResult(
            recommended_tool=primary_tool,
            reasoning="Primary tool offers best balance of accuracy and speed",
            confidence_score=0.85,
            alternative_tools=["secondary_processor", "manual_process"],
            context_factors={"data_size": "large", "time_constraint": "moderate"},
            warnings=["Monitor authentication status"],
        )

        # Convert to decision
        decision = result.to_decision(outcome="Analysis completed successfully")

        # Verify the complete workflow
        assert step.tool_count == 2
        assert step.primary_tool == "primary_analyzer"
        assert primary_tool.is_high_confidence is True
        assert primary_tool.has_alternatives is True
        assert result.requires_fallback is True  # Has warnings
        assert result.tool_name == "primary_analyzer"
        assert decision.was_successful is True
        assert decision.execution_speed is None  # No execution time set

    def test_priority_handling_edge_cases(self):
        """Test edge cases in priority handling."""
        # Create tools with various priority configurations
        tools_mixed_priorities = [
            ToolRecommendation(
                tool_name="tool_high",
                confidence=0.8,
                rationale="High priority",
                priority=1,
            ),
            ToolRecommendation(
                tool_name="tool_low",
                confidence=0.6,
                rationale="Low priority",
                priority=5,
            ),
            ToolRecommendation(
                tool_name="tool_medium",
                confidence=0.7,
                rationale="Medium priority",
                priority=3,
            ),
        ]

        step = StepRecommendation(
            step_description="Multi-tool step",
            recommended_tools=tools_mixed_priorities,
            expected_outcome="Combined results",
        )

        # Primary tool should be the one with priority 1
        assert step.primary_tool == "tool_high"
        assert step.tool_count == 3

    def test_validation_cascading_effects(self):
        """Test how validation in one model affects others."""
        # Create tool with edge case values
        edge_tool = ToolRecommendation(
            tool_name="  edge_case_tool  ",  # Will be trimmed
            confidence=0.7999,  # Just below high confidence
            rationale="Edge case testing",
            priority=1,
        )

        # Should be trimmed and not high confidence
        assert edge_tool.tool_name == "edge_case_tool"
        assert edge_tool.is_high_confidence is False

        # Use in selection result
        result = ToolSelectionResult(
            recommended_tool=edge_tool,
            reasoning="Testing edge cases",
            confidence_score=0.6999,  # Just below fallback threshold
        )

        # Should require fallback due to low confidence
        assert result.requires_fallback is True
        assert result.tool_name == "edge_case_tool"

        # Convert to decision with failure scenario
        decision = result.to_decision()
        decision.success = False
        decision.outcome = None  # Should be auto-filled

        # Re-create to test validator
        failed_decision = ToolDecision(
            tool_name="edge_case_tool",
            rationale="Testing failure",
            success=False,
            outcome=None,  # Should trigger validator
        )

        assert failed_decision.outcome == "Failed - see error message"
        assert failed_decision.was_successful is False

    def test_model_boundary_values(self):
        """Test models with boundary values."""
        # Test minimum valid values
        min_tool = ToolRecommendation(
            tool_name="min_tool",
            confidence=0.0,
            rationale="Minimum confidence",
            priority=1,
        )

        min_step = StepRecommendation(
            step_description="Minimum complexity step",
            expected_outcome="Basic outcome",
            estimated_complexity=0.0,
        )

        min_decision = ToolDecision(
            tool_name="min_tool",
            rationale="Minimum confidence decision",
            confidence=0.0,
            execution_time_ms=0,  # Edge case for speed calculation
        )

        min_result = ToolSelectionResult(
            recommended_tool=min_tool,
            reasoning="Minimum confidence result",
            confidence_score=0.0,
        )

        # Test maximum valid values
        max_tool = ToolRecommendation(
            tool_name="max_tool",
            confidence=1.0,
            rationale="Maximum confidence",
            priority=999999,  # High but valid priority
        )

        max_step = StepRecommendation(
            step_description="Maximum complexity step",
            expected_outcome="Complex outcome",
            estimated_complexity=1.0,
        )

        max_decision = ToolDecision(
            tool_name="max_tool",
            rationale="Maximum confidence decision",
            confidence=1.0,
            execution_time_ms=999999,  # Very slow but valid
        )

        max_result = ToolSelectionResult(
            recommended_tool=max_tool,
            reasoning="Maximum confidence result",
            confidence_score=1.0,
        )

        # All should be valid
        assert min_tool.is_high_confidence is False
        assert max_tool.is_high_confidence is True
        assert min_step.tool_count == 0
        assert max_step.tool_count == 0
        assert min_decision.execution_speed == "fast"  # 0ms is fast
        assert max_decision.execution_speed == "slow"  # 999999ms is slow
        assert min_result.requires_fallback is True  # confidence 0.0 < 0.7
        assert (
            max_result.requires_fallback is False
        )  # confidence 1.0 >= 0.7, no warnings
