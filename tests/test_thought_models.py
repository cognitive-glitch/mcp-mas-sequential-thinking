"""
Tests for enhanced thought models with tool recommendations and topic alignment.
"""

import pytest
from pydantic import ValidationError

from src.models.thought_models import (
    StepRecommendation,
    DomainType,
    BranchAnalysis,
    ThoughtSequenceReview,
    ProcessedThought,
    ToolRecommendation,
)
from .conftest import create_test_thought_data, create_test_tool_recommendation


class TestThoughtData:
    """Test the enhanced ThoughtData model."""

    def test_basic_thought_creation(self):
        """Test creating a basic thought with required fields."""
        thought = create_test_thought_data(
            thought="Test thought content",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
        )

        assert thought.thought == "Test thought content"
        assert thought.thoughtNumber == 1
        assert thought.totalThoughts == 5  # MIN_TOTAL_THOUGHTS enforces minimum of 5
        assert thought.nextThoughtNeeded is True
        assert thought.domain == DomainType.GENERAL  # default
        assert thought.confidence_score == 0.5  # default

    def test_enhanced_thought_with_topic_alignment(self):
        """Test thought with topic/subject alignment features."""
        thought = create_test_thought_data(
            thought="Analyze system performance",
            thoughtNumber=2,
            totalThoughts=5,
            nextThoughtNeeded=True,
            topic="Performance Analysis",
            subject="System Optimization",
            domain=DomainType.TECHNICAL,
            keywords=["performance", "optimization", "analysis"],
        )

        assert thought.topic == "Performance Analysis"
        assert thought.subject == "System Optimization"
        assert thought.domain == DomainType.TECHNICAL
        assert "performance" in thought.keywords
        assert len(thought.keywords) == 3

    def test_revision_validation(self):
        """Test revision logic validation."""
        # Valid revision
        revision_thought = create_test_thought_data(
            thought="Revised analysis with more detailed insights and improvements",
            thoughtNumber=3,
            totalThoughts=5,
            nextThoughtNeeded=True,
            isRevision=True,
            revisesThought=2,
        )

        assert revision_thought.isRevision is True
        assert revision_thought.revisesThought == 2

        # Invalid revision - revising future thought
        with pytest.raises(ValidationError):
            create_test_thought_data(
                thought="Invalid revision",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
                isRevision=True,
                revisesThought=3,  # Can't revise future thought
            )

    def test_branching_validation(self):
        """Test branching logic validation."""
        # Valid branch
        branch_thought = create_test_thought_data(
            thought="Branch analysis",
            thoughtNumber=4,
            totalThoughts=5,
            nextThoughtNeeded=True,
            branchFromThought=2,
            branchId="performance-branch",
        )

        assert branch_thought.branchFromThought == 2
        assert branch_thought.branchId == "performance-branch"

        # Invalid branch - missing branch ID
        with pytest.raises(ValidationError):
            create_test_thought_data(
                thought="Invalid branch",
                thoughtNumber=4,
                totalThoughts=5,
                nextThoughtNeeded=True,
                branchFromThought=2,
                # Missing branchId
            )

    def test_keyword_validation(self):
        """Test keyword validation and cleaning."""
        thought = create_test_thought_data(
            thought="Test with keywords",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            keywords=[
                "Valid",
                "UPPERCASE",
                "with spaces",
                "123numbers",
                "special@chars",
                "",
            ],
        )

        # Should clean keywords while preserving case and special chars
        assert "Valid" in thought.keywords  # Case preserved
        assert "UPPERCASE" in thought.keywords  # Case preserved
        assert "with spaces" in thought.keywords  # Spaces allowed
        assert "123numbers" in thought.keywords  # Numbers allowed
        assert "special@chars" in thought.keywords  # Special chars allowed
        assert "" not in thought.keywords  # Empty strings removed

    def test_topic_subject_validation(self):
        """Test topic and subject validation."""
        thought = create_test_thought_data(
            thought="Test topic validation",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            topic="  Valid Topic  ",
            subject="",  # Empty subject should become None
        )

        assert thought.topic == "Valid Topic"  # Stripped
        assert thought.subject is None  # Empty string converted to None

    def test_log_format_output(
        self,
        sample_tool_recommendation,
        sample_step_recommendation,
    ):
        """Test the enhanced log format with topic/subject and tools."""
        thought = create_test_thought_data(
            thought="Complex thought with all features",
            thoughtNumber=2,
            totalThoughts=5,
            nextThoughtNeeded=True,
            topic="System Design",
            subject="Architecture Planning",
            domain=DomainType.TECHNICAL,
            keywords=["design", "architecture"],
            current_step=sample_step_recommendation,
            confidence_score=0.8,
        )

        # Test that we can create a thought with all features
        assert thought.topic == "System Design"
        assert thought.subject == "Architecture Planning"
        assert thought.domain == DomainType.TECHNICAL
        assert thought.confidence_score == 0.8
        assert thought.thought == "Complex thought with all features"

    def test_tool_summary(self, sample_step_recommendation):
        """Test tool summary generation."""
        thought = create_test_thought_data(
            thought="Test with tools",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            current_step=sample_step_recommendation,
        )

        # Test that current_step is set correctly
        assert thought.current_step is not None
        assert (
            thought.current_step.step_description
            == "Analyze current algorithm complexity"
        )
        assert len(thought.current_step.recommended_tools) == 1
        assert thought.current_step.recommended_tools[0].tool_name == "code_analysis"
        assert thought.current_step.recommended_tools[0].confidence == 0.9

    def test_alignment_summary(self):
        """Test topic/subject alignment summary."""
        thought = create_test_thought_data(
            thought="Test alignment",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            topic="Performance",
            subject="Optimization",
            domain=DomainType.TECHNICAL,
            keywords=["perf", "opt", "speed", "efficient", "fast", "quick", "extra"],
        )

        # Test that alignment attributes are set correctly
        assert thought.topic == "Performance"
        assert thought.subject == "Optimization"
        assert thought.domain == DomainType.TECHNICAL
        assert len(thought.keywords) == 7  # All keywords present


class TestToolRecommendation:
    """Test the ToolRecommendation model."""

    def test_tool_recommendation_creation(self):
        """Test creating a tool recommendation."""
        tool_rec = create_test_tool_recommendation(
            tool_name="performance_analyzer",
            confidence=0.85,
            rationale="System shows performance bottlenecks",
            priority=1,
            expected_outcome="Identify slow functions",
            alternatives=["profiler", "benchmark_tool"],
        )

        assert tool_rec.tool_name == "performance_analyzer"
        assert tool_rec.confidence == 0.85
        assert tool_rec.priority == 1
        assert len(tool_rec.alternatives) == 2

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        tool_rec = create_test_tool_recommendation(
            tool_name="test_tool",
            confidence=0.5,
            rationale="Test",
            priority=1,
            expected_outcome="Test outcome",
        )
        assert tool_rec.confidence == 0.5

        # Invalid confidence - too high
        with pytest.raises(ValidationError):
            create_test_tool_recommendation(
                tool_name="test_tool",
                confidence=1.5,  # > 1.0
                rationale="Test",
                priority=1,
                expected_outcome="Test outcome",
            )

        # Invalid confidence - negative
        with pytest.raises(ValidationError):
            create_test_tool_recommendation(
                tool_name="test_tool",
                confidence=-0.1,  # < 0.0
                rationale="Test",
                priority=1,
                expected_outcome="Test outcome",
            )


class TestStepRecommendation:
    """Test the StepRecommendation model."""

    def test_step_recommendation_creation(self, sample_tool_recommendation):
        """Test creating a step recommendation."""
        step_rec = StepRecommendation(
            step_description="Analyze system performance",
            recommended_tools=[sample_tool_recommendation],
            expected_outcome="Performance bottlenecks identified",
            dependencies=["requirements_gathered"],
            estimated_complexity=0.7,
        )

        assert step_rec.step_description == "Analyze system performance"
        assert len(step_rec.recommended_tools) == 1
        assert len(step_rec.dependencies) == 1
        assert step_rec.estimated_complexity == 0.7


class TestProcessedThought:
    """Test the ProcessedThought model."""

    def test_processed_thought_creation(self, sample_thought_data):
        """Test creating a processed thought result."""
        # Add current_step to sample_thought_data to allow tool_recommendations_generated
        sample_thought_data.current_step = StepRecommendation(
            step_description="Analyze performance metrics",
            recommended_tools=[
                ToolRecommendation(
                    tool_name="performance_analyzer",
                    confidence=0.9,
                    rationale="Need to measure performance",
                    priority=1,
                    suggested_inputs=None,
                )
            ],
            expected_outcome="Performance analysis complete",
            estimated_complexity=0.7,
        )

        processed = ProcessedThought(
            thought_data=sample_thought_data,
            coordinator_response="Primary team analysis complete",
            reflection_response="Reflection feedback provided",
            integrated_response="Combined analysis with feedback",
            next_step_guidance="Continue with implementation",
            execution_time_ms=1500,
            token_usage={"primary": 150, "reflection": 75},
            success=True,
            tool_recommendations_generated=True,
            reflection_applied=True,
            context_updated=True,
        )

        assert processed.success is True
        assert processed.execution_time_ms == 1500
        assert processed.tool_recommendations_generated is True
        assert processed.reflection_applied is True
        assert processed.context_updated is True


# SessionContext class has been removed


class TestThoughtSequenceReview:
    """Test the ThoughtSequenceReview model."""

    def test_sequence_review_creation(self):
        """Test creating a thought sequence review."""
        BranchAnalysis(
            branchId="test-branch",
            thoughtCount=3,
            avgConfidence=0.75,
            keyThemes=["Insight 1", "Insight 2"],
            divergencePoint=1,
            effectiveness=0.8,
            recommendation="Continue development",
        )

        review = ThoughtSequenceReview(
            totalThoughts=10,
            branches=["test-branch", "main-branch"],
            summary="Comprehensive analysis of thought sequence",
            keyInsights=["Key insight 1", "Key insight 2"],
            strengthsIdentified=["Pattern 1"],
            areasForImprovement=["Area 1"],
            overallCoherence=0.8,
            recommendedNextSteps=["Step 1", "Step 2"],
        )

        assert review.totalThoughts == 10
        assert review.overallCoherence == 0.8
        assert len(review.branches) == 2
        assert "test-branch" in review.branches
        assert len(review.keyInsights) == 2
