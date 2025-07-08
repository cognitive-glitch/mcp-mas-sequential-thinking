"""
Tests for ThoughtProcessor class - Fixed version.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional, List

from src.handlers.thought_processor import ThoughtProcessor, generate_sequence_review
from src.models.thought_models import (
    ThoughtData,
    ProcessedThought,
    ThoughtSequenceReview,
)
from src.exceptions import TeamProcessingError


class MockAppContext:
    """Mock application context for testing."""

    def __init__(self):
        self.teams_initialized = True
        self.primary_team = Mock()
        self.reflection_team = Mock()
        self.shared_context = Mock()
        self.error_handler = Mock()
        self.thought_history: List[ThoughtData] = []

        # Setup circuit breakers
        self.error_handler.circuit_breakers = {
            "team_processing": Mock(),
            "reflection": Mock(),
        }

        # Make context callable as async methods
        self.add_thought = AsyncMock()
        self.get_relevant_context = AsyncMock(return_value={"test": "context"})

    async def add_thought(self, thought_data: ThoughtData) -> None:
        """Add thought to history."""
        self.thought_history.append(thought_data)

    async def get_relevant_context(self, thought: str) -> dict:
        """Get relevant context for thought."""
        return {"context": "relevant_data", "history": len(self.thought_history)}


def create_valid_thought_data(
    thought_content: str = "This is a valid thought with sufficient length for testing purposes",
    thought_number: int = 1,
    total_thoughts: int = 5,
    next_needed: bool = True,
    branch_id: Optional[str] = None,
    branch_from: Optional[int] = None,
    confidence: float = 0.7,
) -> ThoughtData:
    """Create a valid ThoughtData instance for testing."""
    kwargs = {
        "thought": thought_content,
        "thoughtNumber": thought_number,
        "totalThoughts": total_thoughts,
        "nextThoughtNeeded": next_needed,
        "confidence_score": confidence,
    }

    if branch_id and branch_from:
        kwargs["branchId"] = branch_id
        kwargs["branchFromThought"] = branch_from

    return ThoughtData(**kwargs)


class TestThoughtProcessor:
    """Test the ThoughtProcessor class."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_context = MockAppContext()
        self.processor = ThoughtProcessor(self.mock_context)

    def test_thought_processor_initialization(self):
        """Test ThoughtProcessor initialization."""
        assert self.processor.context == self.mock_context

    @pytest.mark.asyncio
    async def test_process_thought_success(self):
        """Test successful thought processing."""
        # Setup mock responses
        primary_response = Mock(content="Primary team analysis complete")
        reflection_response = Mock(content="Reflection team feedback provided")

        self.mock_context.primary_team.arun = AsyncMock(return_value=primary_response)
        self.mock_context.reflection_team.arun = AsyncMock(
            return_value=reflection_response
        )

        # Create test thought
        thought_data = create_valid_thought_data()

        # Process thought
        result = await self.processor.process_thought(thought_data)

        # Verify result
        assert isinstance(result, ProcessedThought)
        assert result.content == "Primary team analysis complete"
        assert result.confidence_score > 0
        assert result.processing_time > 0

        # Verify context interactions
        self.mock_context.add_thought.assert_called_once()
        self.mock_context.get_relevant_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_thought_no_primary_team(self):
        """Test processing when primary team is not initialized."""
        self.mock_context.primary_team = None

        thought_data = create_valid_thought_data(next_needed=False)

        with pytest.raises(TeamProcessingError) as exc_info:
            await self.processor.process_thought(thought_data)

        assert "Primary team not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_thought_primary_team_failure(self):
        """Test handling of primary team processing failure."""
        self.mock_context.primary_team.arun = AsyncMock(
            side_effect=Exception("Primary team error")
        )

        thought_data = create_valid_thought_data(next_needed=False)

        with pytest.raises(TeamProcessingError) as exc_info:
            await self.processor.process_thought(thought_data)

        assert "Primary team error" in str(exc_info.value)

        # Verify circuit breaker recorded failure
        self.mock_context.error_handler.circuit_breakers[
            "team_processing"
        ].record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_thought_with_reflection_disabled(self):
        """Test processing with reflection disabled."""
        with patch("src.handlers.thought_processor.ENABLE_REFLECTION", False):
            primary_response = Mock(content="Primary only response")
            self.mock_context.primary_team.arun = AsyncMock(
                return_value=primary_response
            )

            thought_data = create_valid_thought_data(next_needed=False)

            result = await self.processor.process_thought(thought_data)

            assert result.content == "Primary only response"
            assert result.reflection_feedback is None

            # Reflection team should not be called
            self.mock_context.reflection_team.arun.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_thought_reflection_failure(self):
        """Test handling of reflection team failure."""
        # Primary team succeeds
        primary_response = Mock(content="Primary success")
        self.mock_context.primary_team.arun = AsyncMock(return_value=primary_response)

        # Reflection team fails
        self.mock_context.reflection_team.arun = AsyncMock(
            side_effect=Exception("Reflection error")
        )

        with patch("src.handlers.thought_processor.ENABLE_REFLECTION", True):
            thought_data = create_valid_thought_data(next_needed=False)

            # Should still return result but without reflection
            result = await self.processor.process_thought(thought_data)

            assert result.content == "Primary success"
            assert result.reflection_feedback is None

    @pytest.mark.asyncio
    async def test_process_thought_no_reflection_team(self):
        """Test processing when reflection team is not available."""
        self.mock_context.reflection_team = None

        primary_response = Mock(content="Primary response")
        self.mock_context.primary_team.arun = AsyncMock(return_value=primary_response)

        with patch("src.handlers.thought_processor.ENABLE_REFLECTION", True):
            thought_data = create_valid_thought_data(next_needed=False)

            result = await self.processor.process_thought(thought_data)

            assert result.content == "Primary response"
            assert result.reflection_feedback is None

    def test_create_next_step_guidance_complete_sequence(self):
        """Test next step guidance for completed sequence."""
        thought_data = create_valid_thought_data(
            thought_number=5, total_thoughts=5, next_needed=False
        )

        primary_response = "Analysis complete"

        guidance = self.processor._create_next_step_guidance(
            thought_data, primary_response
        )

        assert "sequence is complete" in guidance

    def test_create_next_step_guidance_continue_sequence(self):
        """Test next step guidance for continuing sequence."""
        thought_data = create_valid_thought_data(
            thought_number=2, total_thoughts=5, next_needed=True
        )

        primary_response = "Continue analysis"

        guidance = self.processor._create_next_step_guidance(
            thought_data, primary_response
        )

        assert "Continue to thought 3 of 5" in guidance

    def test_create_next_step_guidance_with_tool_recommendations(self):
        """Test next step guidance with tool recommendations."""
        thought_data = create_valid_thought_data(
            thought_number=1, total_thoughts=5, next_needed=True
        )
        # Add current_step with recommended tools
        thought_data.current_step = {"recommended_tools": ["tool1", "tool2"]}

        primary_response = "Use recommended tools"

        guidance = self.processor._create_next_step_guidance(
            thought_data, primary_response
        )

        assert "recommended tool usage" in guidance

    def test_create_next_step_guidance_with_next_steps(self):
        """Test next step guidance when response mentions next steps."""
        thought_data = create_valid_thought_data(
            thought_number=1, total_thoughts=5, next_needed=True
        )

        primary_response = "Analysis shows the next step should focus on optimization"

        guidance = self.processor._create_next_step_guidance(
            thought_data, primary_response
        )

        assert "next steps outlined" in guidance

    def test_create_next_step_guidance_fallback(self):
        """Test next step guidance fallback when no specific guidance available."""
        thought_data = create_valid_thought_data(
            thought_number=1, total_thoughts=5, next_needed=True
        )

        primary_response = "Simple analysis"

        guidance = self.processor._create_next_step_guidance(
            thought_data, primary_response
        )

        assert "Continue to thought 2" in guidance


class TestGenerateSequenceReview:
    """Test the generate_sequence_review function."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_context = MockAppContext()

    @pytest.mark.asyncio
    async def test_generate_sequence_review_basic(self):
        """Test basic sequence review generation."""
        # Add some thoughts to history
        thoughts = [
            create_valid_thought_data(
                thought_content="First comprehensive thought for analysis",
                thought_number=i,
                total_thoughts=5,
                next_needed=i < 5,
                confidence=0.8 + (i * 0.02),
            )
            for i in range(1, 4)
        ]

        self.mock_context.thought_history = thoughts

        review = await generate_sequence_review(self.mock_context)

        assert isinstance(review, ThoughtSequenceReview)
        assert review.totalThoughts == 3
        assert len(review.keyInsights) > 0
        assert review.overallCoherence > 0

    @pytest.mark.asyncio
    async def test_generate_sequence_review_with_branch_filter(self):
        """Test sequence review with branch filtering."""
        # Add thoughts with different branches
        thoughts = [
            create_valid_thought_data(
                thought_content="Main branch comprehensive thought analysis",
                thought_number=1,
                total_thoughts=5,
                next_needed=True,
                branch_id="main",
                branch_from=0,  # Can't use same number, use 0 as base
            ),
            create_valid_thought_data(
                thought_content="Alternative branch comprehensive thought analysis",
                thought_number=2,
                total_thoughts=5,
                next_needed=True,
                branch_id="alternative",
                branch_from=1,
            ),
        ]

        self.mock_context.thought_history = thoughts

        review = await generate_sequence_review(self.mock_context, branch_id="main")

        assert review.totalThoughts == 1  # Only main branch thoughts

    @pytest.mark.asyncio
    async def test_generate_sequence_review_with_quality_threshold(self):
        """Test sequence review with quality threshold filtering."""
        thoughts = [
            create_valid_thought_data(
                thought_content="High quality comprehensive thought with detailed analysis",
                thought_number=1,
                total_thoughts=5,
                confidence=0.9,
            ),
            create_valid_thought_data(
                thought_content="Low quality basic thought with minimal content here",
                thought_number=2,
                total_thoughts=5,
                confidence=0.3,
            ),
            create_valid_thought_data(
                thought_content="Medium quality thought with reasonable analysis depth",
                thought_number=3,
                total_thoughts=5,
                confidence=0.7,
            ),
        ]

        self.mock_context.thought_history = thoughts

        review = await generate_sequence_review(
            self.mock_context, min_quality_threshold=0.6
        )

        # Should only include thoughts with confidence >= 0.6
        assert review.totalThoughts == 2

    @pytest.mark.asyncio
    async def test_generate_sequence_review_empty_history(self):
        """Test sequence review with empty thought history."""
        self.mock_context.thought_history = []

        review = await generate_sequence_review(self.mock_context)

        assert review.totalThoughts == 0
        assert len(review.keyInsights) == 0
        assert review.overallCoherence == 0.0

    @pytest.mark.asyncio
    async def test_generate_sequence_review_branch_detection(self):
        """Test sequence review branch detection."""
        thoughts = [
            create_valid_thought_data(
                thought_content="Main thought with comprehensive analysis content",
                thought_number=1,
                total_thoughts=5,
                next_needed=True,
                branch_id="main",
                branch_from=0,
            ),
            create_valid_thought_data(
                thought_content="Branch A thought with detailed examination of alternatives",
                thought_number=2,
                total_thoughts=5,
                next_needed=True,
                branch_id="branch_a",
                branch_from=1,
            ),
            create_valid_thought_data(
                thought_content="Branch B thought exploring different approaches thoroughly",
                thought_number=3,
                total_thoughts=5,
                next_needed=True,
                branch_id="branch_b",
                branch_from=1,
            ),
        ]

        self.mock_context.thought_history = thoughts

        review = await generate_sequence_review(self.mock_context)

        # Should detect all branches
        expected_branches = {"main", "branch_a", "branch_b"}
        assert set(review.branches) == expected_branches

    @pytest.mark.asyncio
    async def test_generate_sequence_review_coherence_calculation(self):
        """Test sequence review coherence calculation."""
        # High coherence: all high confidence thoughts
        high_coherence_thoughts = [
            create_valid_thought_data(
                thought_content=f"High quality thought number {i} with comprehensive analysis",
                thought_number=i,
                total_thoughts=5,
                next_needed=i < 5,
                confidence=0.9,
            )
            for i in range(1, 4)
        ]

        self.mock_context.thought_history = high_coherence_thoughts

        review = await generate_sequence_review(self.mock_context)

        assert review.overallCoherence >= 0.8  # Should be high

        # Low coherence: mixed confidence thoughts
        low_coherence_thoughts = [
            create_valid_thought_data(
                thought_content=f"Variable quality thought number {i} with different levels",
                thought_number=i,
                total_thoughts=5,
                next_needed=i < 5,
                confidence=0.2 + (i * 0.1),
            )
            for i in range(1, 4)
        ]

        self.mock_context.thought_history = low_coherence_thoughts

        review = await generate_sequence_review(self.mock_context)

        assert review.overallCoherence < 0.6  # Should be lower
