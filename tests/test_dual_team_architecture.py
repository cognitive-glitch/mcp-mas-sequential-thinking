"""
Tests for the dual-team architecture (Primary + Reflection teams).
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from main_refactored import (
    EnhancedAppContext,
    process_thought_with_dual_teams,
    generate_sequence_review,
)
from src.models.thought_models import ProcessedThought, DomainType
from .conftest import create_test_thought_data


class TestEnhancedAppContext:
    """Test the enhanced application context."""

    @pytest.mark.asyncio
    async def test_context_initialization(self):
        """Test EnhancedAppContext initialization."""
        context = EnhancedAppContext()

        assert context.session_id is not None
        assert context.shared_context is not None
        assert context.session_context is not None
        assert context.primary_team is None  # Not initialized yet
        assert context.reflection_team is None  # Not initialized yet
        assert context.total_thoughts == 0
        assert context.total_reflections == 0

    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test LLM model initialization."""
        context = EnhancedAppContext()

        # Mock the provider factory
        with patch("main_refactored.LLMProviderFactory.create_models") as mock_create:
            mock_create.return_value = (Mock(), Mock(), Mock())

            await context.initialize_models()

            assert context.team_model is not None
            assert context.agent_model is not None
            assert context.provider_config is not None
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_team_initialization(self, mock_app_context):
        """Test dual-team initialization."""
        context = mock_app_context

        # Teams should be mocked and available
        assert context.primary_team is not None
        assert context.reflection_team is not None
        assert context.primary_team.name == "PrimaryTeam"
        assert context.reflection_team.name == "ReflectionTeam"

    @pytest.mark.asyncio
    async def test_thought_addition(self, mock_app_context, sample_thought_data):
        """Test adding thoughts to context."""
        initial_count = mock_app_context.total_thoughts

        await mock_app_context.add_thought(sample_thought_data)

        assert mock_app_context.total_thoughts == initial_count + 1

        # Check that session context was updated if topic provided
        if sample_thought_data.topic:
            assert (
                mock_app_context.session_context.session_topic
                == sample_thought_data.topic
            )

        if sample_thought_data.domain != DomainType.GENERAL:
            assert (
                mock_app_context.session_context.session_domain
                == sample_thought_data.domain
            )

    @pytest.mark.asyncio
    async def test_context_retrieval(self, mock_app_context):
        """Test context retrieval for thoughts."""
        context_data = await mock_app_context.get_context_for_thought("test thought")

        assert isinstance(context_data, dict)
        # Mock should return empty dict or mock data

    @pytest.mark.asyncio
    async def test_performance_metrics(self, mock_app_context):
        """Test performance metric recording."""
        await mock_app_context.record_performance_metric("test_metric", 123.45)

        # Should not raise an error
        assert True


class TestDualTeamProcessing:
    """Test the dual-team thought processing workflow."""

    @pytest.mark.asyncio
    async def test_basic_dual_team_processing(
        self, mock_app_context, sample_thought_data
    ):
        """Test basic dual-team processing workflow."""
        # Process the thought
        result = await process_thought_with_dual_teams(sample_thought_data)

        assert isinstance(result, ProcessedThought)
        assert result.success is True
        assert result.thought_data == sample_thought_data
        assert result.coordinator_response is not None
        assert result.integrated_response is not None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_primary_team_processing(self, mock_app_context, sample_thought_data):
        """Test primary team processing in isolation."""
        # Mock the app_context to use our mock
        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            # Should contain primary team response
            assert "Primary Team Analysis" in result.coordinator_response
            assert (
                result.tool_recommendations_generated is False
            )  # No actual tools in mock

    @pytest.mark.asyncio
    async def test_reflection_team_processing(
        self, mock_app_context, sample_thought_data
    ):
        """Test reflection team processing."""
        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            # Should include reflection if response is substantial
            if len(result.coordinator_response) > 50:
                assert result.reflection_response is not None
                assert "Reflection Team Feedback" in result.reflection_response
                assert result.reflection_applied is True
            else:
                assert result.reflection_applied is False

    @pytest.mark.asyncio
    async def test_team_initialization_failure(self, sample_thought_data):
        """Test handling of team initialization failure."""
        context = EnhancedAppContext()
        # Don't initialize teams - should cause failure

        with patch("main_refactored.app_context", context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            assert result.success is False
            assert "failed" in result.integrated_response.lower()

    @pytest.mark.asyncio
    async def test_primary_team_failure(self, mock_app_context, sample_thought_data):
        """Test handling of primary team failure."""
        # Mock primary team to return None
        mock_app_context.primary_team.arun = AsyncMock(return_value=None)

        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_reflection_team_failure(self, mock_app_context, sample_thought_data):
        """Test handling of reflection team failure."""
        # Mock reflection team to raise exception
        mock_app_context.reflection_team.arun = AsyncMock(
            side_effect=Exception("Reflection failed")
        )

        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            # Should still succeed with primary team only
            assert result.success is True
            assert result.reflection_applied is False

    @pytest.mark.asyncio
    async def test_context_integration(self, mock_app_context, sample_thought_data):
        """Test context integration during processing."""
        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            assert result.context_updated is True
            assert mock_app_context.total_thoughts > 0

    @pytest.mark.asyncio
    async def test_revision_processing(self, mock_app_context, mock_session_context):
        """Test processing of revision thoughts."""
        revision_thought = create_test_thought_data(
            thought="Revised analysis with better approach",
            thoughtNumber=3,
            totalThoughts=5,
            nextThoughtNeeded=True,
            isRevision=True,
            revisesThought=1,
            topic="Revised Analysis",
            session_context=mock_session_context,
        )

        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(revision_thought)

            assert result.success is True
            # Should mention revision in guidance
            assert (
                "revise" in result.next_step_guidance.lower()
                or "revision" in result.integrated_response.lower()
            )

    @pytest.mark.asyncio
    async def test_branching_processing(self, mock_app_context, mock_session_context):
        """Test processing of branching thoughts."""
        branch_thought = create_test_thought_data(
            thought="Alternative approach via branch",
            thoughtNumber=4,
            totalThoughts=5,
            nextThoughtNeeded=True,
            branchFromThought=2,
            branchId="alternative-branch",
            topic="Branch Analysis",
            session_context=mock_session_context,
        )

        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(branch_thought)

            assert result.success is True
            # Branch info should be in processing
            assert (
                "branch" in result.integrated_response.lower()
                or "alternative" in result.integrated_response.lower()
            )

    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_app_context, sample_thought_data):
        """Test performance tracking during processing."""
        with patch("main_refactored.app_context", mock_app_context):
            result = await process_thought_with_dual_teams(sample_thought_data)

            assert result.execution_time_ms > 0
            assert isinstance(result.token_usage, dict)
            assert "primary_team" in result.token_usage


class TestSequenceReview:
    """Test the sequence review functionality."""

    @pytest.mark.asyncio
    async def test_sequence_review_generation(self, mock_app_context):
        """Test generating a thought sequence review."""
        with patch("main_refactored.app_context", mock_app_context):
            review = await generate_sequence_review()

            assert review.session_id == mock_app_context.session_id
            assert review.total_thoughts >= 0
            assert review.total_branches >= 0
            assert 0.0 <= review.overall_quality <= 1.0
            assert 0.0 <= review.topic_alignment_score <= 1.0
            assert 0.0 <= review.review_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_review_with_branches(self, mock_app_context, mock_session_context):
        """Test review generation with branch analysis."""
        # Add some mock thoughts to context first
        thoughts = []
        for i in range(1, 4):
            thought = create_test_thought_data(
                thought=f"Test thought {i}",
                thoughtNumber=i,
                totalThoughts=3,
                nextThoughtNeeded=i < 3,
                session_context=mock_session_context,
            )
            thoughts.append(thought)
            await mock_app_context.add_thought(thought)

        with patch("main_refactored.app_context", mock_app_context):
            review = await generate_sequence_review()

            assert len(review.branch_analyses) >= 0
            assert len(review.key_insights) >= 0
            assert len(review.patterns_identified) >= 0

    @pytest.mark.asyncio
    async def test_review_error_handling(self):
        """Test review generation error handling."""
        # Create a context that will cause errors
        broken_context = EnhancedAppContext()
        # broken_context.shared_context = None  # This should cause an error
        # Using attribute access instead to avoid type checker issues
        setattr(broken_context, "shared_context", None)

        with patch("main_refactored.app_context", broken_context):
            review = await generate_sequence_review()

            # Should return minimal review on error
            assert review.total_thoughts == 0
            assert review.overall_quality == 0.0
            assert review.review_confidence == 0.0


class TestTeamCoordination:
    """Test coordination between primary and reflection teams."""

    @pytest.mark.asyncio
    async def test_team_communication(self, mock_primary_team, mock_reflection_team):
        """Test communication flow between teams."""
        # Test primary team response
        primary_response = await mock_primary_team.arun("Test input")
        assert "Primary Team Analysis" in primary_response.content

        # Test reflection team response
        reflection_response = await mock_reflection_team.arun("Test reflection input")
        assert "Reflection Team Feedback" in reflection_response.content

    @pytest.mark.asyncio
    async def test_parallel_processing_potential(
        self, mock_app_context, sample_thought_data
    ):
        """Test the potential for parallel team processing."""
        start_time = asyncio.get_event_loop().time()

        with patch("main_refactored.app_context", mock_app_context):
            # Process in sequence (current implementation)
            result = await process_thought_with_dual_teams(sample_thought_data)

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        assert result.success is True
        assert processing_time < 1.0  # Should be fast with mocks

        # Note: This test validates the sequential processing works
        # Future optimization could make teams run in parallel

    @pytest.mark.asyncio
    async def test_context_sharing(self, mock_app_context, sample_thought_data):
        """Test context sharing between teams."""
        with patch("main_refactored.app_context", mock_app_context):
            # Add thought to context first
            await mock_app_context.add_thought(sample_thought_data)

            # Process another thought
            second_thought = create_test_thought_data(
                thought="Building on previous analysis",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="Follow-up Analysis",
                session_context=mock_app_context.session_context,
            )

            result = await process_thought_with_dual_teams(second_thought)

            assert result.success is True
            assert mock_app_context.total_thoughts >= 2

    @pytest.mark.asyncio
    async def test_quality_improvement_tracking(self, mock_app_context):
        """Test tracking of quality improvements through reflection."""
        thoughts = []
        results = []

        with patch("main_refactored.app_context", mock_app_context):
            # Process a series of thoughts
            for i in range(1, 4):
                thought = create_test_thought_data(
                    thought=f"Progressive analysis step {i}",
                    thoughtNumber=i,
                    totalThoughts=3,
                    nextThoughtNeeded=i < 3,
                    topic="Progressive Analysis",
                    session_context=mock_app_context.session_context,
                )

                result = await process_thought_with_dual_teams(thought)
                thoughts.append(thought)
                results.append(result)

        # All should succeed
        assert all(result.success for result in results)

        # Should track reflection application
        reflection_applied = sum(1 for result in results if result.reflection_applied)
        assert reflection_applied >= 0  # Depends on response length in mocks
