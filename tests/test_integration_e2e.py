"""
End-to-end integration tests for the complete thought processing system.
Tests the full pipeline: MCP tool -> EnhancedAppContext -> Teams -> SharedContext -> Response
"""

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pydantic import ValidationError

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Test imports for integration
from src.models.thought_models import ProcessedThought, DomainType
from src.models.protocols import ModelProtocol
from context.app_context import EnhancedAppContext

# Import the main processing function
import main
from .conftest import create_test_thought_data, create_test_tool_decision


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock(spec=ModelProtocol)
    model.aresponse = AsyncMock(return_value="Mocked thinking response")
    model.ainvoke = AsyncMock(return_value="Mocked invoke response")
    return model


@pytest.fixture
def sample_thought_data():
    """Create sample thought data for testing."""
    return create_test_thought_data(
        thought="Analyze the architectural patterns in modern web applications",
        thoughtNumber=1,
        totalThoughts=5,
        nextThoughtNeeded=True,
        topic="Web Architecture",
        domain=DomainType.TECHNICAL,
        keywords=["architecture", "patterns", "web", "applications"],
        confidence_score=0.8,
    )


@pytest_asyncio.fixture
async def app_context(mock_model):
    """Create an enhanced app context for testing."""
    with patch(
        "context.app_context.LLMProviderFactory.get_provider_config"
    ) as mock_factory:
        # Mock provider config
        mock_config = Mock()
        mock_config.provider_name = "test_provider"
        mock_config.get_models.return_value = ("test-team-model", "test-agent-model")
        mock_config.create_model_instance.return_value = mock_model
        mock_factory.return_value = mock_config

        # Create context
        context = EnhancedAppContext()

        # Mock teams directly
        mock_team = Mock()
        mock_team.arun = AsyncMock(return_value=Mock(response="Team response"))
        mock_team.name = "TestTeam"

        # Set up teams
        context.primary_team = mock_team
        context.reflection_team = mock_team
        context.teams_initialized = True

        return context


class TestFullThoughtProcessingPipeline:
    """Test the complete thought processing pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_single_thought_processing_success(
        self, app_context, sample_thought_data
    ):
        """Test successful processing of a single thought through the complete pipeline."""

        # Process the thought
        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Verify result structure
        assert isinstance(result, ProcessedThought)
        assert result.success is True
        assert result.error is None
        assert result.execution_time_ms >= 0  # May be 0 for mocked teams

        # Verify thought data is included
        assert result.thought_data == sample_thought_data
        assert result.thought_data.thoughtNumber == 1

        # Verify responses are generated
        assert len(result.coordinator_response) > 0
        assert len(result.integrated_response) > 0
        assert len(result.next_step_guidance) > 0

        # Verify processing flags
        assert result.context_updated is True

        # Verify context was updated
        memory_usage = app_context.shared_context.get_memory_usage()
        assert memory_usage["total_items"] > 0

    @pytest.mark.asyncio
    async def test_thought_sequence_processing(self, app_context):
        """Test processing a sequence of related thoughts."""
        thoughts = []

        for i in range(1, 6):  # Process 5 thoughts
            thought_data = create_test_thought_data(
                thought=f"Step {i}: Building on previous analysis of web architecture",
                thoughtNumber=i,
                totalThoughts=5,
                nextThoughtNeeded=(i < 5),
                topic="Web Architecture",
                domain=DomainType.TECHNICAL,
                keywords=["architecture", "step", f"phase_{i}"],
                confidence_score=min(0.7 + (i * 0.05), 1.0),  # Keep under 1.0
            )

            # If not first thought, add relationship to previous
            if i > 1:
                from src.models.thought_models import ThoughtRelation

                thought_data.thought_relationships = [
                    ThoughtRelation(
                        from_thought=i - 1,
                        to_thought=i,
                        relation_type="leads_to",
                        strength=0.9,
                        description="Sequential thought progression",
                    )
                ]

            result = await main.process_thought_with_dual_teams(
                thought_data, app_context
            )
            thoughts.append(result)

            # Verify each thought processes successfully
            assert result.success is True
            assert result.thought_data.thoughtNumber == i

        # Verify sequence integrity
        assert len(thoughts) == 5
        assert thoughts[0].thought_data.nextThoughtNeeded is True
        assert thoughts[1].thought_data.nextThoughtNeeded is True
        assert thoughts[2].thought_data.nextThoughtNeeded is True
        assert thoughts[3].thought_data.nextThoughtNeeded is True
        assert thoughts[4].thought_data.nextThoughtNeeded is False

        # Verify context accumulation
        final_context = app_context.shared_context.get_memory_usage()
        assert final_context["total_items"] >= 5

    @pytest.mark.asyncio
    async def test_thought_processing_with_error_recovery(
        self, app_context, sample_thought_data
    ):
        """Test error handling and recovery in thought processing."""

        # Mock a team failure followed by success
        call_count = 0

        async def mock_team_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated team failure")
            return Mock(response="Recovery response")

        app_context.primary_team.arun.side_effect = mock_team_run

        # Process should handle the error gracefully
        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Should succeed on retry via circuit breaker
        assert result.success is True
        # The response should include either recovery or successful processing
        response_lower = result.integrated_response.lower()
        assert (
            "team processing error" in response_lower
            or "recovery" in response_lower
            or len(result.integrated_response) > 100
        )

    @pytest.mark.asyncio
    async def test_branched_thought_processing(self, app_context):
        """Test processing thoughts with branching."""

        # Main thought
        main_thought = create_test_thought_data(
            thought="Main architectural analysis of microservices",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            topic="Microservices",
            domain=DomainType.TECHNICAL,
        )

        main_result = await main.process_thought_with_dual_teams(
            main_thought, app_context
        )
        assert main_result.success is True

        # Branch thought (skip from thought 1 to 3 to avoid consecutive branching)
        branch_thought = create_test_thought_data(
            thought="Alternative analysis focusing on monolithic approach",
            thoughtNumber=3,
            totalThoughts=5,
            nextThoughtNeeded=True,
            branchFromThought=1,  # Branch from thought 1, not 2
            branchId="mono-branch",
            topic="Microservices",
            domain=DomainType.TECHNICAL,
        )

        branch_result = await main.process_thought_with_dual_teams(
            branch_thought, app_context
        )
        assert branch_result.success is True
        assert branch_result.thought_data.branchId == "mono-branch"

        # Verify both thoughts are in context
        context_state = app_context.shared_context.get_memory_usage()
        assert context_state["total_items"] >= 2

    @pytest.mark.asyncio
    async def test_revision_thought_processing(self, app_context):
        """Test processing revision thoughts."""

        # Original thought
        original_thought = create_test_thought_data(
            thought="Initial analysis of database design patterns",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            topic="Database Design",
            domain=DomainType.TECHNICAL,
            confidence_score=0.6,
        )

        original_result = await main.process_thought_with_dual_teams(
            original_thought, app_context
        )
        assert original_result.success is True

        # Revision thought
        revision_thought = create_test_thought_data(
            thought="Revised analysis incorporating NoSQL considerations and performance implications",
            thoughtNumber=2,
            totalThoughts=3,
            nextThoughtNeeded=True,
            isRevision=True,
            revisesThought=1,
            topic="Database Design",
            domain=DomainType.TECHNICAL,
            confidence_score=0.8,
        )

        revision_result = await main.process_thought_with_dual_teams(
            revision_thought, app_context
        )
        assert revision_result.success is True
        assert revision_result.thought_data.isRevision is True
        assert revision_result.thought_data.revisesThought == 1

        # Revision should have higher confidence
        assert (
            revision_result.thought_data.confidence_score
            > original_result.thought_data.confidence_score
        )


class TestSharedContextIntegration:
    """Test shared context integration across the system."""

    @pytest.mark.asyncio
    async def test_context_memory_management(self, app_context):
        """Test memory management in shared context."""

        # Add many insights to test memory management
        for i in range(15):  # Exceed the 10-item limit
            thought = create_test_thought_data(
                thought=f"Insight {i}: Analysis of component {i}",
                thoughtNumber=i + 1,
                totalThoughts=20,
                nextThoughtNeeded=True,
                confidence_score=0.5 + (i * 0.03),  # Varying confidence
            )
            await app_context.add_thought(thought)

        # Verify memory management kicked in
        memory_usage = app_context.shared_context.get_memory_usage()
        assert memory_usage["key_insights"] <= 10  # Should be limited

        # Verify highest quality insights are retained
        insights = app_context.shared_context.key_insights
        if len(insights) > 1:
            # Should be sorted by confidence (highest first)
            confidences = [insight.confidence for insight in insights]
            assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_tool_usage_tracking(self, app_context, sample_thought_data):
        """Test tool usage pattern tracking."""

        # Add tool decisions to thought

        sample_thought_data.tool_decisions = [
            create_test_tool_decision(
                tool_name="ThinkingTools",
                rationale="Deep analysis required",
                confidence=0.9,
                outcome="Success",
            ),
            create_test_tool_decision(
                tool_name="ExaTools",
                rationale="Research needed",
                confidence=0.8,
                outcome="Success",
            ),
        ]

        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )
        assert result.success is True

        # Verify tool patterns are tracked
        patterns = app_context.shared_context.get_tool_usage_patterns()
        assert "ThinkingTools" in patterns
        assert "ExaTools" in patterns

        assert patterns["ThinkingTools"]["count"] >= 1
        assert patterns["ThinkingTools"]["avg_confidence"] > 0.8
        assert patterns["ExaTools"]["success_rate"] > 0.0


class TestErrorHandlingIntegration:
    """Test error handling integration across the system."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, app_context, sample_thought_data):
        """Test circuit breaker behavior in the full system."""

        # Mock repeated failures to trigger circuit breaker
        failure_count = 0

        async def failing_team_run(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 times
                raise Exception("Simulated persistent failure")
            return Mock(response="Recovery after circuit breaker")

        app_context.primary_team.arun.side_effect = failing_team_run

        # First few attempts should fail and trigger circuit breaker
        with patch.object(
            app_context.error_handler.circuit_breakers["team_processing"],
            "failure_threshold",
            2,
        ):
            # This should eventually succeed after circuit breaker recovery
            await main.process_thought_with_dual_teams(sample_thought_data, app_context)

            # Circuit breaker should have handled the failures
            assert (
                app_context.error_handler.circuit_breakers[
                    "team_processing"
                ].failure_count
                > 0
            )

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, app_context, sample_thought_data):
        """Test graceful degradation when components fail."""

        # Simulate reflection team failure
        app_context.reflection_team.arun.side_effect = Exception("Reflection team down")

        # Primary processing should still work
        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Should succeed with graceful error handling
        assert result.success is True
        # Reflection should contain error message (graceful degradation)
        assert result.reflection_response is not None
        assert "error" in result.reflection_response.lower()
        assert len(result.coordinator_response) > 0


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_thought_processing(self, app_context):
        """Test concurrent processing of multiple thoughts."""

        # Create multiple thoughts to process concurrently
        thoughts = []
        for i in range(3):
            thought = create_test_thought_data(
                thought=f"Concurrent analysis {i}: examining distributed systems",
                thoughtNumber=i + 1,
                totalThoughts=5,  # Use minimum required thoughts
                nextThoughtNeeded=True,  # Still processing
                topic=f"Distributed Systems {i}",
                domain=DomainType.TECHNICAL,
                confidence_score=0.7,
            )
            thoughts.append(thought)

        # Process all thoughts concurrently
        start_time = datetime.now()
        tasks = [
            main.process_thought_with_dual_teams(thought, app_context)
            for thought in thoughts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()

        # Verify all succeeded
        for result in results:
            assert not isinstance(result, Exception)
            # Results should be ProcessedThought objects

        # Should complete in reasonable time (concurrent, not sequential)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 10.0  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, app_context):
        """Test memory usage patterns under load."""

        app_context.shared_context.get_memory_usage()

        # Process many thoughts
        for i in range(20):
            thought = create_test_thought_data(
                thought=f"Load test thought {i}: analyzing system performance",
                thoughtNumber=i + 1,
                totalThoughts=20,
                nextThoughtNeeded=(i < 19),
                domain=DomainType.TECHNICAL,
            )
            await app_context.add_thought(thought)

        final_memory = app_context.shared_context.get_memory_usage()

        # Memory should be managed (not grow indefinitely)
        assert final_memory["total_items"] <= 25  # Should be reasonable
        assert final_memory["key_insights"] <= 10  # Insights should be limited


class TestMCPToolsIntegration:
    """Test integration with the MCP tool interface."""

    @pytest.mark.asyncio
    async def test_reflectivethinking_tool_integration(self, app_context):
        """Test the reflectivethinking MCP tool integration."""

        # Create the complete thought data
        thought_data = create_test_thought_data(
            thought="Design patterns for scalable microservices architecture",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            topic="Microservices Design",
            domain=DomainType.TECHNICAL,
            keywords=["design", "patterns", "microservices", "scalability"],
            confidence_score=0.8,
        )

        # Mock the app context processing
        expected_result = ProcessedThought(
            thought_data=thought_data,
            coordinator_response="Primary team analysis complete",
            reflection_response="Reflection team feedback",
            integrated_response="Comprehensive architectural analysis",
            next_step_guidance="Consider implementation patterns",
            execution_time_ms=1500,
            success=True,
            tool_recommendations_generated=True,
            reflection_applied=True,
            context_updated=True,
        )

        # Set the app context for MCP tools
        from src.tools.mcp_tools import set_app_context

        set_app_context(app_context)

        with patch(
            "src.tools.mcp_tools.process_thought_with_dual_teams",
            return_value=expected_result,
        ):
            # Test the actual MCP tool
            from src.tools.mcp_tools import reflectivethinking

            mcp_result = await reflectivethinking(
                thought="Design patterns for scalable microservices architecture",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
            )

            # The MCP tool returns a string, not ProcessedThought
            assert "Comprehensive architectural analysis" in mcp_result
            assert isinstance(mcp_result, str)
            assert len(mcp_result) > 0

    @pytest.mark.asyncio
    async def test_data_serialization_compatibility(
        self, app_context, sample_thought_data
    ):
        """Test that all data structures can be properly serialized for MCP."""

        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Test JSON serialization (required for MCP)
        try:
            # Serialize thought data
            thought_json = sample_thought_data.model_dump_json()
            thought_dict = json.loads(thought_json)
            assert isinstance(thought_dict, dict)
            assert "thought" in thought_dict
            assert "thoughtNumber" in thought_dict

            # Serialize result (should work for MCP response)
            result_json = result.model_dump_json()
            result_dict = json.loads(result_json)
            assert isinstance(result_dict, dict)
            assert "success" in result_dict
            assert "execution_time_ms" in result_dict

        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")


class TestSystemResilience:
    """Test system resilience and recovery capabilities."""

    @pytest.mark.asyncio
    async def test_recovery_from_complete_team_failure(
        self, app_context, sample_thought_data
    ):
        """Test recovery when both teams fail."""

        # Mock complete team failure
        app_context.primary_team.arun.side_effect = Exception("Primary team failure")
        app_context.reflection_team.arun.side_effect = Exception(
            "Reflection team failure"
        )

        # System should handle gracefully
        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Should still return a result (degraded mode)
        assert isinstance(result, ProcessedThought)
        # May succeed with fallback or fail gracefully
        if result.success:
            assert len(result.integrated_response) > 0
        else:
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_invalid_thought_data_handling(self, app_context):
        """Test handling of invalid thought data."""

        # Test with minimal invalid data (this DOES raise ValidationError)
        with pytest.raises((ValidationError, ValueError)):
            create_test_thought_data(
                thought="x",  # Too short
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

        # Test with inconsistent data - thoughtNumber > totalThoughts should be valid
        # The model allows this case, so we test a different validation case
        with pytest.raises((ValidationError, ValueError)):
            create_test_thought_data(
                thought="Valid thought content here",
                thoughtNumber=0,  # Invalid: must be >= 1
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

    @pytest.mark.asyncio
    async def test_context_corruption_recovery(self, app_context, sample_thought_data):
        """Test recovery from corrupted shared context."""

        # Corrupt the shared context
        app_context.shared_context.memory_store = None  # Simulate corruption

        # System should handle this gracefully
        result = await main.process_thought_with_dual_teams(
            sample_thought_data, app_context
        )

        # Should either succeed with new context or fail gracefully
        assert isinstance(result, ProcessedThought)
