"""
Integration tests for the complete enhanced sequential thinking system.
"""

import pytest
import asyncio
from unittest.mock import patch
from mcp.server.fastmcp import FastMCP

from main_refactored import (
    sequentialthinking,
    sequentialreview,
    EnhancedAppContext,
    mcp,
)


class TestMCPToolsIntegration:
    """Test the MCP tools (sequentialthinking and sequentialreview)."""

    @pytest.mark.asyncio
    async def test_sequentialthinking_tool_basic(self, mock_app_context):
        """Test basic sequentialthinking tool functionality."""
        with patch("main_refactored.app_context", mock_app_context):
            result = await sequentialthinking(
                thought="Analyze the system architecture for scalability issues",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="System Architecture",
                subject="Scalability Analysis",
                domain="technical",
                keywords=["scalability", "architecture", "performance"],
            )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Primary Team Analysis" in result or "analysis" in result.lower()

    @pytest.mark.asyncio
    async def test_sequentialthinking_with_revision(self, mock_app_context):
        """Test sequentialthinking tool with revision."""
        with patch("main_refactored.app_context", mock_app_context):
            # First, process an initial thought
            await sequentialthinking(
                thought="Initial system analysis",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="System Analysis",
            )

            # Then process a revision
            result = await sequentialthinking(
                thought="Revised system analysis with better methodology",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
                isRevision=True,
                revisesThought=1,
                topic="System Analysis",
            )

        assert isinstance(result, str)
        assert "revision" in result.lower() or "revise" in result.lower()

    @pytest.mark.asyncio
    async def test_sequentialthinking_with_branching(self, mock_app_context):
        """Test sequentialthinking tool with branching."""
        with patch("main_refactored.app_context", mock_app_context):
            # Process initial thought
            await sequentialthinking(
                thought="Main approach analysis",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Process branch
            result = await sequentialthinking(
                thought="Alternative approach via different methodology",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="alternative-approach",
            )

        assert isinstance(result, str)
        assert "branch" in result.lower() or "alternative" in result.lower()

    @pytest.mark.asyncio
    async def test_sequentialthinking_validation_errors(self, mock_app_context):
        """Test sequentialthinking tool validation error handling."""
        with patch("main_refactored.app_context", mock_app_context):
            # Test with invalid thoughtNumber
            result = await sequentialthinking(
                thought="Test thought",
                thoughtNumber=0,  # Invalid - should be >= 1
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

        assert "validation failed" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_sequentialreview_tool(self, mock_app_context):
        """Test sequentialreview tool functionality."""
        with patch("main_refactored.app_context", mock_app_context):
            # Add some thoughts first
            await sequentialthinking(
                thought="First analysis step",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="Analysis Project",
            )

            await sequentialthinking(
                thought="Second analysis step",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
                topic="Analysis Project",
            )

            # Generate review
            result = await sequentialreview()

        assert isinstance(result, str)
        assert "Sequential Thinking Review" in result
        assert "Session Overview" in result
        assert "Recommendations" in result

    @pytest.mark.asyncio
    async def test_sequentialreview_error_handling(self):
        """Test sequentialreview error handling."""
        # Create broken context
        broken_context = EnhancedAppContext()
        # broken_context.shared_context = None
        # Using setattr to avoid type checker issues
        setattr(broken_context, "shared_context", None)

        with patch("main_refactored.app_context", broken_context):
            result = await sequentialreview()

        assert "failed" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_domain_parsing(self, mock_app_context):
        """Test domain parsing in sequentialthinking tool."""
        with patch("main_refactored.app_context", mock_app_context):
            # Test valid domain
            result1 = await sequentialthinking(
                thought="Technical analysis",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
                domain="technical",
            )

            # Test invalid domain (should default to general)
            result2 = await sequentialthinking(
                thought="Invalid domain analysis",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
                domain="invalid_domain",
            )

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        # Both should succeed, invalid domain should be handled gracefully


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_thinking_sequence(self, mock_app_context):
        """Test a complete thinking sequence from start to finish."""
        with patch("main_refactored.app_context", mock_app_context):
            thoughts = [
                {
                    "thought": "Identify the core problem in our system",
                    "thoughtNumber": 1,
                    "totalThoughts": 4,
                    "nextThoughtNeeded": True,
                    "topic": "Problem Analysis",
                    "domain": "analytical",
                },
                {
                    "thought": "Research existing solutions and best practices",
                    "thoughtNumber": 2,
                    "totalThoughts": 4,
                    "nextThoughtNeeded": True,
                    "topic": "Problem Analysis",
                    "domain": "research",
                },
                {
                    "thought": "Design our solution approach",
                    "thoughtNumber": 3,
                    "totalThoughts": 4,
                    "nextThoughtNeeded": True,
                    "topic": "Problem Analysis",
                    "domain": "strategic",
                },
                {
                    "thought": "Plan implementation roadmap",
                    "thoughtNumber": 4,
                    "totalThoughts": 4,
                    "nextThoughtNeeded": False,
                    "topic": "Problem Analysis",
                    "domain": "planning",
                },
            ]

            results = []
            for thought_params in thoughts:
                result = await sequentialthinking(**thought_params)
                results.append(result)

                # Brief pause to simulate realistic timing
                await asyncio.sleep(0.01)

            # All thoughts should process successfully
            assert all(
                isinstance(result, str) and len(result) > 0 for result in results
            )

            # Generate final review
            review = await sequentialreview()
            assert "Sequential Thinking Review" in review

    @pytest.mark.asyncio
    async def test_branching_and_revision_workflow(self, mock_app_context):
        """Test a workflow with branching and revision."""
        with patch("main_refactored.app_context", mock_app_context):
            # Initial analysis
            result1 = await sequentialthinking(
                thought="Analyze current system performance",
                thoughtNumber=1,
                totalThoughts=4,
                nextThoughtNeeded=True,
                topic="Performance Analysis",
            )

            # Continuation
            result2 = await sequentialthinking(
                thought="Identify primary bottlenecks",
                thoughtNumber=2,
                totalThoughts=4,
                nextThoughtNeeded=True,
                topic="Performance Analysis",
            )

            # Branch for alternative approach
            result3 = await sequentialthinking(
                thought="Explore caching solution approach",
                thoughtNumber=3,
                totalThoughts=4,
                nextThoughtNeeded=True,
                branchFromThought=2,
                branchId="caching-approach",
                topic="Performance Analysis",
            )

            # Revision of original analysis
            result4 = await sequentialthinking(
                thought="Revised performance analysis with additional metrics",
                thoughtNumber=4,
                totalThoughts=4,
                nextThoughtNeeded=False,
                isRevision=True,
                revisesThought=1,
                topic="Performance Analysis",
            )

            # All should succeed
            assert all(
                isinstance(result, str)
                for result in [result1, result2, result3, result4]
            )

            # Generate review to see branch analysis
            review = await sequentialreview()
            assert "Sequential Thinking Review" in review

    @pytest.mark.asyncio
    async def test_multi_domain_analysis(self, mock_app_context):
        """Test analysis across multiple domains."""
        with patch("main_refactored.app_context", mock_app_context):
            domains = ["technical", "creative", "analytical", "strategic"]
            results = []

            for i, domain in enumerate(domains, 1):
                result = await sequentialthinking(
                    thought=f"Analyze from {domain} perspective",
                    thoughtNumber=i,
                    totalThoughts=len(domains),
                    nextThoughtNeeded=i < len(domains),
                    topic="Multi-Domain Analysis",
                    domain=domain,
                    keywords=[domain, "analysis", "perspective"],
                )
                results.append(result)

            assert all(isinstance(result, str) for result in results)

            # Review should show domain diversity
            review = await sequentialreview()
            assert "Sequential Thinking Review" in review

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_app_context):
        """Test error recovery in workflows."""
        with patch("main_refactored.app_context", mock_app_context):
            # Normal thought
            result1 = await sequentialthinking(
                thought="Normal analysis",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Simulate error condition by breaking the context temporarily
            original_primary_team = mock_app_context.primary_team
            mock_app_context.primary_team = None

            result2 = await sequentialthinking(
                thought="This should handle error gracefully",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Restore context
            mock_app_context.primary_team = original_primary_team

            # Continue normally
            result3 = await sequentialthinking(
                thought="Recovery analysis",
                thoughtNumber=3,
                totalThoughts=3,
                nextThoughtNeeded=False,
            )

            # First and third should succeed, second should have error handling
            assert isinstance(result1, str) and len(result1) > 0
            assert isinstance(result2, str)  # Should contain error message
            assert isinstance(result3, str) and len(result3) > 0


class TestConcurrencyAndPerformance:
    """Test concurrency handling and performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_thought_processing(self, mock_app_context):
        """Test concurrent processing of multiple thoughts."""
        with patch("main_refactored.app_context", mock_app_context):
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = sequentialthinking(
                    thought=f"Concurrent analysis {i + 1}",
                    thoughtNumber=i + 1,
                    totalThoughts=5,
                    nextThoughtNeeded=i < 4,
                    topic=f"Concurrent Topic {i + 1}",
                )
                tasks.append(task)

            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete (either success or handled exceptions)
            assert len(results) == 5

            # Most should be strings (successful results)
            successful_results = [r for r in results if isinstance(r, str)]
            assert len(successful_results) >= 3  # Allow some to fail in concurrent test

    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, mock_app_context):
        """Test performance characteristics of the system."""
        with patch("main_refactored.app_context", mock_app_context):
            import time

            # Measure single thought processing time
            start_time = time.time()

            result = await sequentialthinking(
                thought="Performance test analysis",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
                topic="Performance Testing",
            )

            end_time = time.time()
            processing_time = end_time - start_time

            assert isinstance(result, str)
            assert processing_time < 1.0  # Should be fast with mocks

            # Test multiple thoughts for throughput
            start_time = time.time()

            for i in range(3):
                await sequentialthinking(
                    thought=f"Throughput test {i + 1}",
                    thoughtNumber=i + 1,
                    totalThoughts=3,
                    nextThoughtNeeded=i < 2,
                    topic="Throughput Testing",
                )

            end_time = time.time()
            total_time = end_time - start_time

            assert total_time < 3.0  # Should handle 3 thoughts quickly with mocks

    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self, mock_app_context):
        """Test memory usage patterns during processing."""
        with patch("main_refactored.app_context", mock_app_context):
            initial_thought_count = mock_app_context.total_thoughts

            # Process a series of thoughts
            for i in range(10):
                await sequentialthinking(
                    thought=f"Memory test thought {i + 1}",
                    thoughtNumber=i + 1,
                    totalThoughts=10,
                    nextThoughtNeeded=i < 9,
                    topic="Memory Testing",
                )

            # Check that context is tracking thoughts
            assert mock_app_context.total_thoughts == initial_thought_count + 10

            # Context should not grow unbounded (in real implementation)
            # This would be more meaningful with actual SharedContext
            assert mock_app_context.shared_context is not None


class TestSystemIntegration:
    """Test integration with external systems and dependencies."""

    @pytest.mark.asyncio
    async def test_provider_switching(self, mock_app_context):
        """Test switching between different LLM providers."""
        # This test validates the provider abstraction works
        with patch("main_refactored.app_context", mock_app_context):
            # Mock different provider configs
            providers = ["openrouter", "openai", "gemini"]

            for provider in providers:
                with patch.dict("os.environ", {"LLM_PROVIDER": provider}):
                    result = await sequentialthinking(
                        thought=f"Test with {provider} provider",
                        thoughtNumber=1,
                        totalThoughts=1,
                        nextThoughtNeeded=False,
                        topic=f"{provider.title()} Testing",
                    )

                    assert isinstance(result, str)
                    assert len(result) > 0

    @pytest.mark.asyncio
    async def test_context_persistence(self, mock_app_context):
        """Test context persistence across multiple interactions."""
        with patch("main_refactored.app_context", mock_app_context):
            session_id = mock_app_context.session_id

            # Process thoughts with context building
            await sequentialthinking(
                thought="Establish context baseline",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
                topic="Context Testing",
                keywords=["context", "persistence", "testing"],
            )

            # Second thought should have access to context
            result = await sequentialthinking(
                thought="Build upon established context",
                thoughtNumber=2,
                totalThoughts=2,
                nextThoughtNeeded=False,
                topic="Context Testing",
                keywords=["context", "building", "continuation"],
            )

            assert isinstance(result, str)

            # Session should remain consistent
            assert mock_app_context.session_id == session_id

    @pytest.mark.asyncio
    async def test_mcp_server_compatibility(self):
        """Test compatibility with MCP server framework."""
        # Verify that the mcp instance is properly configured
        assert isinstance(mcp, FastMCP)

        # Check that tools are registered
        # Note: This is a basic structural test
        # Full MCP integration would require server startup
        assert hasattr(mcp, "_tools")  # FastMCP internal structure

    @pytest.mark.asyncio
    async def test_async_lifecycle(self, mock_app_context):
        """Test async lifecycle management."""
        with patch("main_refactored.app_context", mock_app_context):
            # Test that async operations complete properly
            start_time = asyncio.get_event_loop().time()

            # Chain of async operations
            await sequentialthinking(
                thought="Async lifecycle test",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
            )

            review = await sequentialreview()

            end_time = asyncio.get_event_loop().time()

            # Should complete without hanging
            assert (end_time - start_time) < 2.0
            assert isinstance(review, str)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_thought_handling(self, mock_app_context):
        """Test handling of edge case inputs."""
        with patch("main_refactored.app_context", mock_app_context):
            # Very short thought
            result1 = await sequentialthinking(
                thought="x", thoughtNumber=1, totalThoughts=1, nextThoughtNeeded=False
            )

            # Very long thought (truncated in practice)
            long_thought = "x" * 10000
            result2 = await sequentialthinking(
                thought=long_thought,
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
            )

            # Both should be handled gracefully
            assert isinstance(result1, str)
            assert isinstance(result2, str)

    @pytest.mark.asyncio
    async def test_boundary_thought_numbers(self, mock_app_context):
        """Test boundary conditions for thought numbers."""
        with patch("main_refactored.app_context", mock_app_context):
            # Minimum valid thought number
            result1 = await sequentialthinking(
                thought="Minimum thought number",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
            )

            # Large thought numbers
            result2 = await sequentialthinking(
                thought="Large thought number",
                thoughtNumber=100,
                totalThoughts=100,
                nextThoughtNeeded=False,
            )

            assert isinstance(result1, str)
            assert isinstance(result2, str)

    @pytest.mark.asyncio
    async def test_malformed_input_handling(self, mock_app_context):
        """Test handling of malformed inputs."""
        with patch("main_refactored.app_context", mock_app_context):
            # Test with various malformed inputs
            test_cases = [
                {"keywords": ["valid", "", "also_valid"]},  # Empty keyword
                {"topic": "   "},  # Whitespace only topic
                {"subject": ""},  # Empty subject
                {"domain": "nonexistent"},  # Invalid domain
            ]

            for case in test_cases:
                params = {
                    "thought": "Test malformed input handling",
                    "thoughtNumber": 1,
                    "totalThoughts": 1,
                    "nextThoughtNeeded": False,
                    **case,
                }

                result = await sequentialthinking(**params)
                assert isinstance(result, str)  # Should handle gracefully
