"""
Advanced tests for SharedContext including memory management,
graph algorithms, and performance characteristics.
"""

import pytest
import asyncio
import networkx as nx

from src.context.shared_context import SharedContext
from src.models.thought_models import (
    ToolDecision,
    ThoughtRelation,
)
from .conftest import create_test_thought_data


class TestSharedContextAdvanced:
    """Advanced tests for SharedContext functionality."""

    @pytest.mark.asyncio
    async def test_memory_eviction_fifo(self):
        """Test FIFO eviction when memory limit is reached."""
        context = SharedContext(max_memory_items=5)

        # Add items beyond limit
        for i in range(10):
            await context.update_context(f"key_{i}", f"value_{i}")

        # Should only have last 5 items
        assert len(context.memory_store) == 5

        # Check FIFO eviction - oldest should be gone
        for i in range(5):
            value = await context.get_context(f"key_{i}")
            assert value is None  # Should be evicted

        # Recent items should exist
        for i in range(5, 10):
            value = await context.get_context(f"key_{i}")
            assert value == f"value_{i}"

    @pytest.mark.asyncio
    async def test_insight_limit_enforcement(self):
        """Test insight limit enforcement."""
        context = SharedContext(max_insights=3)

        # Add insights beyond limit
        for i in range(5):
            await context.add_insight(
                f"Insight {i}", source_thought=i, confidence=0.8, category="test"
            )

        # Should only have last 3
        assert len(context.key_insights) == 3

        # Check it's the most recent ones
        assert context.key_insights[0].content == "Insight 2"
        assert context.key_insights[1].content == "Insight 3"
        assert context.key_insights[2].content == "Insight 4"

    @pytest.mark.asyncio
    async def test_thought_graph_algorithms(self, mock_session_context):
        """Test graph algorithms on thought relationships."""
        context = SharedContext()

        # Create a complex thought graph
        # 1 -> 2 -> 3
        # 1 -> 4 -> 5
        # 3 -> 6
        # 5 -> 6 (convergence)

        thoughts = []
        for i in range(1, 7):
            thought = create_test_thought_data(
                thought=f"Test thought number {i} with sufficient content",
                thoughtNumber=i,
                totalThoughts=6,
                nextThoughtNeeded=(i < 6),
                session_context=mock_session_context,
            )
            thoughts.append(thought)
            await context.update_from_thought(thought)

        # Add explicit relationships
        thought_relations = [
            ThoughtRelation(
                from_thought=1, to_thought=2, relation_type="leads_to", strength=0.9
            ),
            ThoughtRelation(
                from_thought=2, to_thought=3, relation_type="leads_to", strength=0.8
            ),
            ThoughtRelation(
                from_thought=1, to_thought=4, relation_type="leads_to", strength=0.7
            ),
            ThoughtRelation(
                from_thought=4, to_thought=5, relation_type="leads_to", strength=0.8
            ),
            ThoughtRelation(
                from_thought=3, to_thought=6, relation_type="leads_to", strength=0.9
            ),
            ThoughtRelation(
                from_thought=5, to_thought=6, relation_type="leads_to", strength=0.8
            ),
        ]

        for relation in thought_relations:
            context.thought_graph.add_edge(
                relation.from_thought,
                relation.to_thought,
                relation_type=relation.relation_type,
                strength=relation.strength,
            )

        # Test path finding
        path_1_to_6 = await context.get_thought_path(1, 6)
        assert path_1_to_6 is not None
        assert len(path_1_to_6) >= 3  # At least 3 nodes in path

        # Test multiple paths exist
        all_paths = list(nx.all_simple_paths(context.thought_graph, 1, 6))
        assert len(all_paths) == 2  # Two paths from 1 to 6

    @pytest.mark.asyncio
    async def test_relevant_context_keyword_matching(self, mock_session_context):
        """Test sophisticated keyword matching in context retrieval."""
        context = SharedContext()

        # Add thoughts with specific keywords
        thoughts_data = [
            (
                "machine learning algorithms optimization",
                ["machine", "learning", "algorithms", "optimization"],
            ),
            (
                "deep learning neural networks",
                ["deep", "learning", "neural", "networks"],
            ),
            (
                "optimization techniques for performance",
                ["optimization", "techniques", "performance"],
            ),
            ("database query optimization", ["database", "query", "optimization"]),
        ]

        for i, (thought_text, keywords) in enumerate(thoughts_data):
            thought = create_test_thought_data(
                thought=thought_text,
                thoughtNumber=i + 1,
                keywords=keywords,
                session_context=mock_session_context,
            )
            await context.update_from_thought(thought)

        # Test keyword matching
        relevant = await context.get_relevant_context(
            "optimization algorithms for machine learning"
        )

        # Should prioritize thoughts with multiple matching keywords
        assert len(relevant["recent_thoughts"]) > 0

        # First thought should have most keyword overlap
        first_thought = relevant["recent_thoughts"][0]
        assert "optimization" in first_thought["thought"].lower()

    @pytest.mark.asyncio
    async def test_performance_metrics_statistics(self):
        """Test performance metrics tracking and statistics."""
        context = SharedContext()

        # Record various metrics
        processing_times = [100, 150, 200, 120, 180, 90, 250, 130, 170, 110]
        for time_ms in processing_times:
            await context.record_performance("processing_time", time_ms)

        token_counts = [1000, 1500, 1200, 1800, 1300]
        for tokens in token_counts:
            await context.record_performance("token_usage", tokens)

        # Get statistics
        stats = await context.get_performance_summary()

        # Check processing time stats
        assert "processing_time" in stats
        proc_stats = stats["processing_time"]
        assert proc_stats["count"] == 10
        assert proc_stats["mean"] == sum(processing_times) / len(processing_times)
        assert proc_stats["min"] == 90
        assert proc_stats["max"] == 250

        # Check token usage stats
        assert "token_usage" in stats
        token_stats = stats["token_usage"]
        assert token_stats["count"] == 5
        assert token_stats["mean"] == sum(token_counts) / len(token_counts)

    @pytest.mark.asyncio
    async def test_concurrent_context_updates(self):
        """Test thread-safe concurrent updates to context."""
        context = SharedContext()

        # Define concurrent update tasks
        async def update_task(prefix: str, count: int):
            for i in range(count):
                await context.update_context(f"{prefix}_{i}", f"value_{prefix}_{i}")
                await asyncio.sleep(0.001)  # Small delay to increase contention

        # Run multiple update tasks concurrently
        tasks = [
            update_task("task1", 20),
            update_task("task2", 20),
            update_task("task3", 20),
            update_task("task4", 20),
        ]

        await asyncio.gather(*tasks)

        # Verify all updates succeeded
        for prefix in ["task1", "task2", "task3", "task4"]:
            for i in range(20):
                value = await context.get_context(f"{prefix}_{i}")
                assert value == f"value_{prefix}_{i}"

    @pytest.mark.asyncio
    async def test_complex_revision_chains(self, mock_session_context):
        """Test handling of complex revision chains."""
        context = SharedContext()

        # Create initial thought
        thought1 = create_test_thought_data(
            thought="Initial approach",
            thoughtNumber=1,
            session_context=mock_session_context,
        )
        await context.update_from_thought(thought1)

        # Create revision chain: 2 revises 1, 3 revises 2, 4 revises 3
        for i in range(2, 5):
            thought = create_test_thought_data(
                thought=f"This is revision {i - 1} providing substantial new insights and improvements to the previous thought",
                thoughtNumber=i,
                isRevision=True,
                revisesThought=i - 1,
                session_context=mock_session_context,
            )
            await context.update_from_thought(thought)

        # Check revision chain in graph
        assert context.thought_graph.has_edge(2, 1)
        assert context.thought_graph.has_edge(3, 2)
        assert context.thought_graph.has_edge(4, 3)

        # Find revision path
        path = await context.get_thought_path(4, 1)
        assert path == [4, 3, 2, 1]

    @pytest.mark.asyncio
    async def test_branch_merge_detection(self, mock_session_context):
        """Test detection of branch merges in thought graph."""
        context = SharedContext()

        # Create branching structure
        # 1 -> 2 (branch A) -> 4
        # 1 -> 3 (branch B) -> 4 (merge)

        thought1 = create_test_thought_data(
            thoughtNumber=1,
            session_context=mock_session_context,
        )
        await context.update_from_thought(thought1)

        # Continue linearly to thought 2
        thought2 = create_test_thought_data(
            thoughtNumber=2,
            session_context=mock_session_context,
        )
        await context.update_from_thought(thought2)

        # Branch A from thought 1
        thought3 = create_test_thought_data(
            thoughtNumber=3,
            branchFromThought=1,
            branchId="approach-a",
            session_context=mock_session_context,
        )
        await context.update_from_thought(thought3)

        # Branch B from thought 1
        thought4 = create_test_thought_data(
            thoughtNumber=4,
            branchFromThought=1,
            branchId="approach-b",
            session_context=mock_session_context,
        )
        await context.update_from_thought(thought4)

        # Merge point
        thought5 = create_test_thought_data(
            thoughtNumber=5,
            session_context=mock_session_context,
            thought_relationships=[
                ThoughtRelation(
                    from_thought=3,
                    to_thought=5,
                    relation_type="merges_to",
                    strength=0.8,
                ),
                ThoughtRelation(
                    from_thought=4,
                    to_thought=5,
                    relation_type="merges_to",
                    strength=0.7,
                ),
            ],
        )
        await context.update_from_thought(thought5)

        # Check merge detection - thought 5 should have two predecessors (3 and 4)
        predecessors = list(context.thought_graph.predecessors(5))
        assert len(predecessors) == 2
        assert 3 in predecessors
        assert 4 in predecessors

    @pytest.mark.asyncio
    async def test_tool_usage_patterns(self, mock_session_context):
        """Test analysis of tool usage patterns."""
        context = SharedContext()

        # Simulate tool usage over multiple thoughts
        tool_decisions = [
            ("code_analysis", 0.9, "Found performance issues", 1200),
            ("profiler", 0.85, "Identified bottlenecks", 2500),
            ("code_analysis", 0.7, "Reviewed changes", 1000),
            ("test_runner", 0.95, "All tests passed", 3000),
            ("code_analysis", 0.8, "Final review", 1100),
        ]

        for i, (tool, confidence, outcome, time_ms) in enumerate(tool_decisions):
            decision = ToolDecision(
                tool_name=tool,
                rationale=f"Used for step {i + 1}",
                alternatives_considered=[],
                confidence=confidence,
                outcome=outcome,
                execution_time_ms=time_ms,
            )

            thought = create_test_thought_data(
                thoughtNumber=i + 1,
                tool_decisions=[decision],
                session_context=mock_session_context,
            )
            await context.update_from_thought(thought)

        # Get relevant context for tool-related query
        relevant = await context.get_relevant_context("analyze code performance")

        # Check tool patterns
        assert "tool_patterns" in relevant
        patterns = relevant["tool_patterns"]

        # code_analysis should be most used
        assert patterns[0]["tool"] == "code_analysis"
        assert patterns[0]["recent_uses"] == 3

    @pytest.mark.asyncio
    async def test_memory_usage_reporting(self):
        """Test accurate memory usage reporting."""
        context = SharedContext()

        # Add various data
        for i in range(10):
            await context.update_context(f"key_{i}", f"value_{i}")

        for i in range(5):
            await context.add_insight(f"Insight {i}", i, 0.8)

        for i in range(3):
            decision = ToolDecision(
                tool_name=f"tool_{i}",
                rationale="Test",
                alternatives_considered=[],
                confidence=0.8,
                outcome="Test outcome",
                execution_time_ms=1000,
            )
            context.tool_usage_history.append(decision)

        # Get memory usage
        usage = context.get_memory_usage()

        assert usage["memory_store_items"] == 10
        assert usage["insights_count"] == 5
        assert usage["tool_history_count"] == 3
        assert usage["thought_nodes"] == 0  # No thoughts added
        assert usage["performance_metrics"] == 0  # No metrics recorded

    @pytest.mark.asyncio
    async def test_context_cleanup(self):
        """Test context cleanup functionality."""
        context = SharedContext()

        # Add data
        await context.update_context("key", "value")
        await context.add_insight("Test insight", 1)
        context.tool_usage_history.append(
            ToolDecision(
                tool_name="test",
                rationale="Test",
                alternatives_considered=[],
                confidence=0.8,
                outcome="Test outcome",
                execution_time_ms=1000,
            )
        )

        # Clear context
        context.clear()

        # Verify everything is cleared
        assert len(context.memory_store) == 0
        assert len(context.key_insights) == 0
        assert len(context.tool_usage_history) == 0
        assert context.thought_graph.number_of_nodes() == 0
        assert context.access_count == 0

    @pytest.mark.asyncio
    async def test_timestamp_tracking(self):
        """Test timestamp tracking for context operations."""
        context = SharedContext()

        # Record creation time
        created_at = context.created_at

        # Wait a bit and update
        await asyncio.sleep(0.1)
        await context.update_context("key", "value")

        # Last updated should be after creation
        assert context.last_updated > created_at

        # Update again
        prev_update = context.last_updated
        await asyncio.sleep(0.1)
        await context.add_insight("Test", 1)

        # Should update timestamp
        assert context.last_updated > prev_update

    @pytest.mark.asyncio
    async def test_access_count_tracking(self):
        """Test access count tracking."""
        context = SharedContext()

        initial_count = context.access_count

        # Access context multiple times
        for i in range(5):
            await context.get_context(f"key_{i}")

        assert context.access_count == initial_count + 5
