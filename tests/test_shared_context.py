"""
Tests for SharedContext memory persistence and graph operations.
"""

import pytest
import asyncio
import networkx as nx

from src.context.shared_context import SharedContext, Insight
from src.models.thought_models import ToolDecision
from .conftest import create_test_thought_data


class TestSharedContext:
    """Test the SharedContext memory persistence system."""

    @pytest.mark.asyncio
    async def test_context_initialization(self):
        """Test SharedContext initialization."""
        context = SharedContext()

        assert isinstance(context.memory_store, dict)
        assert isinstance(context.thought_graph, nx.DiGraph)
        assert len(context.key_insights) == 0
        assert len(context.tool_usage_history) == 0
        assert context.thought_graph.number_of_nodes() == 0

    @pytest.mark.asyncio
    async def test_basic_context_operations(self):
        """Test basic context update and retrieval."""
        context = SharedContext()

        # Test update
        await context.update_context("test_key", "test_value")

        # Test retrieval
        value = await context.get_context("test_key")
        assert value == "test_value"

        # Test default value
        default_value = await context.get_context("nonexistent", "default")
        assert default_value == "default"

    @pytest.mark.asyncio
    async def test_thought_integration(self, sample_thought_data):
        """Test adding thoughts to context and graph building."""
        context = SharedContext()

        # Add first thought
        await context.update_from_thought(sample_thought_data)

        # Check graph was updated
        assert context.thought_graph.number_of_nodes() == 1
        assert 1 in context.thought_graph.nodes()

        # Check node data
        node_data = context.thought_graph.nodes[1]
        assert node_data["thought_content"] == sample_thought_data.thought
        assert node_data["domain"] == sample_thought_data.domain.value
        assert node_data["topic"] == sample_thought_data.topic

    @pytest.mark.asyncio
    async def test_thought_relationships(self, mock_session_context):
        """Test thought relationships and graph connections."""
        context = SharedContext()

        # Create a sequence of related thoughts
        thought1 = create_test_thought_data(
            thought="First thought",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            session_context=mock_session_context,
        )

        thought2 = create_test_thought_data(
            thought="Second thought building on first",
            thoughtNumber=2,
            totalThoughts=3,
            nextThoughtNeeded=True,
            session_context=mock_session_context,
        )

        thought3 = create_test_thought_data(
            thought="Revised first thought",
            thoughtNumber=3,
            totalThoughts=3,
            nextThoughtNeeded=False,
            isRevision=True,
            revisesThought=1,
            session_context=mock_session_context,
        )

        # Add thoughts to context
        await context.update_from_thought(thought1)
        await context.update_from_thought(thought2)
        await context.update_from_thought(thought3)

        # Check graph structure
        assert context.thought_graph.number_of_nodes() == 3
        assert context.thought_graph.number_of_edges() >= 1

        # Check revision relationship
        assert context.thought_graph.has_edge(3, 1)  # Revision edge
        edge_data = context.thought_graph.get_edge_data(3, 1)
        assert edge_data["relation_type"] == "revises"

    @pytest.mark.asyncio
    async def test_branching_support(self, mock_session_context):
        """Test branching thought support in graph."""
        context = SharedContext()

        # Create main sequence
        main_thought = create_test_thought_data(
            thought="Main analysis",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            session_context=mock_session_context,
        )

        # Create branch
        branch_thought = create_test_thought_data(
            thought="Alternative approach",
            thoughtNumber=2,
            totalThoughts=3,
            nextThoughtNeeded=True,
            branchFromThought=1,
            branchId="alternative-branch",
            session_context=mock_session_context,
        )

        await context.update_from_thought(main_thought)
        await context.update_from_thought(branch_thought)

        # Check branch relationship
        assert context.thought_graph.has_edge(2, 1)  # Branch edge
        edge_data = context.thought_graph.get_edge_data(2, 1)
        assert edge_data["relation_type"] == "branches_from"
        assert edge_data["branch_id"] == "alternative-branch"

    @pytest.mark.asyncio
    async def test_relevant_context_retrieval(self, mock_session_context):
        """Test retrieval of relevant context for new thoughts."""
        context = SharedContext()

        # Add several thoughts with different topics
        thoughts = [
            create_test_thought_data(
                thought="Performance analysis of database queries",
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="Performance",
                keywords=["database", "performance", "queries"],
                session_context=mock_session_context,
            ),
            create_test_thought_data(
                thought="Security review of authentication system",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="Security",
                keywords=["security", "authentication", "review"],
                session_context=mock_session_context,
            ),
            create_test_thought_data(
                thought="Database optimization strategies",
                thoughtNumber=3,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="Optimization",
                keywords=["database", "optimization", "performance"],
                session_context=mock_session_context,
            ),
        ]

        for thought in thoughts:
            await context.update_from_thought(thought)

        # Retrieve context relevant to database performance
        relevant = await context.get_relevant_context(
            "database performance optimization", max_items=5
        )

        assert "recent_thoughts" in relevant
        assert "keywords" in relevant
        assert "tool_patterns" in relevant

        # Should prioritize thoughts 1 and 3 (database/performance related)
        recent_thoughts = relevant["recent_thoughts"]
        assert len(recent_thoughts) >= 2

    @pytest.mark.asyncio
    async def test_insight_management(self):
        """Test insight addition and retrieval."""
        context = SharedContext()

        # Add insights
        await context.add_insight(
            "Database queries are the main bottleneck",
            source_thought=1,
            confidence=0.9,
            category="performance",
        )

        await context.add_insight(
            "Caching could improve response time by 50%",
            source_thought=2,
            confidence=0.8,
            category="optimization",
        )

        # Check insights were added
        assert len(context.key_insights) == 2

        insight1 = context.key_insights[0]
        assert insight1.content == "Database queries are the main bottleneck"
        assert insight1.source_thought == 1
        assert insight1.confidence == 0.9
        assert insight1.category == "performance"

    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance metrics recording and summary."""
        context = SharedContext()

        # Record performance metrics
        await context.record_performance("processing_time", 1500.0)
        await context.record_performance("processing_time", 1200.0)
        await context.record_performance("processing_time", 1800.0)

        await context.record_performance("token_usage", 150.0)
        await context.record_performance("token_usage", 200.0)

        # Get performance summary
        summary = await context.get_performance_summary()

        assert "processing_time" in summary
        assert "token_usage" in summary

        processing_stats = summary["processing_time"]
        assert processing_stats["count"] == 3
        assert processing_stats["mean"] == 1500.0  # (1500 + 1200 + 1800) / 3
        assert processing_stats["min"] == 1200.0
        assert processing_stats["max"] == 1800.0

        token_stats = summary["token_usage"]
        assert token_stats["count"] == 2
        assert token_stats["mean"] == 175.0  # (150 + 200) / 2

    @pytest.mark.asyncio
    async def test_thought_path_finding(self, mock_session_context):
        """Test finding paths between thoughts in the graph."""
        context = SharedContext()

        # Create a chain of thoughts
        thoughts = []
        for i in range(1, 5):
            thought = create_test_thought_data(
                thought=f"Thought {i}",
                thoughtNumber=i,
                totalThoughts=4,
                nextThoughtNeeded=i < 4,
                session_context=mock_session_context,
            )
            thoughts.append(thought)
            await context.update_from_thought(thought)

        # Find path from first to last thought
        path = await context.get_thought_path(1, 4)

        # Should have a path since thoughts form a sequence
        assert path is not None
        assert len(path) >= 2
        assert path[0] == 1
        assert path[-1] == 4

    @pytest.mark.asyncio
    async def test_cycle_detection(self, mock_session_context):
        """Test detection of circular reasoning patterns."""
        context = SharedContext()

        # Create thoughts that could form a cycle
        thought1 = create_test_thought_data(
            thought="Initial analysis",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            session_context=mock_session_context,
        )

        thought2 = create_test_thought_data(
            thought="Refinement of analysis",
            thoughtNumber=2,
            totalThoughts=3,
            nextThoughtNeeded=True,
            session_context=mock_session_context,
        )

        thought3 = create_test_thought_data(
            thought="Back to initial approach",
            thoughtNumber=3,
            totalThoughts=3,
            nextThoughtNeeded=False,
            isRevision=True,
            revisesThought=1,
            session_context=mock_session_context,
        )

        await context.update_from_thought(thought1)
        await context.update_from_thought(thought2)
        await context.update_from_thought(thought3)

        # Check for cycles
        cycles = await context.identify_cycles()

        # This simple case shouldn't have cycles, but the method should work
        assert isinstance(cycles, list)

    @pytest.mark.asyncio
    async def test_memory_management(self, sample_thought_data):
        """Test memory management and cleanup."""
        context = SharedContext(max_memory_items=5, max_insights=3)

        # Add data beyond limits to test cleanup
        for i in range(10):
            await context.update_context(f"key_{i}", f"value_{i}")
            await context.add_insight(f"Insight {i}", i, 0.8, "test")

        # Should have enforced limits
        assert len(context.memory_store) <= 5
        assert len(context.key_insights) <= 3

        # Test clear functionality
        context.clear()
        assert len(context.memory_store) == 0
        assert len(context.key_insights) == 0
        assert context.thought_graph.number_of_nodes() == 0

        # Test memory usage statistics
        await context.update_from_thought(sample_thought_data)
        stats = context.get_memory_usage()
        assert "memory_store_items" in stats
        assert "thought_nodes" in stats
        assert stats["thought_nodes"] == 1

    @pytest.mark.asyncio
    async def test_tool_decision_tracking(self, mock_session_context):
        """Test tracking of tool usage decisions."""
        context = SharedContext()

        # Create thought with tool decisions
        tool_decision = ToolDecision(
            tool_name="code_analyzer",
            rationale="Need to analyze code complexity",
            alternatives_considered=["profiler", "benchmark"],
            confidence=0.85,
            outcome="Identified 3 performance bottlenecks",
            execution_time_ms=2500,
        )

        thought = create_test_thought_data(
            thought="Code analysis complete",
            thoughtNumber=1,
            totalThoughts=2,
            nextThoughtNeeded=True,
            tool_decisions=[tool_decision],
            session_context=mock_session_context,
        )

        await context.update_from_thought(thought)

        # Check tool usage was tracked
        assert len(context.tool_usage_history) == 1
        tracked_decision = context.tool_usage_history[0]
        assert tracked_decision.tool_name == "code_analyzer"
        assert tracked_decision.confidence == 0.85

    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_session_context):
        """Test thread-safe concurrent access to context."""
        context = SharedContext()

        # Create multiple thoughts to add concurrently
        thoughts = []
        for i in range(1, 6):
            thought = create_test_thought_data(
                thought=f"Concurrent thought {i}",
                thoughtNumber=i,
                totalThoughts=5,
                nextThoughtNeeded=i < 5,
                session_context=mock_session_context,
            )
            thoughts.append(thought)

        # Add thoughts concurrently
        tasks = [context.update_from_thought(thought) for thought in thoughts]
        await asyncio.gather(*tasks)

        # Verify all thoughts were added correctly
        assert context.thought_graph.number_of_nodes() == 5

        # Verify graph integrity
        for i in range(1, 6):
            assert i in context.thought_graph.nodes()


class TestInsight:
    """Test the Insight data structure."""

    def test_insight_creation(self):
        """Test creating an insight."""
        from datetime import datetime

        insight = Insight(
            content="Database optimization needed",
            source_thought=2,
            confidence=0.9,
            category="performance",
            timestamp=datetime.now(),
        )

        assert insight.content == "Database optimization needed"
        assert insight.source_thought == 2
        assert insight.confidence == 0.9
        assert insight.category == "performance"
        assert insight.timestamp is not None

    def test_insight_serialization(self):
        """Test insight to_dict method."""
        from datetime import datetime

        insight = Insight(
            content="Test insight",
            source_thought=1,
            confidence=0.8,
            category="test",
            timestamp=datetime.now(),
        )

        data = insight.to_dict()

        assert data["content"] == "Test insight"
        assert data["source_thought"] == 1
        assert data["confidence"] == 0.8
        assert data["category"] == "test"
        assert "timestamp" in data
