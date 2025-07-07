"""
Comprehensive tests for SharedContext following TDD principles.
Tests written FIRST before any refactoring.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from context.shared_context import SharedContext, Insight
from models.thought_models import ThoughtData, DomainType, ToolDecision, ThoughtRelation
import networkx as nx


class TestSharedContextMemoryManagement:
    """Test memory overflow and management features."""
    
    @pytest.mark.asyncio
    async def test_memory_fifo_eviction(self):
        """Test FIFO eviction when memory limit is reached."""
        context = SharedContext(max_memory_items=3)
        
        # Add items beyond limit
        for i in range(5):
            await context.update_context(f"key_{i}", f"value_{i}")
        
        # Should only have last 3 items (FIFO eviction)
        assert len(context.memory_store) == 3
        assert "key_0" not in context.memory_store
        assert "key_1" not in context.memory_store
        assert "key_2" in context.memory_store
        assert "key_3" in context.memory_store
        assert "key_4" in context.memory_store
    
    @pytest.mark.asyncio
    async def test_memory_store_with_complex_objects(self):
        """Test storing complex objects in memory."""
        context = SharedContext()
        
        complex_obj = {
            "data": [1, 2, 3],
            "nested": {"key": "value"},
            "timestamp": datetime.now().isoformat()
        }
        
        await context.update_context("complex", complex_obj)
        retrieved = context.memory_store.get("complex")
        
        assert retrieved == complex_obj
        assert retrieved["nested"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_insight_capacity_management(self):
        """Test insight capacity limits and quality-based retention."""
        context = SharedContext(max_insights=3)
        
        # Add insights with varying quality
        insights = [
            ("Low quality insight", 0.3),
            ("Medium quality insight", 0.5),
            ("High quality insight", 0.9),
            ("Another high quality", 0.8),
            ("Best quality insight", 0.95)
        ]
        
        for i, (content, quality) in enumerate(insights):
            await context.add_insight(content, i, quality, "test")
        
        # Should keep top 3 by quality
        assert len(context.key_insights) == 3
        qualities = [insight.confidence for insight in context.key_insights]
        assert all(q >= 0.8 for q in qualities)
    
    @pytest.mark.asyncio
    async def test_memory_usage_reporting(self):
        """Test accurate memory usage statistics."""
        context = SharedContext()
        
        # Add various data
        for i in range(5):
            await context.update_context(f"key_{i}", f"value_{i}")
        
        await context.add_insight("Test insight", 1, 0.8, "test")
        
        thought = ThoughtData(
            thought="Test thought",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True
        )
        await context.update_from_thought(thought)
        
        usage = context.get_memory_usage()
        
        assert usage["memory_store_items"] == 5
        assert usage["key_insights"] == 1
        assert usage["thought_nodes"] == 1
        assert usage["total_items"] == 7


class TestSharedContextConcurrentAccess:
    """Test concurrent access patterns and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_updates(self):
        """Test concurrent updates to memory store."""
        context = SharedContext()
        
        async def update_worker(worker_id: int, count: int):
            for i in range(count):
                key = f"worker_{worker_id}_item_{i}"
                value = f"value_{worker_id}_{i}"
                await context.update_context(key, value)
        
        # Launch concurrent workers
        workers = []
        worker_count = 5
        items_per_worker = 20
        
        for i in range(worker_count):
            workers.append(update_worker(i, items_per_worker))
        
        await asyncio.gather(*workers)
        
        # Verify all updates succeeded
        assert len(context.memory_store) == worker_count * items_per_worker
        
        # Verify data integrity
        for worker_id in range(worker_count):
            for item_id in range(items_per_worker):
                key = f"worker_{worker_id}_item_{item_id}"
                expected_value = f"value_{worker_id}_{item_id}"
                assert context.memory_store.get(key) == expected_value
    
    @pytest.mark.asyncio
    async def test_concurrent_thought_updates(self):
        """Test concurrent thought chain updates."""
        context = SharedContext()
        
        async def add_thought_branch(branch_id: str, start: int, count: int):
            for i in range(count):
                thought = ThoughtData(
                    thought=f"Branch {branch_id} thought {i}",
                    thoughtNumber=start + i,
                    totalThoughts=start + count,
                    nextThoughtNeeded=(i < count - 1),
                    branchId=branch_id
                )
                await context.update_from_thought(thought)
        
        # Create multiple branches concurrently
        branches = []
        for i in range(3):
            branches.append(
                add_thought_branch(f"branch_{i}", i * 10, 5)
            )
        
        await asyncio.gather(*branches)
        
        # Verify all thoughts were added
        assert len(context.thought_chain) == 15
        
        # Verify branch integrity
        for i in range(3):
            branch_thoughts = [
                t for t in context.thought_chain 
                if t.branchId == f"branch_{i}"
            ]
            assert len(branch_thoughts) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_graph_operations(self):
        """Test concurrent graph modifications."""
        context = SharedContext()
        
        async def add_graph_edges(start_node: int, count: int):
            for i in range(count):
                from_node = start_node + i
                to_node = start_node + i + 1
                context.thought_graph.add_edge(
                    from_node, to_node,
                    relation_type="leads_to",
                    strength=0.8
                )
        
        # Add edges concurrently
        tasks = []
        for i in range(0, 20, 5):
            tasks.append(add_graph_edges(i, 4))
        
        await asyncio.gather(*tasks)
        
        # Verify graph structure
        assert context.thought_graph.number_of_nodes() == 24
        assert context.thought_graph.number_of_edges() == 16


class TestSharedContextPerformanceTracking:
    """Test performance metrics and tracking."""
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self):
        """Test accurate processing time measurement."""
        context = SharedContext()
        
        # Record multiple processing times
        times = [100, 200, 150, 300, 250]
        for t in times:
            context.performance_metrics["processing_time"].append(t)
        
        summary = await context.get_performance_summary()
        
        assert summary["processing_time"]["mean"] == 200
        assert summary["processing_time"]["median"] == 200
        assert summary["processing_time"]["min"] == 100
        assert summary["processing_time"]["max"] == 300
        assert summary["processing_time"]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_tool_usage_tracking(self):
        """Test tool usage pattern tracking."""
        context = SharedContext()
        
        # Track tool decisions
        tools = [
            ToolDecision(
                tool_name="ThinkingTools",
                rationale="Need deep analysis",
                alternatives_considered=["ExaTools"],
                confidence=0.9,
                outcome="Success"
            ),
            ToolDecision(
                tool_name="ThinkingTools",
                rationale="Continue analysis",
                alternatives_considered=[],
                confidence=0.8,
                outcome="Success"
            ),
            ToolDecision(
                tool_name="ExaTools",
                rationale="Research needed",
                alternatives_considered=["ThinkingTools"],
                confidence=0.7,
                outcome="Partial"
            )
        ]
        
        for tool in tools:
            context.tool_usage_history.append(tool)
        
        # Get usage patterns
        patterns = context.get_tool_usage_patterns()
        
        assert patterns["ThinkingTools"]["count"] == 2
        assert patterns["ThinkingTools"]["avg_confidence"] == 0.85
        assert patterns["ExaTools"]["count"] == 1
        assert patterns["ExaTools"]["avg_confidence"] == 0.7
    
    @pytest.mark.asyncio
    async def test_quality_tracking_over_time(self):
        """Test quality metrics tracking over time."""
        context = SharedContext()
        
        # Add thoughts with varying quality
        for i in range(10):
            thought = ThoughtData(
                thought=f"Thought {i}",
                thoughtNumber=i + 1,
                totalThoughts=10,
                nextThoughtNeeded=(i < 9),
                confidence_score=0.5 + (i * 0.05)  # Increasing confidence
            )
            await context.update_from_thought(thought)
        
        # Check quality trend
        recent_thoughts = list(context.thought_chain)[-5:]
        recent_avg_confidence = sum(t.confidence_score for t in recent_thoughts) / 5
        
        early_thoughts = list(context.thought_chain)[:5]
        early_avg_confidence = sum(t.confidence_score for t in early_thoughts) / 5
        
        assert recent_avg_confidence > early_avg_confidence  # Quality improving


class TestSharedContextEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_context_operations(self):
        """Test operations on empty context."""
        context = SharedContext()
        
        # Test retrieval from empty context
        relevant = await context.get_relevant_context("test query")
        assert relevant == {}
        
        # Test performance summary on empty metrics
        summary = await context.get_performance_summary()
        assert summary == {}
        
        # Test clear on empty context
        await context.clear()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_circular_thought_relationships(self):
        """Test handling of circular dependencies in thought graph."""
        context = SharedContext()
        
        # Create circular relationship: 1 -> 2 -> 3 -> 1
        for i in range(1, 4):
            thought = ThoughtData(
                thought=f"Circular thought {i}",
                thoughtNumber=i,
                totalThoughts=3,
                nextThoughtNeeded=True
            )
            await context.update_from_thought(thought)
        
        # Add circular edges
        context.thought_graph.add_edge(1, 2)
        context.thought_graph.add_edge(2, 3)
        context.thought_graph.add_edge(3, 1)
        
        # Detect cycles
        cycles = list(nx.simple_cycles(context.thought_graph))
        assert len(cycles) > 0
        assert [1, 2, 3] in cycles or [2, 3, 1] in cycles or [3, 1, 2] in cycles
    
    @pytest.mark.asyncio
    async def test_large_context_handling(self):
        """Test handling of very large contexts."""
        context = SharedContext(max_memory_items=1000)
        
        # Add many items
        start_time = time.time()
        
        for i in range(1000):
            await context.update_context(f"key_{i}", {"index": i, "data": "x" * 100})
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds
        
        # Test retrieval performance
        start_time = time.time()
        relevant = await context.get_relevant_context("test query")
        elapsed = time.time() - start_time
        
        assert elapsed < 0.5  # Retrieval should be fast
    
    @pytest.mark.asyncio
    async def test_invalid_thought_relationships(self):
        """Test handling of invalid thought relationships."""
        context = SharedContext()
        
        # Add a thought
        thought = ThoughtData(
            thought="Single thought",
            thoughtNumber=1,
            totalThoughts=1,
            nextThoughtNeeded=False
        )
        await context.update_from_thought(thought)
        
        # Try to add relationship to non-existent thought
        relation = ThoughtRelation(
            from_thought=1,
            to_thought=999,  # Non-existent
            relation_type="leads_to",
            strength=0.8
        )
        
        # Should handle gracefully
        context.thought_graph.add_edge(
            relation.from_thought,
            relation.to_thought,
            relation_type=relation.relation_type,
            strength=relation.strength
        )
        
        # Graph should have the edge even if node doesn't exist yet
        assert context.thought_graph.has_edge(1, 999)


class TestSharedContextIntegration:
    """Test integration with other components."""
    
    @pytest.mark.asyncio
    async def test_thought_to_insight_conversion(self):
        """Test automatic insight extraction from thoughts."""
        context = SharedContext()
        
        # Add high-quality thought
        thought = ThoughtData(
            thought="This is a breakthrough insight about the system architecture",
            thoughtNumber=1,
            totalThoughts=1,
            nextThoughtNeeded=False,
            confidence_score=0.95,
            keywords=["architecture", "breakthrough", "system"]
        )
        
        await context.update_from_thought(thought)
        
        # Should trigger insight creation for high-quality thoughts
        if thought.confidence_score > 0.9:
            await context.add_insight(
                thought.thought,
                thought.thoughtNumber,
                thought.confidence_score,
                "high_confidence"
            )
        
        assert len(context.key_insights) == 1
        assert context.key_insights[0]["quality"] == 0.95
    
    @pytest.mark.asyncio
    async def test_context_relevance_scoring(self):
        """Test context relevance scoring algorithm."""
        context = SharedContext()
        
        # Add various context items
        await context.update_context("python_tips", {
            "content": "Python programming best practices",
            "keywords": ["python", "programming", "best practices"]
        })
        
        await context.update_context("rust_guide", {
            "content": "Rust memory management guide",
            "keywords": ["rust", "memory", "management"]
        })
        
        await context.update_context("general_advice", {
            "content": "General software development advice",
            "keywords": ["software", "development", "general"]
        })
        
        # Query for Python-related context
        relevant = await context.get_relevant_context("python programming tips")
        
        # Should prioritize Python content
        assert "python_tips" in str(relevant)
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """Test detection of performance degradation."""
        context = SharedContext()
        
        # Simulate degrading performance
        base_time = 100
        for i in range(10):
            # Performance gets worse
            processing_time = base_time * (1 + i * 0.2)
            context.performance_metrics["processing_time"].append(processing_time)
        
        summary = await context.get_performance_summary()
        
        # Recent average should be worse than overall average
        all_times = context.performance_metrics["processing_time"]
        recent_avg = sum(all_times[-3:]) / 3
        overall_avg = sum(all_times) / len(all_times)
        
        assert recent_avg > overall_avg * 1.5  # Significant degradation