"""
Simple in-memory context system for maintaining state during a single execution.
No persistence, no session management - just clean, fast memory storage.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import logging
from collections import defaultdict
import networkx as nx
from src.models.thought_models import ThoughtData, ToolDecision

logger = logging.getLogger(__name__)


class Insight:
    """Represents a key insight discovered during thinking."""

    def __init__(
        self,
        content: str,
        source_thought: int,
        confidence: float,
        category: str,
        timestamp: datetime,
    ):
        self.content = content
        self.source_thought = source_thought
        self.confidence = confidence
        self.category = category
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_thought": self.source_thought,
            "confidence": self.confidence,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
        }


class SharedContext:
    """
    Simple in-memory shared context for multi-agent coordination.

    Maintains state only for the current execution - no persistence, no sessions.
    Memory is automatically cleaned up when the process ends.
    """

    def __init__(
        self,
        max_memory_items: int = 500,
        max_insights: int = 50,
        max_thought_nodes: int = 200,
    ):
        self.max_memory_items = max_memory_items
        self.max_insights = max_insights
        self.max_thought_nodes = max_thought_nodes
        self._lock = asyncio.Lock()

        # In-memory storage
        self.memory_store: Dict[str, Any] = {}
        self.tool_usage_history: List[ToolDecision] = []
        self.thought_graph = nx.DiGraph()
        self.key_insights: List[Insight] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

        # Metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        self.access_count = 0

    async def update_context(self, key: str, value: Any) -> None:
        """Thread-safe context update with memory limit enforcement."""
        async with self._lock:
            self.memory_store[key] = value
            self.last_updated = datetime.now(timezone.utc)

            # Simple memory limit enforcement (FIFO eviction)
            if len(self.memory_store) > self.max_memory_items:
                oldest_keys = list(self.memory_store.keys())[: -self.max_memory_items]
                for old_key in oldest_keys:
                    del self.memory_store[old_key]
                logger.debug(f"Evicted {len(oldest_keys)} old memory entries")

            logger.debug(f"Updated context key '{key}'")

    async def get_context(self, key: str, default: Any = None) -> Any:
        """Thread-safe context retrieval."""
        async with self._lock:
            self.access_count += 1
            return self.memory_store.get(key, default)

    async def update_from_thought(self, thought_data: ThoughtData) -> None:
        """Updates context based on a thought."""
        async with self._lock:
            # Add to thought graph
            self.thought_graph.add_node(
                thought_data.thoughtNumber,
                thought=thought_data.thought,
                thought_content=thought_data.thought,  # For backward compatibility with tests
                confidence=thought_data.confidence_score,
                timestamp=datetime.now(timezone.utc),
                domain=thought_data.domain.value if thought_data.domain else None,
                topic=thought_data.topic,
            )

            # Add relationships
            if thought_data.isRevision and thought_data.revisesThought:
                self.thought_graph.add_edge(
                    thought_data.thoughtNumber,
                    thought_data.revisesThought,
                    relation_type="revises",
                )

            if thought_data.branchFromThought:
                self.thought_graph.add_edge(
                    thought_data.thoughtNumber,
                    thought_data.branchFromThought,
                    relation_type="branches_from",
                    branch_id=thought_data.branchId,
                )

            for relation in thought_data.thought_relationships:
                self.thought_graph.add_edge(
                    relation.from_thought,
                    relation.to_thought,
                    relation_type=relation.relation_type,
                    strength=relation.strength,
                )

            # Store tool decisions (keep only recent ones)
            self.tool_usage_history.extend(thought_data.tool_decisions)
            if len(self.tool_usage_history) > 100:  # Keep last 100 tool decisions
                self.tool_usage_history = self.tool_usage_history[-100:]

            # Evict old thought nodes if graph gets too large
            if self.thought_graph.number_of_nodes() > self.max_thought_nodes:
                # Get oldest nodes by thought number (assuming sequential numbering)
                all_nodes = sorted(self.thought_graph.nodes())
                nodes_to_remove = all_nodes[: -self.max_thought_nodes]

                # Remove nodes and their edges
                for node in nodes_to_remove:
                    self.thought_graph.remove_node(node)

                logger.debug(
                    f"Evicted {len(nodes_to_remove)} old thought nodes from graph"
                )

            # Update snapshot
            if thought_data.context_snapshot:
                for k, v in thought_data.context_snapshot.items():
                    snapshot_key = f"snapshot_{thought_data.thoughtNumber}_{k}"
                    await self.update_context(snapshot_key, v)

            self.last_updated = datetime.now(timezone.utc)

    async def get_relevant_context(
        self, thought: str, max_items: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieves context relevant to the given thought.
        Uses simple keyword matching and recency.
        """
        async with self._lock:
            relevant = {
                "recent_thoughts": [],
                "related_insights": [],
                "tool_patterns": [],
                "graph_neighbors": [],
                "keywords": [],
            }

            # Get recent thoughts from graph
            if self.thought_graph.number_of_nodes() > 0:
                recent_nodes = sorted(
                    self.thought_graph.nodes(data=True),
                    key=lambda x: x[0],
                    reverse=True,
                )[:max_items]

                relevant["recent_thoughts"] = [
                    {
                        "number": node[0],
                        "thought": node[1].get("thought", ""),
                        "confidence": node[1].get("confidence", 0.5),
                    }
                    for node in recent_nodes
                ]

            # Get related insights (simple keyword match)
            thought_words = set(thought.lower().split())
            related_insights = []

            for insight in self.key_insights:
                insight_words = set(insight.content.lower().split())
                overlap = len(thought_words & insight_words)
                if overlap > 2:  # At least 3 common words
                    related_insights.append((overlap, insight))

            related_insights.sort(key=lambda x: x[0], reverse=True)
            relevant["related_insights"] = [
                ins.to_dict() for _, ins in related_insights[: max_items // 2]
            ]

            # Analyze tool usage patterns
            tool_counts = defaultdict(int)
            for decision in self.tool_usage_history[-20:]:  # Last 20 uses
                tool_counts[decision.tool_name] += 1

            relevant["tool_patterns"] = [
                {"tool": tool, "recent_uses": count}
                for tool, count in sorted(
                    tool_counts.items(), key=lambda x: x[1], reverse=True
                )
            ]

            # Extract keywords from thought
            thought_words = set(thought.lower().split())
            relevant["keywords"] = list(thought_words)[:10]  # Top 10 keywords

            return relevant

    async def add_insight(
        self,
        content: str,
        source_thought: int,
        confidence: float = 0.7,
        category: str = "general",
    ) -> None:
        """Adds a key insight to the context."""
        async with self._lock:
            insight = Insight(
                content=content,
                source_thought=source_thought,
                confidence=confidence,
                category=category,
                timestamp=datetime.now(timezone.utc),
            )
            self.key_insights.append(insight)

            # Keep only top insights by quality/confidence
            if len(self.key_insights) > self.max_insights:
                # Sort by confidence (quality) and keep top N
                self.key_insights.sort(key=lambda x: x.confidence, reverse=True)
                self.key_insights = self.key_insights[: self.max_insights]
                logger.debug(f"Trimmed insights to top {self.max_insights} by quality")

            self.last_updated = datetime.now(timezone.utc)
            logger.info(f"Added insight: {content[:50]}...")

    async def record_performance(self, metric_name: str, value: float) -> None:
        """Records a performance metric."""
        async with self._lock:
            self.performance_metrics[metric_name].append(value)

            # Keep only recent metrics (last 100 per metric)
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name] = self.performance_metrics[
                    metric_name
                ][-100:]

    async def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Gets summary statistics for performance metrics."""
        async with self._lock:
            summary = {}
            for metric, values in self.performance_metrics.items():
                if values:
                    summary[metric] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                        "median": sorted(values)[len(values) // 2] if values else 0,
                    }
            return summary

    def get_tool_usage_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyzes tool usage patterns from history."""
        patterns = {}

        # Group by tool name
        tool_groups = defaultdict(list)
        for decision in self.tool_usage_history:
            tool_groups[decision.tool_name].append(decision)

        # Calculate patterns for each tool
        for tool_name, decisions in tool_groups.items():
            confidences = [d.confidence for d in decisions]
            patterns[tool_name] = {
                "count": len(decisions),
                "avg_confidence": sum(confidences) / len(confidences)
                if confidences
                else 0,
                "success_rate": sum(1 for d in decisions if d.outcome == "Success")
                / len(decisions)
                if decisions
                else 0,
                "alternatives": list(
                    set(alt for d in decisions for alt in d.alternatives_considered)
                ),
            }

        return patterns

    async def get_thought_path(
        self, from_thought: int, to_thought: int
    ) -> Optional[List[int]]:
        """Finds the path between two thoughts in the graph."""
        async with self._lock:
            try:
                path = nx.shortest_path(self.thought_graph, from_thought, to_thought)
                return path
            except nx.NetworkXNoPath:
                return None

    async def identify_cycles(self) -> List[List[int]]:
        """Identifies cycles in the thought graph."""
        async with self._lock:
            return list(nx.simple_cycles(self.thought_graph))

    def get_memory_usage(self) -> Dict[str, int]:
        """Returns current memory usage statistics."""
        return {
            "memory_store_items": len(self.memory_store),
            "tool_history_count": len(self.tool_usage_history),
            "thought_nodes": self.thought_graph.number_of_nodes(),
            "thought_edges": self.thought_graph.number_of_edges(),
            "key_insights": len(self.key_insights),
            "performance_metrics": sum(
                len(values) for values in self.performance_metrics.values()
            ),
            "total_items": len(self.memory_store)
            + len(self.key_insights)
            + self.thought_graph.number_of_nodes(),
        }

    async def clear(self) -> None:
        """Clear all stored data (useful for cleanup)."""
        async with self._lock:
            self.memory_store.clear()
            self.tool_usage_history.clear()
            self.thought_graph.clear()
            self.key_insights.clear()
            self.performance_metrics.clear()
            self.access_count = 0
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Shared context cleared")
