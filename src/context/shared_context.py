"""
Shared context system for maintaining state across agents and thoughts.
"""

import asyncio
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
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
        timestamp: datetime
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
            "timestamp": self.timestamp.isoformat()
        }


class SharedContext:
    """
    Thread-safe shared context for multi-agent coordination.
    
    Maintains:
    - Key-value memory store
    - Tool usage history
    - Thought relationships graph
    - Key insights
    - Performance metrics
    """
    
    def __init__(self, backend: Literal["memory", "redis"] = "memory"):
        self.backend = backend
        self._lock = asyncio.Lock()
        
        # In-memory storage
        self.memory_store: Dict[str, Any] = {}
        self.tool_usage_history: List[ToolDecision] = []
        self.thought_graph = nx.DiGraph()
        self.key_insights: List[Insight] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.access_count = 0
        
        # Redis backend (future enhancement)
        if backend == "redis":
            logger.warning("Redis backend not yet implemented, using memory")
            self.backend = "memory"
    
    async def update_context(self, key: str, value: Any) -> None:
        """Thread-safe context update."""
        async with self._lock:
            self.memory_store[key] = value
            self.last_updated = datetime.utcnow()
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
                confidence=thought_data.confidence_score,
                timestamp=datetime.utcnow()
            )
            
            # Add relationships
            if thought_data.isRevision and thought_data.revisesThought:
                self.thought_graph.add_edge(
                    thought_data.thoughtNumber,
                    thought_data.revisesThought,
                    relation_type="revises"
                )
            
            if thought_data.branchFromThought:
                self.thought_graph.add_edge(
                    thought_data.branchFromThought,
                    thought_data.thoughtNumber,
                    relation_type="branches_to"
                )
            
            for relation in thought_data.thought_relationships:
                self.thought_graph.add_edge(
                    relation.from_thought,
                    relation.to_thought,
                    relation_type=relation.relation_type,
                    strength=relation.strength
                )
            
            # Store tool decisions
            self.tool_usage_history.extend(thought_data.tool_decisions)
            
            # Update snapshot
            if thought_data.context_snapshot:
                for k, v in thought_data.context_snapshot.items():
                    self.memory_store[f"snapshot_{thought_data.thoughtNumber}_{k}"] = v
            
            self.last_updated = datetime.utcnow()
    
    async def get_relevant_context(
        self, 
        thought: str, 
        max_items: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieves context relevant to the given thought.
        
        Uses simple keyword matching and recency for now.
        Could be enhanced with embeddings/semantic search.
        """
        async with self._lock:
            relevant = {
                "recent_thoughts": [],
                "related_insights": [],
                "tool_patterns": [],
                "graph_neighbors": []
            }
            
            # Get recent thoughts from graph
            if self.thought_graph.number_of_nodes() > 0:
                recent_nodes = sorted(
                    self.thought_graph.nodes(data=True),
                    key=lambda x: x[0],
                    reverse=True
                )[:max_items]
                
                relevant["recent_thoughts"] = [
                    {
                        "number": node[0],
                        "thought": node[1].get("thought", ""),
                        "confidence": node[1].get("confidence", 0.5)
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
                ins.to_dict() for _, ins in related_insights[:max_items//2]
            ]
            
            # Analyze tool usage patterns
            tool_counts = defaultdict(int)
            for decision in self.tool_usage_history[-20:]:  # Last 20 uses
                tool_counts[decision.tool_name] += 1
            
            relevant["tool_patterns"] = [
                {"tool": tool, "recent_uses": count}
                for tool, count in sorted(
                    tool_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
            return relevant
    
    async def add_insight(
        self,
        content: str,
        source_thought: int,
        confidence: float = 0.7,
        category: str = "general"
    ) -> None:
        """Adds a key insight to the context."""
        async with self._lock:
            insight = Insight(
                content=content,
                source_thought=source_thought,
                confidence=confidence,
                category=category,
                timestamp=datetime.utcnow()
            )
            self.key_insights.append(insight)
            logger.info(f"Added insight: {content[:50]}...")
    
    async def record_performance(
        self,
        metric_name: str,
        value: float
    ) -> None:
        """Records a performance metric."""
        async with self._lock:
            self.performance_metrics[metric_name].append(value)
    
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
                        "count": len(values)
                    }
            return summary
    
    async def get_thought_path(
        self,
        from_thought: int,
        to_thought: int
    ) -> Optional[List[int]]:
        """Finds the path between two thoughts in the graph."""
        async with self._lock:
            try:
                path = nx.shortest_path(
                    self.thought_graph,
                    from_thought,
                    to_thought
                )
                return path
            except nx.NetworkXNoPath:
                return None
    
    async def identify_cycles(self) -> List[List[int]]:
        """Identifies cycles in the thought graph."""
        async with self._lock:
            return list(nx.simple_cycles(self.thought_graph))
    
    async def export_state(self) -> Dict[str, Any]:
        """Exports the full context state."""
        async with self._lock:
            return {
                "memory_store": self.memory_store,
                "tool_usage_history": [
                    decision.model_dump() for decision in self.tool_usage_history
                ],
                "thought_graph": nx.node_link_data(self.thought_graph),
                "key_insights": [ins.to_dict() for ins in self.key_insights],
                "performance_metrics": dict(self.performance_metrics),
                "metadata": {
                    "created_at": self.created_at.isoformat(),
                    "last_updated": self.last_updated.isoformat(),
                    "access_count": self.access_count
                }
            }
    
    async def import_state(self, state: Dict[str, Any]) -> None:
        """Imports a previously exported state."""
        async with self._lock:
            self.memory_store = state.get("memory_store", {})
            
            # Reconstruct tool decisions
            self.tool_usage_history = [
                ToolDecision(**decision)
                for decision in state.get("tool_usage_history", [])
            ]
            
            # Reconstruct graph
            if "thought_graph" in state:
                self.thought_graph = nx.node_link_graph(state["thought_graph"])
            
            # Reconstruct insights
            self.key_insights = [
                Insight(
                    content=ins["content"],
                    source_thought=ins["source_thought"],
                    confidence=ins["confidence"],
                    category=ins["category"],
                    timestamp=datetime.fromisoformat(ins["timestamp"])
                )
                for ins in state.get("key_insights", [])
            ]
            
            self.performance_metrics = defaultdict(
                list,
                state.get("performance_metrics", {})
            )
            
            metadata = state.get("metadata", {})
            if "created_at" in metadata:
                self.created_at = datetime.fromisoformat(metadata["created_at"])
            if "access_count" in metadata:
                self.access_count = metadata["access_count"]
            
            self.last_updated = datetime.utcnow()