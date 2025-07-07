"""
Backward-compatible imports for thought models.
This module maintains backward compatibility after splitting models into separate modules.
All original imports should continue to work while new code can import from specific modules.
"""

# Core models - fundamental data structures
from .core_models import (
    DomainType,
    ThoughtRelation,
    ThoughtData,
    ProcessedThought,
)

# Analysis models - quality assessment and reflection
from .analysis_models import (
    ReflectionFeedback,
    QualityIndicators,
    BranchAnalysis,
    ThoughtSequenceReview,
)

# Tool models - tool recommendations and decisions
from .tool_models import (
    ToolRecommendation,
    StepRecommendation,
    ToolDecision,
    ToolSelectionResult,
)

# Legacy PriorityLevel enum for compatibility (can be deprecated later)
from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for tool recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Export all models for backward compatibility
__all__ = [
    # Core models
    "DomainType",
    "ThoughtRelation",
    "ThoughtData",
    "ProcessedThought",
    # Analysis models
    "ReflectionFeedback",
    "QualityIndicators",
    "BranchAnalysis",
    "ThoughtSequenceReview",
    # Tool models
    "ToolRecommendation",
    "StepRecommendation",
    "ToolDecision",
    "ToolSelectionResult",
    # Legacy
    "PriorityLevel",
]
