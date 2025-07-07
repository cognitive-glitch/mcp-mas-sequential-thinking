"""
Core models for thought processing system.
Contains fundamental data structures for thoughts and their relationships.
"""

from typing import Any, ClassVar, Dict, List, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    computed_field,
    field_serializer,
)
from enum import Enum
import time
import re


class DomainType(str, Enum):
    """Domain types for topic alignment."""

    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    RESEARCH = "research"
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    GENERAL = "general"


class ThoughtRelation(BaseModel):
    """Represents a relationship between two thoughts in the graph."""

    from_thought: int = Field(..., description="Source thought number")
    to_thought: int = Field(..., description="Target thought number")
    relation_type: str = Field(
        ...,
        description="Type of relation (e.g., 'leads_to', 'contradicts', 'supports')",
    )
    strength: float = Field(
        0.5, ge=0.0, le=1.0, description="Strength of the relationship"
    )
    description: Optional[str] = Field(None, description="Optional description")

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: str) -> str:
        """Ensure relation type is meaningful."""
        valid_types = {
            "leads_to",
            "supports",
            "contradicts",
            "elaborates",
            "questions",
            "alternatives",
            "depends_on",
        }
        if v not in valid_types:
            # Allow custom types but log warning
            import logging

            logging.warning(f"Non-standard relation type: {v}")
        return v


class ThoughtData(BaseModel):
    """
    Enhanced ThoughtData model with backward compatibility.
    Core data structure representing a single thought in the thinking process.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # Required core fields
    thought: str = Field(..., description="The actual thought content")
    thoughtNumber: int = Field(..., ge=1, description="Sequential thought number")
    totalThoughts: int = Field(..., ge=1, description="Total expected thoughts")
    nextThoughtNeeded: bool = Field(..., description="Whether more thoughts are needed")

    # Optional metadata fields
    isRevision: bool = Field(
        False, description="Whether this revises a previous thought"
    )
    revisesThought: Optional[int] = Field(
        None, description="Which thought is being revised"
    )
    branchId: Optional[str] = Field(None, description="Branch identifier if branching")
    branchFromThought: Optional[int] = Field(
        None, description="Source thought for branch"
    )
    needsMoreThoughts: bool = Field(
        False, description="Indicates need for extended thinking"
    )
    confidence_score: float = Field(
        0.7, ge=0.0, le=1.0, description="Confidence in this thought"
    )

    # Enhanced fields for topic alignment and tool selection
    topic: Optional[str] = Field(
        None, description="Current topic or subject being explored"
    )
    subject: Optional[str] = Field(
        None, description="Specific subject area (legacy, use topic)"
    )
    domain: DomainType = Field(DomainType.GENERAL, description="Domain classification")
    keywords: List[str] = Field(
        default_factory=list, description="Key terms in this thought"
    )
    entities: List[str] = Field(
        default_factory=list, description="Named entities mentioned"
    )
    concepts: List[str] = Field(
        default_factory=list, description="Abstract concepts discussed"
    )

    # Tool recommendation fields (moved to separate models but kept reference)
    current_step: Optional[Any] = Field(None, description="Current step recommendation")
    previous_steps: List[Any] = Field(
        default_factory=list, description="History of previous steps"
    )
    tool_decisions: List[Any] = Field(
        default_factory=list, description="Tool selection decisions"
    )

    # Reflection and quality fields
    reflection_feedback: Optional[Any] = Field(
        None, description="Feedback from reflection team"
    )
    quality_indicators: Optional[Any] = Field(None, description="Quality metrics")

    # Relationships
    thought_relationships: List[ThoughtRelation] = Field(
        default_factory=list, description="Relationships to other thoughts"
    )

    # Context and metadata
    context_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of context at thought time"
    )
    timestamp_ms: Optional[int] = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Creation timestamp in milliseconds",
    )

    # Type stubs
    MIN_THOUGHT_LENGTH: ClassVar[int] = 10
    MIN_TOTAL_THOUGHTS: ClassVar[int] = 5

    @field_validator("thought")
    @classmethod
    def validate_thought_content(cls, v: str) -> str:
        """Ensure thought has meaningful content."""
        if len(v.strip()) < cls.MIN_THOUGHT_LENGTH:
            raise ValueError(
                f"Thought content must be at least {cls.MIN_THOUGHT_LENGTH} characters long"
            )
        return v.strip()

    @field_validator("keywords", "entities", "concepts")
    @classmethod
    def clean_string_lists(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate string lists."""
        if not v:
            return []
        # Remove empty strings and duplicates while preserving order
        seen = set()
        result = []
        for item in v:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    @model_validator(mode="after")
    def validate_thought_consistency(self) -> "ThoughtData":
        """Ensure internal consistency of thought data."""
        # Validate revision logic
        if self.isRevision and self.revisesThought is None:
            raise ValueError("isRevision requires revisesThought to be specified")

        if self.revisesThought is not None:
            self.isRevision = True
            if self.revisesThought >= self.thoughtNumber:
                raise ValueError("Cannot revise a future thought")
            if self.revisesThought < 1:
                raise ValueError("revisesThought must be >= 1")

        # Validate branch logic
        if self.branchId and self.branchFromThought is None:
            raise ValueError("branchId requires branchFromThought")

        if self.branchFromThought is not None:
            if not self.branchId:
                self.branchId = f"branch-{self.branchFromThought}-{self.thoughtNumber}"
            if self.branchFromThought >= self.thoughtNumber:
                raise ValueError("Cannot branch from a future thought")
            if self.branchFromThought < 1:
                raise ValueError("branchFromThought must be >= 1")

            # Additional validation: prevent consecutive branching
            if self.branchFromThought == self.thoughtNumber - 1:
                raise ValueError(
                    "Branching from immediately previous thought creates ambiguity. "
                    "Consider revision instead of branching."
                )

        # Validate completion logic
        if not self.nextThoughtNeeded:
            expected_min = max(self.totalThoughts, self.MIN_TOTAL_THOUGHTS)
            progress = self.thoughtNumber / expected_min
            if progress < 0.8 and not self.needsMoreThoughts:
                raise ValueError(
                    f"Ending thought sequence too early - insufficient progress "
                    f"({self.thoughtNumber}/{expected_min} = {progress:.1%})"
                )

        # Auto-populate keywords from thought content
        if not self.keywords and len(self.thought) > 20:
            words = re.findall(r"\b\w{4,}\b", self.thought.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            # Get top 5 most frequent meaningful words
            self.keywords = sorted(
                word_freq.keys(), key=lambda x: word_freq[x], reverse=True
            )[:5]

        # Sync subject and topic for backward compatibility
        if self.subject and not self.topic:
            self.topic = self.subject
        elif self.topic and not self.subject:
            self.subject = self.topic

        return self

    @computed_field
    @property
    def thought_id(self) -> str:
        """Generates a unique identifier for this thought."""
        if self.branchId:
            return f"{self.branchId}-{self.thoughtNumber}"
        return f"main-{self.thoughtNumber}"

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if this thought completes the sequence."""
        return not self.nextThoughtNeeded

    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Calculate progress through the thought sequence."""
        return (self.thoughtNumber / max(self.totalThoughts, 1)) * 100

    @field_serializer("timestamp_ms")
    def serialize_timestamp(self, value: Optional[int]) -> Optional[int]:
        """Ensure timestamp is always included in serialization."""
        return value or int(time.time() * 1000)

    def to_concise_dict(self) -> Dict[str, Any]:
        """Returns a concise representation for logging/display."""
        return {
            "number": self.thoughtNumber,
            "content": self.thought[:100] + "..."
            if len(self.thought) > 100
            else self.thought,
            "complete": self.is_complete,
            "confidence": self.confidence_score,
            "domain": self.domain.value,
        }


class ProcessedThought(BaseModel):
    """Enhanced result of processing a thought through the system."""

    thought_data: "ThoughtData"
    coordinator_response: str
    reflection_response: Optional[str] = None
    integrated_response: str
    next_step_guidance: str
    execution_time_ms: int
    token_usage: Dict[str, int] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    # Enhanced fields
    tool_recommendations_generated: bool = Field(
        False, description="Whether tool recommendations were generated"
    )
    reflection_applied: bool = Field(
        False, description="Whether reflection feedback was applied"
    )
    context_updated: bool = Field(
        False, description="Whether shared context was updated"
    )

    # Enhanced computed fields for ProcessedThought
    @computed_field
    @property
    def processing_efficiency(self) -> float:
        """Calculates processing efficiency based on execution time and content complexity."""
        if self.execution_time_ms <= 0:
            return 0.0

        # Estimate content complexity (simple heuristic)
        content_length = len(self.integrated_response) + len(self.thought_data.thought)
        complexity_factor = min(
            content_length / 1000.0, 3.0
        )  # Normalize to reasonable range

        # Efficiency = complexity handled per second
        time_seconds = self.execution_time_ms / 1000.0
        return complexity_factor / max(time_seconds, 0.1)

    @computed_field
    @property
    def quality_score(self) -> float:
        """Estimates overall processing quality."""
        factors = []

        # Success factor
        factors.append(1.0 if self.success else 0.0)

        # Tool recommendation quality
        if self.tool_recommendations_generated:
            factors.append(0.8)

        # Reflection quality
        if self.reflection_applied and self.reflection_response:
            factors.append(0.9)

        # Context integration
        if self.context_updated:
            factors.append(0.7)

        # Thought data quality (use confidence score as proxy)
        if hasattr(self.thought_data, "confidence_score"):
            factors.append(self.thought_data.confidence_score)

        return sum(factors) / len(factors) if factors else 0.5

    # Enhanced field serializers
    @field_serializer("execution_time_ms")
    def serialize_execution_time(self, value: int) -> int:
        """Ensures execution time is non-negative."""
        return max(value, 0)

    @field_serializer("integrated_response")
    def serialize_integrated_response(self, value: str) -> str:
        """Cleans and formats integrated response."""
        return value.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Enhanced validation
    @model_validator(mode="after")
    def validate_processing_consistency(self) -> "ProcessedThought":
        """Validates processing result consistency."""

        # Error state consistency
        if not self.success and not self.error:
            self.error = "Processing failed with unknown error"
        elif self.success and self.error:
            # Success but error present - clear error or mark as warning
            if "warning" not in self.error.lower():
                self.success = False

        # Tool recommendations consistency
        # Only validate if explicitly set to True and we have thought_data
        if self.tool_recommendations_generated and hasattr(self, "thought_data"):
            if self.thought_data and self.thought_data.current_step is None:
                # Only change if there are no tool decisions either
                if not self.thought_data.tool_decisions:
                    self.tool_recommendations_generated = False

        # Reflection consistency
        if self.reflection_applied and not self.reflection_response:
            self.reflection_applied = False

        # Response content validation
        if self.success and len(self.integrated_response.strip()) < 10:
            raise ValueError(
                "Successful processing should produce substantial response"
            )

        return self


# Export all classes
__all__ = [
    "DomainType",
    "ThoughtRelation",
    "ThoughtData",
    "ProcessedThought",
]
