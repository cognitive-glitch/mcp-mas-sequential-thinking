"""
Tool-related models for thought processing system.
Contains models for tool recommendations, decisions, and selection results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, computed_field


class ToolRecommendation(BaseModel):
    """Recommendation for a specific tool."""

    tool_name: str = Field(..., description="Name of the recommended tool")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this recommendation"
    )
    rationale: str = Field(..., description="Why this tool is recommended")
    priority: int = Field(..., ge=1, description="Priority order (1=highest)")
    alternatives: List[str] = Field(
        default_factory=list, description="Alternative tools that could be used"
    )
    suggested_inputs: Optional[Dict[str, Any]] = Field(
        None, description="Suggested parameters for the tool"
    )
    expected_benefits: List[str] = Field(
        default_factory=list, description="Expected benefits of using this tool"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Known limitations or constraints"
    )

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is not empty."""
        if not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence recommendation."""
        return self.confidence >= 0.8

    @computed_field
    @property
    def has_alternatives(self) -> bool:
        """Check if alternatives are available."""
        return len(self.alternatives) > 0


class StepRecommendation(BaseModel):
    """Recommendation for a processing step."""

    step_description: str = Field(..., description="What needs to be done")
    recommended_tools: List[ToolRecommendation] = Field(
        default_factory=list, description="Tools recommended for this step"
    )
    expected_outcome: str = Field(..., description="Expected result from this step")
    next_step_conditions: List[str] = Field(
        default_factory=list, description="Conditions for next steps"
    )
    estimated_complexity: float = Field(
        0.5, ge=0.0, le=1.0, description="Estimated complexity of this step"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Dependencies on previous steps"
    )

    @computed_field
    @property
    def tool_count(self) -> int:
        """Number of tools recommended for this step."""
        return len(self.recommended_tools)

    @computed_field
    @property
    def primary_tool(self) -> Optional[str]:
        """Get the primary (highest priority) tool name."""
        if not self.recommended_tools:
            return None
        # Sort by priority (lower number = higher priority)
        sorted_tools = sorted(self.recommended_tools, key=lambda t: t.priority)
        return sorted_tools[0].tool_name

    @field_validator("recommended_tools")
    @classmethod
    def validate_tool_priorities(
        cls, v: List[ToolRecommendation]
    ) -> List[ToolRecommendation]:
        """Ensure tools have unique priorities."""
        if not v:
            return v

        priorities = [tool.priority for tool in v]
        if len(priorities) != len(set(priorities)):
            # Auto-fix duplicate priorities
            for i, tool in enumerate(v):
                tool.priority = i + 1
        return v


class ToolDecision(BaseModel):
    """Record of a tool selection decision."""

    tool_name: str = Field(..., description="Name of the selected tool")
    rationale: str = Field(..., description="Reasoning for this selection")
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Other tools that were considered"
    )
    confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Confidence in this decision"
    )
    outcome: Optional[str] = Field(None, description="Outcome after tool execution")
    execution_time_ms: Optional[int] = Field(
        None, description="Execution time in milliseconds"
    )
    success: bool = Field(True, description="Whether the tool execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @computed_field
    @property
    def was_successful(self) -> bool:
        """Check if the tool execution was successful."""
        return self.success and self.error_message is None

    @computed_field
    @property
    def execution_speed(self) -> Optional[str]:
        """Categorize execution speed."""
        if self.execution_time_ms is None:
            return None
        if self.execution_time_ms < 100:
            return "fast"
        elif self.execution_time_ms < 1000:
            return "moderate"
        else:
            return "slow"

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure outcome aligns with success status."""
        values = info.data
        if not values.get("success") and not v:
            return "Failed - see error message"
        return v


class ToolSelectionResult(BaseModel):
    """Result of tool selection process."""

    recommended_tool: ToolRecommendation = Field(
        ..., description="The recommended tool"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the selection")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in selection"
    )
    alternative_tools: List[str] = Field(
        default_factory=list, description="Alternative tools if primary fails"
    )
    context_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Context factors considered"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Any warnings or caveats"
    )

    @computed_field
    @property
    def requires_fallback(self) -> bool:
        """Check if fallback planning is recommended."""
        return self.confidence_score < 0.7 or len(self.warnings) > 0

    @computed_field
    @property
    def tool_name(self) -> str:
        """Quick access to recommended tool name."""
        return self.recommended_tool.tool_name

    def to_decision(self, outcome: Optional[str] = None) -> ToolDecision:
        """Convert selection result to a decision record."""
        return ToolDecision(
            tool_name=self.recommended_tool.tool_name,
            rationale=self.reasoning,
            alternatives_considered=self.alternative_tools,
            confidence=self.confidence_score,
            outcome=outcome,
            execution_time_ms=None,
            success=True,
            error_message=None,
        )


# Export all classes
__all__ = [
    "ToolRecommendation",
    "StepRecommendation",
    "ToolDecision",
    "ToolSelectionResult",
]
