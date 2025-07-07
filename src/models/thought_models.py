"""
Enhanced Pydantic models for thought data structures with tool recommendations and topic alignment.
Cherry-picked and enhanced from schema.ts with reflection capabilities.
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
    ValidationInfo,
)
from enum import Enum
import time
import hashlib
import re


class PriorityLevel(str, Enum):
    """Priority levels for tool recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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


class ToolRecommendation(BaseModel):
    """Enhanced tool recommendation from schema.ts with reflection support."""

    tool_name: str = Field(..., description="Name of the recommended tool")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in recommendation (0-1)"
    )
    rationale: str = Field(..., description="Reasoning for recommending this tool")
    priority: int = Field(..., ge=1, description="Execution order priority (1=highest)")
    suggested_inputs: Optional[Dict[str, Any]] = Field(
        None, description="Suggested parameters for tool"
    )
    alternatives: List[str] = Field(
        default_factory=list, description="Alternative tools that could be used"
    )
    expected_outcome: str = Field(
        ..., description="What we expect this tool to achieve"
    )
    risk_assessment: Optional[str] = Field(
        None, description="Potential risks or limitations"
    )
    execution_time_estimate: Optional[int] = Field(
        None, description="Estimated execution time in ms"
    )

    # Computed fields using Pydantic v2
    @computed_field
    @property
    def effectiveness_score(self) -> float:
        """Computes effectiveness score combining confidence and priority."""
        # Higher confidence and lower priority (1 is highest) = higher effectiveness
        priority_factor = 1.0 / max(self.priority, 1)  # Avoid division by zero
        return (self.confidence * 0.7) + (priority_factor * 0.3)

    @computed_field
    @property
    def risk_level(self) -> str:
        """Computes risk level based on confidence and risk assessment."""
        if self.confidence >= 0.8 and (
            not self.risk_assessment or "low" in self.risk_assessment.lower()
        ):
            return "low"
        elif self.confidence >= 0.6:
            return "medium"
        else:
            return "high"

    # Enhanced field serializers
    @field_serializer("rationale")
    def serialize_rationale(self, value: str) -> str:
        """Ensures rationale is properly formatted for MCP output."""
        return value.strip().replace("\n", " ").replace("\r", "")

    @field_serializer("confidence")
    def serialize_confidence(self, value: float) -> float:
        """Rounds confidence to 2 decimal places for consistency."""
        return round(value, 2)


class StepRecommendation(BaseModel):
    """Step-level recommendation with tool orchestration."""

    step_description: str = Field(..., description="What needs to be done in this step")
    recommended_tools: List[ToolRecommendation] = Field(
        ..., description="Tools recommended for this step"
    )
    expected_outcome: str = Field(..., description="Expected result of this step")
    next_step_conditions: List[str] = Field(
        default_factory=list, description="Conditions for proceeding to next step"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Dependencies on previous steps"
    )
    validation_criteria: List[str] = Field(
        default_factory=list, description="How to validate step completion"
    )

    # Computed fields
    @computed_field
    @property
    def complexity_score(self) -> float:
        """Computes complexity based on tool count and dependencies."""
        tool_factor = min(len(self.recommended_tools) / 5.0, 1.0)  # Normalize to 0-1
        dependency_factor = min(len(self.dependencies) / 3.0, 1.0)  # Normalize to 0-1
        return (tool_factor * 0.6) + (dependency_factor * 0.4)

    @computed_field
    @property
    def avg_tool_confidence(self) -> float:
        """Average confidence of all recommended tools."""
        if not self.recommended_tools:
            return 0.0
        return sum(tool.confidence for tool in self.recommended_tools) / len(
            self.recommended_tools
        )

    # Enhanced validators
    @field_validator("recommended_tools")
    @classmethod
    def validate_tool_priorities(
        cls, tools: List[ToolRecommendation]
    ) -> List[ToolRecommendation]:
        """Validates that tool priorities are unique and properly ordered."""
        if not tools:
            return tools

        priorities = [tool.priority for tool in tools]
        if len(set(priorities)) != len(priorities):
            # Auto-fix duplicate priorities
            for i, tool in enumerate(tools):
                tool.priority = i + 1

        return sorted(tools, key=lambda x: x.priority)


class SessionContext(BaseModel):
    """Context for the current session including available tools and state."""

    session_id: str = Field(..., description="Unique session identifier")
    available_tools: List[str] = Field(
        default_factory=list, description="Tools available in this session"
    )
    session_topic: Optional[str] = Field(
        None, description="Main topic/subject of the session"
    )
    session_domain: DomainType = Field(
        DomainType.GENERAL, description="Primary domain of the session"
    )
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User preferences for tool selection"
    )
    success_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Success metrics for this session"
    )


class ToolDecision(BaseModel):
    """Tracks tool usage decisions for reflection."""

    tool_name: str = Field(..., description="Name of the tool used")
    rationale: str = Field(..., description="Reasoning for using this tool")
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Other tools considered"
    )
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence in tool choice"
    )
    outcome: Optional[str] = Field(None, description="Result of tool usage")
    execution_time_ms: Optional[int] = Field(
        None, description="Time taken to execute tool"
    )


class ThoughtRelation(BaseModel):
    """Represents relationships between thoughts."""

    from_thought: int = Field(..., ge=1, description="Source thought number")
    to_thought: int = Field(..., ge=1, description="Target thought number")
    relation_type: str = Field(
        ..., description="Type of relationship (supports, contradicts, refines, etc)"
    )
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Strength of relationship")


class ReflectionFeedback(BaseModel):
    """Feedback from the reflection team."""

    strengths: List[str] = Field(
        default_factory=list, description="Identified strengths"
    )
    weaknesses: List[str] = Field(
        default_factory=list, description="Identified weaknesses"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    patterns_detected: List[str] = Field(
        default_factory=list, description="Detected patterns"
    )
    overall_quality: float = Field(
        0.5, ge=0.0, le=1.0, description="Overall quality score"
    )
    cognitive_biases: List[str] = Field(
        default_factory=list, description="Detected cognitive biases"
    )


class ThoughtData(BaseModel):
    """
    Enhanced thought data structure with tool recommendations, topic alignment, and reflection support.
    Cherry-picked features from schema.ts and enhanced with reflection capabilities.
    """

    # Core fields
    thought: str = Field(
        ...,
        description="Content of the current thought or step",
        min_length=1,
        max_length=10000,
    )
    thoughtNumber: int = Field(
        ..., description="Sequence number of this thought (starting from 1)", ge=1
    )
    totalThoughts: int = Field(
        ..., description="Estimated total number of thoughts required", ge=1
    )
    nextThoughtNeeded: bool = Field(
        ..., description="Indicates if another thought step is expected"
    )

    # Topic/Subject Alignment (NEW - User requested)
    topic: Optional[str] = Field(
        None, description="Main topic or subject matter of this thought", max_length=200
    )
    subject: Optional[str] = Field(
        None, description="Specific subject area being addressed", max_length=200
    )
    domain: DomainType = Field(
        DomainType.GENERAL, description="Problem domain for better alignment"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Key terms relevant to this thought"
    )

    # Revision and branching
    isRevision: bool = Field(
        False, description="Flags if this thought revises a previous one"
    )
    revisesThought: Optional[int] = Field(
        None, description="The thoughtNumber being revised", ge=1
    )
    branchFromThought: Optional[int] = Field(
        None, description="The thoughtNumber from which this branches", ge=1
    )
    branchId: Optional[str] = Field(
        None,
        description="Unique identifier for the branch",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    needsMoreThoughts: bool = Field(
        False, description="Flags if more thoughts needed beyond estimate"
    )

    # Tool Recommendation System (Enhanced from schema.ts)
    current_step: Optional[StepRecommendation] = Field(
        None, description="Current step recommendation with tools"
    )
    previous_steps: List[StepRecommendation] = Field(
        default_factory=list, description="Steps already completed"
    )
    remaining_steps: List[str] = Field(
        default_factory=list, description="High-level descriptions of upcoming steps"
    )

    # Enhanced fields for reflection
    context_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Current context state"
    )
    tool_decisions: List[ToolDecision] = Field(
        default_factory=list, description="Tool usage decisions"
    )
    reflection_feedback: Optional[ReflectionFeedback] = Field(
        None, description="Feedback from reflection team"
    )
    confidence_score: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence in this thought"
    )
    thought_relationships: List[ThoughtRelation] = Field(
        default_factory=list, description="Relationships to other thoughts"
    )

    # Session Integration
    session_context: Optional[SessionContext] = Field(
        None, description="Session context for tool alignment"
    )

    # Metadata
    timestamp_ms: Optional[int] = Field(
        None, description="Timestamp when thought was created"
    )
    processing_time_ms: Optional[int] = Field(
        None, description="Time taken to process this thought"
    )

    # Advanced computed fields using Pydantic v2
    @computed_field
    @property
    def thought_id(self) -> str:
        """Generates a unique identifier for this thought."""
        content_hash = hashlib.md5(self.thought.encode()).hexdigest()[:8]
        return f"thought_{self.thoughtNumber}_{content_hash}"

    @computed_field
    @property
    def content_complexity(self) -> float:
        """Analyzes content complexity based on multiple factors."""
        text = self.thought.lower()

        # Word count factor (0-1)
        word_count = len(text.split())
        word_factor = min(word_count / 100.0, 1.0)  # Normalize to 100 words

        # Technical terms factor
        technical_terms = [
            "algorithm",
            "implementation",
            "optimization",
            "analysis",
            "architecture",
        ]
        tech_count = sum(1 for term in technical_terms if term in text)
        tech_factor = min(tech_count / 5.0, 1.0)

        # Question complexity (more questions = more complexity)
        question_count = (
            text.count("?") + text.count("how") + text.count("why") + text.count("what")
        )
        question_factor = min(question_count / 5.0, 1.0)

        return (word_factor * 0.4) + (tech_factor * 0.4) + (question_factor * 0.2)

    @computed_field
    @property
    def topic_alignment_score(self) -> float:
        """Computes how well this thought aligns with stated topic/subject."""
        if not self.topic and not self.subject:
            return 0.5  # Neutral when no topic specified

        text = self.thought.lower()
        alignment_score = 0.0
        checks = 0

        if self.topic:
            checks += 1
            topic_words = self.topic.lower().split()
            topic_mentions = sum(1 for word in topic_words if word in text)
            alignment_score += min(topic_mentions / max(len(topic_words), 1), 1.0)

        if self.subject:
            checks += 1
            subject_words = self.subject.lower().split()
            subject_mentions = sum(1 for word in subject_words if word in text)
            alignment_score += min(subject_mentions / max(len(subject_words), 1), 1.0)

        if self.keywords:
            checks += 1
            keyword_mentions = sum(
                1 for keyword in self.keywords if keyword.lower() in text
            )
            alignment_score += min(keyword_mentions / max(len(self.keywords), 1), 1.0)

        return alignment_score / max(checks, 1)

    @computed_field
    @property
    def overall_quality_estimate(self) -> float:
        """Estimates overall thought quality from multiple factors."""
        factors = []

        # Content complexity (balanced - not too simple, not too complex)
        complexity = self.content_complexity
        complexity_score = 1.0 - abs(complexity - 0.6)  # Optimal around 0.6
        factors.append(complexity_score)

        # Topic alignment
        factors.append(self.topic_alignment_score)

        # Confidence score
        factors.append(self.confidence_score)

        # Tool recommendation quality
        if self.current_step and self.current_step.recommended_tools:
            avg_tool_confidence = self.current_step.avg_tool_confidence
            factors.append(avg_tool_confidence)

        # Reflection feedback if available
        if self.reflection_feedback:
            factors.append(self.reflection_feedback.overall_quality)

        return sum(factors) / len(factors)

    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Calculates progress percentage through thought sequence."""
        return (self.thoughtNumber / self.totalThoughts) * 100

    @computed_field
    @property
    def is_final_thought(self) -> bool:
        """Determines if this is likely the final thought."""
        return not self.nextThoughtNeeded and self.thoughtNumber >= self.totalThoughts

    # Enhanced field serializers for MCP compatibility
    @field_serializer("timestamp_ms")
    def serialize_timestamp(self, value: Optional[int]) -> Optional[int]:
        """Ensures timestamp is properly formatted or generates current time."""
        if value is None:
            return int(time.time() * 1000)
        return value

    @field_serializer("thought")
    def serialize_thought_content(self, value: str) -> str:
        """Cleans and formats thought content for MCP output."""
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r"\s+", " ", value.strip())
        # Ensure reasonable length for MCP
        if len(cleaned) > 8000:
            cleaned = cleaned[:7950] + "... [truncated]"
        return cleaned

    @field_serializer("keywords")
    def serialize_keywords(self, value: List[str]) -> List[str]:
        """Ensures keywords are properly formatted and deduplicated."""
        # Remove duplicates while preserving order and case
        seen = set()
        unique_keywords = []
        for keyword in value:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen:
                seen.add(keyword_lower)
                unique_keywords.append(keyword.strip())
        return unique_keywords[:15]  # Limit to 15 keywords

    @field_serializer("confidence_score")
    def serialize_confidence(self, value: float) -> float:
        """Rounds confidence to 3 decimal places for precision."""
        return round(value, 3)

    # Configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,  # Allow modifications for reflection updates
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "thought": "Analyze the core problem structure for code optimization",
                    "thoughtNumber": 1,
                    "totalThoughts": 5,
                    "nextThoughtNeeded": True,
                    "topic": "Performance Optimization",
                    "subject": "Algorithm Analysis",
                    "domain": "technical",
                    "keywords": ["performance", "optimization", "algorithm"],
                    "confidence_score": 0.8,
                    "context_snapshot": {"problem_type": "optimization"},
                    "current_step": {
                        "step_description": "Analyze current algorithm complexity",
                        "recommended_tools": [
                            {
                                "tool_name": "code_analysis",
                                "confidence": 0.9,
                                "rationale": "Need to understand current performance bottlenecks",
                                "priority": 1,
                                "expected_outcome": "Identify O(n) complexity issues",
                            }
                        ],
                        "expected_outcome": "Understanding of performance characteristics",
                    },
                }
            ]
        },
    )

    # Class-level constants
    MIN_TOTAL_THOUGHTS: ClassVar[int] = 5

    @field_validator("totalThoughts")
    @classmethod
    def validate_total_thoughts_minimum(cls, v: int) -> int:
        """Ensures totalThoughts meets minimum requirement."""
        if v < cls.MIN_TOTAL_THOUGHTS:
            return cls.MIN_TOTAL_THOUGHTS
        return v

    @field_validator("revisesThought")
    @classmethod
    def validate_revises_thought(
        cls, v: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        """Validates revision logic."""
        is_revision = info.data.get("isRevision", False)
        thought_number = info.data.get("thoughtNumber")

        if v is not None:
            if not is_revision:
                raise ValueError("revisesThought requires isRevision=True")
            if thought_number and v >= thought_number:
                raise ValueError("revisesThought must be less than thoughtNumber")
        elif is_revision:
            raise ValueError("isRevision=True requires revisesThought")

        return v

    @field_validator("branchId")
    @classmethod
    def validate_branch_id(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Validates branch logic."""
        branch_from = info.data.get("branchFromThought")

        if v is not None and branch_from is None:
            raise ValueError("branchId requires branchFromThought")
        elif branch_from is not None and v is None:
            raise ValueError("branchFromThought requires branchId")

        return v

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validates and cleans keywords."""
        if len(v) > 20:  # Reasonable limit
            v = v[:20]
        # Clean keywords while preserving case and allowing special characters
        cleaned = []
        for keyword in v:
            if keyword and isinstance(keyword, str):
                clean_keyword = keyword.strip()
                if clean_keyword and len(clean_keyword) <= 50:
                    cleaned.append(clean_keyword)
        return cleaned

    @field_validator("topic", "subject")
    @classmethod
    def validate_topic_subject(cls, v: Optional[str]) -> Optional[str]:
        """Validates topic and subject fields."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

    @field_validator("thought")
    @classmethod
    def validate_thought_content(cls, v: str) -> str:
        """Enhanced thought content validation."""
        v = v.strip()

        # Check for minimum meaningful content
        if len(v) < 10:
            raise ValueError("Thought content must be at least 10 characters long")

        # Check for suspicious patterns that might indicate poor quality
        if v.lower() in ["test", "testing", "debug", "placeholder"]:
            raise ValueError("Please provide meaningful thought content")

        # Check for excessive repetition
        words = v.lower().split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                raise ValueError("Thought content appears to be too repetitive")

        return v

    @field_validator("thoughtNumber", "totalThoughts")
    @classmethod
    def validate_thought_numbers(cls, v: int, info: ValidationInfo) -> int:
        """Enhanced validation for thought numbers."""
        field_name = info.field_name

        if field_name == "thoughtNumber":
            if v > 1000:  # Reasonable upper limit
                raise ValueError("thoughtNumber exceeds reasonable limit (1000)")
        elif field_name == "totalThoughts":
            if v > 1000:  # Reasonable upper limit
                raise ValueError("totalThoughts exceeds reasonable limit (1000)")
            # Get thoughtNumber from context if available
            thought_num = info.data.get("thoughtNumber")
            if thought_num and v < thought_num:
                # Auto-adjust totalThoughts if it's too low
                return max(v, thought_num)

        return v

    @model_validator(mode="after")
    def validate_comprehensive_logic(self) -> "ThoughtData":
        """Comprehensive validation of thought logic and relationships."""

        # Validate branch relationships
        if self.branchFromThought is not None:
            if self.branchFromThought >= self.thoughtNumber:
                raise ValueError("branchFromThought must be less than thoughtNumber")

            # Branch should have meaningful difference from parent
            if self.branchFromThought == self.thoughtNumber - 1 and not self.isRevision:
                raise ValueError("Consecutive branching may indicate revision instead")

        # Validate revision logic
        if self.isRevision and self.revisesThought:
            if self.revisesThought >= self.thoughtNumber:
                raise ValueError("Cannot revise future or current thought")

            # Revision should be meaningful (different content)
            if len(self.thought) < 20:
                raise ValueError("Revision should provide substantial new content")

        # Validate thought relationships don't create cycles
        if self.thought_relationships:
            for rel in self.thought_relationships:
                if rel.from_thought == rel.to_thought:
                    raise ValueError("Thought cannot relate to itself")

                # Validate relationship strength
                if rel.strength < 0.1:
                    raise ValueError("Relationship strength too weak to be meaningful")

        # Validate tool recommendation consistency
        if self.current_step and self.current_step.recommended_tools:
            high_confidence_tools = [
                t for t in self.current_step.recommended_tools if t.confidence > 0.8
            ]
            if high_confidence_tools and self.confidence_score < 0.5:
                # If we have high-confidence tools, thought confidence should be reasonable
                self.confidence_score = max(self.confidence_score, 0.6)

        # Validate final thought logic
        if not self.nextThoughtNeeded and self.thoughtNumber < self.totalThoughts:
            # Auto-adjust if final thought doesn't match expectations
            if self.thoughtNumber >= max(
                3, self.totalThoughts * 0.8
            ):  # At least 80% complete
                self.totalThoughts = self.thoughtNumber
            else:
                raise ValueError(
                    "Ending thought sequence too early - insufficient progress"
                )

        return self

    def to_log_format(self) -> str:
        """Enhanced log format with topic/subject information."""
        prefix = "Thought"
        context = ""

        if self.isRevision:
            prefix = "🔄 Revision"
            context = f" (revising #{self.revisesThought})"
        elif self.branchFromThought:
            prefix = "🌿 Branch"
            context = f" (from #{self.branchFromThought}, ID: {self.branchId})"

        # Add topic/subject info
        topic_info = ""
        if self.topic or self.subject:
            parts = []
            if self.topic:
                parts.append(f"Topic: {self.topic}")
            if self.subject:
                parts.append(f"Subject: {self.subject}")
            if self.domain != DomainType.GENERAL:
                parts.append(f"Domain: {self.domain.value}")
            topic_info = f" [{', '.join(parts)}]"

        quality = ""
        if self.reflection_feedback:
            quality = f" [Quality: {self.reflection_feedback.overall_quality:.2f}]"

        confidence = f" [Confidence: {self.confidence_score:.2f}]"

        # Add tool recommendation summary
        tools_info = ""
        if self.current_step and self.current_step.recommended_tools:
            tool_names = [
                tool.tool_name for tool in self.current_step.recommended_tools[:3]
            ]
            tools_info = f" [Tools: {', '.join(tool_names)}{'...' if len(self.current_step.recommended_tools) > 3 else ''}]"

        return f"{prefix} {self.thoughtNumber}/{self.totalThoughts}{context}{topic_info}{quality}{confidence}{tools_info}\n{self.thought}"

    def get_tool_summary(self) -> Dict[str, Any]:
        """Returns summary of tool recommendations for this thought."""
        if not self.current_step:
            return {"tools": [], "step_description": None}

        return {
            "step_description": self.current_step.step_description,
            "tools": [
                {
                    "name": tool.tool_name,
                    "confidence": tool.confidence,
                    "priority": tool.priority,
                    "rationale": tool.rationale[:100] + "..."
                    if len(tool.rationale) > 100
                    else tool.rationale,
                }
                for tool in sorted(
                    self.current_step.recommended_tools, key=lambda x: x.priority
                )
            ],
            "expected_outcome": self.current_step.expected_outcome,
        }

    def get_alignment_summary(self) -> Dict[str, str]:
        """Returns topic/subject alignment summary."""
        return {
            "topic": self.topic or "Not specified",
            "subject": self.subject or "Not specified",
            "domain": self.domain.value,
            "keywords": ", ".join(self.keywords[:5])
            + ("..." if len(self.keywords) > 5 else ""),
        }


class BranchAnalysis(BaseModel):
    """Analysis of a thought branch for the review tool."""

    branch_id: str = Field(..., description="Branch identifier")
    branch_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality of this branch"
    )
    thought_count: int = Field(
        ..., ge=0, description="Number of thoughts in this branch"
    )
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights from this branch"
    )
    completion_status: str = Field(
        ..., description="Whether branch is complete, abandoned, or ongoing"
    )
    recommendation: str = Field(..., description="Recommendation for this branch")


class ThoughtSequenceReview(BaseModel):
    """Comprehensive review of a thought sequence for reflectivereview tool."""

    session_id: str = Field(..., description="Session identifier")
    total_thoughts: int = Field(..., ge=0, description="Total number of thoughts")
    total_branches: int = Field(..., ge=0, description="Total number of branches")
    overall_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Overall sequence quality"
    )

    # Analysis sections
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights discovered"
    )
    patterns_identified: List[str] = Field(
        default_factory=list, description="Patterns in thinking"
    )
    quality_trends: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics over time"
    )
    tool_effectiveness: Dict[str, float] = Field(
        default_factory=dict, description="Tool usage effectiveness"
    )

    # Branch analysis
    branch_analyses: List[BranchAnalysis] = Field(
        default_factory=list, description="Analysis of each branch"
    )
    best_branch: Optional[str] = Field(
        None, description="ID of the most promising branch"
    )

    # Recommendations
    next_steps: List[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    areas_for_improvement: List[str] = Field(
        default_factory=list, description="Areas needing improvement"
    )
    topic_alignment_score: float = Field(
        0.5, ge=0.0, le=1.0, description="How well aligned with stated topic"
    )

    # Metadata
    review_timestamp: Optional[int] = Field(
        None, description="When this review was generated"
    )
    review_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Confidence in this review"
    )


class ProcessedThought(BaseModel):
    """Enhanced result of processing a thought through the system."""

    thought_data: ThoughtData
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

        # Thought data quality
        if hasattr(self.thought_data, "overall_quality_estimate"):
            factors.append(self.thought_data.overall_quality_estimate)

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
