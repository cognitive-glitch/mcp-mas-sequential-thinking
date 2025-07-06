"""
Enhanced Pydantic models for thought data structures with tool recommendations and topic alignment.
Cherry-picked and enhanced from schema.ts with reflection capabilities.
"""

from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo
from enum import Enum


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
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation (0-1)")
    rationale: str = Field(..., description="Reasoning for recommending this tool")
    priority: int = Field(..., ge=1, description="Execution order priority (1=highest)")
    suggested_inputs: Optional[Dict[str, Any]] = Field(None, description="Suggested parameters for tool")
    alternatives: List[str] = Field(default_factory=list, description="Alternative tools that could be used")
    expected_outcome: str = Field(..., description="What we expect this tool to achieve")
    risk_assessment: Optional[str] = Field(None, description="Potential risks or limitations")
    execution_time_estimate: Optional[int] = Field(None, description="Estimated execution time in ms")


class StepRecommendation(BaseModel):
    """Step-level recommendation with tool orchestration."""
    step_description: str = Field(..., description="What needs to be done in this step")
    recommended_tools: List[ToolRecommendation] = Field(..., description="Tools recommended for this step")
    expected_outcome: str = Field(..., description="Expected result of this step")
    next_step_conditions: List[str] = Field(default_factory=list, description="Conditions for proceeding to next step")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on previous steps")
    validation_criteria: List[str] = Field(default_factory=list, description="How to validate step completion")


class SessionContext(BaseModel):
    """Context for the current session including available tools and state."""
    session_id: str = Field(..., description="Unique session identifier")
    available_tools: List[str] = Field(default_factory=list, description="Tools available in this session")
    session_topic: Optional[str] = Field(None, description="Main topic/subject of the session")
    session_domain: DomainType = Field(DomainType.GENERAL, description="Primary domain of the session")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences for tool selection")
    success_metrics: Dict[str, float] = Field(default_factory=dict, description="Success metrics for this session")


class ToolDecision(BaseModel):
    """Tracks tool usage decisions for reflection."""
    tool_name: str = Field(..., description="Name of the tool used")
    rationale: str = Field(..., description="Reasoning for using this tool")
    alternatives_considered: List[str] = Field(default_factory=list, description="Other tools considered")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in tool choice")
    outcome: Optional[str] = Field(None, description="Result of tool usage")
    execution_time_ms: Optional[int] = Field(None, description="Time taken to execute tool")


class ThoughtRelation(BaseModel):
    """Represents relationships between thoughts."""
    from_thought: int = Field(..., ge=1, description="Source thought number")
    to_thought: int = Field(..., ge=1, description="Target thought number")
    relation_type: str = Field(..., description="Type of relationship (supports, contradicts, refines, etc)")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Strength of relationship")


class ReflectionFeedback(BaseModel):
    """Feedback from the reflection team."""
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    patterns_detected: List[str] = Field(default_factory=list, description="Detected patterns")
    overall_quality: float = Field(0.5, ge=0.0, le=1.0, description="Overall quality score")
    cognitive_biases: List[str] = Field(default_factory=list, description="Detected cognitive biases")


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
        max_length=10000
    )
    thoughtNumber: int = Field(
        ...,
        description="Sequence number of this thought (starting from 1)",
        ge=1
    )
    totalThoughts: int = Field(
        ...,
        description="Estimated total number of thoughts required",
        ge=1
    )
    nextThoughtNeeded: bool = Field(
        ...,
        description="Indicates if another thought step is expected"
    )
    
    # Topic/Subject Alignment (NEW - User requested)
    topic: Optional[str] = Field(
        None,
        description="Main topic or subject matter of this thought",
        max_length=200
    )
    subject: Optional[str] = Field(
        None, 
        description="Specific subject area being addressed",
        max_length=200
    )
    domain: DomainType = Field(
        DomainType.GENERAL,
        description="Problem domain for better alignment"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms relevant to this thought"
    )
    
    # Revision and branching
    isRevision: bool = Field(
        False,
        description="Flags if this thought revises a previous one"
    )
    revisesThought: Optional[int] = Field(
        None,
        description="The thoughtNumber being revised",
        ge=1
    )
    branchFromThought: Optional[int] = Field(
        None,
        description="The thoughtNumber from which this branches",
        ge=1
    )
    branchId: Optional[str] = Field(
        None,
        description="Unique identifier for the branch",
        pattern=r"^[a-zA-Z0-9_-]+$"
    )
    needsMoreThoughts: bool = Field(
        False,
        description="Flags if more thoughts needed beyond estimate"
    )
    
    # Tool Recommendation System (Enhanced from schema.ts)
    current_step: Optional[StepRecommendation] = Field(
        None,
        description="Current step recommendation with tools"
    )
    previous_steps: List[StepRecommendation] = Field(
        default_factory=list,
        description="Steps already completed"
    )
    remaining_steps: List[str] = Field(
        default_factory=list,
        description="High-level descriptions of upcoming steps"
    )
    
    # Enhanced fields for reflection
    context_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current context state"
    )
    tool_decisions: List[ToolDecision] = Field(
        default_factory=list,
        description="Tool usage decisions"
    )
    reflection_feedback: Optional[ReflectionFeedback] = Field(
        None,
        description="Feedback from reflection team"
    )
    confidence_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this thought"
    )
    thought_relationships: List[ThoughtRelation] = Field(
        default_factory=list,
        description="Relationships to other thoughts"
    )
    
    # Session Integration
    session_context: Optional[SessionContext] = Field(
        None,
        description="Session context for tool alignment"
    )
    
    # Metadata
    timestamp_ms: Optional[int] = Field(
        None,
        description="Timestamp when thought was created"
    )
    processing_time_ms: Optional[int] = Field(
        None,
        description="Time taken to process this thought"
    )
    
    # Configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,  # Allow modifications for reflection updates
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [{
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
                    "recommended_tools": [{
                        "tool_name": "code_analysis",
                        "confidence": 0.9,
                        "rationale": "Need to understand current performance bottlenecks",
                        "priority": 1,
                        "expected_outcome": "Identify O(n) complexity issues"
                    }],
                    "expected_outcome": "Understanding of performance characteristics"
                }
            }]
        }
    )
    
    # Class-level constants
    MIN_TOTAL_THOUGHTS: ClassVar[int] = 5
    
    @field_validator('totalThoughts')
    @classmethod
    def validate_total_thoughts_minimum(cls, v: int) -> int:
        """Ensures totalThoughts meets minimum requirement."""
        if v < cls.MIN_TOTAL_THOUGHTS:
            return cls.MIN_TOTAL_THOUGHTS
        return v
    
    @field_validator('revisesThought')
    @classmethod
    def validate_revises_thought(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validates revision logic."""
        is_revision = info.data.get('isRevision', False)
        thought_number = info.data.get('thoughtNumber')
        
        if v is not None:
            if not is_revision:
                raise ValueError('revisesThought requires isRevision=True')
            if thought_number and v >= thought_number:
                raise ValueError('revisesThought must be less than thoughtNumber')
        elif is_revision:
            raise ValueError('isRevision=True requires revisesThought')
        
        return v
    
    @field_validator('branchId')
    @classmethod
    def validate_branch_id(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validates branch logic."""
        branch_from = info.data.get('branchFromThought')
        
        if v is not None and branch_from is None:
            raise ValueError('branchId requires branchFromThought')
        elif branch_from is not None and v is None:
            raise ValueError('branchFromThought requires branchId')
        
        return v
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validates and cleans keywords."""
        if len(v) > 20:  # Reasonable limit
            v = v[:20]
        # Clean and normalize keywords
        cleaned = []
        for keyword in v:
            if keyword and isinstance(keyword, str):
                clean_keyword = keyword.strip().lower()
                if clean_keyword and len(clean_keyword) <= 50:
                    cleaned.append(clean_keyword)
        return cleaned
    
    @field_validator('topic', 'subject')
    @classmethod
    def validate_topic_subject(cls, v: Optional[str]) -> Optional[str]:
        """Validates topic and subject fields."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v
    
    @model_validator(mode='after')
    def validate_thought_relationships(self) -> 'ThoughtData':
        """Validates thought number relationships."""
        # Validate branch relationships
        if self.branchFromThought is not None:
            if self.branchFromThought >= self.thoughtNumber:
                raise ValueError('branchFromThought must be less than thoughtNumber')
        
        # Validate thought relationships don't create cycles
        if self.thought_relationships:
            # Simple cycle detection - could be enhanced
            for rel in self.thought_relationships:
                if rel.from_thought == rel.to_thought:
                    raise ValueError('Thought cannot relate to itself')
        
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
            tool_names = [tool.tool_name for tool in self.current_step.recommended_tools[:3]]
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
                    "rationale": tool.rationale[:100] + "..." if len(tool.rationale) > 100 else tool.rationale
                }
                for tool in sorted(self.current_step.recommended_tools, key=lambda x: x.priority)
            ],
            "expected_outcome": self.current_step.expected_outcome
        }
    
    def get_alignment_summary(self) -> Dict[str, str]:
        """Returns topic/subject alignment summary."""
        return {
            "topic": self.topic or "Not specified",
            "subject": self.subject or "Not specified", 
            "domain": self.domain.value,
            "keywords": ", ".join(self.keywords[:5]) + ("..." if len(self.keywords) > 5 else "")
        }


class BranchAnalysis(BaseModel):
    """Analysis of a thought branch for the review tool."""
    branch_id: str = Field(..., description="Branch identifier")
    branch_quality: float = Field(..., ge=0.0, le=1.0, description="Overall quality of this branch")
    thought_count: int = Field(..., ge=0, description="Number of thoughts in this branch")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from this branch")
    completion_status: str = Field(..., description="Whether branch is complete, abandoned, or ongoing")
    recommendation: str = Field(..., description="Recommendation for this branch")


class ThoughtSequenceReview(BaseModel):
    """Comprehensive review of a thought sequence for sequentialreview tool."""
    session_id: str = Field(..., description="Session identifier")
    total_thoughts: int = Field(..., ge=0, description="Total number of thoughts")
    total_branches: int = Field(..., ge=0, description="Total number of branches")
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Overall sequence quality")
    
    # Analysis sections
    key_insights: List[str] = Field(default_factory=list, description="Key insights discovered")
    patterns_identified: List[str] = Field(default_factory=list, description="Patterns in thinking")
    quality_trends: Dict[str, float] = Field(default_factory=dict, description="Quality metrics over time")
    tool_effectiveness: Dict[str, float] = Field(default_factory=dict, description="Tool usage effectiveness")
    
    # Branch analysis
    branch_analyses: List[BranchAnalysis] = Field(default_factory=list, description="Analysis of each branch")
    best_branch: Optional[str] = Field(None, description="ID of the most promising branch")
    
    # Recommendations
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas needing improvement")
    topic_alignment_score: float = Field(0.5, ge=0.0, le=1.0, description="How well aligned with stated topic")
    
    # Metadata
    review_timestamp: Optional[int] = Field(None, description="When this review was generated")
    review_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence in this review")


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
    tool_recommendations_generated: bool = Field(False, description="Whether tool recommendations were generated")
    reflection_applied: bool = Field(False, description="Whether reflection feedback was applied")
    context_updated: bool = Field(False, description="Whether shared context was updated")