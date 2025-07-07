"""
Analysis models for thought processing system.
Contains models for reflection, quality assessment, and thought sequence analysis.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, computed_field, field_validator


class ReflectionFeedback(BaseModel):
    """Feedback from the reflection team about thinking quality."""

    strengths: List[str] = Field(
        default_factory=list, description="Identified strengths in the thinking"
    )
    weaknesses: List[str] = Field(
        default_factory=list, description="Identified weaknesses or gaps"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )
    patterns_detected: List[str] = Field(
        default_factory=list, description="Thinking patterns observed"
    )
    overall_quality: float = Field(
        0.7, ge=0.0, le=1.0, description="Overall quality assessment"
    )
    cognitive_biases: List[str] = Field(
        default_factory=list, description="Potential cognitive biases detected"
    )
    missed_opportunities: List[str] = Field(
        default_factory=list, description="Opportunities not explored"
    )

    @computed_field
    @property
    def needs_improvement(self) -> bool:
        """Determines if significant improvement is needed."""
        return self.overall_quality < 0.6 or len(self.weaknesses) > len(self.strengths)

    @computed_field
    @property
    def key_insight(self) -> Optional[str]:
        """Extracts the most important insight from feedback."""
        if self.patterns_detected:
            return f"Pattern: {self.patterns_detected[0]}"
        elif self.strengths:
            return f"Strength: {self.strengths[0]}"
        elif self.weaknesses:
            return f"Weakness: {self.weaknesses[0]}"
        return None


class QualityIndicators(BaseModel):
    """Quality metrics for thought evaluation."""

    clarity_score: float = Field(
        0.7, ge=0.0, le=1.0, description="How clear and understandable the thought is"
    )
    depth_score: float = Field(
        0.7, ge=0.0, le=1.0, description="Depth of analysis and exploration"
    )
    coherence_score: float = Field(
        0.7, ge=0.0, le=1.0, description="Logical consistency and flow"
    )
    relevance_score: float = Field(
        0.7, ge=0.0, le=1.0, description="Relevance to the topic/problem"
    )
    innovation_score: float = Field(
        0.5, ge=0.0, le=1.0, description="Novelty and creative insights"
    )
    completeness_score: float = Field(
        0.7, ge=0.0, le=1.0, description="How complete the analysis is"
    )

    @computed_field
    @property
    def overall_quality_estimate(self) -> float:
        """Calculates weighted overall quality score."""
        weights = {
            "clarity": 0.2,
            "depth": 0.2,
            "coherence": 0.2,
            "relevance": 0.25,
            "innovation": 0.05,
            "completeness": 0.1,
        }

        weighted_sum = (
            self.clarity_score * weights["clarity"]
            + self.depth_score * weights["depth"]
            + self.coherence_score * weights["coherence"]
            + self.relevance_score * weights["relevance"]
            + self.innovation_score * weights["innovation"]
            + self.completeness_score * weights["completeness"]
        )

        return round(weighted_sum, 2)

    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Estimates progress based on quality indicators."""
        # Higher quality generally indicates more progress
        return min(self.overall_quality_estimate * 100, 95.0)

    @computed_field
    @property
    def is_final_thought(self) -> bool:
        """Determines if this represents final thought quality."""
        return self.completeness_score >= 0.9 and self.overall_quality_estimate >= 0.8


class BranchAnalysis(BaseModel):
    """Analysis of a thought branch."""

    branchId: str = Field(..., description="Identifier for the branch")
    thoughtCount: int = Field(..., ge=0, description="Number of thoughts in branch")
    avgConfidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average confidence across branch"
    )
    keyThemes: List[str] = Field(
        default_factory=list, description="Main themes in the branch"
    )
    divergencePoint: int = Field(
        ..., description="Thought number where branch diverged"
    )
    convergencePoints: List[int] = Field(
        default_factory=list, description="Points where branch reconnects"
    )
    effectiveness: float = Field(
        0.5, ge=0.0, le=1.0, description="Branch effectiveness rating"
    )
    recommendation: str = Field("continue", description="Recommendation for the branch")

    @computed_field
    @property
    def is_productive(self) -> bool:
        """Determines if the branch is productive."""
        return self.effectiveness >= 0.6 and self.avgConfidence >= 0.7

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: str) -> str:
        """Ensure recommendation is valid."""
        valid_recommendations = {"continue", "merge", "abandon", "prioritize"}
        if v not in valid_recommendations:
            raise ValueError(f"Recommendation must be one of {valid_recommendations}")
        return v


class ThoughtSequenceReview(BaseModel):
    """Comprehensive review of a thought sequence."""

    totalThoughts: int = Field(..., ge=0, description="Total thoughts analyzed")
    branches: List[str] = Field(default_factory=list, description="Branch identifiers")
    summary: str = Field(..., description="Executive summary of the sequence")
    keyInsights: List[str] = Field(
        default_factory=list, description="Key insights discovered"
    )
    strengthsIdentified: List[str] = Field(
        default_factory=list, description="Strengths in thinking process"
    )
    areasForImprovement: List[str] = Field(
        default_factory=list, description="Areas needing improvement"
    )
    overallCoherence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall coherence score"
    )
    recommendedNextSteps: List[str] = Field(
        default_factory=list, description="Recommended next actions"
    )
    patternAnalysis: Dict[str, int] = Field(
        default_factory=dict, description="Frequency of thinking patterns"
    )
    toolEffectiveness: Dict[str, float] = Field(
        default_factory=dict, description="Effectiveness of tools used"
    )

    @computed_field
    @property
    def quality_rating(self) -> str:
        """Provides a quality rating based on coherence."""
        if self.overallCoherence >= 0.9:
            return "Excellent"
        elif self.overallCoherence >= 0.7:
            return "Good"
        elif self.overallCoherence >= 0.5:
            return "Satisfactory"
        else:
            return "Needs Improvement"

    @computed_field
    @property
    def has_multiple_branches(self) -> bool:
        """Checks if the sequence has explored multiple branches."""
        return len(self.branches) > 1

    @computed_field
    @property
    def insight_density(self) -> float:
        """Calculates insights per thought ratio."""
        if self.totalThoughts == 0:
            return 0.0
        return len(self.keyInsights) / self.totalThoughts


# Export all classes
__all__ = [
    "ReflectionFeedback",
    "QualityIndicators",
    "BranchAnalysis",
    "ThoughtSequenceReview",
]
