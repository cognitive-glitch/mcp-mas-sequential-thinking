"""
Tests for analysis models.
"""

import pytest
from pydantic import ValidationError

from src.models.analysis_models import (
    ReflectionFeedback,
    QualityIndicators,
    BranchAnalysis,
    ThoughtSequenceReview,
)


class TestReflectionFeedback:
    """Test the ReflectionFeedback model."""

    def test_reflection_feedback_default_initialization(self):
        """Test ReflectionFeedback with default values."""
        feedback = ReflectionFeedback(overall_quality=0.7)

        assert feedback.strengths == []
        assert feedback.weaknesses == []
        assert feedback.suggestions == []
        assert feedback.patterns_detected == []
        assert feedback.overall_quality == 0.7
        assert feedback.cognitive_biases == []
        assert feedback.missed_opportunities == []

    def test_reflection_feedback_with_values(self):
        """Test ReflectionFeedback with provided values."""
        feedback = ReflectionFeedback(
            strengths=["Clear thinking", "Good analysis"],
            weaknesses=["Lacks depth", "Too rushed"],
            suggestions=["Take more time", "Consider alternatives"],
            patterns_detected=["Linear thinking", "Solution-focused"],
            overall_quality=0.8,
            cognitive_biases=["Confirmation bias"],
            missed_opportunities=["Could explore edge cases"],
        )

        assert len(feedback.strengths) == 2
        assert len(feedback.weaknesses) == 2
        assert len(feedback.suggestions) == 2
        assert len(feedback.patterns_detected) == 2
        assert feedback.overall_quality == 0.8
        assert len(feedback.cognitive_biases) == 1
        assert len(feedback.missed_opportunities) == 1

    def test_reflection_feedback_overall_quality_validation(self):
        """Test overall_quality field validation."""
        # Valid range
        feedback = ReflectionFeedback(overall_quality=0.0)
        assert feedback.overall_quality == 0.0

        feedback = ReflectionFeedback(overall_quality=1.0)
        assert feedback.overall_quality == 1.0

        feedback = ReflectionFeedback(overall_quality=0.5)
        assert feedback.overall_quality == 0.5

        # Invalid range should raise ValidationError
        with pytest.raises(ValidationError):
            ReflectionFeedback(overall_quality=-0.1)

        with pytest.raises(ValidationError):
            ReflectionFeedback(overall_quality=1.1)

    def test_needs_improvement_computed_field(self):
        """Test needs_improvement computed field."""
        # Low quality -> needs improvement
        feedback = ReflectionFeedback(overall_quality=0.5)
        assert feedback.needs_improvement is True

        # More weaknesses than strengths -> needs improvement
        feedback = ReflectionFeedback(
            overall_quality=0.8,
            strengths=["One strength"],
            weaknesses=["Weakness 1", "Weakness 2"],
        )
        assert feedback.needs_improvement is True

        # High quality and more strengths -> doesn't need improvement
        feedback = ReflectionFeedback(
            overall_quality=0.8,
            strengths=["Strength 1", "Strength 2"],
            weaknesses=["One weakness"],
        )
        assert feedback.needs_improvement is False

        # Equal strengths and weaknesses, high quality -> doesn't need improvement
        feedback = ReflectionFeedback(
            overall_quality=0.7, strengths=["Strength 1"], weaknesses=["Weakness 1"]
        )
        assert feedback.needs_improvement is False

    def test_key_insight_computed_field(self):
        """Test key_insight computed field."""
        # No insights -> None
        feedback = ReflectionFeedback(overall_quality=0.7)
        assert feedback.key_insight is None

        # Patterns detected -> pattern insight
        feedback = ReflectionFeedback(
            overall_quality=0.7,
            patterns_detected=["Pattern 1", "Pattern 2"],
            strengths=["Strength"],
            weaknesses=["Weakness"],
        )
        assert feedback.key_insight == "Pattern: Pattern 1"

        # No patterns but strengths -> strength insight
        feedback = ReflectionFeedback(
            overall_quality=0.7,
            strengths=["Main strength", "Secondary strength"],
            weaknesses=["Weakness"],
        )
        assert feedback.key_insight == "Strength: Main strength"

        # No patterns or strengths but weaknesses -> weakness insight
        feedback = ReflectionFeedback(
            overall_quality=0.7, weaknesses=["Critical weakness", "Minor weakness"]
        )
        assert feedback.key_insight == "Weakness: Critical weakness"


class TestQualityIndicators:
    """Test the QualityIndicators model."""

    def test_quality_indicators_default_initialization(self):
        """Test QualityIndicators with default values."""
        quality = QualityIndicators(
            clarity_score=0.7,
            depth_score=0.7,
            coherence_score=0.7,
            relevance_score=0.7,
            innovation_score=0.5,
            completeness_score=0.7,
        )

        assert quality.clarity_score == 0.7
        assert quality.depth_score == 0.7
        assert quality.coherence_score == 0.7
        assert quality.relevance_score == 0.7
        assert quality.innovation_score == 0.5
        assert quality.completeness_score == 0.7

    def test_quality_indicators_with_values(self):
        """Test QualityIndicators with provided values."""
        quality = QualityIndicators(
            clarity_score=0.9,
            depth_score=0.8,
            coherence_score=0.85,
            relevance_score=0.95,
            innovation_score=0.6,
            completeness_score=0.8,
        )

        assert quality.clarity_score == 0.9
        assert quality.depth_score == 0.8
        assert quality.coherence_score == 0.85
        assert quality.relevance_score == 0.95
        assert quality.innovation_score == 0.6
        assert quality.completeness_score == 0.8

    def test_quality_indicators_score_validation(self):
        """Test score field validation (0.0 <= score <= 1.0)."""
        # Valid boundary values
        quality = QualityIndicators(
            clarity_score=0.0,
            depth_score=1.0,
            coherence_score=0.5,
            relevance_score=0.7,
            innovation_score=0.5,
            completeness_score=0.7,
        )
        assert quality.clarity_score == 0.0
        assert quality.depth_score == 1.0
        assert quality.coherence_score == 0.5

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            QualityIndicators(
                clarity_score=-0.1,
                depth_score=0.7,
                coherence_score=0.7,
                relevance_score=0.7,
                innovation_score=0.5,
                completeness_score=0.7,
            )

        with pytest.raises(ValidationError):
            QualityIndicators(
                clarity_score=0.7,
                depth_score=1.1,
                coherence_score=0.7,
                relevance_score=0.7,
                innovation_score=0.5,
                completeness_score=0.7,
            )

    def test_overall_quality_estimate_computed_field(self):
        """Test overall_quality_estimate computed field."""
        # Test default values
        quality = QualityIndicators(
            clarity_score=0.7,
            depth_score=0.7,
            coherence_score=0.7,
            relevance_score=0.7,
            innovation_score=0.5,
            completeness_score=0.7,
        )
        expected = round(
            0.7 * 0.2 + 0.7 * 0.2 + 0.7 * 0.2 + 0.7 * 0.25 + 0.5 * 0.05 + 0.7 * 0.1, 2
        )
        assert quality.overall_quality_estimate == expected

        # Test perfect scores
        quality = QualityIndicators(
            clarity_score=1.0,
            depth_score=1.0,
            coherence_score=1.0,
            relevance_score=1.0,
            innovation_score=1.0,
            completeness_score=1.0,
        )
        assert quality.overall_quality_estimate == 1.0

        # Test zero scores
        quality = QualityIndicators(
            clarity_score=0.0,
            depth_score=0.0,
            coherence_score=0.0,
            relevance_score=0.0,
            innovation_score=0.0,
            completeness_score=0.0,
        )
        assert quality.overall_quality_estimate == 0.0

        # Test weighted calculation with mixed scores
        quality = QualityIndicators(
            clarity_score=0.8,  # weight 0.2
            depth_score=0.6,  # weight 0.2
            coherence_score=0.9,  # weight 0.2
            relevance_score=1.0,  # weight 0.25
            innovation_score=0.4,  # weight 0.05
            completeness_score=0.7,  # weight 0.1
        )
        expected = round(
            0.8 * 0.2 + 0.6 * 0.2 + 0.9 * 0.2 + 1.0 * 0.25 + 0.4 * 0.05 + 0.7 * 0.1, 2
        )
        assert quality.overall_quality_estimate == expected

    def test_progress_percentage_computed_field(self):
        """Test progress_percentage computed field."""
        # Test with default values
        quality = QualityIndicators(
            clarity_score=0.7,
            depth_score=0.7,
            coherence_score=0.7,
            relevance_score=0.7,
            innovation_score=0.5,
            completeness_score=0.7,
        )
        expected_overall = round(
            0.7 * 0.2 + 0.7 * 0.2 + 0.7 * 0.2 + 0.7 * 0.25 + 0.5 * 0.05 + 0.7 * 0.1, 2
        )
        expected_progress = min(expected_overall * 100, 95.0)
        assert quality.progress_percentage == expected_progress

        # Test with high quality (should cap at 95%)
        quality = QualityIndicators(
            clarity_score=1.0,
            depth_score=1.0,
            coherence_score=1.0,
            relevance_score=1.0,
            innovation_score=1.0,
            completeness_score=1.0,
        )
        assert quality.progress_percentage == 95.0

        # Test with low quality
        quality = QualityIndicators(
            clarity_score=0.1,
            depth_score=0.1,
            coherence_score=0.1,
            relevance_score=0.1,
            innovation_score=0.1,
            completeness_score=0.1,
        )
        assert quality.progress_percentage == 10.0  # 0.1 * 100

    def test_is_final_thought_computed_field(self):
        """Test is_final_thought computed field."""
        # Not final: low completeness
        quality = QualityIndicators(
            completeness_score=0.8,  # < 0.9
            clarity_score=1.0,
            depth_score=1.0,
            coherence_score=1.0,
            relevance_score=1.0,
            innovation_score=1.0,
        )
        assert quality.is_final_thought is False

        # Not final: low overall quality
        quality = QualityIndicators(
            completeness_score=0.95,  # >= 0.9
            clarity_score=0.1,  # This will make overall < 0.8
            depth_score=0.1,
            coherence_score=0.1,
            relevance_score=0.1,
            innovation_score=0.1,
        )
        assert quality.is_final_thought is False

        # Final: both conditions met
        quality = QualityIndicators(
            completeness_score=0.95,  # >= 0.9
            clarity_score=1.0,
            depth_score=1.0,
            coherence_score=1.0,
            relevance_score=1.0,
            innovation_score=1.0,  # overall will be 1.0 >= 0.8
        )
        assert quality.is_final_thought is True


class TestBranchAnalysis:
    """Test the BranchAnalysis model."""

    def test_branch_analysis_initialization(self):
        """Test BranchAnalysis initialization."""
        branch = BranchAnalysis(
            branchId="main",
            thoughtCount=5,
            avgConfidence=0.8,
            divergencePoint=2,
            effectiveness=0.5,
            recommendation="continue",
        )

        assert branch.branchId == "main"
        assert branch.thoughtCount == 5
        assert branch.avgConfidence == 0.8
        assert branch.keyThemes == []
        assert branch.divergencePoint == 2
        assert branch.convergencePoints == []
        assert branch.effectiveness == 0.5  # default
        assert branch.recommendation == "continue"  # default

    def test_branch_analysis_with_optional_fields(self):
        """Test BranchAnalysis with all fields."""
        branch = BranchAnalysis(
            branchId="alternative",
            thoughtCount=3,
            avgConfidence=0.9,
            keyThemes=["optimization", "efficiency"],
            divergencePoint=1,
            convergencePoints=[4, 7],
            effectiveness=0.85,
            recommendation="prioritize",
        )

        assert branch.branchId == "alternative"
        assert branch.thoughtCount == 3
        assert branch.avgConfidence == 0.9
        assert branch.keyThemes == ["optimization", "efficiency"]
        assert branch.divergencePoint == 1
        assert branch.convergencePoints == [4, 7]
        assert branch.effectiveness == 0.85
        assert branch.recommendation == "prioritize"

    def test_branch_analysis_field_validation(self):
        """Test BranchAnalysis field validation."""
        # Valid thought count
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=0,  # >= 0
            avgConfidence=0.5,
            divergencePoint=1,
            effectiveness=0.5,
            recommendation="continue",
        )
        assert branch.thoughtCount == 0

        # Invalid thought count
        with pytest.raises(ValidationError):
            BranchAnalysis(
                branchId="test",
                thoughtCount=-1,  # < 0
                avgConfidence=0.5,
                divergencePoint=1,
                effectiveness=0.5,
                recommendation="continue",
            )

        # Valid avgConfidence range
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.0,
            divergencePoint=1,
            effectiveness=0.5,
            recommendation="continue",
        )
        assert branch.avgConfidence == 0.0

        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=1.0,
            divergencePoint=1,
            effectiveness=0.5,
            recommendation="continue",
        )
        assert branch.avgConfidence == 1.0

        # Invalid avgConfidence
        with pytest.raises(ValidationError):
            BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=-0.1,
                divergencePoint=1,
                effectiveness=0.5,
                recommendation="continue",
            )

        with pytest.raises(ValidationError):
            BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=1.1,
                divergencePoint=1,
                effectiveness=0.5,
                recommendation="continue",
            )

        # Valid effectiveness range
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.5,
            divergencePoint=1,
            effectiveness=0.0,
            recommendation="continue",
        )
        assert branch.effectiveness == 0.0

        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.5,
            divergencePoint=1,
            effectiveness=1.0,
            recommendation="continue",
        )
        assert branch.effectiveness == 1.0

        # Invalid effectiveness
        with pytest.raises(ValidationError):
            BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=0.5,
                divergencePoint=1,
                effectiveness=-0.1,
                recommendation="continue",
            )

        with pytest.raises(ValidationError):
            BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=0.5,
                divergencePoint=1,
                effectiveness=1.1,
                recommendation="continue",
            )

    def test_is_productive_computed_field(self):
        """Test is_productive computed field."""
        # Not productive: low effectiveness
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.8,  # >= 0.7
            divergencePoint=1,
            effectiveness=0.5,  # < 0.6
            recommendation="continue",
        )
        assert branch.is_productive is False

        # Not productive: low confidence
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.6,  # < 0.7
            divergencePoint=1,
            effectiveness=0.7,  # >= 0.6
            recommendation="continue",
        )
        assert branch.is_productive is False

        # Not productive: both low
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.6,  # < 0.7
            divergencePoint=1,
            effectiveness=0.5,  # < 0.6
            recommendation="continue",
        )
        assert branch.is_productive is False

        # Productive: both conditions met
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.8,  # >= 0.7
            divergencePoint=1,
            effectiveness=0.7,  # >= 0.6
            recommendation="continue",
        )
        assert branch.is_productive is True

        # Edge case: exactly at thresholds
        branch = BranchAnalysis(
            branchId="test",
            thoughtCount=1,
            avgConfidence=0.7,  # == 0.7
            divergencePoint=1,
            effectiveness=0.6,  # == 0.6
            recommendation="continue",
        )
        assert branch.is_productive is True

    def test_validate_recommendation_field_validator(self):
        """Test validate_recommendation field validator."""
        # Valid recommendations
        valid_recommendations = ["continue", "merge", "abandon", "prioritize"]

        for rec in valid_recommendations:
            branch = BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=0.5,
                divergencePoint=1,
                effectiveness=0.5,
                recommendation=rec,
            )
            assert branch.recommendation == rec

        # Invalid recommendation
        with pytest.raises(ValidationError) as exc_info:
            BranchAnalysis(
                branchId="test",
                thoughtCount=1,
                avgConfidence=0.5,
                divergencePoint=1,
                effectiveness=0.5,
                recommendation="invalid_choice",
            )

        assert "Recommendation must be one of" in str(exc_info.value)
        assert "continue" in str(exc_info.value)
        assert "merge" in str(exc_info.value)
        assert "abandon" in str(exc_info.value)
        assert "prioritize" in str(exc_info.value)


class TestThoughtSequenceReview:
    """Test the ThoughtSequenceReview model."""

    def test_thought_sequence_review_initialization(self):
        """Test ThoughtSequenceReview initialization."""
        review = ThoughtSequenceReview(
            totalThoughts=10,
            summary="Comprehensive analysis completed",
            overallCoherence=0.85,
        )

        assert review.totalThoughts == 10
        assert review.branches == []
        assert review.summary == "Comprehensive analysis completed"
        assert review.keyInsights == []
        assert review.strengthsIdentified == []
        assert review.areasForImprovement == []
        assert review.overallCoherence == 0.85
        assert review.recommendedNextSteps == []
        assert review.patternAnalysis == {}
        assert review.toolEffectiveness == {}

    def test_thought_sequence_review_with_all_fields(self):
        """Test ThoughtSequenceReview with all fields."""
        review = ThoughtSequenceReview(
            totalThoughts=15,
            branches=["main", "alternative", "experimental"],
            summary="Multi-branch analysis with good insights",
            keyInsights=["Key insight 1", "Key insight 2"],
            strengthsIdentified=["Thorough analysis", "Creative thinking"],
            areasForImprovement=["Need more depth", "Consider edge cases"],
            overallCoherence=0.75,
            recommendedNextSteps=["Continue analysis", "Implement solution"],
            patternAnalysis={"linear": 5, "recursive": 3, "creative": 7},
            toolEffectiveness={"search": 0.8, "analyze": 0.9, "synthesize": 0.7},
        )

        assert review.totalThoughts == 15
        assert len(review.branches) == 3
        assert len(review.keyInsights) == 2
        assert len(review.strengthsIdentified) == 2
        assert len(review.areasForImprovement) == 2
        assert review.overallCoherence == 0.75
        assert len(review.recommendedNextSteps) == 2
        assert len(review.patternAnalysis) == 3
        assert len(review.toolEffectiveness) == 3

    def test_thought_sequence_review_field_validation(self):
        """Test ThoughtSequenceReview field validation."""
        # Valid totalThoughts
        review = ThoughtSequenceReview(
            totalThoughts=0,  # >= 0
            summary="Empty analysis",
            overallCoherence=0.0,
        )
        assert review.totalThoughts == 0

        # Invalid totalThoughts
        with pytest.raises(ValidationError):
            ThoughtSequenceReview(
                totalThoughts=-1,  # < 0
                summary="Invalid",
                overallCoherence=0.5,
            )

        # Valid overallCoherence range
        review = ThoughtSequenceReview(
            totalThoughts=1, summary="Test", overallCoherence=0.0
        )
        assert review.overallCoherence == 0.0

        review = ThoughtSequenceReview(
            totalThoughts=1, summary="Test", overallCoherence=1.0
        )
        assert review.overallCoherence == 1.0

        # Invalid overallCoherence
        with pytest.raises(ValidationError):
            ThoughtSequenceReview(
                totalThoughts=1, summary="Test", overallCoherence=-0.1
            )

        with pytest.raises(ValidationError):
            ThoughtSequenceReview(totalThoughts=1, summary="Test", overallCoherence=1.1)

    def test_quality_rating_computed_field(self):
        """Test quality_rating computed field."""
        # Excellent: >= 0.9
        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Excellent analysis", overallCoherence=0.95
        )
        assert review.quality_rating == "Excellent"

        review = ThoughtSequenceReview(
            totalThoughts=5,
            summary="Excellent analysis",
            overallCoherence=0.9,  # exactly 0.9
        )
        assert review.quality_rating == "Excellent"

        # Good: >= 0.7 and < 0.9
        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Good analysis", overallCoherence=0.85
        )
        assert review.quality_rating == "Good"

        review = ThoughtSequenceReview(
            totalThoughts=5,
            summary="Good analysis",
            overallCoherence=0.7,  # exactly 0.7
        )
        assert review.quality_rating == "Good"

        # Satisfactory: >= 0.5 and < 0.7
        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Satisfactory analysis", overallCoherence=0.6
        )
        assert review.quality_rating == "Satisfactory"

        review = ThoughtSequenceReview(
            totalThoughts=5,
            summary="Satisfactory analysis",
            overallCoherence=0.5,  # exactly 0.5
        )
        assert review.quality_rating == "Satisfactory"

        # Needs Improvement: < 0.5
        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Poor analysis", overallCoherence=0.4
        )
        assert review.quality_rating == "Needs Improvement"

        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Poor analysis", overallCoherence=0.0
        )
        assert review.quality_rating == "Needs Improvement"

    def test_has_multiple_branches_computed_field(self):
        """Test has_multiple_branches computed field."""
        # No branches
        review = ThoughtSequenceReview(
            totalThoughts=5, summary="Single branch", overallCoherence=0.7
        )
        assert review.has_multiple_branches is False

        # Single branch
        review = ThoughtSequenceReview(
            totalThoughts=5,
            branches=["main"],
            summary="Single branch",
            overallCoherence=0.7,
        )
        assert review.has_multiple_branches is False

        # Multiple branches
        review = ThoughtSequenceReview(
            totalThoughts=5,
            branches=["main", "alternative"],
            summary="Multiple branches",
            overallCoherence=0.7,
        )
        assert review.has_multiple_branches is True

        review = ThoughtSequenceReview(
            totalThoughts=5,
            branches=["main", "alt1", "alt2", "experimental"],
            summary="Many branches",
            overallCoherence=0.7,
        )
        assert review.has_multiple_branches is True

    def test_insight_density_computed_field(self):
        """Test insight_density computed field."""
        # Zero thoughts -> 0.0 density
        review = ThoughtSequenceReview(
            totalThoughts=0,
            keyInsights=["Insight 1"],
            summary="Empty",
            overallCoherence=0.0,
        )
        assert review.insight_density == 0.0

        # No insights -> 0.0 density
        review = ThoughtSequenceReview(
            totalThoughts=5, keyInsights=[], summary="No insights", overallCoherence=0.5
        )
        assert review.insight_density == 0.0

        # Normal ratio
        review = ThoughtSequenceReview(
            totalThoughts=10,
            keyInsights=["Insight 1", "Insight 2"],  # 2 insights
            summary="Some insights",
            overallCoherence=0.7,
        )
        assert review.insight_density == 0.2  # 2/10

        # High density
        review = ThoughtSequenceReview(
            totalThoughts=4,
            keyInsights=["I1", "I2", "I3", "I4"],  # 4 insights
            summary="High density",
            overallCoherence=0.9,
        )
        assert review.insight_density == 1.0  # 4/4

        # Very high density (more insights than thoughts - edge case)
        review = ThoughtSequenceReview(
            totalThoughts=2,
            keyInsights=["I1", "I2", "I3", "I4", "I5"],  # 5 insights
            summary="Very high density",
            overallCoherence=0.8,
        )
        assert review.insight_density == 2.5  # 5/2


class TestAnalysisModelsIntegration:
    """Test integration between analysis models."""

    def test_models_work_together(self):
        """Test that analysis models work together in realistic scenarios."""
        # Create reflection feedback
        feedback = ReflectionFeedback(
            strengths=["Clear reasoning", "Good structure"],
            weaknesses=["Lacks creativity"],
            suggestions=["Consider alternatives"],
            patterns_detected=["Analytical thinking"],
            overall_quality=0.8,
            cognitive_biases=["Anchoring bias"],
            missed_opportunities=["Didn't explore edge cases"],
        )

        # Create quality indicators
        quality = QualityIndicators(
            clarity_score=0.9,
            depth_score=0.7,
            coherence_score=0.8,
            relevance_score=0.85,
            innovation_score=0.4,
            completeness_score=0.8,
        )

        # Create branch analysis
        branch = BranchAnalysis(
            branchId="main_analysis",
            thoughtCount=8,
            avgConfidence=0.75,
            keyThemes=["problem_solving", "optimization"],
            divergencePoint=3,
            convergencePoints=[7],
            effectiveness=0.7,
            recommendation="continue",
        )

        # Create sequence review
        review = ThoughtSequenceReview(
            totalThoughts=8,
            branches=["main_analysis"],
            summary="Solid analytical progression with room for innovation",
            keyInsights=[
                "Problem structure is clear",
                "Solution space is well-defined",
            ],
            strengthsIdentified=["Logical flow", "Clear reasoning"],
            areasForImprovement=[
                "More creative alternatives",
                "Edge case consideration",
            ],
            overallCoherence=0.75,
            recommendedNextSteps=["Explore creative solutions", "Test edge cases"],
            patternAnalysis={"analytical": 6, "creative": 2},
            toolEffectiveness={"analysis": 0.8, "synthesis": 0.6},
        )

        # Verify all models work correctly
        assert feedback.needs_improvement is False  # Good quality, balanced feedback
        assert feedback.key_insight == "Pattern: Analytical thinking"

        assert quality.overall_quality_estimate > 0.7  # Should be good quality
        assert quality.is_final_thought is False  # Not quite ready (innovation low)

        assert branch.is_productive is True  # Good effectiveness and confidence

        assert review.quality_rating == "Good"  # 0.75 coherence
        assert review.has_multiple_branches is False  # Only one branch
        assert review.insight_density == 0.25  # 2 insights / 8 thoughts

    def test_edge_case_scenarios(self):
        """Test edge cases across all models."""
        # Minimum viable instances
        feedback_min = ReflectionFeedback(overall_quality=0.0)
        quality_min = QualityIndicators(
            clarity_score=0.0,
            depth_score=0.0,
            coherence_score=0.0,
            relevance_score=0.0,
            innovation_score=0.0,
            completeness_score=0.0,
        )
        branch_min = BranchAnalysis(
            branchId="minimal",
            thoughtCount=0,
            avgConfidence=0.0,
            divergencePoint=0,
            effectiveness=0.0,
            recommendation="continue",
        )
        review_min = ThoughtSequenceReview(
            totalThoughts=0, summary="Minimal review", overallCoherence=0.0
        )

        # All should work without errors
        assert feedback_min.overall_quality == 0.0
        assert quality_min.overall_quality_estimate == 0.0
        assert branch_min.is_productive is False
        assert review_min.quality_rating == "Needs Improvement"

        # Maximum viable instances
        feedback_max = ReflectionFeedback(overall_quality=1.0)
        quality_max = QualityIndicators(
            clarity_score=1.0,
            depth_score=1.0,
            coherence_score=1.0,
            relevance_score=1.0,
            innovation_score=1.0,
            completeness_score=1.0,
        )
        branch_max = BranchAnalysis(
            branchId="maximal",
            thoughtCount=100,
            avgConfidence=1.0,
            divergencePoint=1,
            effectiveness=1.0,
            recommendation="continue",
        )
        review_max = ThoughtSequenceReview(
            totalThoughts=100, summary="Perfect review", overallCoherence=1.0
        )

        # All should work at maximum values
        assert feedback_max.overall_quality == 1.0
        assert quality_max.overall_quality_estimate == 1.0
        assert branch_max.is_productive is True
        assert review_max.quality_rating == "Excellent"
