"""
Thought processing logic for handling individual and sequences of thoughts.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List

from src.models.thought_models import (
    ThoughtData,
    ProcessedThought,
    ThoughtSequenceReview,
    ReflectionFeedback,
)
from src.models.protocols import AppContextProtocol

# EnhancedAppContext will be properly typed with AppContextProtocol
from exceptions import TeamProcessingError
from config import ENABLE_REFLECTION

logger = logging.getLogger(__name__)


class ThoughtProcessor:
    """Handles processing of individual thoughts and thought sequences."""

    def __init__(self, context: AppContextProtocol):
        """
        Initialize thought processor.

        Args:
            context: Enhanced application context with teams and shared memory
        """
        self.context = context

    async def process_thought(self, thought_data: ThoughtData) -> ProcessedThought:
        """
        Process a single thought through the dual-team system.

        Args:
            thought_data: The thought to process

        Returns:
            ProcessedThought with analysis and recommendations

        Raises:
            TeamProcessingError: If processing fails
        """
        start_time = time.time()

        try:
            # Ensure teams are initialized
            if not self.context.teams_initialized:
                await self.context.initialize_teams()

            # Process through primary team
            primary_response = await self._process_with_primary_team(thought_data)

            # Process through reflection team if enabled
            reflection_response = None
            if ENABLE_REFLECTION and self.context.reflection_team:
                reflection_response = await self._process_with_reflection_team(
                    thought_data, primary_response
                )

            # Create reflection feedback
            reflection_feedback = self._create_reflection_feedback(
                primary_response, reflection_response
            )

            # Update thought data with feedback
            thought_data.reflection_feedback = reflection_feedback

            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)

            # Create processed thought
            return ProcessedThought(
                thought_data=thought_data,
                coordinator_response=primary_response
                or "Primary team processing completed",
                reflection_response=reflection_response,
                integrated_response=self._create_integrated_response(
                    primary_response, reflection_response
                ),
                next_step_guidance=self._create_next_step_guidance(
                    thought_data, primary_response
                ),
                execution_time_ms=processing_time,
                success=True,
                tool_recommendations_generated=bool(thought_data.current_step),
                reflection_applied=bool(reflection_response),
                context_updated=True,
            )

        except Exception as e:
            logger.error(
                "Thought processing failed",
                extra={"thought_number": thought_data.thoughtNumber, "error": str(e)},
            )
            raise TeamProcessingError(
                team_name="thought_processor", reason=str(e), stage="processing"
            )

    async def _process_with_primary_team(self, thought_data: ThoughtData) -> str:
        """Process thought with primary thinking team."""
        try:
            # Update shared context
            await self.context.add_thought(thought_data)

            # Get relevant context
            relevant_context = await self.context.get_relevant_context(
                thought_data.thought
            )

            # Prepare input
            primary_input = self._create_primary_team_input(
                thought_data, relevant_context
            )

            # Process with team
            if not self.context.primary_team:
                raise TeamProcessingError(
                    team_name="primary", reason="Primary team not initialized"
                )

            response = await self.context.primary_team.arun(primary_input)

            # Record success
            self.context.error_handler.circuit_breakers[
                "team_processing"
            ].record_success()

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            self.context.error_handler.circuit_breakers[
                "team_processing"
            ].record_failure()
            raise TeamProcessingError(
                team_name="primary", reason=str(e), stage="primary_processing"
            )

    async def _process_with_reflection_team(
        self, thought_data: ThoughtData, primary_response: str
    ) -> Optional[str]:
        """Process thought with reflection team for meta-analysis."""
        try:
            # Prepare reflection input
            reflection_input = self._create_reflection_team_input(
                thought_data, primary_response
            )

            # Process with team
            if not self.context.reflection_team:
                logger.warning("Reflection team not initialized")
                return None

            response = await self.context.reflection_team.arun(reflection_input)

            # Record success
            self.context.error_handler.circuit_breakers[
                "team_processing"
            ].record_success()

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            logger.warning("Reflection team processing failed", extra={"error": str(e)})
            self.context.error_handler.circuit_breakers[
                "team_processing"
            ].record_failure()
            return None

    def _create_primary_team_input(
        self, thought_data: ThoughtData, relevant_context: Dict[str, Any]
    ) -> str:
        """Create input prompt for primary team."""
        return f"""
Thought #{thought_data.thoughtNumber}/{thought_data.totalThoughts}
Topic: {thought_data.topic or "General"}
Domain: {thought_data.domain.value}

Current Thought: {thought_data.thought}

Context:
- Is Revision: {thought_data.isRevision}
- Branch ID: {thought_data.branchId or "main"}
- Confidence: {thought_data.confidence_score}

Relevant Context:
{json.dumps(relevant_context, indent=2)}

Tool Recommendations:
{thought_data.current_step.model_dump_json(indent=2) if thought_data.current_step else "None"}

Please analyze this thought and provide comprehensive guidance.
"""

    def _create_reflection_team_input(
        self, thought_data: ThoughtData, primary_response: str
    ) -> str:
        """Create input prompt for reflection team."""
        return f"""
Primary Team Response:
{primary_response}

Original Thought: {thought_data.thought}
Thought Number: {thought_data.thoughtNumber}

Please provide meta-analysis of:
1. Quality of the thinking process
2. Potential biases or gaps
3. Tool selection effectiveness
4. Suggestions for improvement
"""

    def _create_reflection_feedback(
        self, primary_response: str, reflection_response: Optional[str]
    ) -> ReflectionFeedback:
        """Create reflection feedback from team responses."""
        # Extract feedback components from responses
        # This is a simplified version - in practice, you'd parse the responses
        return ReflectionFeedback(
            strengths=["Clear analysis", "Systematic approach"],
            weaknesses=["Could explore alternatives more"],
            suggestions=["Consider edge cases", "Validate assumptions"],
            patterns_detected=["Sequential reasoning", "Tool-oriented thinking"],
            overall_quality=0.85,
        )

    def _extract_key_insights(
        self, primary_response: str, reflection_response: Optional[str]
    ) -> List[str]:
        """Extract key insights from team responses."""
        insights = []

        # Simple extraction - in practice, use NLP or structured parsing
        if "key insight" in primary_response.lower():
            insights.append("Primary insight extracted")

        if reflection_response and "pattern" in reflection_response.lower():
            insights.append("Pattern identified in thinking process")

        return insights or ["Analysis completed successfully"]

    def _extract_suggested_actions(self, primary_response: str) -> List[str]:
        """Extract suggested actions from primary team response."""
        # Simple extraction - enhance with proper parsing
        actions = []

        if "recommend" in primary_response.lower():
            actions.append("Follow team recommendations")

        if "next step" in primary_response.lower():
            actions.append("Proceed to next thinking step")

        return actions or ["Continue with current approach"]

    def _extract_tool_recommendations(
        self, thought_data: ThoughtData
    ) -> List[Dict[str, Any]]:
        """Extract tool recommendations from thought data."""
        if not thought_data.current_step:
            return []

        step = thought_data.current_step
        recommendations = []

        if hasattr(step, "recommended_tools"):
            for tool in step.recommended_tools:
                recommendations.append(
                    {
                        "tool": tool.tool_name,
                        "confidence": tool.confidence,
                        "rationale": tool.rationale,
                    }
                )

        return recommendations

    def _create_reflection_summary(
        self, reflection_feedback: Optional[ReflectionFeedback]
    ) -> str:
        """Create a summary of reflection feedback."""
        if not reflection_feedback:
            return "No reflection analysis available"

        return (
            f"Quality: {reflection_feedback.overall_quality:.2f} | "
            f"Strengths: {len(reflection_feedback.strengths)} | "
            f"Suggestions: {len(reflection_feedback.suggestions)}"
        )

    def _create_integrated_response(
        self, primary_response: str, reflection_response: Optional[str]
    ) -> str:
        """Create integrated response from primary and reflection team outputs."""
        if not primary_response:
            return "No analysis available"

        integrated = primary_response

        if reflection_response:
            integrated += f"\n\n### Reflection Analysis\n{reflection_response}"

        return integrated

    def _create_next_step_guidance(
        self, thought_data: ThoughtData, primary_response: str
    ) -> str:
        """Create next step guidance based on thought and analysis."""
        guidance_parts = []

        # Check if sequence is complete
        if not thought_data.nextThoughtNeeded:
            guidance_parts.append("This thought sequence is complete.")
        else:
            guidance_parts.append(
                f"Continue to thought {thought_data.thoughtNumber + 1} of {thought_data.totalThoughts}."
            )

        # Add tool recommendations if available
        if thought_data.current_step:
            guidance_parts.append(
                "Follow the recommended tool usage for optimal results."
            )

        # Extract guidance from primary response
        if "next step" in primary_response.lower():
            guidance_parts.append("Consider the next steps outlined in the analysis.")

        return (
            " ".join(guidance_parts)
            if guidance_parts
            else "Continue with the analysis."
        )


async def generate_sequence_review(
    context: AppContextProtocol,
    branch_id: Optional[str] = None,
    min_quality_threshold: float = 0.0,
) -> ThoughtSequenceReview:
    """
    Generate a review of a thought sequence.

    Args:
        context: Application context with thought history
        branch_id: Optional branch to review
        min_quality_threshold: Minimum quality threshold

    Returns:
        ThoughtSequenceReview with analysis
    """
    try:
        # Get thoughts for review
        if branch_id:
            thoughts = [t for t in context.thought_history if t.branchId == branch_id]
        else:
            thoughts = list(context.thought_history)

        # Filter by quality if threshold set
        if min_quality_threshold > 0:
            thoughts = [
                t for t in thoughts if t.confidence_score >= min_quality_threshold
            ]

        if not thoughts:
            return ThoughtSequenceReview(
                totalThoughts=0,
                branches=[],
                summary="No thoughts found matching criteria",
                keyInsights=[],
                strengthsIdentified=[],
                areasForImprovement=[],
                overallCoherence=0.0,
                recommendedNextSteps=[],
            )

        # Calculate metrics
        total_thoughts = len(thoughts)
        branches = list(set(t.branchId for t in thoughts if t.branchId))
        avg_confidence = sum(t.confidence_score for t in thoughts) / total_thoughts

        # Extract insights from thought graph
        key_insights = (
            context.shared_context.key_insights[-5:]
            if context.shared_context.key_insights
            else []
        )

        return ThoughtSequenceReview(
            totalThoughts=total_thoughts,
            branches=branches,
            summary=f"Reviewed {total_thoughts} thoughts with average confidence {avg_confidence:.2f}",
            keyInsights=[insight.get("content", "") for insight in key_insights],
            strengthsIdentified=["Systematic approach", "Clear progression"],
            areasForImprovement=[
                "Explore more alternatives",
                "Increase depth of analysis",
            ],
            overallCoherence=avg_confidence,
            recommendedNextSteps=[
                "Continue with current approach",
                "Consider branching for alternatives",
            ],
        )

    except Exception as e:
        logger.error("Failed to generate sequence review", extra={"error": str(e)})
        return ThoughtSequenceReview(
            totalThoughts=0,
            branches=[],
            summary=f"Review generation failed: {str(e)}",
            keyInsights=[],
            strengthsIdentified=[],
            areasForImprovement=["Unable to analyze"],
            overallCoherence=0.0,
            recommendedNextSteps=["Retry analysis"],
        )
