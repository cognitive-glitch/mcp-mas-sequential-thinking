"""
MCP Tools and Prompts for Reflective Sequential Thinking

This module contains all MCP (Model Context Protocol) tool and prompt definitions
extracted from main.py to improve modularity and reduce the God Object anti-pattern.

Dependencies:
- FastMCP for @mcp decorators
- App context and shared components from main module
"""

import time
import logging
from typing import Any, Dict, List, Optional

# MCP instance will be injected at runtime
mcp = None
from models.thought_models import (
    ThoughtData,
    DomainType,
)
from exceptions import ValidationError, ErrorType

logger = logging.getLogger(__name__)

# Will be set by main.py during initialization
app_context = None


def set_app_context(context):
    """Set the application context for MCP tools to use."""
    global app_context
    app_context = context


def set_mcp_instance(mcp_instance):
    """Set the FastMCP instance for tool registration."""
    global mcp
    mcp = mcp_instance
    # Register all tools and prompts after MCP instance is set
    _register_mcp_tools()


def _register_mcp_tools():
    """Register all MCP tools and prompts with the FastMCP instance."""
    if mcp is None:
        raise RuntimeError("MCP instance not set. Call set_mcp_instance() first.")

    # Register tools
    mcp.tool()(reflectivethinking)
    mcp.tool()(reflectivereview)

    # Register prompts
    mcp.prompt("sequential-thinking")(sequential_thinking_prompt)
    mcp.prompt("tool-selection")(tool_selection_prompt)
    mcp.prompt("thought-review")(thought_review_prompt)
    mcp.prompt("complex-problem")(complex_problem_prompt)


def validate_thought_input(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    is_revision: bool,
    revises_thought: Optional[int],
    branch_from_thought: Optional[int],
) -> None:
    """Validate thought input parameters."""
    if not thought or len(thought.strip()) < 10:
        raise ValidationError(
            "thought", thought, "Thought must be at least 10 characters long"
        )

    if thought_number < 1:
        raise ValidationError(
            "thought_number", thought_number, "Thought number must be positive"
        )

    if total_thoughts < 5:
        raise ValidationError(
            "total_thoughts", total_thoughts, "Total thoughts must be at least 5"
        )

    if thought_number > total_thoughts:
        raise ValidationError(
            "thought_number",
            thought_number,
            f"Thought number cannot exceed total thoughts ({total_thoughts})",
        )

    if is_revision and not revises_thought:
        raise ValidationError(
            "revises_thought",
            revises_thought,
            "Must specify which thought is being revised",
        )

    if revises_thought and revises_thought >= thought_number:
        raise ValidationError(
            "revises_thought", revises_thought, "Can only revise previous thoughts"
        )


async def process_thought_with_dual_teams(thought_data: ThoughtData, context) -> Any:
    """Process thought using dual team architecture."""
    # Import here to avoid circular imports
    from handlers.thought_processor import ThoughtProcessor

    processor = ThoughtProcessor(context)
    return await processor.process_thought(thought_data)


async def generate_sequence_review(
    context, branch_id: Optional[str] = None, min_quality_threshold: float = 0.0
) -> Any:
    """Generate a comprehensive review of thought sequences."""
    # Import here to avoid circular imports
    from handlers.thought_processor import ThoughtProcessor

    processor = ThoughtProcessor(context)
    return await processor.generate_review(branch_id, min_quality_threshold)


# =============================================================================
# MCP TOOLS
# =============================================================================


async def reflectivethinking(
    thought: str,
    next_thought_needed: bool,
    thought_number: int,
    total_thoughts: int,
    is_revision: bool = False,
    revises_thought: Optional[int] = None,
    branch_from_thought: Optional[int] = None,
    branch_id: Optional[str] = None,
    needs_more_thoughts: bool = False,
    current_step: Optional[Dict[str, Any]] = None,
    previous_steps: Optional[List[Dict[str, Any]]] = None,
    remaining_steps: Optional[List[str]] = None,
) -> str:
    """
    A detailed tool for dynamic and reflective problem-solving through thoughts.
    This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
    Each thought can build on, question, or revise previous insights as understanding deepens.

    Includes:
    - Dual-team processing (Primary thinking team + Reflection team)
    - Integrated tool recommendations based on thought content
    - Support for revision and branching
    - Comprehensive error handling
    """
    logger.info(f"Processing thought #{thought_number}")

    try:
        # Import models needed for conversion
        from models.thought_models import StepRecommendation, ToolRecommendation

        # Convert current_step from dict to StepRecommendation if provided
        step_recommendation = None
        if current_step:
            # Convert tool recommendations
            tool_recs = []
            for tool in current_step.get("recommended_tools", []):
                tool_recs.append(
                    ToolRecommendation(
                        tool_name=tool["tool_name"],
                        confidence=tool["confidence"],
                        rationale=tool["rationale"],
                        priority=tool["priority"],
                        suggested_inputs=tool.get("suggested_inputs"),
                        alternatives=tool.get("alternatives", []),
                        expected_benefits=tool.get("expected_benefits", []),
                        limitations=tool.get("limitations", []),
                    )
                )

            step_recommendation = StepRecommendation(
                step_description=current_step["step_description"],
                recommended_tools=tool_recs,
                expected_outcome=current_step["expected_outcome"],
                next_step_conditions=current_step.get("next_step_conditions", []),
            )

        # Convert previous_steps from list of dicts to list of StepRecommendation objects
        converted_previous_steps = []
        if previous_steps:
            for step_dict in previous_steps:
                try:
                    # Convert tool recommendations for this previous step
                    prev_tool_recs = []
                    for tool in step_dict.get("recommended_tools", []):
                        prev_tool_recs.append(
                            ToolRecommendation(
                                tool_name=tool["tool_name"],
                                confidence=tool["confidence"],
                                rationale=tool["rationale"],
                                priority=tool["priority"],
                                suggested_inputs=tool.get("suggested_inputs"),
                                alternatives=tool.get("alternatives", []),
                                expected_benefits=tool.get("expected_benefits", []),
                                limitations=tool.get("limitations", []),
                            )
                        )
                    
                    # Create StepRecommendation object
                    prev_step_obj = StepRecommendation(
                        step_description=step_dict["step_description"],
                        recommended_tools=prev_tool_recs,
                        expected_outcome=step_dict["expected_outcome"],
                        next_step_conditions=step_dict.get("next_step_conditions", []),
                        estimated_complexity=step_dict.get("estimated_complexity", 0.5),
                        dependencies=step_dict.get("dependencies", []),
                    )
                    converted_previous_steps.append(prev_step_obj)
                    
                except Exception as e:
                    logger.warning(f"Failed to convert previous step: {e}. Skipping invalid step.")
                    # Continue processing other valid steps

        # Create ThoughtData from parameters (only valid fields)
        thought_data = ThoughtData(
            thought=thought,
            thoughtNumber=thought_number,
            totalThoughts=total_thoughts,
            nextThoughtNeeded=next_thought_needed,
            isRevision=is_revision,
            revisesThought=revises_thought,
            branchFromThought=branch_from_thought,
            branchId=branch_id,
            needsMoreThoughts=needs_more_thoughts,
            current_step=step_recommendation,
            previous_steps=converted_previous_steps,
            # Valid ThoughtData fields only
            domain=DomainType.GENERAL,
            keywords=[],
            confidence_score=0.5,
            timestamp_ms=int(time.time() * 1000),
            topic=None,
            subject=None,
            reflection_feedback=None,
        )

        # Process through dual teams
        result = await process_thought_with_dual_teams(thought_data, app_context)

        # Build response
        response_parts = []

        # Add main content
        response_parts.append(result.integrated_response)

        # Add guidance for next steps
        if next_thought_needed:
            response_parts.append(
                f"\n## Next Step Guidance\n{result.next_step_guidance}"
            )

            # Add tool recommendations if available
            if step_recommendation:
                response_parts.append("\n## Recommended Tools for Next Step:")
                for tool in step_recommendation.recommended_tools[:3]:
                    response_parts.append(
                        f"- **{tool.tool_name}** (confidence: {tool.confidence:.2f}): {tool.rationale}"
                    )

        # Add performance metrics if this is a final thought
        if not next_thought_needed:
            metrics = await app_context.get_performance_metrics()
            response_parts.append(
                f"\n## Summary\n"
                f"- Total thoughts: {metrics['total_thoughts']}\n"
                f"- Duration: {metrics['duration_seconds']:.1f}s\n"
                f"- Overall quality: {result.quality_score:.2f}"
            )

        return "\n".join(response_parts)

    except ValidationError as e:
        error_msg = app_context.error_handler.handle_error(
            e, ErrorType.VALIDATION_ERROR, thought_number
        )
        logger.error(f"Validation error: {e}")
        return f"Validation Error: {error_msg}\n\nDetails: {str(e)}"

    except Exception as e:
        error_msg = app_context.error_handler.handle_error(
            e, ErrorType.TEAM_PROCESSING, thought_number
        )
        logger.error(f"Unexpected error: {e}")
        return f"Error: {error_msg}\n\nPlease try again with a simpler thought."


async def reflectivereview() -> str:
    """
    Generate a comprehensive review of the thought sequence with quality metrics.
    """
    try:
        # Generate review
        review = await generate_sequence_review(app_context)

        # Format review for output
        output = [
            "# Thought Sequence Review",
            f"**Total Thoughts**: {review.total_thoughts}",
            f"**Total Branches**: {review.total_branches}",
            f"**Overall Quality**: {review.overall_quality:.2f}",
            f"**Topic Alignment**: {review.topic_alignment_score:.2f}",
            "",
            "## Key Insights",
        ]

        for i, insight in enumerate(review.key_insights, 1):
            output.append(f"{i}. {insight}")

        if review.patterns_identified:
            output.extend(["", "## Patterns Identified"])
            for pattern in review.patterns_identified:
                output.append(f"- {pattern}")

        if review.tool_effectiveness:
            output.extend(["", "## Tool Effectiveness"])
            for tool, effectiveness in review.tool_effectiveness.items():
                output.append(f"- {tool}: {effectiveness:.2f}")

        if review.branch_analyses:
            output.extend(["", "## Branch Analysis"])
            for branch in review.branch_analyses:
                output.append(
                    f"\n### Branch: {branch.branch_id}\n"
                    f"- Quality: {branch.branch_quality:.2f}\n"
                    f"- Thoughts: {branch.thought_count}\n"
                    f"- Status: {branch.completion_status}\n"
                    f"- Recommendation: {branch.recommendation}"
                )

        if review.next_steps:
            output.extend(["", "## Recommended Next Steps"])
            for step in review.next_steps:
                output.append(f"- {step}")

        if review.areas_for_improvement:
            output.extend(["", "## Areas for Improvement"])
            for area in review.areas_for_improvement:
                output.append(f"- {area}")

        output.append(f"\n**Review Confidence**: {review.review_confidence:.2f}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Failed to generate review: {e}")
        return f"Error generating review: {str(e)}"


# =============================================================================
# MCP PROMPTS
# =============================================================================


def sequential_thinking_prompt(problem: str, context: str = ""):
    """
    Starter prompt for sequential thinking that ENCOURAGES non-linear exploration
    using coordinate mode. Returns separate user and assistant messages.
    """
    min_thoughts = 5  # Set a reasonable minimum number of initial thoughts

    user_prompt_text = f"""Initiate a comprehensive sequential thinking process for the following problem:

Problem: {problem}
{f"Context: {context}" if context else ""}"""

    assistant_guidelines = f"""I'll start the sequential thinking process. Here are the guidelines and the process we'll follow using the 'coordinate' mode team:

**Sequential Thinking Goals & Guidelines (Coordinate Mode):**

1. **Estimate Steps:** Analyze the problem complexity. Your initial `totalThoughts` estimate should be at least {min_thoughts}.
2. **First Thought:** Call the 'reflectivethinking' tool with `thoughtNumber: 1`, your estimated `totalThoughts` (at least {min_thoughts}), and `nextThoughtNeeded: True`. Structure your first thought as: "Plan a comprehensive analysis approach for: {problem}"
3. **Encouraged Revision:** Actively look for opportunities to revise previous thoughts if you identify flaws, oversights, or necessary refinements based on later analysis (especially from the Coordinator synthesizing Critic/Analyzer outputs). Use `isRevision: True` and `revisesThought: <thought_number>` when performing a revision. Robust thinking often involves self-correction. Look for 'RECOMMENDATION: Revise thought #X...' in the Coordinator's response.
4. **Encouraged Branching:** Explore alternative paths, perspectives, or solutions where appropriate. Use `branchFromThought: <thought_number>` and `branchId: <unique_branch_name>` to initiate branches. Exploring alternatives is key to thorough analysis. Consider suggestions for branching proposed by the Coordinator (e.g., 'SUGGESTION: Consider branching...').
5. **Dynamic Adjustment:** Increase `totalThoughts` if analysis reveals greater complexity than initially estimated. The thinking should be thorough and comprehensive.
6. **Natural Conclusion:** Continue until analysis is complete and satisfactory. `nextThoughtNeeded: False` indicates completion.

**Process Flow:**
- Call 'reflectivethinking' tool for each thought
- Follow guidance and tool recommendations from the Coordinator
- Look for revision and branching opportunities in responses
- Adjust approach based on team feedback and insights
- Continue until problem is thoroughly analyzed

Let me begin with the first thought."""

    return {
        "user": user_prompt_text,
        "assistant": assistant_guidelines,
    }


def tool_selection_prompt(task: str, available_tools: Optional[str] = None):
    """
    Prompt for intelligent tool selection integrated with reflective thinking.
    Tool recommendations are now part of the main reflectivethinking tool.
    """
    tools_context = f"\nAvailable tools: {available_tools}" if available_tools else ""

    user_prompt_text = f"""I need help with tool selection for this task. Use the reflectivethinking tool with integrated tool recommendations.

Task: {task}{tools_context}

Please analyze this task and recommend appropriate tools with rationale."""

    assistant_guidelines = """I'll help you select the most appropriate tools for your task. Let me analyze it using the tool selection system.

**Tool Selection Process:**

1. **Task Analysis:** First, I'll analyze the task to understand:
   - The primary intent (research, analysis, creation, validation, etc.)
   - Required capabilities
   - Success criteria
   - Constraints or limitations

2. **Tool Mapping:** I'll identify tools that match your needs:
   - Primary tools for core functionality
   - Supporting tools for auxiliary tasks
   - Alternative tools for different approaches

3. **Recommendation:** I'll provide:
   - Ranked tool recommendations with confidence scores
   - Rationale for each suggestion
   - Expected outcomes and trade-offs
   - Suggested usage patterns or sequences

Let me start by analyzing your task."""

    return {
        "user": user_prompt_text,
        "assistant": assistant_guidelines,
    }


def thought_review_prompt():
    """
    Prompt to review and summarize a sequential thinking session.
    """

    user_prompt_text = """Please review and summarize the sequential thinking process.

I'd like to understand:
- Key insights discovered
- Decision points and branches explored
- Revisions made and their impact
- Overall effectiveness of the analysis
- Recommendations for improvement"""

    assistant_guidelines = """I'll review the sequential thinking session and provide a comprehensive summary.

**Review Process:**

1. **Session Analysis:** Using the `reflectivereview` tool, I'll examine:
   - Total thoughts and their progression
   - Branch exploration and outcomes
   - Revision patterns and effectiveness
   - Quality metrics and coherence scores

2. **Key Findings:** I'll identify:
   - Most valuable insights discovered
   - Critical decision points and their impact
   - Patterns in the thinking process
   - Areas where the analysis was particularly strong or weak

3. **Improvement Recommendations:** Based on the analysis, I'll suggest:
   - Ways to enhance future thinking sessions
   - Tools or approaches that could be more effective
   - Process adjustments for similar problems

Let me review the session now."""

    return {
        "user": user_prompt_text,
        "assistant": assistant_guidelines,
    }


def complex_problem_prompt(
    problem: str, constraints: Optional[str] = None, goals: Optional[str] = None
):
    """
    Comprehensive prompt for tackling complex, multi-faceted problems with constraints and goals.
    """
    constraints_text = f"\n\nConstraints:\n{constraints}" if constraints else ""
    goals_text = f"\n\nGoals:\n{goals}" if goals else ""

    user_prompt_text = f"""I have a complex problem that requires deep, systematic analysis:

Problem: {problem}{constraints_text}{goals_text}

Please help me work through this systematically, exploring multiple approaches and considering various perspectives."""

    assistant_guidelines = """I'll help you tackle this complex problem using a comprehensive, multi-faceted approach.

**Approach Overview:**

1. **Problem Decomposition:** I'll break down the problem into:
   - Core components and sub-problems
   - Stakeholder perspectives and interests
   - System interactions and dependencies
   - Critical success factors

2. **Multi-Angle Analysis:** I'll explore:
   - Different solution approaches and methodologies
   - Risk assessment and mitigation strategies
   - Resource requirements and constraints
   - Timeline and implementation considerations

3. **Synthesis and Recommendations:** I'll provide:
   - Integrated solution recommendations
   - Implementation roadmap with priorities
   - Contingency planning for key risks
   - Success metrics and evaluation criteria

**Process:**
- I'll use the reflectivethinking tool for systematic analysis
- Multiple thinking branches will explore different solution paths
- Revisions will refine approaches based on deeper insights
- The analysis will be thorough and consider multiple perspectives

Let me begin with a comprehensive analysis of your complex problem."""

    return {
        "user": user_prompt_text,
        "assistant": assistant_guidelines,
    }
