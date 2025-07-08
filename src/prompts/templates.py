"""
Prompt templates for the reflective sequential thinking system.
"""

from typing import Dict, Any, List, Optional


def get_sequential_thinking_prompt(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    is_revision: bool = False,
    revises_thought: Optional[int] = None,
    branch_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Get prompts for sequential thinking MCP tool.

    Args:
        thought: Current thought content
        thought_number: Current thought number
        total_thoughts: Total expected thoughts
        is_revision: Whether this revises a previous thought
        revises_thought: Which thought is being revised
        branch_id: Optional branch identifier
        context: Additional context

    Returns:
        Dictionary of prompts for different aspects
    """
    prompts = {}

    # Main thinking prompt
    prompts["main"] = f"""
You are engaged in sequential thinking step {thought_number} of {total_thoughts}.

Current thought: {thought}

{"This is a revision of thought #" + str(revises_thought) if is_revision else ""}
{"Branch: " + branch_id if branch_id else ""}

Please provide:
1. Deep analysis of this thought
2. Key insights and connections
3. Potential next steps
4. Tool recommendations if applicable
5. Confidence assessment

Remember:
- Be thorough but concise
- Maintain logical flow from previous thoughts
- Consider multiple perspectives
- Identify any gaps or biases
"""

    # Tool selection prompt
    prompts["tool_selection"] = f"""
Based on the thought: "{thought}"

Recommend appropriate tools from the available set.
Consider:
- The nature of the task
- Required capabilities
- Tool strengths and limitations
- Optimal sequencing

Provide confidence scores and rationale for each recommendation.
"""

    # Reflection prompt
    prompts["reflection"] = f"""
Reflect on thought #{thought_number}: "{thought}"

Analyze:
1. Quality of reasoning
2. Completeness of analysis
3. Potential biases or blind spots
4. Alternative approaches
5. Improvement suggestions

Provide constructive feedback for enhancement.
"""

    return prompts


def get_thought_review_prompt(
    thoughts: List[Dict[str, Any]],
    branch_id: Optional[str] = None,
    focus_area: Optional[str] = None,
) -> str:
    """
    Get prompt for reviewing a sequence of thoughts.

    Args:
        thoughts: List of thoughts to review
        branch_id: Optional branch to focus on
        focus_area: Optional specific aspect to analyze

    Returns:
        Review prompt
    """
    thought_summary = "\n".join(
        [
            f"#{t['number']}: {t['content'][:100]}..."
            for t in thoughts[:10]  # Limit to prevent prompt overflow
        ]
    )

    prompt = f"""
Review the following thought sequence:

{thought_summary}

Total thoughts: {len(thoughts)}
{"Branch: " + branch_id if branch_id else "All branches"}
{"Focus: " + focus_area if focus_area else ""}

Please provide:
1. Overall coherence and flow assessment
2. Key insights and patterns identified
3. Strengths of the thinking process
4. Areas for improvement
5. Recommended next steps

Consider:
- Logical progression
- Depth of analysis
- Coverage of the problem space
- Quality of conclusions
"""

    return prompt


def get_complex_problem_prompt(
    problem_statement: str,
    constraints: Optional[List[str]] = None,
    objectives: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get prompt for tackling complex problems.

    Args:
        problem_statement: The problem to solve
        constraints: List of constraints
        objectives: List of objectives
        context: Additional context

    Returns:
        Complex problem prompt
    """
    constraints_text = "\n".join([f"- {c}" for c in (constraints or [])])
    objectives_text = "\n".join([f"- {o}" for o in (objectives or [])])

    # Build constraint section
    constraints_section = f"Constraints:\n{constraints_text}\n" if constraints else ""
    objectives_section = f"Objectives:\n{objectives_text}\n" if objectives else ""

    prompt = f"""
Complex Problem Analysis

Problem Statement:
{problem_statement}

{constraints_section}
{objectives_section}

Approach this systematically:

1. Problem Decomposition
   - Break down into sub-problems
   - Identify dependencies
   - Map the problem space

2. Analysis Framework
   - Apply relevant mental models
   - Consider multiple perspectives
   - Use appropriate analytical tools

3. Solution Development
   - Generate multiple approaches
   - Evaluate trade-offs
   - Synthesize optimal solution

4. Implementation Strategy
   - Define actionable steps
   - Identify resources needed
   - Anticipate challenges

5. Validation
   - Test assumptions
   - Verify logical consistency
   - Assess feasibility

Provide a comprehensive analysis with clear reasoning at each step.
"""

    return prompt


def get_tool_integration_prompt(
    task: str,
    available_tools: List[str],
    current_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get prompt for tool integration and orchestration.

    Args:
        task: The task to accomplish
        available_tools: List of available tools
        current_context: Current execution context

    Returns:
        Tool integration prompt
    """
    tools_list = "\n".join([f"- {tool}" for tool in available_tools])

    prompt = f"""
Tool Integration Strategy

Task: {task}

Available Tools:
{tools_list}

Design an optimal tool usage strategy:

1. Task Analysis
   - Identify required capabilities
   - Break down into tool-appropriate subtasks
   - Determine dependencies

2. Tool Selection
   - Match tools to subtasks
   - Consider tool strengths/limitations
   - Plan integration points

3. Execution Sequence
   - Define order of operations
   - Plan data flow between tools
   - Handle error scenarios

4. Quality Assurance
   - Validate outputs at each step
   - Ensure coherence across tools
   - Maintain overall objective alignment

Provide specific recommendations with confidence scores and rationale.
"""

    return prompt
