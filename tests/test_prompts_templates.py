"""
Tests for prompt templates module.
"""

from src.prompts.templates import (
    get_sequential_thinking_prompt,
    get_thought_review_prompt,
    get_complex_problem_prompt,
    get_tool_integration_prompt,
)


class TestSequentialThinkingPrompt:
    """Test the sequential thinking prompt generation."""

    def test_basic_sequential_thinking_prompt(self):
        """Test basic sequential thinking prompt generation."""
        prompts = get_sequential_thinking_prompt(
            thought="Analyze the system architecture",
            thought_number=1,
            total_thoughts=5,
        )

        assert isinstance(prompts, dict)
        assert "main" in prompts
        assert "tool_selection" in prompts
        assert "reflection" in prompts

        # Check main prompt content
        main_prompt = prompts["main"]
        assert "sequential thinking step 1 of 5" in main_prompt
        assert "Analyze the system architecture" in main_prompt
        assert "Deep analysis" in main_prompt
        assert "Key insights" in main_prompt

    def test_sequential_thinking_with_revision(self):
        """Test sequential thinking prompt with revision."""
        prompts = get_sequential_thinking_prompt(
            thought="Revised analysis approach",
            thought_number=3,
            total_thoughts=5,
            is_revision=True,
            revises_thought=2,
        )

        main_prompt = prompts["main"]
        assert "revision of thought #2" in main_prompt
        assert "step 3 of 5" in main_prompt

    def test_sequential_thinking_with_branch(self):
        """Test sequential thinking prompt with branch."""
        prompts = get_sequential_thinking_prompt(
            thought="Branch analysis",
            thought_number=2,
            total_thoughts=4,
            branch_id="experimental-approach",
        )

        main_prompt = prompts["main"]
        assert "Branch: experimental-approach" in main_prompt

    def test_sequential_thinking_with_context(self):
        """Test sequential thinking prompt with context."""
        context = {"domain": "technical", "priority": "high"}
        prompts = get_sequential_thinking_prompt(
            thought="Technical review",
            thought_number=1,
            total_thoughts=3,
            context=context,
        )

        # Should generate prompts regardless of context
        assert len(prompts) == 3
        assert all(isinstance(p, str) for p in prompts.values())

    def test_tool_selection_prompt_content(self):
        """Test tool selection prompt specific content."""
        prompts = get_sequential_thinking_prompt(
            thought="Need to analyze code quality", thought_number=1, total_thoughts=3
        )

        tool_prompt = prompts["tool_selection"]
        assert "Need to analyze code quality" in tool_prompt
        assert "appropriate tools" in tool_prompt
        assert "confidence scores" in tool_prompt

    def test_reflection_prompt_content(self):
        """Test reflection prompt specific content."""
        prompts = get_sequential_thinking_prompt(
            thought="Performance optimization strategy",
            thought_number=2,
            total_thoughts=4,
        )

        reflection_prompt = prompts["reflection"]
        assert "thought #2" in reflection_prompt
        assert "Performance optimization strategy" in reflection_prompt
        assert "Quality of reasoning" in reflection_prompt
        assert "Alternative approaches" in reflection_prompt


class TestThoughtReviewPrompt:
    """Test the thought review prompt generation."""

    def test_basic_thought_review_prompt(self):
        """Test basic thought review prompt generation."""
        thoughts = [
            {"number": 1, "content": "First thought about system analysis"},
            {"number": 2, "content": "Second thought about implementation details"},
            {"number": 3, "content": "Third thought about testing strategy"},
        ]

        prompt = get_thought_review_prompt(thoughts)

        assert isinstance(prompt, str)
        assert "Total thoughts: 3" in prompt
        assert "#1: First thought about system analysis" in prompt
        assert "#2: Second thought about implementation" in prompt
        assert "Overall coherence" in prompt
        assert "Key insights" in prompt

    def test_thought_review_with_branch(self):
        """Test thought review prompt with branch filter."""
        thoughts = [
            {"number": 1, "content": "Analysis of approach A"},
            {"number": 2, "content": "Analysis of approach B"},
        ]

        prompt = get_thought_review_prompt(thoughts, branch_id="approach-a")

        assert "Branch: approach-a" in prompt

    def test_thought_review_with_focus_area(self):
        """Test thought review prompt with focus area."""
        thoughts = [
            {"number": 1, "content": "Performance analysis"},
            {"number": 2, "content": "Security considerations"},
        ]

        prompt = get_thought_review_prompt(thoughts, focus_area="security")

        assert "Focus: security" in prompt

    def test_thought_review_truncation(self):
        """Test that long thought lists are truncated."""
        # Create 15 thoughts
        thoughts = [
            {"number": i, "content": f"Thought {i} with some content"}
            for i in range(1, 16)
        ]

        prompt = get_thought_review_prompt(thoughts)

        # Should only show first 10
        assert "#10:" in prompt
        assert "#11:" not in prompt
        assert "Total thoughts: 15" in prompt

    def test_thought_review_content_truncation(self):
        """Test that long thought content is truncated."""
        thoughts = [
            {
                "number": 1,
                "content": "This is a very long thought content that should be truncated after 100 characters to prevent prompt overflow and maintain readability",
            }
        ]

        prompt = get_thought_review_prompt(thoughts)

        # Should truncate after 100 chars and add "..."
        assert (
            "This is a very long thought content that should be truncated after 100 characters to prevent"
            in prompt
        )
        assert "..." in prompt

    def test_empty_thoughts_list(self):
        """Test handling of empty thoughts list."""
        thoughts = []

        prompt = get_thought_review_prompt(thoughts)

        assert "Total thoughts: 0" in prompt
        assert isinstance(prompt, str)


class TestComplexProblemPrompt:
    """Test the complex problem prompt generation."""

    def test_basic_complex_problem_prompt(self):
        """Test basic complex problem prompt generation."""
        prompt = get_complex_problem_prompt(
            "Design a scalable microservices architecture"
        )

        assert isinstance(prompt, str)
        assert "Design a scalable microservices architecture" in prompt
        assert "Complex Problem Analysis" in prompt
        assert "Problem Decomposition" in prompt
        assert "Implementation Strategy" in prompt
        assert "Validation" in prompt

    def test_complex_problem_with_constraints(self):
        """Test complex problem prompt with constraints."""
        constraints = [
            "Must handle 1M+ users",
            "Budget limit of $50k",
            "Use existing tech stack",
        ]

        prompt = get_complex_problem_prompt(
            "Build user management system", constraints=constraints
        )

        assert "Constraints:" in prompt
        assert "- Must handle 1M+ users" in prompt
        assert "- Budget limit of $50k" in prompt
        assert "- Use existing tech stack" in prompt

    def test_complex_problem_with_objectives(self):
        """Test complex problem prompt with objectives."""
        objectives = [
            "Improve user experience",
            "Reduce operational costs",
            "Enhance system reliability",
        ]

        prompt = get_complex_problem_prompt(
            "System optimization project", objectives=objectives
        )

        assert "Objectives:" in prompt
        assert "- Improve user experience" in prompt
        assert "- Reduce operational costs" in prompt
        assert "- Enhance system reliability" in prompt

    def test_complex_problem_with_constraints_and_objectives(self):
        """Test complex problem prompt with both constraints and objectives."""
        constraints = ["Limited time", "Small team"]
        objectives = ["High quality", "Fast delivery"]

        prompt = get_complex_problem_prompt(
            "Development project", constraints=constraints, objectives=objectives
        )

        assert "Constraints:" in prompt
        assert "Objectives:" in prompt
        assert "- Limited time" in prompt
        assert "- High quality" in prompt

    def test_complex_problem_with_context(self):
        """Test complex problem prompt with additional context."""
        context = {"domain": "fintech", "urgency": "high"}

        prompt = get_complex_problem_prompt("Payment system redesign", context=context)

        # Context is accepted but doesn't change prompt structure
        assert "Payment system redesign" in prompt
        assert "Complex Problem Analysis" in prompt

    def test_complex_problem_empty_constraints_objectives(self):
        """Test complex problem prompt with empty constraints and objectives."""
        prompt = get_complex_problem_prompt(
            "Simple problem statement", constraints=[], objectives=[]
        )

        # Empty lists should not show constraint/objective sections
        assert "Constraints:" not in prompt
        assert "Objectives:" not in prompt
        assert "Simple problem statement" in prompt


class TestToolIntegrationPrompt:
    """Test the tool integration prompt generation."""

    def test_basic_tool_integration_prompt(self):
        """Test basic tool integration prompt generation."""
        tools = ["code_analyzer", "test_runner", "performance_profiler"]

        prompt = get_tool_integration_prompt("Optimize application performance", tools)

        assert isinstance(prompt, str)
        assert "Optimize application performance" in prompt
        assert "Tool Integration Strategy" in prompt
        assert "- code_analyzer" in prompt
        assert "- test_runner" in prompt
        assert "- performance_profiler" in prompt
        assert "Task Analysis" in prompt
        assert "Tool Selection" in prompt
        assert "Execution Sequence" in prompt
        assert "Quality Assurance" in prompt

    def test_tool_integration_with_context(self):
        """Test tool integration prompt with context."""
        tools = ["linter", "formatter"]
        context = {"language": "python", "framework": "django"}

        prompt = get_tool_integration_prompt("Code quality improvement", tools, context)

        # Context is accepted but doesn't change core prompt structure
        assert "Code quality improvement" in prompt
        assert "- linter" in prompt
        assert "- formatter" in prompt

    def test_tool_integration_empty_tools(self):
        """Test tool integration prompt with empty tools list."""
        prompt = get_tool_integration_prompt("Manual analysis task", [])

        assert "Manual analysis task" in prompt
        assert "Available Tools:" in prompt
        # Should still provide framework even with no tools
        assert "Tool Selection" in prompt

    def test_tool_integration_single_tool(self):
        """Test tool integration prompt with single tool."""
        prompt = get_tool_integration_prompt("Security scan", ["security_scanner"])

        assert "Security scan" in prompt
        assert "- security_scanner" in prompt
        assert "Tool Integration Strategy" in prompt

    def test_tool_integration_many_tools(self):
        """Test tool integration prompt with many tools."""
        tools = [f"tool_{i}" for i in range(1, 11)]  # 10 tools

        prompt = get_tool_integration_prompt("Complex multi-tool task", tools)

        assert "Complex multi-tool task" in prompt
        # All tools should be listed
        for tool in tools:
            assert f"- {tool}" in prompt

    def test_all_prompts_return_strings(self):
        """Test that all prompt functions return non-empty strings."""
        # Sequential thinking
        seq_prompts = get_sequential_thinking_prompt("test", 1, 3)
        for prompt in seq_prompts.values():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0

        # Thought review
        review_prompt = get_thought_review_prompt([{"number": 1, "content": "test"}])
        assert isinstance(review_prompt, str)
        assert len(review_prompt.strip()) > 0

        # Complex problem
        complex_prompt = get_complex_problem_prompt("test problem")
        assert isinstance(complex_prompt, str)
        assert len(complex_prompt.strip()) > 0

        # Tool integration
        tool_prompt = get_tool_integration_prompt("test task", ["tool1"])
        assert isinstance(tool_prompt, str)
        assert len(tool_prompt.strip()) > 0
