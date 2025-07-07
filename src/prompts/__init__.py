"""Prompt templates for the system."""

from .templates import (
    get_sequential_thinking_prompt,
    get_thought_review_prompt,
    get_complex_problem_prompt,
    get_tool_integration_prompt,
)

__all__ = [
    "get_sequential_thinking_prompt",
    "get_thought_review_prompt",
    "get_complex_problem_prompt",
    "get_tool_integration_prompt",
]
