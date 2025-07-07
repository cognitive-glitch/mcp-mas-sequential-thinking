"""
Tools package for reflective sequential thinking MCP server.

This package contains MCP tool and prompt definitions that were extracted
from main.py to improve modularity and reduce the God Object anti-pattern.
"""

from .mcp_tools import (
    # Tools
    reflectivethinking,
    reflectivereview,
    # Prompts
    sequential_thinking_prompt,
    tool_selection_prompt,
    thought_review_prompt,
    complex_problem_prompt,
    # Utilities
    set_app_context,
    set_mcp_instance,
)

__all__ = [
    "reflectivethinking",
    "reflectivereview",
    "sequential_thinking_prompt",
    "tool_selection_prompt",
    "thought_review_prompt",
    "complex_problem_prompt",
    "set_app_context",
    "set_mcp_instance",
]
