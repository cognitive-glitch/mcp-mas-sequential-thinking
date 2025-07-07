"""
Configuration constants and settings for the Reflective Sequential Thinking MCP server.
"""

import os

# Team configuration
DEFAULT_MAX_AGENTS_PER_TEAM = 8
DEFAULT_AGENT_TIMEOUT = 30.0
AGENT_CONCURRENCY_LIMIT = 3  # Maximum agents to run concurrently

# Model configuration
DEFAULT_TEAM_MODEL = "gpt-4o-mini"
DEFAULT_REFLECTION_MODEL = "gpt-4o-mini"

# Thought validation
MIN_THOUGHT_LENGTH = 10
MIN_TOTAL_THOUGHTS = 5
MAX_KEYWORD_LENGTH = 20

# Context management
MAX_CONTEXT_MEMORY_ITEMS = 100
RELEVANT_CONTEXT_LIMIT = 5
GRAPH_EDGE_SIMILARITY_THRESHOLD = 0.7

# Performance settings
DEFAULT_PROCESSING_TIMEOUT = 120.0
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 60.0

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Provider settings
REFLECTIVE_LLM_PROVIDER = os.getenv("REFLECTIVE_LLM_PROVIDER", "openrouter")

# Feature flags
ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "true").lower() == "true"
ENABLE_SHARED_CONTEXT = os.getenv("ENABLE_SHARED_CONTEXT", "true").lower() == "true"
REFLECTION_DELAY_MS = int(os.getenv("REFLECTION_DELAY_MS", "500"))

# Error messages
ERROR_MESSAGES = {
    "invalid_thought_length": f"Thought must be at least {MIN_THOUGHT_LENGTH} characters long",
    "invalid_total_thoughts": f"Total thoughts must be at least {MIN_TOTAL_THOUGHTS}",
    "ending_too_early": "Cannot end thought sequence before reaching total thoughts",
    "consecutive_branching": "Consecutive branching may indicate revision instead",
    "team_not_initialized": "Teams not initialized. Please ensure environment is properly configured.",
    "model_error": "Model invocation failed. Please check your API credentials.",
    "circuit_breaker_open": "Circuit breaker is open. Too many failures detected.",
}

# Prompt templates
TEAM_INSTRUCTION_TEMPLATE = """
You are part of a {team_type} team working on sequential thinking problems.
Your role is to {role_description}.

Key responsibilities:
{responsibilities}

Guidelines:
- Be thorough and systematic in your analysis
- Consider multiple perspectives
- Provide clear, actionable insights
- Support your conclusions with reasoning
"""


def get_model_id(provider: str, model_type: str = "team") -> str:
    """Get the appropriate model ID based on provider and type."""
    env_key = f"{provider.upper()}_{model_type.upper()}_MODEL_ID"
    default_key = f"DEFAULT_{model_type.upper()}_MODEL"

    return os.getenv(env_key, globals()[default_key])


def validate_config() -> None:
    """Validate configuration settings."""
    if MIN_THOUGHT_LENGTH < 1:
        raise ValueError("MIN_THOUGHT_LENGTH must be positive")

    if MIN_TOTAL_THOUGHTS < 1:
        raise ValueError("MIN_TOTAL_THOUGHTS must be positive")

    if AGENT_CONCURRENCY_LIMIT < 1:
        raise ValueError("AGENT_CONCURRENCY_LIMIT must be positive")

    if MAX_CONTEXT_MEMORY_ITEMS < 10:
        raise ValueError("MAX_CONTEXT_MEMORY_ITEMS must be at least 10")
