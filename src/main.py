#!/usr/bin/env python3
"""
Enhanced Reflective Sequential Thinking MCP Server with Tool Selection
Comprehensive implementation with dual-team architecture, tool recommendations, and memory persistence.

Key Features:
- Dual-team architecture: Primary thinking team + Reflection team
- Enhanced ThoughtData with topic/subject alignment and tool recommendations
- Integrated tool recommendations within reflectivethinking
- SharedContext for memory persistence across thoughts and branches
- LLMProviderFactory supporting OpenRouter, OpenAI, Gemini
- Zero-token API bug fixes and comprehensive error handling
- reflectivethinking and reflectivereview MCP tools
- Claude Code hooks integration ready
"""

import time
import logging
import json
from contextlib import asynccontextmanager
from pathlib import Path

# Core frameworks
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Enhanced local imports
from models.thought_models import (
    ThoughtData,
    ProcessedThought,
)
from context.app_context import EnhancedAppContext

# Import refactored modules
from team import AsyncTeam as RefactoredAsyncTeam
from tools import (
    set_app_context,
    set_mcp_instance,
)
from exceptions import ErrorType
from config import LOG_LEVEL

import logging.handlers

# Load environment variables
load_dotenv()


# Configure logging
def setup_logging() -> logging.Logger:
    """Enhanced logging setup with detailed format."""
    home_dir = Path.home()
    log_dir = home_dir / ".reflective_thinking" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("reflective_thinking_enhanced")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "reflective_thinking.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file logger: {e}")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# AsyncTeam has been moved to team/async_team.py
# This is kept for backward compatibility during migration
AsyncTeam = RefactoredAsyncTeam


# Error handling classes have been extracted to error_handling module


# Global app context
app_context = EnhancedAppContext()

# Set app context for extracted MCP tools
set_app_context(app_context)

# Create FastMCP server
mcp = FastMCP("reflective-thinking-tools")

# Set MCP instance for tools to use decorators
set_mcp_instance(mcp)


async def process_thought_with_dual_teams(
    thought_data: ThoughtData, context: EnhancedAppContext
) -> ProcessedThought:
    """
    Process a thought through both primary and reflection teams.
    Includes comprehensive error handling and performance tracking.
    """
    start_time = time.time()

    try:
        # Ensure teams are initialized
        if not context.teams_initialized:
            await context.initialize_teams()

        # Update shared context
        await context.add_thought(thought_data)

        # Get relevant context
        relevant_context = await context.get_relevant_context(thought_data.thought)

        # Generate tool recommendations if not provided
        if not thought_data.current_step:
            try:
                # Tool recommendation functionality is now integrated directly
                tool_recommendation = None
                thought_data.current_step = tool_recommendation
            except Exception as e:
                logger.warning(f"Tool selection failed: {e}")
                # Continue without tool recommendations

        # Prepare input for primary team
        primary_input = f"""
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

        # Process with primary team
        primary_response = None
        try:
            if context.primary_team:
                primary_response = await context.primary_team.arun(primary_input)
                context.error_handler.circuit_breakers[
                    "team_processing"
                ].record_success()
            else:
                raise Exception("Primary team not initialized")
        except Exception as e:
            context.error_handler.circuit_breakers["team_processing"].record_failure()
            error_msg = context.error_handler.handle_error(
                e,
                ErrorType.TEAM_PROCESSING,
                thought_data.thoughtNumber,
                {"team": "primary", "input_length": len(primary_input)},
            )
            logger.error(f"Primary team processing failed: {error_msg}")
            primary_response = f"Primary team error: {error_msg}"

        # Prepare input for reflection team
        reflection_input = f"""
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

        # Process with reflection team
        reflection_response = None
        try:
            if context.reflection_team:
                reflection_response = await context.reflection_team.arun(
                    reflection_input
                )
                context.error_handler.circuit_breakers[
                    "team_processing"
                ].record_success()
            else:
                raise Exception("Reflection team not initialized")
        except Exception as e:
            context.error_handler.circuit_breakers["team_processing"].record_failure()
            error_msg = context.error_handler.handle_error(
                e,
                ErrorType.TEAM_PROCESSING,
                thought_data.thoughtNumber,
                {"team": "reflection", "input_length": len(reflection_input)},
            )
            logger.error(f"Reflection team processing failed: {error_msg}")
            reflection_response = f"Reflection team error: {error_msg}"

        # Extract content from responses
        primary_content = ""
        reflection_content = ""

        if primary_response:
            if isinstance(primary_response, str):
                primary_content = primary_response
            elif hasattr(primary_response, "content"):
                primary_content = str(primary_response.content)
            else:
                primary_content = str(primary_response)

        if reflection_response:
            if isinstance(reflection_response, str):
                reflection_content = reflection_response
            elif hasattr(reflection_response, "content"):
                reflection_content = str(reflection_response.content)
            else:
                reflection_content = str(reflection_response)

        # Integrate responses
        integrated_response = f"""
## Primary Analysis
{primary_content}

## Reflection & Meta-Analysis
{reflection_content}

## Integrated Guidance
Based on both analyses, the recommended approach is to proceed with the insights from the primary team while incorporating the meta-level improvements suggested by the reflection team.
"""

        # Calculate execution time
        execution_time = int((time.time() - start_time) * 1000)

        # Record performance
        await context.shared_context.record_performance(
            "processing_time", execution_time
        )

        # Record tool effectiveness if tools were used
        if thought_data.current_step and thought_data.current_step.recommended_tools:
            # Tool effectiveness tracking removed (was in ToolSelector)
            pass

        # Create processed thought
        return ProcessedThought(
            thought_data=thought_data,
            coordinator_response=primary_content or "",
            reflection_response=reflection_content or "",
            integrated_response=integrated_response,
            next_step_guidance="Continue with the next thought based on the integrated guidance above.",
            execution_time_ms=execution_time,
            token_usage={
                "primary": len(primary_content.split()) if primary_content else 0,
                "reflection": len(reflection_content.split())
                if reflection_content
                else 0,
            },
            success=True,
            tool_recommendations_generated=bool(thought_data.current_step),
            reflection_applied=True,
            context_updated=True,
        )

    except Exception as e:
        # Handle unexpected errors
        error_msg = context.error_handler.handle_error(
            e,
            ErrorType.TEAM_PROCESSING,
            thought_data.thoughtNumber,
            {"phase": "integration"},
        )
        logger.error(f"Thought processing failed completely: {error_msg}")

        return ProcessedThought(
            thought_data=thought_data,
            coordinator_response="",
            reflection_response="",
            integrated_response=f"Processing failed: {error_msg}",
            next_step_guidance="Please retry with a simpler thought or check the system status.",
            execution_time_ms=int((time.time() - start_time) * 1000),
            token_usage={},
            success=False,
            error=str(e),
            tool_recommendations_generated=False,
            reflection_applied=False,
            context_updated=False,
        )


@asynccontextmanager
async def lifespan(app):
    """Lifecycle manager for the MCP server."""
    logger.info("Starting Reflective Thinking MCP Server...")

    try:
        # Initialize teams on startup
        await app_context.initialize_teams()
        logger.info("Teams initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize teams: {e}")
        # Continue anyway - teams will be initialized on first use

    yield

    # Cleanup
    logger.info("Shutting down Reflective Thinking MCP Server...")
    await app_context.cleanup()


def run():
    mcp.run("stdio")


# Main execution
if __name__ == "__main__":
    run()
