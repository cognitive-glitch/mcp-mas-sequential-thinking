#!/usr/bin/env python3
"""
Basic test to see if MCP tools work at all.
"""

import asyncio
import os
import sys

sys.path.append("src")

from models.thought_models import ThoughtData, DomainType

# Set environment for testing
os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key-for-basic-test"


async def test_basic_mcp():
    """Test basic MCP functionality without complex setup."""
    try:
        from main import reflectivethinking

        # Simple thought data
        thought_data = ThoughtData(
            thought="This is a basic test of the reflective thinking system with sufficient content",
            thoughtNumber=5,  # Make it the final thought
            totalThoughts=5,
            nextThoughtNeeded=False,
            domain=DomainType.GENERAL,
        )

        print("Testing basic reflectivethinking...")
        result = await reflectivethinking(thought_data)
        print(f"Result: {result}")

        return "SUCCESS" if "error" not in result.lower() else "FAILURE"

    except Exception as e:
        print(f"Basic test failed: {e}")
        return "FAILURE"


if __name__ == "__main__":
    result = asyncio.run(test_basic_mcp())
    print(f"Basic MCP test: {result}")
