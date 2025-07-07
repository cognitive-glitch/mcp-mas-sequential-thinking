#!/usr/bin/env python3
"""
Basic test to see if MCP tools work at all.
"""

import os
import sys
import pytest

sys.path.append("src")


# Set environment for testing
os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key-for-basic-test"


@pytest.mark.asyncio
async def test_basic_mcp():
    """Test basic MCP functionality without complex setup."""
    try:
        from tools.mcp_tools import reflectivethinking

        # Simple test with new function signature
        print("Testing basic reflectivethinking...")
        result = await reflectivethinking(
            thought="This is a basic test of the reflective thinking system with sufficient content",
            thought_number=5,  # Make it the final thought
            total_thoughts=5,
            next_thought_needed=False,
        )
        print(f"Result: {result}")

        return "SUCCESS" if "error" not in result.lower() else "FAILURE"

    except Exception as e:
        print(f"Basic test failed: {e}")
        return "FAILURE"


# if __name__ == "__main__":
#     result = asyncio.run(test_basic_mcp())
#     print(f"Basic MCP test: {result}")
