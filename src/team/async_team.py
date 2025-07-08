"""
Async-compatible team implementation for agent coordination.

This module provides the AsyncTeam class which replaces Agno's Team class
to solve async compatibility issues in FastMCP environments.
"""

import asyncio
import json
import logging
from typing import Any, List, Dict, Union
from agno.agent import Agent
from agno.models.message import Message

from config import AGENT_CONCURRENCY_LIMIT, DEFAULT_AGENT_TIMEOUT

logger = logging.getLogger(__name__)


class TeamExecutionError(Exception):
    """Raised when team execution fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AgentExecutionError(Exception):
    """Raised when an individual agent fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MockResponse:
    """Mock response object that mimics Agno's response interface."""

    def __init__(self, content: str):
        self.content = content


class AsyncTeam:
    """
    Simple async-compatible team replacement for Agno Team.

    This class provides concurrent agent execution without using asyncio.run(),
    which is incompatible with FastMCP's existing event loop.
    """

    def __init__(
        self,
        name: str,
        members: List[Agent],
        instructions: List[str],
        model: Any,
        max_concurrency: int = AGENT_CONCURRENCY_LIMIT,
        timeout: float = DEFAULT_AGENT_TIMEOUT,
    ):
        """
        Initialize AsyncTeam.

        Args:
            name: Team name for identification
            members: List of Agent instances
            instructions: Team-level instructions
            model: Language model for synthesis
            max_concurrency: Maximum agents to run concurrently
            timeout: Timeout for individual agent execution
        """
        self.name = name
        self.members = members
        self.instructions = instructions
        self.model = model
        self.max_concurrency = max_concurrency
        self.timeout = timeout

    async def arun(self, input_prompt: Union[str, Dict[str, Any]]) -> MockResponse:
        """
        Async run method that coordinates team members without asyncio.run().

        Args:
            input_prompt: Input prompt for the team

        Returns:
            MockResponse object with synthesized content

        Raises:
            TeamExecutionError: If team execution fails
        """
        try:
            # Convert dict input to string if needed
            prompt_str = (
                json.dumps(input_prompt, indent=2)
                if isinstance(input_prompt, dict)
                else str(input_prompt)
            )

            # Select agents based on concurrency limit
            selected_agents = self.members[: self.max_concurrency]

            # Run agents concurrently
            agent_tasks = [
                self._run_agent_safe(agent, prompt_str) for agent in selected_agents
            ]

            # Wait for all agent responses with timeout
            try:
                agent_responses = await asyncio.wait_for(
                    asyncio.gather(*agent_tasks, return_exceptions=True),
                    timeout=self.timeout * len(selected_agents),
                )
            except asyncio.TimeoutError:
                logger.error(f"Team {self.name} execution timed out")
                raise TeamExecutionError(f"Team {self.name} execution timed out")

            # Filter and format successful responses
            responses = []
            for i, response in enumerate(agent_responses):
                if not isinstance(response, Exception):
                    responses.append(
                        {"agent": selected_agents[i].name, "content": response}
                    )
                else:
                    logger.warning(
                        f"Agent {selected_agents[i].name} failed: {str(response)}"
                    )

            # Synthesize responses
            if responses:
                synthesis = await self._synthesize_responses(responses, prompt_str)
            else:
                raise TeamExecutionError(
                    f"Team {self.name}: All agents failed to process the request"
                )

            return MockResponse(synthesis)

        except TeamExecutionError:
            raise
        except Exception as e:
            logger.error(f"AsyncTeam {self.name} execution failed: {str(e)}")
            raise TeamExecutionError(f"Team {self.name} execution failed: {str(e)}")

    async def _run_agent_safe(self, agent: Agent, prompt: str) -> str:
        """
        Safely run an agent with error handling.

        Args:
            agent: Agent to run
            prompt: Input prompt

        Returns:
            Agent response as string

        Raises:
            AgentExecutionError: If agent execution fails
        """
        try:
            # Create proper Message object for Agno
            messages = [Message(role="user", content=prompt)]

            # Check if agent has a model
            if not agent.model:
                raise AgentExecutionError(f"Agent {agent.name} has no model")

            # Define async function for timeout wrapper
            async def run_model():
                # Type guard: we already checked agent.model is not None
                assert agent.model is not None
                # Try different async methods based on what's available
                if hasattr(agent.model, "aresponse"):
                    return await agent.model.aresponse(messages)
                elif hasattr(agent.model, "ainvoke"):
                    return await agent.model.ainvoke(messages)
                else:
                    raise AgentExecutionError(
                        f"Agent {agent.name} model has no async method"
                    )

            # Use timeout for individual agent (Python 3.10 compatible)
            response = await asyncio.wait_for(run_model(), timeout=self.timeout)

            # Extract content from response
            if hasattr(response, "content"):
                return (
                    str(response.content)
                    if response.content is not None
                    else "No content"
                )
            elif isinstance(response, dict) and "content" in response:
                return (
                    str(response["content"])
                    if response["content"] is not None
                    else "No content"
                )
            else:
                return str(response) if response is not None else "No response"

        except asyncio.TimeoutError:
            raise AgentExecutionError(f"Agent {agent.name} timed out")
        except Exception as e:
            raise AgentExecutionError(f"Agent {agent.name} failed: {str(e)}")

    async def _synthesize_responses(
        self, responses: List[Dict[str, str]], original_prompt: str
    ) -> str:
        """
        Synthesize multiple agent responses into a coherent output.

        Args:
            responses: List of agent responses
            original_prompt: Original input prompt

        Returns:
            Synthesized response as string
        """
        try:
            # Format agent responses
            formatted_responses = "\n\n".join(
                [f"**{r['agent']}**:\n{r['content']}" for r in responses]
            )

            # Create synthesis prompt
            synthesis_prompt = f"""
Based on the following agent responses to the prompt:
"{original_prompt}"

Agent Responses:
{formatted_responses}

Team Instructions:
{" ".join(self.instructions)}

Please synthesize these responses into a coherent, comprehensive answer that:
1. Integrates key insights from all agents
2. Resolves any contradictions
3. Provides a clear, actionable response
4. Maintains the depth and quality of analysis

Synthesis:
"""

            # Use the model to synthesize
            messages = [Message(role="user", content=synthesis_prompt)]

            if hasattr(self.model, "aresponse"):
                response = await self.model.aresponse(messages)
            elif hasattr(self.model, "ainvoke"):
                response = await self.model.ainvoke(messages)
            else:
                # Fallback to simple concatenation
                return f"Team {self.name} Summary:\n\n{formatted_responses}"

            # Extract content
            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict) and "content" in response:
                return response["content"]
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Synthesis failed for team {self.name}: {str(e)}")
            # Fallback to simple formatting
            return f"Team {self.name} Analysis:\n\n" + "\n\n".join(
                [f"{r['agent']}: {r['content']}" for r in responses]
            )

    def __repr__(self) -> str:
        """String representation of the team."""
        return (
            f"AsyncTeam(name='{self.name}', "
            f"members={len(self.members)}, "
            f"max_concurrency={self.max_concurrency})"
        )
