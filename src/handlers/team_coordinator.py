"""
Team coordination and initialization logic.
"""

import logging
from typing import List, Optional, Dict, Any, cast
from agno.agent import Agent
from models.protocols import ModelProtocol

from team import AsyncTeam
from providers.base import LLMProviderFactory
from exceptions import ModelInitializationError
from config import DEFAULT_MAX_AGENTS_PER_TEAM, REFLECTIVE_LLM_PROVIDER

logger = logging.getLogger(__name__)


class TeamCoordinator:
    """Manages team creation and coordination."""

    def __init__(self):
        """Initialize team coordinator."""
        self.provider = REFLECTIVE_LLM_PROVIDER
        self.primary_team: Optional[AsyncTeam] = None
        self.reflection_team: Optional[AsyncTeam] = None
        self.teams_initialized = False

    async def initialize_teams(self) -> None:
        """
        Initialize both primary and reflection teams.

        Raises:
            ModelInitializationError: If model creation fails
            ConfigurationError: If configuration is invalid
        """
        if self.teams_initialized:
            return

        try:
            # Get models
            team_model = await self._get_team_model()
            reflection_model = await self._get_reflection_model()

            # Create primary team
            self.primary_team = await self._create_primary_team(team_model)

            # Create reflection team
            self.reflection_team = await self._create_reflection_team(reflection_model)

            self.teams_initialized = True
            logger.info("Teams initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize teams: {str(e)}")
            raise ModelInitializationError(
                provider=self.provider, model_id="team_models", reason=str(e)
            )

    async def _get_team_model(self) -> ModelProtocol:
        """Get the primary team model."""
        team_model_id = "unknown"
        try:
            config = LLMProviderFactory.get_provider_config(self.provider)
            config.validate()
            team_model_id, _ = config.get_models()
            model = config.create_model_instance(team_model_id)
            return model
        except Exception as e:
            raise ModelInitializationError(
                provider=self.provider,
                model_id=team_model_id,
                reason=str(e),
            )

    async def _get_reflection_model(self) -> ModelProtocol:
        """Get the reflection team model."""
        reflection_model_id = "unknown"
        try:
            config = LLMProviderFactory.get_provider_config(self.provider)
            config.validate()
            _, reflection_model_id = config.get_models()
            model = config.create_model_instance(reflection_model_id)
            return model
        except Exception as e:
            raise ModelInitializationError(
                provider=self.provider,
                model_id=reflection_model_id,
                reason=str(e),
            )

    async def _create_primary_team(self, model: ModelProtocol) -> AsyncTeam:
        """Create the primary thinking team."""
        agents = self._create_primary_agents(model)
        instructions = self._generate_primary_instructions()

        return AsyncTeam(
            name="PrimaryThinkingTeam",
            members=agents,
            instructions=instructions,
            model=model,
        )

    async def _create_reflection_team(self, model: ModelProtocol) -> AsyncTeam:
        """Create the reflection team."""
        agents = self._create_reflection_agents(model)
        instructions = self._generate_reflection_instructions()

        return AsyncTeam(
            name="ReflectionTeam",
            members=agents,
            instructions=instructions,
            model=model,
        )

    def _create_primary_agents(self, model: ModelProtocol) -> List[Agent]:
        """Create agents for the primary thinking team."""
        agent_configs = [
            {
                "name": "Strategic Planner",
                "role": "Plan and structure the thinking process",
                "instructions": [
                    "Break down complex problems into manageable steps",
                    "Identify key objectives and constraints",
                    "Create actionable plans with clear milestones",
                ],
            },
            {
                "name": "Domain Researcher",
                "role": "Gather and analyze domain-specific information",
                "instructions": [
                    "Research relevant concepts and precedents",
                    "Identify domain-specific constraints and opportunities",
                    "Provide expert knowledge and context",
                ],
            },
            {
                "name": "Critical Analyzer",
                "role": "Analyze problems from multiple perspectives",
                "instructions": [
                    "Challenge assumptions and identify biases",
                    "Evaluate trade-offs and alternatives",
                    "Ensure logical consistency and rigor",
                ],
            },
            {
                "name": "Creative Innovator",
                "role": "Generate novel solutions and approaches",
                "instructions": [
                    "Think outside conventional boundaries",
                    "Propose innovative solutions",
                    "Connect disparate concepts creatively",
                ],
            },
            {
                "name": "Practical Implementer",
                "role": "Focus on feasibility and execution",
                "instructions": [
                    "Assess practical constraints and resources",
                    "Develop actionable implementation strategies",
                    "Identify potential obstacles and solutions",
                ],
            },
        ]

        agents = []
        for config in agent_configs[:DEFAULT_MAX_AGENTS_PER_TEAM]:
            agent = Agent(
                name=config["name"],
                role=config["role"],
                goal=f"Excel at {config['role'].lower()}",
                instructions=config["instructions"],
                model=cast(
                    Any, model
                ),  # Cast ModelProtocol to Any for Agent compatibility
            )
            agents.append(agent)

        return agents

    def _create_reflection_agents(self, model: ModelProtocol) -> List[Agent]:
        """Create agents for the reflection team."""
        agent_configs = [
            {
                "name": "Meta-Analyzer",
                "role": "Analyze the thinking process itself",
                "instructions": [
                    "Evaluate the quality of reasoning",
                    "Identify cognitive biases and blind spots",
                    "Assess completeness and depth of analysis",
                ],
            },
            {
                "name": "Pattern Recognizer",
                "role": "Identify patterns and connections",
                "instructions": [
                    "Detect recurring themes and patterns",
                    "Connect insights across different thoughts",
                    "Identify emergent properties and trends",
                ],
            },
            {
                "name": "Quality Assessor",
                "role": "Evaluate output quality and coherence",
                "instructions": [
                    "Assess clarity and logical flow",
                    "Evaluate evidence and argumentation",
                    "Rate confidence and uncertainty levels",
                ],
            },
            {
                "name": "Decision Critic",
                "role": "Critique decisions and recommendations",
                "instructions": [
                    "Challenge proposed solutions",
                    "Identify risks and unintended consequences",
                    "Suggest improvements and alternatives",
                ],
            },
        ]

        agents = []
        for config in agent_configs:
            agent = Agent(
                name=config["name"],
                role=config["role"],
                goal=f"Excel at {config['role'].lower()}",
                instructions=config["instructions"],
                model=cast(
                    Any, model
                ),  # Cast ModelProtocol to Any for Agent compatibility
            )
            agents.append(agent)

        return agents

    def _generate_primary_instructions(self) -> List[str]:
        """Generate instructions for the primary team."""
        return [
            "Work collaboratively to analyze and solve complex problems",
            "Consider multiple perspectives and approaches",
            "Provide clear, actionable insights and recommendations",
            "Balance depth of analysis with practical applicability",
            "Use structured thinking and systematic approaches",
            "Identify and leverage appropriate tools and resources",
            "Maintain intellectual rigor while being creative",
        ]

    def _generate_reflection_instructions(self) -> List[str]:
        """Generate instructions for the reflection team."""
        return [
            "Provide meta-analysis of the thinking process",
            "Identify strengths and weaknesses in the approach",
            "Detect biases, gaps, and missed opportunities",
            "Suggest improvements for future iterations",
            "Evaluate the overall quality and coherence",
            "Ensure alignment with stated objectives",
            "Provide constructive feedback for enhancement",
        ]

    async def generate_adaptive_instructions(
        self, context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate adaptive instructions based on context.

        Args:
            context: Current thinking context

        Returns:
            List of adaptive instructions
        """
        base_instructions = self._generate_primary_instructions()

        # Add context-specific instructions
        if context.get("domain") == "technical":
            base_instructions.append(
                "Pay special attention to technical accuracy and best practices"
            )
        elif context.get("domain") == "creative":
            base_instructions.append(
                "Prioritize innovative and unconventional approaches"
            )

        # Add complexity-based instructions
        if context.get("complexity", 0) > 0.7:
            base_instructions.append(
                "Break down this complex problem into smaller, manageable components"
            )

        # Add revision-specific instructions
        if context.get("is_revision"):
            base_instructions.append(
                "Focus on addressing previous limitations and improving the approach"
            )

        return base_instructions
