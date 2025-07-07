"""
Enhanced application context managing teams, models, and shared state.
Extracted from main.py following TDD principles.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.tools.exa import ExaTools
from agno.tools.thinking import ThinkingTools

from models.thought_models import ThoughtData, DomainType
from providers.base import LLMProviderFactory
from team import AsyncTeam
from context.shared_context import SharedContext
from error_handling import EnhancedErrorHandler
from exceptions import ErrorType

logger = logging.getLogger(__name__)


class EnhancedAppContext:
    """
    Enhanced application context managing teams, models, and shared state.
    Includes tool selection capabilities and comprehensive error handling.
    """

    def __init__(self):
        """Initialize the enhanced application context with all components."""
        self.start_time = datetime.now()

        # Initialize components
        self.shared_context = SharedContext()  # Simple in-memory context
        self.error_handler = EnhancedErrorHandler()

        # Available tools
        self.available_tools = ["ThinkingTools", "ExaTools"]
        self.current_topic = "Reflective Thinking Session"
        self.current_domain = DomainType.GENERAL

        # Initialize provider and models
        try:
            self.provider_config = LLMProviderFactory.get_provider_config()
            self.provider_initialized = True
            logger.info(f"Initialized provider: {self.provider_config.provider_name}")
        except Exception as e:
            self.provider_initialized = False
            error_msg = self.error_handler.handle_error(
                e, ErrorType.TEAM_INITIALIZATION, context={"component": "provider"}
            )
            logger.error(f"Failed to initialize provider: {error_msg}")
            raise

        # Initialize teams
        self.primary_team: Optional[AsyncTeam] = None
        self.reflection_team: Optional[AsyncTeam] = None
        self.teams_initialized = False

        # Thought tracking
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}

        # Available MCP tools tracking
        self.available_mcp_tools: List[str] = []

        logger.info("Enhanced app context initialized")

    async def initialize_teams(self):
        """Initialize both primary and reflection teams asynchronously."""
        if self.teams_initialized:
            return

        try:
            # Get model configurations
            team_model_id, agent_model_id = self.provider_config.get_models()
            logger.info(
                f"Using team model: {team_model_id}, agent model: {agent_model_id}"
            )

            try:
                team_model = self.provider_config.create_model_instance(team_model_id)
                agent_model = self.provider_config.create_model_instance(agent_model_id)
                logger.info("Model instances created successfully")
            except Exception as e:
                logger.error(f"Failed to create model instances: {e}")
                raise

            # Create primary thinking team
            logger.info("Creating primary thinking team...")
            self.primary_team = await self._create_primary_team(team_model, agent_model)
            logger.info("Primary team created successfully")

            # Create reflection team
            logger.info("Creating reflection team...")
            self.reflection_team = await self._create_reflection_team(
                team_model, agent_model
            )
            logger.info("Reflection team created successfully")

            self.teams_initialized = True
            logger.info("Both teams initialized successfully")

        except Exception as e:
            error_msg = self.error_handler.handle_error(
                e, ErrorType.TEAM_INITIALIZATION, context={"stage": "team_creation"}
            )
            logger.error(f"Team initialization failed: {error_msg}")
            raise

    async def _generate_adaptive_coordinator_instructions(self) -> List[str]:
        """Generate context-aware coordinator instructions."""
        base_instructions = [
            "🎯 **PRIMARY THINKING TEAM COORDINATOR**",
            "You orchestrate a team of specialists to process reflective thoughts with precision and intelligence.",
        ]

        # Add context-specific instructions
        if self.thought_history:
            recent_domains = [t.domain for t in self.thought_history[-3:]]
            if all(d == DomainType.TECHNICAL for d in recent_domains):
                base_instructions.append(
                    "Focus on technical accuracy and implementation details."
                )
            elif all(d == DomainType.CREATIVE for d in recent_domains):
                base_instructions.append(
                    "Emphasize creative solutions and innovative approaches."
                )

        # Add performance-based adjustments
        perf_summary = await self.shared_context.get_performance_summary()
        if perf_summary and "processing_time" in perf_summary:
            avg_time = perf_summary["processing_time"].get("mean", 0)
            if avg_time > 3000:  # If average processing > 3 seconds
                base_instructions.append(
                    "Optimize for efficiency while maintaining quality."
                )

        base_instructions.extend(
            [
                "",
                "**Team Coordination Process:**",
                "1. Receive and understand the thought's intent and context",
                "2. Delegate specific aspects to appropriate team members",
                "3. Synthesize all specialist responses into cohesive guidance",
                "4. Ensure responses are actionable and contextually relevant",
                "",
                "**Key Principles:**",
                "- Adaptive thinking based on thought complexity and domain",
                "- Clear, actionable synthesis of specialist insights",
                "- Context awareness across thought sequences and branches",
                "- Quality over speed, but mindful of efficiency",
            ]
        )

        return base_instructions

    def _generate_planner_instructions(
        self, thought_data: Optional[ThoughtData] = None
    ) -> str:
        """Generate adaptive planner instructions based on context."""
        base_prompt = "You are the Strategic Planner in the thinking team."

        if thought_data:
            if thought_data.isRevision:
                base_prompt += (
                    " Focus on revising the previous strategy based on new insights."
                )
            elif thought_data.branchFromThought:
                base_prompt += " Develop an alternative strategic path for this branch."
            elif thought_data.thoughtNumber == 1:
                base_prompt += (
                    " Create the initial strategic framework for this thinking process."
                )
            elif thought_data.needsMoreThoughts:
                base_prompt += (
                    " Extend the strategic plan to accommodate additional analysis."
                )

        base_prompt += """
        Your responsibilities:
        - Develop strategic approaches tailored to the current thought
        - Identify key milestones and decision points
        - Anticipate potential challenges and opportunities
        - Provide clear direction for the team's efforts
        """

        return base_prompt

    def _generate_researcher_instructions(
        self, thought_data: Optional[ThoughtData] = None
    ) -> str:
        """Generate adaptive researcher instructions based on context."""
        base_prompt = "You are the Information Researcher in the thinking team."

        if thought_data and thought_data.keywords:
            keywords_str = ", ".join(thought_data.keywords[:5])
            base_prompt += f" Focus your research on: {keywords_str}."

        base_prompt += """
        Your responsibilities:
        - Gather relevant information and context
        - Identify key facts, patterns, and relationships
        - Highlight important discoveries and insights
        - Provide evidence-based support for analysis
        """

        return base_prompt

    async def _create_primary_team(self, team_model, agent_model) -> AsyncTeam:
        """Create the primary thinking team with specialized agents."""
        planner = Agent(
            name="Planner",
            role="Strategic Planner",
            instructions=self._generate_planner_instructions(),
            model=agent_model,
            tools=[ThinkingTools()],
        )

        researcher = Agent(
            name="Researcher",
            role="Information Gatherer",
            instructions=self._generate_researcher_instructions(),
            model=agent_model,
            tools=[ExaTools()],
        )

        analyzer = Agent(
            name="Analyzer",
            role="Core Analyst",
            instructions="""You are the Core Analyst in the thinking team.
            Your responsibilities:
            - Perform deep analysis of the current thought
            - Identify patterns, relationships, and implications
            - Break down complex problems into components
            - Provide structured analytical insights
            - Recommend appropriate tools for the current step
            """,
            model=agent_model,
            tools=[ThinkingTools()],
        )

        critic = Agent(
            name="Critic",
            role="Quality Controller",
            instructions="""You are the Critical Reviewer in the thinking team.
            Your responsibilities:
            - Identify potential flaws or gaps in reasoning
            - Challenge assumptions constructively
            - Ensure logical consistency
            - Suggest improvements and alternatives
            - Validate tool recommendations
            """,
            model=agent_model,
        )

        synthesizer = Agent(
            name="Synthesizer",
            role="Integration Specialist",
            instructions="""You are the Integration Specialist in the thinking team.
            Your responsibilities:
            - Integrate insights from all team members
            - Create coherent summaries
            - Identify emergent patterns
            - Formulate actionable conclusions
            - Prioritize tool recommendations
            """,
            model=agent_model,
        )

        team = AsyncTeam(
            name="PrimaryThinkingTeam",
            members=[planner, researcher, analyzer, critic, synthesizer],
            instructions=await self._generate_adaptive_coordinator_instructions(),
            model=team_model,  # Add model to team
        )

        # Run the team to ensure it's properly initialized
        try:
            initialization_response = await team.arun("Team initialization check")
            logger.info(
                f"Primary team initialization successful: {type(initialization_response)}"
            )
        except Exception as e:
            logger.error(f"Primary team initialization failed: {e}")
            raise

        return team

    async def _create_reflection_team(self, team_model, agent_model) -> AsyncTeam:
        """Create the reflection team for meta-analysis."""
        meta_analyzer = Agent(
            name="MetaAnalyzer",
            role="Thinking Process Analyst",
            instructions="""You are the Meta-Analyzer in the reflection team.
            Your responsibilities:
            - Analyze the thinking process itself
            - Identify cognitive biases or blind spots
            - Evaluate the quality of reasoning
            - Suggest process improvements
            - Assess tool selection effectiveness
            """,
            model=agent_model,
        )

        pattern_recognizer = Agent(
            name="PatternRecognizer",
            role="Pattern Detection Specialist",
            instructions="""You are the Pattern Recognition Specialist.
            Your responsibilities:
            - Identify recurring patterns in thought sequences
            - Detect successful problem-solving strategies
            - Recognize inefficient thinking loops
            - Suggest pattern-based optimizations
            - Evaluate tool usage patterns
            """,
            model=agent_model,
        )

        quality_assessor = Agent(
            name="QualityAssessor",
            role="Quality Evaluator",
            instructions="""You are the Quality Assessment Specialist.
            Your responsibilities:
            - Evaluate response completeness and accuracy
            - Assess clarity and coherence of thinking
            - Check alignment with original objectives
            - Rate confidence in conclusions
            - Validate tool recommendation quality
            """,
            model=agent_model,
        )

        decision_critic = Agent(
            name="DecisionCritic",
            role="Decision Process Analyst",
            instructions="""You are the Decision Process Critic.
            Your responsibilities:
            - Review decision-making logic
            - Evaluate tool selection decisions
            - Assess risk-benefit trade-offs
            - Identify decision-making biases
            - Suggest decision improvements
            """,
            model=agent_model,
        )

        team = AsyncTeam(
            name="ReflectionTeam",
            members=[
                meta_analyzer,
                pattern_recognizer,
                quality_assessor,
                decision_critic,
            ],
            instructions=[
                "🔍 **REFLECTION TEAM COORDINATOR**",
                "You lead a team that provides meta-analysis of the thinking process.",
                "",
                "**Your Mission:**",
                "- Analyze HOW the thinking is being done, not just WHAT",
                "- Identify strengths and weaknesses in the approach",
                "- Suggest improvements for future thinking steps",
                "- Ensure quality and completeness of analysis",
                "- Evaluate tool selection and usage effectiveness",
                "",
                "Synthesize your team's insights into actionable feedback.",
            ],
            model=team_model,  # Add model to team
        )

        # Run the team to ensure it's properly initialized
        try:
            initialization_response = await team.arun(
                "Reflection team initialization check"
            )
            logger.info(
                f"Reflection team initialization successful: {type(initialization_response)}"
            )
        except Exception as e:
            logger.error(f"Reflection team initialization failed: {e}")
            raise

        return team

    async def add_thought(self, thought_data: ThoughtData):
        """Add a thought to the context and update tracking."""
        self.thought_history.append(thought_data)

        # Update shared context
        await self.shared_context.update_from_thought(thought_data)

        # Track branches
        if thought_data.branchFromThought and thought_data.branchId:
            if thought_data.branchId not in self.branches:
                self.branches[thought_data.branchId] = []
            self.branches[thought_data.branchId].append(thought_data)

        # Update context if topic/domain changes
        if thought_data.topic and thought_data.topic != self.current_topic:
            self.current_topic = thought_data.topic

        if thought_data.domain != self.current_domain:
            self.current_domain = thought_data.domain

    async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
        """Get context relevant to the current thought."""
        return await self.shared_context.get_relevant_context(thought)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the session."""
        perf_summary = await self.shared_context.get_performance_summary()
        error_summary = self.error_handler.get_error_summary()

        return {
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_thoughts": len(self.thought_history),
            "total_branches": len(self.branches),
            "performance": perf_summary,
            "errors": error_summary,
            "memory_usage": self.shared_context.get_memory_usage(),
        }

    def update_available_tools(self, tools: List[str]):
        """Update the list of available MCP tools."""
        self.available_mcp_tools = tools
        self.available_tools = tools
        logger.info(f"Updated available tools: {tools}")

    async def cleanup(self):
        """Cleanup resources."""
        await self.shared_context.clear()
        logger.info("Enhanced app context cleaned up")
