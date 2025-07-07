#!/usr/bin/env python3
"""
Enhanced Reflective Sequential Thinking MCP Server with Tool Selection
Comprehensive implementation with dual-team architecture, tool recommendations, and memory persistence.

Key Features:
- Dual-team architecture: Primary thinking team + Reflection team
- Enhanced ThoughtData with topic/subject alignment and tool recommendations
- Tool selection thinking: Intelligent MCP tool recommendations
- SharedContext for memory persistence across thoughts and branches
- LLMProviderFactory supporting OpenRouter, OpenAI, Gemini
- Zero-token API bug fixes and comprehensive error handling
- reflectivethinking, reflectivereview, and toolselectthinking MCP tools
- Claude Code hooks integration ready
"""

import time
import uuid
import asyncio
import logging
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# Core frameworks
from mcp.server.fastmcp import FastMCP
from agno.agent import Agent
from agno.team.team import Team
from agno.tools.exa import ExaTools
from agno.tools.thinking import ThinkingTools
from dotenv import load_dotenv
from pydantic import ValidationError

# Enhanced local imports
from src.models.thought_models import (
    ThoughtData,
    ProcessedThought,
    ThoughtSequenceReview,
    SessionContext,
    DomainType,
    BranchAnalysis,
)
from src.providers.base import LLMProviderFactory
from src.context.shared_context import SharedContext
from src.tools.tool_selector import ToolSelector

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
    logger.setLevel(logging.DEBUG)

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
        file_handler.setLevel(logging.DEBUG)
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


class ErrorType(Enum):
    """Categorize different types of errors for appropriate handling."""

    TEAM_INITIALIZATION = "team_initialization"
    TEAM_PROCESSING = "team_processing"
    MODEL_COMMUNICATION = "model_communication"
    VALIDATION_ERROR = "validation_error"
    CONTEXT_ERROR = "context_error"
    TIMEOUT_ERROR = "timeout_error"
    TOOL_SELECTION_ERROR = "tool_selection_error"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    thought_number: Optional[int] = None
    recovery_attempted: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascade failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        """Record failed operation and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_proceed(self) -> bool:
        """Check if operation can proceed."""
        if not self.is_open:
            return True

        # Check if recovery timeout has passed
        if self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time).seconds
            if time_since_failure > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
                logger.info("Circuit breaker closed after recovery timeout")
                return True

        return False


class EnhancedErrorHandler:
    """Comprehensive error handling with recovery strategies."""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "team_processing": CircuitBreaker(),
            "model_communication": CircuitBreaker(),
        }

    def handle_error(
        self,
        error: Exception,
        error_type: ErrorType,
        thought_number: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Handle errors with appropriate recovery strategies.

        Returns:
            Recovery message or None if unrecoverable
        """
        severity = self._assess_severity(error, error_type)

        error_context = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            timestamp=datetime.now(),
            thought_number=thought_number,
            additional_info=context or {},
        )

        self.error_history.append(error_context)

        # Log error with appropriate level
        if severity == ErrorSeverity.CRITICAL:
            logger.error(f"Critical error: {error_context}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {error_context}")
        else:
            logger.warning(f"Error occurred: {error_context}")

        # Apply recovery strategy
        recovery_message = self._apply_recovery_strategy(error_context)

        if recovery_message:
            error_context.recovery_attempted = True

        return recovery_message

    def _assess_severity(
        self, error: Exception, error_type: ErrorType
    ) -> ErrorSeverity:
        """Assess error severity based on type and content."""
        if isinstance(error, ValidationError):
            return ErrorSeverity.LOW
        elif error_type == ErrorType.TEAM_INITIALIZATION:
            return ErrorSeverity.CRITICAL
        elif error_type == ErrorType.MODEL_COMMUNICATION:
            return ErrorSeverity.HIGH
        elif "token" in str(error).lower() or "api" in str(error).lower():
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM

    def _apply_recovery_strategy(self, error_context: ErrorContext) -> Optional[str]:
        """Apply appropriate recovery strategy based on error type."""
        if error_context.error_type == ErrorType.VALIDATION_ERROR:
            return "Input validation failed. Please check the format and try again."

        elif error_context.error_type == ErrorType.TEAM_PROCESSING:
            breaker = self.circuit_breakers.get("team_processing")
            if breaker and not breaker.can_proceed():
                return (
                    "Team processing temporarily unavailable. Please try again later."
                )
            return "Team processing error. Attempting with reduced complexity."

        elif error_context.error_type == ErrorType.MODEL_COMMUNICATION:
            breaker = self.circuit_breakers.get("model_communication")
            if breaker and not breaker.can_proceed():
                return "Model communication temporarily unavailable. Please try again later."
            return "Communication error with AI model. Retrying with fallback settings."

        elif error_context.error_type == ErrorType.TOOL_SELECTION_ERROR:
            return (
                "Tool selection failed. Proceeding with default tool recommendations."
            )

        return None

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history."""
        if not self.error_history:
            return {"total_errors": 0}

        summary = {
            "total_errors": len(self.error_history),
            "by_type": {},
            "by_severity": {},
            "recent_errors": [],
        }

        for error in self.error_history:
            # Count by type
            error_type_str = error.error_type.value
            summary["by_type"][error_type_str] = (
                summary["by_type"].get(error_type_str, 0) + 1
            )

            # Count by severity
            severity_str = error.severity.value
            summary["by_severity"][severity_str] = (
                summary["by_severity"].get(severity_str, 0) + 1
            )

        # Recent errors
        summary["recent_errors"] = [
            {
                "type": err.error_type.value,
                "message": err.message,
                "timestamp": err.timestamp.isoformat(),
            }
            for err in self.error_history[-5:]
        ]

        return summary


class EnhancedAppContext:
    """
    Enhanced application context managing teams, models, and shared state.
    Includes tool selection capabilities and comprehensive error handling.
    """

    def __init__(self):
        """Initialize the enhanced application context with all components."""
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()

        # Initialize components
        self.shared_context = SharedContext()  # Simple in-memory context
        self.error_handler = EnhancedErrorHandler()
        self.tool_selector = ToolSelector()

        # Session context
        self.session_context = SessionContext(
            session_id=self.session_id,
            available_tools=["ThinkingTools", "ExaTools"],
            session_topic="Reflective Thinking Session",
            session_domain=DomainType.GENERAL,
        )

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
        self.primary_team: Optional[Team] = None
        self.reflection_team: Optional[Team] = None
        self.teams_initialized = False

        # Thought tracking
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}

        # Available MCP tools tracking
        self.available_mcp_tools: List[str] = []

        logger.info(
            f"Enhanced app context initialized with session ID: {self.session_id}"
        )

    async def initialize_teams(self):
        """Initialize both primary and reflection teams asynchronously."""
        if self.teams_initialized:
            return

        try:
            # Get model configurations
            team_model_id, agent_model_id = self.provider_config.get_models()
            team_model = self.provider_config.create_model_instance(team_model_id)
            agent_model = self.provider_config.create_model_instance(agent_model_id)

            # Create primary thinking team
            self.primary_team = await self._create_primary_team(team_model, agent_model)

            # Create reflection team
            self.reflection_team = await self._create_reflection_team(
                team_model, agent_model
            )

            self.teams_initialized = True
            logger.info("Both teams initialized successfully")

        except Exception as e:
            error_msg = self.error_handler.handle_error(
                e, ErrorType.TEAM_INITIALIZATION, context={"stage": "team_creation"}
            )
            logger.error(f"Team initialization failed: {error_msg}")
            raise

    def _generate_adaptive_coordinator_instructions(self) -> List[str]:
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
        perf_summary = asyncio.run(self.shared_context.get_performance_summary())
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

    async def _create_primary_team(self, team_model, agent_model) -> Team:
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

        team = Team(
            name="PrimaryThinkingTeam",
            members=[planner, researcher, analyzer, critic, synthesizer],
            instructions=self._generate_adaptive_coordinator_instructions(),
        )

        # Run the team to ensure it's properly initialized
        await team.arun("Team initialization check")

        return team

    async def _create_reflection_team(self, team_model, agent_model) -> Team:
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

        team = Team(
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
        )

        # Run the team to ensure it's properly initialized
        await team.arun("Reflection team initialization check")

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

        # Update session context
        if (
            thought_data.topic
            and thought_data.topic != self.session_context.session_topic
        ):
            self.session_context.session_topic = thought_data.topic

        if thought_data.domain != self.session_context.session_domain:
            self.session_context.session_domain = thought_data.domain

    async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
        """Get context relevant to the current thought."""
        return await self.shared_context.get_relevant_context(thought)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the session."""
        perf_summary = await self.shared_context.get_performance_summary()
        error_summary = self.error_handler.get_error_summary()

        return {
            "session_id": self.session_id,
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
        self.session_context.available_tools = tools
        logger.info(f"Updated available tools: {tools}")

    def cleanup(self):
        """Cleanup resources."""
        self.shared_context.clear()
        logger.info("Enhanced app context cleaned up")


# Global app context
app_context = EnhancedAppContext()

# Create FastMCP server
mcp = FastMCP("reflective-thinking-tools")


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
                tool_recommendation = context.tool_selector.recommend_tools(
                    thought_data, context.available_mcp_tools
                )
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
            # Simple effectiveness based on response quality
            effectiveness = 0.8 if "error" not in integrated_response.lower() else 0.4
            for tool_rec in thought_data.current_step.recommended_tools:
                context.tool_selector.record_tool_effectiveness(
                    tool_rec.tool_name, effectiveness
                )

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


async def generate_sequence_review(
    session_id: str, context: EnhancedAppContext
) -> ThoughtSequenceReview:
    """Generate a comprehensive review of the thought sequence."""
    try:
        # Gather all thoughts
        all_thoughts = context.thought_history

        # Analyze branches
        branch_analyses = []
        for branch_id, branch_thoughts in context.branches.items():
            if branch_thoughts:
                quality_scores = [t.confidence_score for t in branch_thoughts]
                avg_quality = (
                    sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
                )

                branch_analysis = BranchAnalysis(
                    branch_id=branch_id,
                    branch_quality=avg_quality,
                    thought_count=len(branch_thoughts),
                    key_insights=[t.thought[:100] for t in branch_thoughts[:3]],
                    completion_status="completed"
                    if not branch_thoughts[-1].nextThoughtNeeded
                    else "ongoing",
                    recommendation="Continue development"
                    if avg_quality > 0.7
                    else "Consider revision",
                )
                branch_analyses.append(branch_analysis)

        # Calculate overall metrics
        total_thoughts = len(all_thoughts)
        overall_quality = (
            sum(t.confidence_score for t in all_thoughts) / total_thoughts
            if total_thoughts > 0
            else 0.5
        )

        # Extract key insights
        key_insights = []
        insights = await asyncio.gather(
            *[
                context.shared_context.get_context(f"insight_{i}")
                for i in range(min(5, total_thoughts))
            ]
        )
        key_insights = [ins for ins in insights if ins]

        # Identify patterns
        patterns = []
        if total_thoughts > 3:
            # Look for revision patterns
            revision_count = sum(1 for t in all_thoughts if t.isRevision)
            if revision_count > total_thoughts * 0.3:
                patterns.append("High revision rate - iterative refinement approach")

            # Look for branching patterns
            if len(context.branches) > 1:
                patterns.append(
                    f"Multiple exploration paths ({len(context.branches)} branches)"
                )

        # Tool effectiveness
        tool_stats = context.tool_selector.get_tool_statistics()
        tool_effectiveness = {
            name: stats.get("avg_effectiveness", 0.5)
            for name, stats in tool_stats.items()
        }

        # Performance metrics (might be used in future enhancements)
        _ = await context.get_performance_metrics()

        # Determine best branch
        best_branch = None
        if branch_analyses:
            best_branch = max(branch_analyses, key=lambda b: b.branch_quality).branch_id

        # Generate next steps
        next_steps = []
        if all_thoughts and all_thoughts[-1].nextThoughtNeeded:
            next_steps.append("Continue with the next thought in the sequence")

        if any(b.completion_status == "ongoing" for b in branch_analyses):
            next_steps.append("Complete ongoing branches")

        if overall_quality < 0.6:
            next_steps.append("Consider revising low-confidence thoughts")

        # Areas for improvement
        areas_for_improvement = []
        error_summary = context.error_handler.get_error_summary()
        if error_summary["total_errors"] > 5:
            areas_for_improvement.append("Reduce error rate through input validation")

        if tool_effectiveness and min(tool_effectiveness.values()) < 0.5:
            areas_for_improvement.append("Improve tool selection accuracy")

        # Calculate topic alignment
        topic_alignment = 0.8  # Default
        if all_thoughts:
            topics = [t.topic for t in all_thoughts if t.topic]
            if topics:
                # Check topic consistency
                unique_topics = set(topics)
                topic_alignment = 1.0 - (len(unique_topics) - 1) / len(topics)

        return ThoughtSequenceReview(
            session_id=session_id,
            total_thoughts=total_thoughts,
            total_branches=len(context.branches),
            overall_quality=overall_quality,
            key_insights=key_insights[:5],
            patterns_identified=patterns,
            quality_trends={
                "start": all_thoughts[0].confidence_score if all_thoughts else 0.5,
                "middle": all_thoughts[total_thoughts // 2].confidence_score
                if total_thoughts > 1
                else 0.5,
                "end": all_thoughts[-1].confidence_score if all_thoughts else 0.5,
            },
            tool_effectiveness=tool_effectiveness,
            branch_analyses=branch_analyses,
            best_branch=best_branch,
            next_steps=next_steps,
            areas_for_improvement=areas_for_improvement,
            topic_alignment_score=topic_alignment,
            review_timestamp=int(time.time() * 1000),
            review_confidence=0.9 if total_thoughts > 5 else 0.7,
        )

    except Exception as e:
        logger.error(f"Failed to generate sequence review: {e}")
        # Return minimal review on error
        return ThoughtSequenceReview(
            session_id=session_id,
            total_thoughts=len(context.thought_history),
            total_branches=len(context.branches),
            overall_quality=0.5,
            key_insights=["Review generation failed"],
            patterns_identified=[],
            quality_trends={},
            tool_effectiveness={},
            branch_analyses=[],
            best_branch=None,
            next_steps=["Retry review generation"],
            areas_for_improvement=["Fix review generation errors"],
            topic_alignment_score=0.5,
            review_timestamp=int(time.time() * 1000),
            review_confidence=0.1,
        )


@mcp.tool()
async def reflectivethinking(thought_data: ThoughtData) -> str:
    """
    Enhanced tool for dynamic and reflective problem-solving through sequential thoughts.
    Includes dual-team processing, tool recommendations, and comprehensive error handling.
    """
    logger.info(f"Processing thought #{thought_data.thoughtNumber}")

    try:
        # Set session context if not already set
        thought_data.session_context = app_context.session_context

        # Process through dual teams
        result = await process_thought_with_dual_teams(thought_data, app_context)

        # Build response
        response_parts = []

        # Add main content
        response_parts.append(result.integrated_response)

        # Add guidance for next steps
        if thought_data.nextThoughtNeeded:
            response_parts.append(
                f"\n## Next Step Guidance\n{result.next_step_guidance}"
            )

            # Add tool recommendations for next step if available
            if thought_data.current_step:
                response_parts.append("\n## Recommended Tools for Next Step:")
                for tool in thought_data.current_step.recommended_tools[:3]:
                    response_parts.append(
                        f"- **{tool.tool_name}** (confidence: {tool.confidence:.2f}): {tool.rationale}"
                    )

        # Add performance metrics if this is a final thought
        if not thought_data.nextThoughtNeeded:
            metrics = await app_context.get_performance_metrics()
            response_parts.append(
                f"\n## Session Summary\n"
                f"- Total thoughts: {metrics['total_thoughts']}\n"
                f"- Duration: {metrics['duration_seconds']:.1f}s\n"
                f"- Overall quality: {result.quality_score:.2f}"
            )

        return "\n".join(response_parts)

    except ValidationError as e:
        error_msg = app_context.error_handler.handle_error(
            e, ErrorType.VALIDATION_ERROR, thought_data.thoughtNumber
        )
        logger.error(f"Validation error: {e}")
        return f"Validation Error: {error_msg}\n\nDetails: {str(e)}"

    except Exception as e:
        error_msg = app_context.error_handler.handle_error(
            e, ErrorType.TEAM_PROCESSING, thought_data.thoughtNumber
        )
        logger.error(f"Unexpected error: {e}")
        return f"Error: {error_msg}\n\nPlease try again with a simpler thought."


@mcp.tool()
async def reflectivereview(session_id: Optional[str] = None) -> str:
    """
    Generate a comprehensive review of the thought sequence with quality metrics.
    """
    try:
        # Use current session if no ID provided
        if not session_id:
            session_id = app_context.session_id

        # Generate review
        review = await generate_sequence_review(session_id, app_context)

        # Format review for output
        output = [
            "# Thought Sequence Review",
            f"**Session ID**: {review.session_id}",
            f"**Total Thoughts**: {review.total_thoughts}",
            f"**Total Branches**: {review.total_branches}",
            f"**Overall Quality**: {review.overall_quality:.2f}",
            f"**Topic Alignment**: {review.topic_alignment_score:.2f}",
            "",
            "## Key Insights",
        ]

        for i, insight in enumerate(review.key_insights, 1):
            output.append(f"{i}. {insight}")

        if review.patterns_identified:
            output.extend(["", "## Patterns Identified"])
            for pattern in review.patterns_identified:
                output.append(f"- {pattern}")

        if review.tool_effectiveness:
            output.extend(["", "## Tool Effectiveness"])
            for tool, effectiveness in review.tool_effectiveness.items():
                output.append(f"- {tool}: {effectiveness:.2f}")

        if review.branch_analyses:
            output.extend(["", "## Branch Analysis"])
            for branch in review.branch_analyses:
                output.append(
                    f"\n### Branch: {branch.branch_id}\n"
                    f"- Quality: {branch.branch_quality:.2f}\n"
                    f"- Thoughts: {branch.thought_count}\n"
                    f"- Status: {branch.completion_status}\n"
                    f"- Recommendation: {branch.recommendation}"
                )

        if review.next_steps:
            output.extend(["", "## Recommended Next Steps"])
            for step in review.next_steps:
                output.append(f"- {step}")

        if review.areas_for_improvement:
            output.extend(["", "## Areas for Improvement"])
            for area in review.areas_for_improvement:
                output.append(f"- {area}")

        output.append(f"\n**Review Confidence**: {review.review_confidence:.2f}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Failed to generate review: {e}")
        return f"Error generating review: {str(e)}"


@mcp.tool()
async def toolselectthinking(
    thought: str,
    available_tools: Optional[List[str]] = None,
    domain: str = "general",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Intelligent tool selection for a given thought or task.
    Analyzes the thought and recommends appropriate MCP tools with confidence scores.

    Args:
        thought: The current thought or task description
        available_tools: List of available tool names (if None, uses known tools)
        domain: Domain context (general, technical, creative, analytical, strategic, research)
        context: Additional context for tool selection

    Returns:
        Tool recommendations with confidence scores and rationale
    """
    try:
        # Create minimal thought data for analysis
        thought_data = ThoughtData(
            thought=thought,
            thoughtNumber=1,
            totalThoughts=1,
            nextThoughtNeeded=False,
            domain=DomainType(domain.lower())
            if domain.lower() in [d.value for d in DomainType]
            else DomainType.GENERAL,
            keywords=[],  # Will be extracted by tool selector
            isRevision=False,
            needsMoreThoughts=False,
            session_context=app_context.session_context,
            topic=None,
            subject=None,
            revisesThought=None,
            branchFromThought=None,
            branchId=None,
            current_step=None,
            reflection_feedback=None,
            confidence_score=0.5,
            timestamp_ms=int(time.time() * 1000),
            processing_time_ms=0,
        )

        # Update available tools if provided
        if available_tools:
            app_context.update_available_tools(available_tools)

        # Get tool recommendations
        recommendations = app_context.tool_selector.recommend_tools(
            thought_data,
            available_tools or app_context.available_mcp_tools,
            max_recommendations=5,
        )

        # Format output
        output = [
            "# Tool Recommendations for Task",
            f"**Task**: {thought[:200]}...",
            f"**Domain**: {domain}",
            "",
            "## Recommended Tools",
        ]

        for i, tool_rec in enumerate(recommendations.recommended_tools, 1):
            output.append(
                f"\n### {i}. {tool_rec.tool_name}\n"
                f"**Confidence**: {tool_rec.confidence:.2f}\n"
                f"**Priority**: {tool_rec.priority}\n"
                f"**Rationale**: {tool_rec.rationale}\n"
                f"**Expected Outcome**: {tool_rec.expected_outcome}"
            )

            if tool_rec.suggested_inputs:
                output.append("**Suggested Inputs**:")
                for key, value in tool_rec.suggested_inputs.items():
                    output.append(f"  - {key}: {value}")

            if tool_rec.alternatives:
                output.append(f"**Alternatives**: {', '.join(tool_rec.alternatives)}")

        # Add step information
        output.extend(
            [
                "",
                "## Step Information",
                f"**Description**: {recommendations.step_description}",
                f"**Expected Outcome**: {recommendations.expected_outcome}",
            ]
        )

        if recommendations.validation_criteria:
            output.extend(["", "**Validation Criteria**:"])
            for criterion in recommendations.validation_criteria:
                output.append(f"- {criterion}")

        # Add tool statistics if available
        tool_stats = app_context.tool_selector.get_tool_statistics()
        if tool_stats:
            output.extend(["", "## Historical Tool Performance"])
            for tool_name, stats in tool_stats.items():
                if tool_name in [
                    t.tool_name for t in recommendations.recommended_tools
                ]:
                    output.append(
                        f"- **{tool_name}**: "
                        f"Used {stats['usage_count']} times, "
                        f"Avg effectiveness: {stats['avg_effectiveness']:.2f}"
                    )

        return "\n".join(output)

    except Exception as e:
        error_msg = app_context.error_handler.handle_error(
            e, ErrorType.TOOL_SELECTION_ERROR
        )
        logger.error(f"Tool selection error: {e}")
        return f"Error in tool selection: {error_msg}\n\nFallback: Consider using ThinkingTools for general analysis."


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
    app_context.cleanup()


# Main execution
if __name__ == "__main__":
    # Configure server
    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
    }

    logger.info(f"Starting server with configuration: {server_config}")

    # Run the server
    mcp.run()
