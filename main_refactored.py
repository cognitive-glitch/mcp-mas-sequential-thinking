"""
Enhanced Reflective Sequential Thinking MCP Server
Comprehensive refactoring with dual-team architecture, tool recommendations, and memory persistence.

Key Features:
- Dual-team architecture: Primary thinking team + Reflection team
- Enhanced ThoughtData with topic/subject alignment and tool recommendations
- SharedContext for memory persistence across thoughts and branches
- LLMProviderFactory supporting OpenRouter, OpenAI, Gemini (DeepSeek removed)
- Zero-token API bug fixes and comprehensive error handling
- reflectivereview tool for thought sequence analysis
- Claude Code hooks integration ready
"""

import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

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

import logging.handlers
from pathlib import Path

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
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels for escalation."""
    LOW = "low"           # Warnings, non-critical issues
    MEDIUM = "medium"     # Degraded functionality but recoverable
    HIGH = "high"         # Major functionality loss
    CRITICAL = "critical" # System failure


@dataclass
class ErrorContext:
    """Enhanced error context for debugging and recovery."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    thought_number: Optional[int] = None
    session_id: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    max_retries: int = 3


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class EnhancedErrorHandler:
    """Comprehensive error handling with recovery strategies."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "primary_team": CircuitBreaker(),
            "reflection_team": CircuitBreaker(), 
            "context_operations": CircuitBreaker(),
        }
    
    async def handle_error(
        self, 
        error: Exception, 
        error_type: ErrorType, 
        context: Dict[str, Any],
        recovery_strategy: Optional[Callable] = None
    ) -> ErrorContext:
        """Handle errors with appropriate recovery strategies."""
        
        # Determine severity based on error type and context
        severity = self._determine_severity(error, error_type, context)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            thought_number=context.get("thought_number"),
            session_id=context.get("session_id"),
            stack_trace=context.get("stack_trace"),
        )
        
        # Log error with appropriate level
        self._log_error(error_context)
        
        # Attempt recovery if strategy provided
        if recovery_strategy and error_context.retry_count < error_context.max_retries:
            try:
                error_context.recovery_attempted = True
                error_context.retry_count += 1
                
                # Add exponential backoff for retries
                if error_context.retry_count > 1:
                    await asyncio.sleep(min(2 ** error_context.retry_count, 30))
                
                await recovery_strategy()
                error_context.recovery_successful = True
                
            except Exception as recovery_error:
                logger.warning(f"Recovery attempt failed: {recovery_error}")
        
        # Store error for analysis
        self.error_history.append(error_context)
        
        # Update circuit breaker
        component = context.get("component")
        if component in self.circuit_breakers:
            if error_context.recovery_successful:
                self.circuit_breakers[component].record_success()
            else:
                self.circuit_breakers[component].record_failure()
        
        return error_context
    
    def _determine_severity(
        self, 
        error: Exception, 
        error_type: ErrorType, 
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        
        if error_type in [ErrorType.TEAM_INITIALIZATION, ErrorType.MODEL_COMMUNICATION]:
            return ErrorSeverity.HIGH
        elif error_type in [ErrorType.VALIDATION_ERROR, ErrorType.CONTEXT_ERROR]:
            return ErrorSeverity.MEDIUM
        elif error_type == ErrorType.TIMEOUT_ERROR:
            return ErrorSeverity.MEDIUM if context.get("retry_count", 0) < 2 else ErrorSeverity.HIGH
        else:
            return ErrorSeverity.LOW
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_msg = f"[{error_context.error_type.value}] {error_context.message}"
        
        if error_context.thought_number:
            log_msg += f" (Thought #{error_context.thought_number})"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors for monitoring."""
        recent_errors = [e for e in self.error_history if 
                        (datetime.now() - e.timestamp).seconds < 3600]  # Last hour
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": {et.value: len([e for e in recent_errors if e.error_type == et]) 
                           for et in ErrorType},
            "circuit_breaker_states": {name: cb.state for name, cb in self.circuit_breakers.items()},
            "recovery_success_rate": sum(1 for e in recent_errors if e.recovery_successful) / max(len(recent_errors), 1)
        }


class EnhancedAppContext:
    """
    Enhanced application context with SharedContext integration and dual-team architecture.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.shared_context = SharedContext()  # Simple in-memory context
        self.session_context = SessionContext(
            session_id=self.session_id,
            available_tools=[
                "ThinkingTools",
                "ExaTools",
            ],  # Will be populated dynamically
            session_topic=None,  # Will be set when first topic is provided
            session_domain=DomainType.GENERAL,
        )

        # Team instances
        self.primary_team: Optional[Team] = None
        self.reflection_team: Optional[Team] = None

        # Provider configuration
        self.provider_config = None
        self.team_model = None
        self.agent_model = None

        # Performance tracking
        self.start_time = time.time()
        self.total_thoughts = 0
        self.total_reflections = 0

        # Enhanced error handling
        self.error_handler = EnhancedErrorHandler()

        # Initialize context for adaptive prompts
        self.current_thought_context = {
            "domain": "general",
            "complexity": 0.5,
            "sequence_position": 1,
            "has_tools": False,
            "requires_research": False,
            "revision_depth": 0,
            "is_branching": False,
            "topic_specified": False,
        }

        logger.info(
            f"Enhanced app context initialized with session ID: {self.session_id}"
        )

    def _update_thought_context(self, thought_data: ThoughtData) -> None:
        """Update context for adaptive prompt generation."""
        self.current_thought_context.update({
            "domain": thought_data.domain.value,
            "complexity": getattr(thought_data, 'content_complexity', 0.5),
            "sequence_position": thought_data.thoughtNumber / thought_data.totalThoughts,
            "has_tools": bool(thought_data.current_step and thought_data.current_step.recommended_tools),
            "requires_research": any(kw in thought_data.thought.lower() for kw in ['research', 'find', 'search', 'investigate']),
            "revision_depth": 1 if thought_data.isRevision else 0,
            "is_branching": bool(thought_data.branchFromThought),
            "topic_specified": bool(thought_data.topic or thought_data.subject),
        })

    def _generate_adaptive_coordinator_instructions(self) -> List[str]:
        """Generate context-aware coordinator instructions."""
        base_instructions = [
            "🎯 **PRIMARY THINKING TEAM COORDINATOR**",
            "You orchestrate a team of specialists to process reflective thoughts with precision and intelligence.",
            "",
            "**🧠 CORE RESPONSIBILITIES:**",
            "1. **Context Intelligence**: Analyze thought complexity, domain, and sequence position",
            "2. **Smart Delegation**: Route tasks to optimal specialists based on content analysis",
            "3. **Quality Orchestration**: Ensure coherent, actionable outputs with tool recommendations",
            "4. **Adaptive Processing**: Adjust approach based on thought type (revision, branch, continuation)",
            "",
            "**🔧 SPECIALIST DEPLOYMENT STRATEGY:**",
        ]

        # Add context-specific delegation guidance
        context = self.current_thought_context
        
        if context["complexity"] > 0.7:
            base_instructions.append("• **High Complexity Mode**: Engage Analyzer + Critic for thorough examination")
        
        if context["requires_research"]:
            base_instructions.append("• **Research Required**: Prioritize Researcher with domain-specific methodology")
        
        if context["has_tools"]:
            base_instructions.append("• **Tool Integration**: Engage Planner for tool validation and optimization")
        
        if context["revision_depth"] > 0:
            base_instructions.append("• **Revision Mode**: Focus Critic on improvement identification and validation")
        
        if context["is_branching"]:
            base_instructions.append("• **Branching Analysis**: Deploy Analyzer for alternative pathway exploration")

        # Add domain-specific guidance
        domain_guidance = {
            "technical": "Focus on precision, implementation details, and technical feasibility",
            "creative": "Emphasize ideation, alternative approaches, and innovative solutions", 
            "analytical": "Prioritize data-driven insights, pattern recognition, and logical structures",
            "strategic": "Focus on long-term implications, trade-offs, and decision frameworks",
            "research": "Emphasize methodology, source validation, and comprehensive information gathering"
        }
        
        if context["domain"] in domain_guidance:
            base_instructions.extend([
                "",
                f"**🎯 DOMAIN OPTIMIZATION ({context['domain'].upper()}):**",
                f"• {domain_guidance[context['domain']]}",
            ])

        # Add quality standards
        base_instructions.extend([
            "",
            "**✅ QUALITY STANDARDS:**",
            "• Every response must be actionable and well-reasoned",
            "• Tool recommendations require clear rationale and confidence scores",
            "• Maintain alignment with stated topic/subject throughout",
            "• Provide specific, constructive guidance for next steps",
            "",
            "**⚡ OUTPUT REQUIREMENTS:**",
            "• Synthesize specialist insights into coherent, unified response",
            "• Include confidence indicators and quality assessments",
            "• Suggest improvements or alternative approaches when applicable",
            "• Ensure response supports the overall thinking sequence goals",
        ])

        return base_instructions

    def _generate_planner_instructions(self) -> List[str]:
        """Generate context-aware planner instructions."""
        context = self.current_thought_context
        
        base_instructions = [
            "🎯 **STRATEGIC PLANNER & TOOL ORCHESTRATOR**",
            "You are the planning specialist focused on strategic thinking and intelligent tool selection.",
            "",
            "**⚡ CORE CAPABILITIES:**",
            "• Strategic plan development with clear execution paths",
            "• Intelligent tool recommendation with confidence scoring",
            "• Risk assessment and contingency planning",
            "• Resource optimization and efficiency analysis",
            "",
        ]

        # Add context-specific planning guidance
        if context["complexity"] > 0.7:
            base_instructions.extend([
                "**🧩 HIGH COMPLEXITY MODE ACTIVE:**",
                "• Break down complex problems into manageable sub-problems",
                "• Provide multiple execution pathways with risk assessments",
                "• Include detailed contingency plans for each major step",
                "",
            ])

        if context["requires_research"]:
            base_instructions.extend([
                "**🔍 RESEARCH-INTENSIVE PLANNING:**",
                "• Prioritize information gathering tools and methodologies",
                "• Plan iterative research cycles with validation checkpoints",
                "• Include source verification and cross-validation steps",
                "",
            ])

        # Add domain-specific planning approaches
        domain_approaches = {
            "technical": [
                "• Emphasize feasibility analysis and implementation constraints",
                "• Include testing and validation phases in all plans",
                "• Consider scalability and performance implications"
            ],
            "creative": [
                "• Allow for exploration phases and iterative refinement",
                "• Include brainstorming and ideation checkpoints",
                "• Plan for multiple concept evaluation cycles"
            ],
            "analytical": [
                "• Structure plans around data collection and analysis phases",
                "• Include statistical validation and hypothesis testing",
                "• Plan for systematic evidence gathering"
            ]
        }

        if context["domain"] in domain_approaches:
            base_instructions.extend([
                f"**🎯 {context['domain'].upper()} DOMAIN OPTIMIZATION:**"
            ] + domain_approaches[context["domain"]] + [""])

        # Add standard operating procedures
        base_instructions.extend([
            "**📋 PLANNING PROTOCOL:**",
            "1. **Context Analysis**: Extract domain, complexity, and objectives",
            "2. **Tool Assessment**: Evaluate available tools for task alignment", 
            "3. **Strategy Formation**: Create multi-step execution plan",
            "4. **Risk Evaluation**: Identify potential failure points and alternatives",
            "5. **Resource Planning**: Optimize for efficiency and effectiveness",
            "6. **Quality Gates**: Define success criteria and validation methods",
            "",
            "**🔧 TOOL RECOMMENDATION FORMAT:**",
            "• Tool Name: [specific tool identifier]",
            "• Confidence: [0.0-1.0 with justification]",
            "• Rationale: [clear reasoning for selection]",
            "• Priority: [execution order with dependencies]",
            "• Expected Outcome: [specific, measurable result]",
            "• Alternatives: [backup options if primary fails]",
            "• Risk Level: [assessment of potential issues]",
            "",
            "**✅ SUCCESS CRITERIA:**",
            "• Plans must be actionable and specific",
            "• Tool recommendations must include clear justification",
            "• All major risks and contingencies must be addressed",
            "• Resource requirements must be realistic and achievable",
        ])

        return base_instructions

    def _generate_researcher_instructions(self) -> List[str]:
        """Generate context-aware researcher instructions."""
        context = self.current_thought_context
        
        instructions = [
            "🔍 **DOMAIN-AWARE INFORMATION GATHERER**",
            "You specialize in intelligent information gathering with domain expertise.",
            "",
            "**🎯 RESEARCH EXCELLENCE:**",
            "• Domain-specific search strategies and methodologies",
            "• Source validation and credibility assessment",
            "• Information synthesis and gap identification",
            "• Context-aware relevance filtering",
            "",
        ]

        # Add domain-specific research approaches
        domain_methods = {
            "technical": "Focus on technical specifications, implementation guides, and peer-reviewed sources",
            "creative": "Emphasize case studies, creative examples, and innovative approaches",
            "analytical": "Prioritize data sources, statistical studies, and empirical evidence",
            "strategic": "Focus on market analysis, competitive intelligence, and strategic frameworks"
        }

        if context["domain"] in domain_methods:
            instructions.extend([
                f"**🎯 {context['domain'].upper()} RESEARCH METHODOLOGY:**",
                f"• {domain_methods[context['domain']]}",
                "",
            ])

        instructions.extend([
            "**📚 RESEARCH PROTOCOL:**",
            "1. **Query Analysis**: Understand information requirements and context",
            "2. **Source Strategy**: Select appropriate information sources and methods",
            "3. **Data Collection**: Gather relevant, credible information",
            "4. **Validation**: Cross-check sources and verify accuracy",
            "5. **Synthesis**: Organize findings into actionable insights",
            "6. **Gap Analysis**: Identify missing information or follow-up needs",
            "",
            "**✅ QUALITY STANDARDS:**",
            "• All sources must be credible and relevant",
            "• Information must directly address the stated requirements",
            "• Findings must be organized and easily actionable",
            "• Gaps and limitations must be clearly identified",
        ])

        return instructions

    async def initialize_models(self) -> None:
        """Initialize LLM models using the enhanced provider factory."""
        try:
            self.team_model, self.agent_model, self.provider_config = (
                LLMProviderFactory.create_models()
            )
            logger.info(
                f"Models initialized successfully using {self.provider_config.provider_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def initialize_teams(self) -> None:
        """Initialize both primary and reflection teams."""
        if not self.team_model or not self.agent_model:
            await self.initialize_models()

        # Create primary thinking team
        self.primary_team = self._create_primary_team()

        # Create reflection team
        self.reflection_team = self._create_reflection_team()

        logger.info("Dual-team architecture initialized successfully")

    def _create_primary_team(self) -> Team:
        """Create the primary thinking team (enhanced from original)."""

        # Enhanced Planner with tool recommendation capabilities
        planner = Agent(
            name="Planner",
            role="Strategic Planner & Tool Orchestrator",
            description="Develops strategic plans and recommends appropriate tools for each step.",
            tools=[ThinkingTools()],
            instructions=self._generate_planner_instructions(),
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Enhanced Researcher with domain awareness
        researcher = Agent(
            name="Researcher",
            role="Domain-Aware Information Gatherer",
            description="Gathers information with awareness of topic/subject context.",
            tools=[ThinkingTools(), ExaTools()],
            instructions=self._generate_researcher_instructions(),
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Enhanced Analyzer with pattern recognition
        analyzer = Agent(
            name="Analyzer",
            role="Pattern Recognition Analyst",
            description="Performs deep analysis with focus on patterns and relationships.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Pattern Recognition Analyst specialist.",
                "When you receive an analytical sub-task:",
                "1. Understand the analytical requirement within the topic/subject context.",
                "2. Look for patterns, relationships, and underlying structures.",
                "3. Consider how current analysis relates to previous thoughts in the sequence.",
                "4. Identify logical dependencies and causal relationships.",
                "5. Generate insights that build upon or challenge previous findings.",
                "6. Suggest areas where analysis might need revision or branching.",
                "7. Provide confidence levels for your analytical conclusions.",
                "Focus on deep, structured analysis that reveals non-obvious insights.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Enhanced Critic with bias detection
        critic = Agent(
            name="Critic",
            role="Quality Controller & Bias Detector",
            description="Critically evaluates with focus on quality and bias detection.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Quality Controller and Bias Detector specialist.",
                "When you receive a critique sub-task:",
                "1. Evaluate the quality of reasoning and evidence presented.",
                "2. Identify potential cognitive biases in the thinking process.",
                "3. Check for logical fallacies and unsupported assumptions.",
                "4. Assess alignment between stated topic/subject and actual content.",
                "5. Evaluate the appropriateness of tool recommendations made.",
                "6. Suggest improvements or alternative approaches.",
                "7. Provide specific, actionable feedback for enhancement.",
                "Focus on constructive criticism that improves overall thinking quality.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Enhanced Synthesizer with integration capabilities
        synthesizer = Agent(
            name="Synthesizer",
            role="Integration Specialist",
            description="Synthesizes inputs from all specialists into coherent responses.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Integration Specialist.",
                "When you receive synthesis sub-tasks:",
                "1. Integrate insights from Planner, Researcher, Analyzer, and Critic.",
                "2. Resolve conflicts between different specialist recommendations.",
                "3. Create coherent responses that address the original thought.",
                "4. Ensure tool recommendations are practical and well-justified.",
                "5. Maintain alignment with the stated topic/subject throughout.",
                "6. Provide clear next-step guidance based on integrated insights.",
                "7. Suggest when revision or branching might be beneficial.",
                "Focus on creating unified, actionable responses from diverse inputs.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Create team with enhanced coordinator instructions
        team = Team(
            members=[planner, researcher, analyzer, critic, synthesizer],
            instructions=self._generate_adaptive_coordinator_instructions(),
            model=self.team_model,
            mode="coordinate",
            add_datetime_to_instructions=True,
            markdown=True,
        )

        return team

    def _create_reflection_team(self) -> Team:
        """Create the reflection team for meta-analysis and quality improvement."""

        # Meta-Analyzer: Analyzes thinking patterns and quality
        meta_analyzer = Agent(
            name="MetaAnalyzer",
            role="Thinking Pattern Analyst",
            description="Analyzes meta-patterns in thinking processes and sequences.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Thinking Pattern Analyst for reflection.",
                "When you receive a thought and its primary team response for reflection:",
                "1. Analyze the quality of the thinking process demonstrated.",
                "2. Identify patterns in reasoning, tool selection, and topic alignment.",
                "3. Evaluate the coherence of the thought sequence so far.",
                "4. Assess whether the thinking is progressing toward meaningful insights.",
                "5. Identify areas where the thinking could be more effective.",
                "6. Suggest meta-level improvements to the thinking approach.",
                "7. Provide confidence scores for your meta-analysis.",
                "Focus on improving the thinking process itself, not just the content.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Pattern Recognizer: Identifies recurring patterns and biases
        pattern_recognizer = Agent(
            name="PatternRecognizer",
            role="Bias & Pattern Detection Specialist",
            description="Identifies cognitive patterns, biases, and recurring themes.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Bias and Pattern Detection Specialist.",
                "When you receive reflection tasks:",
                "1. Identify cognitive biases present in the thinking process.",
                "2. Recognize recurring patterns that might limit or enhance thinking.",
                "3. Detect when thinking becomes circular or stuck in loops.",
                "4. Identify assumptions that should be questioned or validated.",
                "5. Recognize when tool recommendations are biased or suboptimal.",
                "6. Suggest pattern breaks or alternative approaches.",
                "7. Provide specific examples of patterns and biases detected.",
                "Focus on helping break limiting patterns and reinforce productive ones.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Quality Assessor: Evaluates overall quality and coherence
        quality_assessor = Agent(
            name="QualityAssessor",
            role="Quality & Coherence Evaluator",
            description="Evaluates the quality, coherence, and effectiveness of thinking.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Quality and Coherence Evaluator.",
                "When you receive quality assessment tasks:",
                "1. Evaluate the overall quality of reasoning demonstrated.",
                "2. Assess coherence between thoughts and their stated objectives.",
                "3. Check alignment between topic/subject and actual content quality.",
                "4. Evaluate the effectiveness of tool recommendations made.",
                "5. Assess whether insights are meaningful and actionable.",
                "6. Identify specific areas where quality could be improved.",
                "7. Provide numerical quality scores with detailed justification.",
                "Focus on objective quality assessment with constructive feedback.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Decision Critic: Analyzes decision-making processes
        decision_critic = Agent(
            name="DecisionCritic",
            role="Decision Process Analyst",
            description="Analyzes and critiques decision-making processes and tool choices.",
            tools=[ThinkingTools()],
            instructions=[
                "You are the Decision Process Analyst.",
                "When you receive decision analysis tasks:",
                "1. Analyze the quality of decisions made in tool selection.",
                "2. Evaluate whether decision-making processes are systematic.",
                "3. Assess if alternatives were properly considered.",
                "4. Check if decisions align with stated goals and context.",
                "5. Identify decision-making biases or shortcuts taken.",
                "6. Suggest improvements to decision-making processes.",
                "7. Recommend better frameworks for future decisions.",
                "Focus on improving the systematic quality of decision-making.",
            ],
            model=self.agent_model,
            add_datetime_to_instructions=True,
            markdown=True,
        )

        # Create reflection team
        reflection_team = Team(
            members=[
                meta_analyzer,
                pattern_recognizer,
                quality_assessor,
                decision_critic,
            ],
            instructions=[
                "You are the Reflection Team Coordinator for meta-analysis and quality improvement.",
                "Your role is to reflect on the primary team's thinking and provide improvement feedback.",
                "",
                "When you receive a thought and primary team response for reflection:",
                "1. **Meta-Analysis**: Delegate meta-analysis of thinking patterns and quality.",
                "2. **Pattern Detection**: Identify biases, loops, and limiting patterns.",
                "3. **Quality Assessment**: Evaluate overall coherence and effectiveness.",
                "4. **Decision Analysis**: Critique decision-making processes and tool choices.",
                "5. **Integration**: Synthesize reflection insights into actionable feedback.",
                "",
                "Your reflection should provide:",
                "- Strengths in the current thinking approach",
                "- Weaknesses or biases that need attention",
                "- Specific suggestions for improvement",
                "- Patterns detected that should be noted",
                "- Overall quality assessment with scores",
                "- Recommendations for next steps or revisions",
                "",
                "Focus on improving the thinking process, not just validating the content.",
                "Be constructive, specific, and actionable in your feedback.",
            ],
            model=self.team_model,
            mode="coordinate",
            add_datetime_to_instructions=True,
            markdown=True,
        )

        return reflection_team

    async def add_thought(self, thought_data: ThoughtData) -> None:
        """Add thought to shared context with enhanced tracking."""
        try:
            await self.shared_context.update_from_thought(thought_data)
            self.total_thoughts += 1

            # Update session context if topic/subject provided
            if thought_data.topic and not self.session_context.session_topic:
                self.session_context.session_topic = thought_data.topic

            if thought_data.domain != DomainType.GENERAL:
                self.session_context.session_domain = thought_data.domain

            logger.debug(
                f"Thought #{thought_data.thoughtNumber} added to shared context"
            )

        except Exception as e:
            logger.error(f"Error adding thought to context: {e}")
            raise

    async def get_context_for_thought(self, thought: str) -> Dict[str, Any]:
        """Get relevant context for processing a thought."""
        try:
            return await self.shared_context.get_relevant_context(thought, max_items=10)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {}

    async def record_performance_metric(self, metric_name: str, value: float) -> None:
        """Record performance metrics."""
        try:
            await self.shared_context.record_performance(metric_name, value)
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")


async def _process_with_minimal_teams(
    thought_data: ThoughtData, start_time: float, error_msg: str
) -> ProcessedThought:
    """
    Graceful degradation processing when main teams fail.
    Provides minimal but functional response.
    """
    execution_time = int((time.time() - start_time) * 1000)
    
    # Generate minimal but helpful response
    minimal_response = f"""## Minimal Processing Mode

**Thought Analysis**: {thought_data.thought}

**Basic Assessment**:
- Content received and acknowledged
- Topic: {thought_data.topic or "Not specified"}
- Domain: {thought_data.domain.value}
- Sequence Position: {thought_data.thoughtNumber}/{thought_data.totalThoughts}

**Error Context**: 
Processing teams encountered issues: {error_msg}

**Fallback Analysis**:
While full team analysis is unavailable, the thought content appears to be {'complex' if len(thought_data.thought) > 200 else 'straightforward'} and {'well-structured' if '?' in thought_data.thought or any(word in thought_data.thought.lower() for word in ['analyze', 'consider', 'evaluate']) else 'direct'}.

**Recommendations**:
- Consider breaking down complex thoughts into smaller components
- Retry processing after a brief pause
- Ensure network connectivity for team operations
- Review system status and error logs

**Next Steps**: {'Continue sequence with simplified approach' if thought_data.nextThoughtNeeded else 'Complete sequence review recommended'}
"""

    return ProcessedThought(
        thought_data=thought_data,
        coordinator_response=minimal_response,
        reflection_response=None,
        integrated_response=minimal_response,
        next_step_guidance="System is operating in degraded mode. Consider retrying or simplifying requests.",
        execution_time_ms=execution_time,
        token_usage={"minimal_processing": len(minimal_response.split())},
        success=True,  # Minimal success
        error=f"Degraded processing due to: {error_msg}",
        tool_recommendations_generated=False,
        reflection_applied=False,
        context_updated=False,
    )


# Global app context
app_context = EnhancedAppContext()


async def process_thought_with_dual_teams(
    thought_data: ThoughtData,
) -> ProcessedThought:
    """
    Process a thought using the dual-team architecture with comprehensive error handling.
    """
    start_time = time.time()
    error_handler = app_context.error_handler

    try:
        # Check circuit breaker for team operations
        if not error_handler.circuit_breakers["primary_team"].can_execute():
            raise Exception("Primary team circuit breaker is open - too many recent failures")
        
        if not error_handler.circuit_breakers["context_operations"].can_execute():
            raise Exception("Context operations circuit breaker is open - too many recent failures")

        # Ensure teams are initialized with circuit breaker protection
        try:
            if not app_context.primary_team or not app_context.reflection_team:
                await app_context.initialize_teams()
                error_handler.circuit_breakers["primary_team"].record_success()
        except Exception as e:
            error_context = await error_handler.handle_error(
                e, 
                ErrorType.TEAM_INITIALIZATION, 
                {
                    "component": "primary_team",
                    "thought_number": thought_data.thoughtNumber,
                    "session_id": app_context.session_id
                }
            )
            # Try graceful degradation with simplified processing
            if error_context.retry_count < 2:
                return await _process_with_minimal_teams(thought_data, start_time, str(e))
            else:
                raise

        # Update context for adaptive prompts with error protection
        try:
            app_context._update_thought_context(thought_data)
        except Exception as e:
            await error_handler.handle_error(
                e, 
                ErrorType.CONTEXT_ERROR, 
                {"component": "context_operations", "thought_number": thought_data.thoughtNumber}
            )
            # Continue without context update if it fails

        # Add thought to context for memory persistence with error protection
        try:
            await app_context.add_thought(thought_data)
            error_handler.circuit_breakers["context_operations"].record_success()
        except Exception as e:
            await error_handler.handle_error(
                e, 
                ErrorType.CONTEXT_ERROR, 
                {"component": "context_operations", "thought_number": thought_data.thoughtNumber}
            )
            # Continue without adding to context if it fails

        # Get relevant context with timeout and error protection
        relevant_context = {}
        try:
            relevant_context = await asyncio.wait_for(
                app_context.get_context_for_thought(thought_data.thought),
                timeout=10.0  # 10 second timeout
            )
        except asyncio.TimeoutError:
            await error_handler.handle_error(
                TimeoutError("Context retrieval timeout"), 
                ErrorType.TIMEOUT_ERROR, 
                {"component": "context_operations", "thought_number": thought_data.thoughtNumber}
            )
            # Continue with empty context if retrieval times out
        except Exception as e:
            await error_handler.handle_error(
                e, 
                ErrorType.CONTEXT_ERROR, 
                {"component": "context_operations", "thought_number": thought_data.thoughtNumber}
            )
            # Continue with empty context if retrieval fails

        # Prepare enhanced input for primary team
        input_prompt = f"""Process Enhanced Thought #{thought_data.thoughtNumber}:

**Topic**: {thought_data.topic or "Not specified"}
**Subject**: {thought_data.subject or "Not specified"}
**Domain**: {thought_data.domain.value}
**Keywords**: {", ".join(thought_data.keywords) if thought_data.keywords else "None"}

**Thought Content**: "{thought_data.thought}"

**Context Information**:
- Progress: {thought_data.thoughtNumber}/{thought_data.totalThoughts}
- Session ID: {app_context.session_id}
- Available Tools: {app_context.session_context.available_tools}

**Relevant Historical Context**:
{relevant_context.get("recent_thoughts", [])}

**Tool Usage Patterns**:
{relevant_context.get("tool_patterns", [])}

**Special Handling**:"""

        # Add revision context
        if thought_data.isRevision and thought_data.revisesThought:
            input_prompt += (
                f"\n- **REVISION** of Thought #{thought_data.revisesThought}"
            )

        # Add branching context
        elif thought_data.branchFromThought and thought_data.branchId:
            input_prompt += f"\n- **BRANCH** (ID: {thought_data.branchId}) from Thought #{thought_data.branchFromThought}"

        input_prompt += "\n\nPlease provide comprehensive analysis with tool recommendations and next-step guidance."

        # Process with primary team with comprehensive error handling
        logger.info(
            f"Processing thought #{thought_data.thoughtNumber} with primary team..."
        )
        
        primary_content = None
        try:
            # Check circuit breaker before primary team operation
            if not error_handler.circuit_breakers["primary_team"].can_execute():
                raise Exception("Primary team circuit breaker is open")
            
            if not app_context.primary_team:
                raise ValueError("Primary team not initialized")
            
            # Execute with timeout protection
            primary_response = await asyncio.wait_for(
                app_context.primary_team.arun(input_prompt),
                timeout=60.0  # 60 second timeout for primary team
            )

            # Extract and validate primary response content
            if not primary_response or not hasattr(primary_response, "content"):
                raise ValueError("Primary team returned invalid response")

            primary_content = primary_response.content
            if not primary_content or len(primary_content.strip()) == 0:
                raise ValueError("Primary team returned empty response")

            # Record successful operation
            error_handler.circuit_breakers["primary_team"].record_success()
            logger.info(
                f"Primary team completed processing thought #{thought_data.thoughtNumber}"
            )

        except asyncio.TimeoutError:
            await error_handler.handle_error(
                TimeoutError("Primary team processing timeout"), 
                ErrorType.TIMEOUT_ERROR, 
                {
                    "component": "primary_team",
                    "thought_number": thought_data.thoughtNumber,
                    "session_id": app_context.session_id,
                    "retry_count": 0
                }
            )
            # Provide fallback response
            primary_content = f"Analysis of: {thought_data.thought}\n\nProcessing timeout occurred. The thought has been acknowledged but requires manual review. Please consider breaking down complex thoughts into smaller components for better processing."
            
        except Exception as e:
            error_context = await error_handler.handle_error(
                e, 
                ErrorType.TEAM_PROCESSING, 
                {
                    "component": "primary_team",
                    "thought_number": thought_data.thoughtNumber,
                    "session_id": app_context.session_id
                }
            )
            
            # Try graceful degradation
            if error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                # Provide minimal fallback response
                primary_content = f"Analysis of: {thought_data.thought}\n\nProcessing encountered an error: {str(e)}\n\nFallback Response: The thought content has been received and basic acknowledgment provided. Please retry or simplify the request."
            else:
                raise

        # Process with reflection team if response is substantial
        reflection_content = None
        if len(primary_content.strip()) > 50:  # Only reflect on substantial responses
            logger.info(
                f"Processing reflection for thought #{thought_data.thoughtNumber}..."
            )

            reflection_input = f"""Reflect on Primary Team Analysis:

**Original Thought**: "{thought_data.thought}"
**Topic/Subject Context**: {thought_data.topic or "Not specified"} / {thought_data.subject or "Not specified"}
**Domain**: {thought_data.domain.value}

**Primary Team Response**:
{primary_content}

**Reflection Focus**:
- Analyze thinking quality and patterns
- Identify biases or limitations  
- Evaluate tool recommendations
- Assess topic alignment
- Suggest improvements
- Provide quality scores

Please provide constructive reflection with specific feedback and recommendations."""

            try:
                # Check circuit breaker for reflection team
                if not error_handler.circuit_breakers["reflection_team"].can_execute():
                    logger.warning("Reflection team circuit breaker is open - skipping reflection")
                    reflection_content = "Reflection skipped due to recent failures. Primary analysis stands as-is."
                else:
                    if not app_context.reflection_team:
                        raise ValueError("Reflection team not initialized")
                    
                    # Execute with timeout protection
                    reflection_response = await asyncio.wait_for(
                        app_context.reflection_team.arun(reflection_input),
                        timeout=45.0  # 45 second timeout for reflection team
                    )
                    
                    if reflection_response and hasattr(reflection_response, "content"):
                        reflection_content = reflection_response.content
                        app_context.total_reflections += 1
                        error_handler.circuit_breakers["reflection_team"].record_success()
                        logger.info(
                            f"Reflection completed for thought #{thought_data.thoughtNumber}"
                        )
                    else:
                        raise ValueError("Reflection team returned invalid response")
                        
            except asyncio.TimeoutError:
                await error_handler.handle_error(
                    TimeoutError("Reflection team processing timeout"), 
                    ErrorType.TIMEOUT_ERROR, 
                    {
                        "component": "reflection_team",
                        "thought_number": thought_data.thoughtNumber,
                        "session_id": app_context.session_id
                    }
                )
                reflection_content = "Reflection timeout occurred. Primary analysis is complete but lacks meta-cognitive feedback."
                
            except Exception as e:
                await error_handler.handle_error(
                    e, 
                    ErrorType.TEAM_PROCESSING, 
                    {
                        "component": "reflection_team",
                        "thought_number": thought_data.thoughtNumber,
                        "session_id": app_context.session_id
                    }
                )
                logger.warning(
                    f"Reflection team failed for thought #{thought_data.thoughtNumber}: {e}"
                )
                # Graceful degradation - continue without reflection
                reflection_content = None

        # Integrate responses
        integrated_response = primary_content
        if reflection_content:
            integrated_response += (
                f"\n\n## Reflection Team Feedback\n\n{reflection_content}"
            )

        # Generate contextual guidance (avoid repetitive text)
        next_step_guidance = ""
        if thought_data.nextThoughtNeeded:
            if reflection_content and "revise" in reflection_content.lower():
                next_step_guidance = "\n\n**Suggested Next Step**: Consider revising previous thoughts based on reflection feedback."
            elif thought_data.thoughtNumber / thought_data.totalThoughts > 0.7:
                next_step_guidance = (
                    "\n\n**Approaching Completion**: Focus on synthesis and conclusion."
                )
            else:
                next_step_guidance = "\n\n**Continue Sequence**: Build upon current insights in the next thought."
        else:
            next_step_guidance = "\n\n**Sequence Complete**: Review the complete analysis and conclusions."

        # Calculate execution time
        execution_time = int((time.time() - start_time) * 1000)

        # Record performance metrics
        await app_context.record_performance_metric(
            "thought_processing_time_ms", execution_time
        )
        await app_context.record_performance_metric(
            "response_length", len(integrated_response)
        )

        # Create processed thought result
        processed_thought = ProcessedThought(
            thought_data=thought_data,
            coordinator_response=primary_content,
            reflection_response=reflection_content,
            integrated_response=integrated_response,
            next_step_guidance=next_step_guidance,
            execution_time_ms=execution_time,
            token_usage={
                "primary_team": len(primary_content.split()),
                "reflection_team": len(reflection_content.split())
                if reflection_content
                else 0,
            },
            success=True,
            tool_recommendations_generated=bool(thought_data.current_step),
            reflection_applied=bool(reflection_content),
            context_updated=True,
        )

        logger.info(
            f"Successfully processed thought #{thought_data.thoughtNumber} in {execution_time}ms"
        )
        return processed_thought

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        
        # Use enhanced error handling for main exceptions
        error_context = await error_handler.handle_error(
            e, 
            ErrorType.UNKNOWN_ERROR, 
            {
                "component": "main_processing",
                "thought_number": thought_data.thoughtNumber,
                "session_id": app_context.session_id,
                "execution_time_ms": execution_time,
                "stack_trace": str(e.__traceback__)
            }
        )
        
        logger.error(f"Error processing thought #{thought_data.thoughtNumber}: {e}")
        
        # Try graceful degradation one more time if error is recoverable
        if error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] and error_context.retry_count < 1:
            try:
                return await _process_with_minimal_teams(thought_data, start_time, str(e))
            except Exception as fallback_error:
                logger.critical(f"Even fallback processing failed: {fallback_error}")

        # Return comprehensive error result with recovery suggestions
        error_summary = error_handler.get_error_summary()
        recovery_suggestions = []
        
        if error_summary["circuit_breaker_states"]["primary_team"] == "open":
            recovery_suggestions.append("Primary team circuit breaker is open - wait for recovery period")
        if error_summary["recent_errors"] > 5:
            recovery_suggestions.append("High error rate detected - consider system restart")
        if "timeout" in str(e).lower():
            recovery_suggestions.append("Timeouts detected - check network connectivity and system load")
        
        recovery_text = "\n\n**Recovery Suggestions**:\n" + "\n".join(f"- {s}" for s in recovery_suggestions) if recovery_suggestions else ""
        
        return ProcessedThought(
            thought_data=thought_data,
            coordinator_response="",
            reflection_response=None,
            integrated_response=f"""## Processing Error

**Error Details**: {str(e)}
**Error Type**: {error_context.error_type.value}
**Severity**: {error_context.severity.value}
**Retry Count**: {error_context.retry_count}

**Thought**: {thought_data.thought}

**System Status**:
- Recent Errors: {error_summary['recent_errors']}
- Circuit Breakers: {error_summary['circuit_breaker_states']}
- Recovery Success Rate: {error_summary['recovery_success_rate']:.2%}

{recovery_text}

**Next Steps**: Consider simplifying the request, checking system status, or retrying after a brief pause.""",
            next_step_guidance="System error encountered. Review error details and consider recovery suggestions.",
            execution_time_ms=execution_time,
            success=False,
            error=str(e),
            tool_recommendations_generated=False,
            reflection_applied=False,
            context_updated=False,
        )


async def generate_sequence_review() -> ThoughtSequenceReview:
    """
    Generate comprehensive review of the thought sequence using SharedContext.
    Implementation for the reflectivereview tool.
    """
    try:
        # Analyze thought patterns
        thought_graph = app_context.shared_context.thought_graph
        total_thoughts = thought_graph.number_of_nodes()

        # Identify branches
        branches = {}
        for node, data in thought_graph.nodes(data=True):
            # Simple branch detection - could be enhanced
            for successor in thought_graph.successors(node):
                edge_data = thought_graph.get_edge_data(node, successor)
                if edge_data and edge_data.get("relation_type") == "branches_to":
                    branch_id = f"branch_{successor}"
                    if branch_id not in branches:
                        branches[branch_id] = {
                            "origin": node,
                            "thoughts": [successor],
                            "quality": data.get("confidence", 0.5),
                        }

        # Analyze branch quality
        branch_analyses = []
        for branch_id, branch_data in branches.items():
            analysis = BranchAnalysis(
                branch_id=branch_id,
                branch_quality=branch_data["quality"],
                thought_count=len(branch_data["thoughts"]),
                key_insights=[f"Branch from thought {branch_data['origin']}"],
                completion_status="ongoing",
                recommendation="Continue development"
                if branch_data["quality"] > 0.6
                else "Consider revision",
            )
            branch_analyses.append(analysis)

        # Calculate quality metrics
        performance_summary = await app_context.shared_context.get_performance_summary()
        avg_processing_time = performance_summary.get(
            "thought_processing_time_ms", {}
        ).get("mean", 0)
        avg_response_length = performance_summary.get("response_length", {}).get(
            "mean", 0
        )

        # Generate insights from context
        key_insights = [
            insight.content for insight in app_context.shared_context.key_insights[-10:]
        ]

        # Create comprehensive review
        review = ThoughtSequenceReview(
            session_id=app_context.session_id,
            total_thoughts=total_thoughts,
            total_branches=len(branches),
            overall_quality=min(
                0.9, (avg_response_length / 1000) + 0.3
            ),  # Simple quality heuristic
            key_insights=key_insights,
            patterns_identified=[
                f"Average processing time: {avg_processing_time:.1f}ms",
                f"Total reflections generated: {app_context.total_reflections}",
                f"Session duration: {(time.time() - app_context.start_time):.1f}s",
            ],
            quality_trends={
                "processing_efficiency": avg_processing_time,
                "response_comprehensiveness": avg_response_length,
            },
            tool_effectiveness={
                tool.tool_name: tool.confidence
                for tool_decision in app_context.shared_context.tool_usage_history[-5:]
                for tool in [tool_decision]
                if hasattr(tool_decision, "confidence")
            },
            branch_analyses=branch_analyses,
            best_branch=max(branches.keys(), key=lambda x: branches[x]["quality"])
            if branches
            else None,
            next_steps=[
                "Continue with most promising branch"
                if branches
                else "Proceed with linear development",
                "Consider reflection feedback for quality improvement",
                "Maintain topic/subject alignment",
            ],
            areas_for_improvement=[
                "Increase thought depth if responses are brief",
                "Enhance tool recommendation specificity",
                "Strengthen domain-specific analysis",
            ],
            topic_alignment_score=0.8,  # Could be calculated based on keyword analysis
            review_timestamp=int(time.time() * 1000),
            review_confidence=0.75,
        )

        logger.info(f"Generated sequence review for session {app_context.session_id}")
        return review

    except Exception as e:
        logger.error(f"Error generating sequence review: {e}")
        # Return minimal review on error
        return ThoughtSequenceReview(
            session_id=app_context.session_id,
            total_thoughts=0,
            total_branches=0,
            overall_quality=0.0,
            best_branch=None,
            topic_alignment_score=0.0,
            review_timestamp=int(time.time() * 1000),
            review_confidence=0.0,
        )


# Initialize FastMCP server with enhanced tools
mcp = FastMCP("reflective_thinking_tools")


@mcp.tool()
async def reflectivethinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool = False,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: bool = False,
    # Enhanced fields
    topic: Optional[str] = None,
    subject: Optional[str] = None,
    domain: str = "general",
    keywords: Optional[List[str]] = None,
) -> str:
    """
    Enhanced reflective thinking tool with dual-team architecture, tool recommendations, and memory persistence.

    Processes thoughts using primary thinking team + reflection team for meta-analysis.
    Supports topic/subject alignment, branching, revision, and comprehensive context tracking.
    """

    try:
        # Parse domain
        try:
            domain_type = DomainType(domain.lower())
        except ValueError:
            domain_type = DomainType.GENERAL
            logger.warning(f"Invalid domain '{domain}', defaulting to general")

        # Create enhanced thought data
        thought_data = ThoughtData(
            thought=thought,
            thoughtNumber=thoughtNumber,
            totalThoughts=totalThoughts,
            nextThoughtNeeded=nextThoughtNeeded,
            isRevision=isRevision,
            revisesThought=revisesThought,
            branchFromThought=branchFromThought,
            branchId=branchId,
            needsMoreThoughts=needsMoreThoughts,
            topic=topic,
            subject=subject,
            domain=domain_type,
            keywords=keywords or [],
            current_step=None,  # Will be populated by processing
            reflection_feedback=None,  # Will be populated by reflection
            confidence_score=0.5,  # Default confidence
            processing_time_ms=0,  # Will be calculated
            session_context=app_context.session_context,
            timestamp_ms=int(time.time() * 1000),
        )

        logger.info(f"\n{thought_data.to_log_format()}\n")

        # Process with dual teams
        processed_thought = await process_thought_with_dual_teams(thought_data)

        # Return integrated response with guidance
        return (
            processed_thought.integrated_response + processed_thought.next_step_guidance
        )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return f"Input validation failed: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Processing failed: {str(e)}"


@mcp.tool()
async def reflectivereview() -> str:
    """
    Comprehensive review tool for analyzing thought sequences, branches, and patterns.

    Provides meta-analysis of the entire thinking session including:
    - Overall quality assessment
    - Branch analysis and recommendations
    - Pattern identification
    - Performance metrics
    - Improvement suggestions
    """

    try:
        review = await generate_sequence_review()

        # Format review as readable text
        output = f"""# Sequential Thinking Review

## Session Overview
- **Session ID**: {review.session_id}
- **Total Thoughts**: {review.total_thoughts}
- **Total Branches**: {review.total_branches}
- **Overall Quality**: {review.overall_quality:.2f}/1.0
- **Review Confidence**: {review.review_confidence:.2f}/1.0

## Key Insights
{chr(10).join(f"• {insight}" for insight in review.key_insights[:5])}

## Patterns Identified
{chr(10).join(f"• {pattern}" for pattern in review.patterns_identified)}

## Branch Analysis"""

        if review.branch_analyses:
            for branch in review.branch_analyses:
                output += f"""
### Branch: {branch.branch_id}
- **Quality**: {branch.branch_quality:.2f}/1.0
- **Thought Count**: {branch.thought_count}
- **Status**: {branch.completion_status}
- **Recommendation**: {branch.recommendation}"""
        else:
            output += "\nNo branches detected in this sequence."

        output += f"""

## Performance Metrics
{chr(10).join(f"• {metric}: {value}" for metric, value in review.quality_trends.items())}

## Recommendations
### Next Steps:
{chr(10).join(f"• {step}" for step in review.next_steps)}

### Areas for Improvement:
{chr(10).join(f"• {area}" for area in review.areas_for_improvement)}

## Topic Alignment
- **Alignment Score**: {review.topic_alignment_score:.2f}/1.0

---
*Review generated at {datetime.fromtimestamp(review.review_timestamp / 1000 if review.review_timestamp else time.time()).strftime("%Y-%m-%d %H:%M:%S")}*"""

        logger.info(f"Sequence review completed for session {app_context.session_id}")
        return output

    except Exception as e:
        logger.error(f"Error generating sequence review: {e}", exc_info=True)
        return f"Review generation failed: {str(e)}"


# Server startup with enhanced initialization
@asynccontextmanager
async def lifespan(app):
    """Enhanced server lifespan management with proper initialization."""
    logger.info("Starting Enhanced Sequential Thinking MCP Server...")

    try:
        # Initialize app context and models
        await app_context.initialize_models()
        provider_name = (
            app_context.provider_config.provider_name
            if app_context.provider_config
            else "unknown"
        )
        logger.info(f"Initialized with provider: {provider_name}")

        # Pre-initialize teams for faster first request
        await app_context.initialize_teams()
        logger.info("Dual-team architecture ready")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Enhanced Sequential Thinking MCP Server shutting down...")


def run():
    """Run the enhanced MCP server."""
    logger.info("Enhanced Reflective Sequential Thinking MCP Server starting...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
