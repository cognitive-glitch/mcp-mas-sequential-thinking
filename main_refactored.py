"""
Enhanced Reflective Sequential Thinking MCP Server
Comprehensive refactoring with dual-team architecture, tool recommendations, and memory persistence.

Key Features:
- Dual-team architecture: Primary thinking team + Reflection team
- Enhanced ThoughtData with topic/subject alignment and tool recommendations
- SharedContext for memory persistence across thoughts and branches
- LLMProviderFactory supporting OpenRouter, OpenAI, Gemini (DeepSeek removed)
- Zero-token API bug fixes and comprehensive error handling
- sequentialreview tool for thought sequence analysis
- Claude Code hooks integration ready
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

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

import logging
import logging.handlers
from pathlib import Path

# Load environment variables
load_dotenv()


# Configure logging
def setup_logging() -> logging.Logger:
    """Enhanced logging setup with detailed format."""
    home_dir = Path.home()
    log_dir = home_dir / ".sequential_thinking" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sequential_thinking_enhanced")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "sequential_thinking.log",
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


class EnhancedAppContext:
    """
    Enhanced application context with SharedContext integration and dual-team architecture.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.shared_context = SharedContext()
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

        logger.info(
            f"Enhanced app context initialized with session ID: {self.session_id}"
        )

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
            instructions=[
                "You are the Strategic Planner and Tool Orchestrator specialist.",
                "When you receive a sub-task from the Team Coordinator:",
                "1. Understand the specific planning requirement and topic/subject context.",
                "2. Analyze the available tools and recommend the most appropriate ones.",
                "3. Create step-by-step plans with tool recommendations including:",
                "   - Tool name and confidence level (0-1)",
                "   - Rationale for each tool choice",
                "   - Priority order for tool execution",
                "   - Expected outcomes from each tool",
                "   - Alternative tools if primary choice fails",
                "4. Consider the problem domain (technical, creative, analytical, etc.).",
                "5. Identify potential revision/branching points in your plan.",
                "6. Return a structured response with clear tool recommendations.",
                "Focus on practical, executable plans with intelligent tool selection.",
            ],
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
            instructions=[
                "You are the Domain-Aware Information Gatherer specialist.",
                "When you receive a research sub-task:",
                "1. Analyze the topic, subject, and domain context provided.",
                "2. Tailor your search strategy to the specific domain (technical, creative, etc.).",
                "3. Use appropriate keywords and search terms based on the context.",
                "4. Validate information relevance to the stated topic/subject.",
                "5. Structure findings with domain-specific considerations.",
                "6. Note any domain-specific information gaps or limitations.",
                "7. Provide recommendations for follow-up research if needed.",
                "Focus on domain-appropriate research methodologies and validation.",
            ],
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
            instructions=[
                "You are the Primary Thinking Team Coordinator for reflective sequential thinking.",
                "Your role is to process thoughts with tool recommendations and topic alignment.",
                "",
                "When you receive a thought to process:",
                "1. **Context Analysis**: Extract topic, subject, domain, and keywords from the thought.",
                "2. **Task Delegation**: Delegate appropriate sub-tasks to your specialists:",
                "   - **Planner**: For strategic planning and tool recommendation",
                "   - **Researcher**: For information gathering with domain awareness",
                "   - **Analyzer**: For pattern recognition and deep analysis",
                "   - **Critic**: For quality control and bias detection",
                "   - **Synthesizer**: For integration and coherent response generation",
                "3. **Tool Orchestration**: Ensure tool recommendations are practical and justified.",
                "4. **Quality Assurance**: Verify alignment with stated topic/subject.",
                "5. **Response Integration**: Synthesize specialist inputs into a coherent response.",
                "",
                "Always consider:",
                "- The thought's position in the sequence (revision, branch, continuation)",
                "- Topic/subject alignment throughout the process",
                "- Tool recommendation quality and justification",
                "- Opportunities for improvement or alternative approaches",
                "",
                "Provide responses that are actionable, well-reasoned, and aligned with the thinking sequence goals.",
            ],
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


# Global app context
app_context = EnhancedAppContext()


async def process_thought_with_dual_teams(
    thought_data: ThoughtData,
) -> ProcessedThought:
    """
    Process a thought using the dual-team architecture with comprehensive error handling.
    """
    start_time = time.time()

    try:
        # Ensure teams are initialized
        if not app_context.primary_team or not app_context.reflection_team:
            await app_context.initialize_teams()

        # Add thought to context for memory persistence
        await app_context.add_thought(thought_data)

        # Get relevant context
        relevant_context = await app_context.get_context_for_thought(
            thought_data.thought
        )

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

        # Process with primary team
        logger.info(
            f"Processing thought #{thought_data.thoughtNumber} with primary team..."
        )
        if not app_context.primary_team:
            raise ValueError("Primary team not initialized")
        primary_response = await app_context.primary_team.arun(input_prompt)

        # Extract and validate primary response content
        if not primary_response or not hasattr(primary_response, "content"):
            raise ValueError("Primary team returned invalid response")

        primary_content = primary_response.content
        if not primary_content or len(primary_content.strip()) == 0:
            raise ValueError("Primary team returned empty response")

        logger.info(
            f"Primary team completed processing thought #{thought_data.thoughtNumber}"
        )

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
                if not app_context.reflection_team:
                    raise ValueError("Reflection team not initialized")
                reflection_response = await app_context.reflection_team.arun(
                    reflection_input
                )
                if reflection_response and hasattr(reflection_response, "content"):
                    reflection_content = reflection_response.content
                    app_context.total_reflections += 1
                    logger.info(
                        f"Reflection completed for thought #{thought_data.thoughtNumber}"
                    )
            except Exception as e:
                logger.warning(
                    f"Reflection team failed for thought #{thought_data.thoughtNumber}: {e}"
                )
                reflection_content = f"Reflection unavailable due to error: {str(e)}"

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
        logger.error(f"Error processing thought #{thought_data.thoughtNumber}: {e}")

        # Return error result
        return ProcessedThought(
            thought_data=thought_data,
            coordinator_response="",
            reflection_response=None,
            integrated_response=f"Processing failed: {str(e)}",
            next_step_guidance="Please try again or revise the thought.",
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
    Implementation for the sequentialreview tool.
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
mcp = FastMCP("enhanced_sequential_thinking")


@mcp.tool()
async def sequentialthinking(
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
    Enhanced sequential thinking tool with dual-team architecture, tool recommendations, and memory persistence.

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
async def sequentialreview() -> str:
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
    mcp.run()


if __name__ == "__main__":
    run()
