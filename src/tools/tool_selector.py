"""
Tool selection and recommendation engine for the reflective thinking MCP server.
Analyzes available MCP tools and recommends appropriate ones for each thinking step.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from models.thought_models import (
    ThoughtData,
    ToolRecommendation,
    StepRecommendation,
    DomainType,
)

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools for better organization and selection."""

    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATION = "creation"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    UTILITY = "utility"
    THINKING = "thinking"
    UNKNOWN = "unknown"


@dataclass
class ToolProfile:
    """Profile of an available MCP tool with metadata for selection."""

    name: str
    description: str
    category: ToolCategory
    keywords: List[str]
    typical_use_cases: List[str]
    confidence_baseline: float = 0.5  # Base confidence for this tool


class ToolSelector:
    """
    Intelligent tool selection engine that analyzes thinking steps
    and recommends appropriate MCP tools.
    """

    def __init__(self):
        self.tool_profiles: Dict[str, ToolProfile] = self._initialize_known_tools()
        self.usage_history: List[Tuple[str, float]] = []  # (tool_name, effectiveness)

    def _initialize_known_tools(self) -> Dict[str, ToolProfile]:
        """Initialize profiles for commonly available MCP tools."""
        return {
            # Thinking tools
            "ThinkingTools": ToolProfile(
                name="ThinkingTools",
                description="Deep thinking and reflection capabilities",
                category=ToolCategory.THINKING,
                keywords=["think", "reflect", "analyze", "consider", "ponder"],
                typical_use_cases=[
                    "Complex problem decomposition",
                    "Strategic planning",
                    "Deep analysis",
                ],
                confidence_baseline=0.7,
            ),
            # Research tools
            "ExaTools": ToolProfile(
                name="ExaTools",
                description="Advanced search and research capabilities",
                category=ToolCategory.RESEARCH,
                keywords=["search", "research", "find", "discover", "explore"],
                typical_use_cases=[
                    "Finding relevant information",
                    "Research tasks",
                    "Knowledge gathering",
                ],
                confidence_baseline=0.8,
            ),
            "tavily_search": ToolProfile(
                name="tavily_search",
                description="Web search for current information",
                category=ToolCategory.RESEARCH,
                keywords=["search", "web", "current", "latest", "news"],
                typical_use_cases=[
                    "Current events research",
                    "Latest information",
                    "Web content discovery",
                ],
                confidence_baseline=0.75,
            ),
            # File and code tools
            "file_read": ToolProfile(
                name="file_read",
                description="Read files from the filesystem",
                category=ToolCategory.UTILITY,
                keywords=["read", "file", "open", "load", "content"],
                typical_use_cases=[
                    "Reading source code",
                    "Loading configuration",
                    "Accessing documentation",
                ],
                confidence_baseline=0.9,
            ),
            "file_write": ToolProfile(
                name="file_write",
                description="Write files to the filesystem",
                category=ToolCategory.CREATION,
                keywords=["write", "save", "create", "output", "store"],
                typical_use_cases=[
                    "Saving results",
                    "Creating new files",
                    "Writing code",
                ],
                confidence_baseline=0.85,
            ),
            # Analysis tools
            "code_analysis": ToolProfile(
                name="code_analysis",
                description="Analyze code structure and quality",
                category=ToolCategory.ANALYSIS,
                keywords=["code", "analyze", "review", "quality", "structure"],
                typical_use_cases=[
                    "Code review",
                    "Finding bugs",
                    "Performance analysis",
                ],
                confidence_baseline=0.8,
            ),
            # Documentation tools
            "search_docs": ToolProfile(
                name="search_docs",
                description="Search technical documentation",
                category=ToolCategory.RESEARCH,
                keywords=["docs", "documentation", "api", "reference", "guide"],
                typical_use_cases=[
                    "API reference lookup",
                    "Finding examples",
                    "Understanding frameworks",
                ],
                confidence_baseline=0.85,
            ),
        }

    def analyze_thought_intent(self, thought: str) -> Dict[str, float]:
        """
        Analyze the intent of a thought to determine what kind of tools might be needed.
        Returns a dictionary of intent categories with confidence scores.
        """
        thought_lower = thought.lower()

        intents = {
            "research": 0.0,
            "analysis": 0.0,
            "creation": 0.0,
            "validation": 0.0,
            "planning": 0.0,
            "implementation": 0.0,
        }

        # Research indicators
        research_keywords = [
            "search",
            "find",
            "discover",
            "explore",
            "research",
            "look up",
            "investigate",
        ]
        intents["research"] = sum(
            1 for kw in research_keywords if kw in thought_lower
        ) / len(research_keywords)

        # Analysis indicators
        analysis_keywords = [
            "analyze",
            "examine",
            "review",
            "evaluate",
            "assess",
            "understand",
            "study",
        ]
        intents["analysis"] = sum(
            1 for kw in analysis_keywords if kw in thought_lower
        ) / len(analysis_keywords)

        # Creation indicators
        creation_keywords = [
            "create",
            "build",
            "implement",
            "write",
            "develop",
            "generate",
            "make",
        ]
        intents["creation"] = sum(
            1 for kw in creation_keywords if kw in thought_lower
        ) / len(creation_keywords)

        # Validation indicators
        validation_keywords = [
            "test",
            "verify",
            "validate",
            "check",
            "ensure",
            "confirm",
            "prove",
        ]
        intents["validation"] = sum(
            1 for kw in validation_keywords if kw in thought_lower
        ) / len(validation_keywords)

        # Planning indicators
        planning_keywords = [
            "plan",
            "design",
            "architect",
            "structure",
            "organize",
            "strategy",
        ]
        intents["planning"] = sum(
            1 for kw in planning_keywords if kw in thought_lower
        ) / len(planning_keywords)

        # Implementation indicators
        implementation_keywords = [
            "implement",
            "code",
            "program",
            "execute",
            "run",
            "deploy",
        ]
        intents["implementation"] = sum(
            1 for kw in implementation_keywords if kw in thought_lower
        ) / len(implementation_keywords)

        # Normalize scores
        total = sum(intents.values())
        if total > 0:
            intents = {k: v / total for k, v in intents.items()}

        return intents

    def calculate_tool_relevance(
        self,
        tool_profile: ToolProfile,
        thought: str,
        intents: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate how relevant a tool is for the current thought.
        Returns a confidence score between 0 and 1.
        """
        relevance = tool_profile.confidence_baseline
        thought_lower = thought.lower()

        # Keyword matching
        keyword_matches = sum(1 for kw in tool_profile.keywords if kw in thought_lower)
        keyword_score = (
            keyword_matches / len(tool_profile.keywords) if tool_profile.keywords else 0
        )
        relevance = (relevance + keyword_score) / 2

        # Intent alignment
        intent_alignment = 0.0
        if tool_profile.category == ToolCategory.RESEARCH:
            intent_alignment = intents.get("research", 0.0)
        elif tool_profile.category == ToolCategory.ANALYSIS:
            intent_alignment = intents.get("analysis", 0.0)
        elif tool_profile.category == ToolCategory.CREATION:
            intent_alignment = intents.get("creation", 0.0) + intents.get(
                "implementation", 0.0
            )
        elif tool_profile.category == ToolCategory.VALIDATION:
            intent_alignment = intents.get("validation", 0.0)
        elif tool_profile.category == ToolCategory.THINKING:
            intent_alignment = intents.get("planning", 0.0) + intents.get(
                "analysis", 0.0
            )

        relevance = (relevance + intent_alignment) / 2

        # Context-based adjustments
        if context:
            domain = context.get("domain", DomainType.GENERAL)
            if domain == DomainType.TECHNICAL and tool_profile.category in [
                ToolCategory.ANALYSIS,
                ToolCategory.CREATION,
            ]:
                relevance *= 1.2
            elif (
                domain == DomainType.RESEARCH
                and tool_profile.category == ToolCategory.RESEARCH
            ):
                relevance *= 1.3

        # Historical performance adjustment
        tool_history = [
            eff for name, eff in self.usage_history if name == tool_profile.name
        ]
        if tool_history:
            avg_effectiveness = sum(tool_history) / len(tool_history)
            relevance = (relevance + avg_effectiveness) / 2

        return min(relevance, 1.0)

    def recommend_tools(
        self,
        thought_data: ThoughtData,
        available_tools: Optional[List[str]] = None,
        max_recommendations: int = 3,
    ) -> StepRecommendation:
        """
        Recommend tools for the current thinking step.

        Args:
            thought_data: Current thought data
            available_tools: List of available tool names (if None, uses all known tools)
            max_recommendations: Maximum number of tools to recommend

        Returns:
            StepRecommendation with tool recommendations
        """
        thought = thought_data.thought
        intents = self.analyze_thought_intent(thought)

        # Filter to available tools
        if available_tools:
            tool_profiles = {
                name: profile
                for name, profile in self.tool_profiles.items()
                if name in available_tools
            }
        else:
            tool_profiles = self.tool_profiles

        # Calculate relevance for each tool
        tool_scores: List[Tuple[str, ToolProfile, float]] = []

        context = {
            "domain": thought_data.domain,
            "thought_number": thought_data.thoughtNumber,
            "is_revision": thought_data.isRevision,
            "keywords": thought_data.keywords,
        }

        for name, profile in tool_profiles.items():
            relevance = self.calculate_tool_relevance(
                profile, thought, intents, context
            )
            if relevance > 0.3:  # Minimum threshold
                tool_scores.append((name, profile, relevance))

        # Sort by relevance
        tool_scores.sort(key=lambda x: x[2], reverse=True)

        # Create recommendations
        recommendations: List[ToolRecommendation] = []

        for i, (name, profile, score) in enumerate(tool_scores[:max_recommendations]):
            # Find alternatives
            alternatives = [
                alt_name
                for alt_name, alt_profile, alt_score in tool_scores[
                    max_recommendations : max_recommendations + 2
                ]
                if alt_profile.category == profile.category
            ]

            recommendation = ToolRecommendation(
                tool_name=name,
                confidence=score,
                rationale=self._generate_rationale(profile, thought, intents),
                priority=i + 1,
                expected_outcome=profile.typical_use_cases[0]
                if profile.typical_use_cases
                else "Process the current step",
                alternatives=alternatives,
                suggested_inputs=self._suggest_inputs(name, thought_data),
                risk_assessment=self._assess_risk(profile, thought_data),
                execution_time_estimate=self._estimate_execution_time(profile),
            )
            recommendations.append(recommendation)

        # Determine expected outcome and next conditions
        primary_intent = (
            max(intents.items(), key=lambda x: x[1])[0] if intents else "general"
        )

        expected_outcome = self._generate_expected_outcome(primary_intent, thought)
        next_conditions = self._generate_next_conditions(primary_intent, thought_data)

        return StepRecommendation(
            step_description=f"Step {thought_data.thoughtNumber}: {thought[:100]}...",
            recommended_tools=recommendations,
            expected_outcome=expected_outcome,
            dependencies=[],
            validation_criteria=next_conditions,
        )

    def _generate_rationale(
        self, profile: ToolProfile, thought: str, intents: Dict[str, float]
    ) -> str:
        """Generate a rationale for why this tool is recommended."""
        primary_intent = (
            max(intents.items(), key=lambda x: x[1])[0] if intents else "general"
        )

        if profile.category == ToolCategory.RESEARCH:
            return f"Use {profile.name} to gather information and research relevant to: {thought[:50]}..."
        elif profile.category == ToolCategory.ANALYSIS:
            return f"Apply {profile.name} to analyze and understand the complexities of the current step"
        elif profile.category == ToolCategory.CREATION:
            return f"Utilize {profile.name} to create or implement the solution for this step"
        elif profile.category == ToolCategory.THINKING:
            return f"Engage {profile.name} for deep reflection and strategic planning"
        else:
            return f"Use {profile.name} to support the {primary_intent} intent of this step"

    def _suggest_inputs(
        self, tool_name: str, thought_data: ThoughtData
    ) -> Optional[Dict[str, Any]]:
        """Suggest inputs for a specific tool based on the current thought."""
        suggestions = {}

        if tool_name == "file_read" and "file" in thought_data.thought.lower():
            # Try to extract filename from thought
            suggestions["path"] = "path/to/relevant/file"

        elif tool_name == "tavily_search":
            # Extract key terms for search
            keywords = thought_data.keywords[:3] if thought_data.keywords else []
            if keywords:
                suggestions["query"] = " ".join(keywords)

        elif (
            tool_name == "code_analysis" and thought_data.domain == DomainType.TECHNICAL
        ):
            suggestions["focus"] = (
                "performance"
                if "performance" in thought_data.thought.lower()
                else "quality"
            )

        return suggestions if suggestions else None

    def _generate_expected_outcome(self, primary_intent: str, thought: str) -> str:
        """Generate expected outcome based on intent."""
        intent_outcomes = {
            "research": "Comprehensive information gathered and synthesized",
            "analysis": "Deep understanding of the problem structure and components",
            "creation": "Solution or artifact created successfully",
            "validation": "Verification completed with results documented",
            "planning": "Clear plan or strategy formulated",
            "implementation": "Code or solution implemented and ready",
        }

        return intent_outcomes.get(
            primary_intent, "Step completed with insights documented"
        )

    def _generate_next_conditions(
        self, primary_intent: str, thought_data: ThoughtData
    ) -> List[str]:
        """Generate conditions to check before proceeding to next step."""
        conditions = []

        if primary_intent == "research":
            conditions.extend(
                [
                    "Verify information accuracy and relevance",
                    "Check if additional sources are needed",
                    "Summarize key findings",
                ]
            )
        elif primary_intent == "analysis":
            conditions.extend(
                [
                    "Ensure all aspects have been examined",
                    "Identify any gaps in understanding",
                    "Document insights and patterns",
                ]
            )
        elif primary_intent == "creation":
            conditions.extend(
                [
                    "Validate the created solution",
                    "Check for edge cases",
                    "Ensure quality standards are met",
                ]
            )

        # Add thought-specific conditions
        if thought_data.isRevision:
            conditions.append("Confirm revision addresses original concerns")

        if thought_data.branchFromThought:
            conditions.append("Compare branch results with main path")

        return conditions[:3]  # Limit to 3 conditions

    def record_tool_effectiveness(self, tool_name: str, effectiveness: float):
        """Record how effective a tool was for future recommendations."""
        self.usage_history.append((tool_name, effectiveness))

        # Keep history manageable
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]

    def get_tool_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about tool usage and effectiveness."""
        stats = {}

        for name in self.tool_profiles:
            tool_uses = [(n, e) for n, e in self.usage_history if n == name]
            if tool_uses:
                effectiveness_scores = [e for _, e in tool_uses]
                stats[name] = {
                    "usage_count": len(tool_uses),
                    "avg_effectiveness": sum(effectiveness_scores)
                    / len(effectiveness_scores),
                    "min_effectiveness": min(effectiveness_scores),
                    "max_effectiveness": max(effectiveness_scores),
                }

        return stats

    def _assess_risk(
        self, profile: ToolProfile, thought_data: ThoughtData
    ) -> Optional[str]:
        """Assess potential risks for using a tool."""
        risks = []

        if profile.category == ToolCategory.CREATION:
            if thought_data.confidence_score < 0.7:
                risks.append(
                    "Low confidence in current understanding may lead to suboptimal implementation"
                )
        elif profile.category == ToolCategory.RESEARCH and thought_data.isRevision:
            risks.append("Research tools may yield different results on revision")
        elif (
            profile.category == ToolCategory.ANALYSIS
            and len(thought_data.tool_decisions) > 5
        ):
            risks.append(
                "Multiple previous analyses may indicate complexity requiring careful interpretation"
            )

        return "; ".join(risks) if risks else None

    def _estimate_execution_time(self, profile: ToolProfile) -> int:
        """Estimate execution time in milliseconds based on tool category."""
        estimates = {
            ToolCategory.THINKING: 3000,  # Deep thinking takes time
            ToolCategory.RESEARCH: 2000,  # Web searches can be slow
            ToolCategory.ANALYSIS: 1500,  # Code analysis is moderately fast
            ToolCategory.CREATION: 1000,  # File operations are fast
            ToolCategory.VALIDATION: 2500,  # Tests take time
            ToolCategory.UTILITY: 500,  # Utilities are quick
            ToolCategory.COMMUNICATION: 1000,  # Moderate speed
            ToolCategory.UNKNOWN: 1500,  # Default estimate
        }

        return estimates.get(profile.category, 1500)
