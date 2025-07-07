"""
Unit tests for the ToolSelector component.
Tests tool recommendation logic, scoring, and intent analysis.
"""

import pytest

from src.tools.tool_selector import (
    ToolSelector,
    ToolCategory,
    ToolProfile,
)
from src.models.thought_models import (
    DomainType,
    ToolRecommendation,
    StepRecommendation,
)
from .conftest import create_test_thought_data


class TestToolSelector:
    """Test the ToolSelector class."""

    @pytest.fixture
    def tool_selector(self):
        """Create a ToolSelector instance for testing."""
        return ToolSelector()

    @pytest.fixture
    def custom_tool_profiles(self):
        """Create custom tool profiles for testing."""
        return {
            "custom_analyzer": ToolProfile(
                name="custom_analyzer",
                description="Custom analysis tool",
                category=ToolCategory.ANALYSIS,
                keywords=["analyze", "custom", "inspect"],
                typical_use_cases=["Custom analysis", "Deep inspection"],
                confidence_baseline=0.85,
            ),
            "data_processor": ToolProfile(
                name="data_processor",
                description="Process and transform data",
                category=ToolCategory.UTILITY,
                keywords=["data", "process", "transform"],
                typical_use_cases=["Data transformation", "ETL operations"],
                confidence_baseline=0.75,
            ),
        }

    def test_tool_selector_initialization(self, tool_selector):
        """Test ToolSelector initialization with default tools."""
        assert isinstance(tool_selector.tool_profiles, dict)
        assert len(tool_selector.tool_profiles) > 0

        # Check some default tools exist
        assert "ThinkingTools" in tool_selector.tool_profiles
        assert "ExaTools" in tool_selector.tool_profiles
        assert "file_read" in tool_selector.tool_profiles

    def test_analyze_thought_intent_research(self, tool_selector):
        """Test intent analysis for research-focused thoughts."""
        thought = "I need to search for information about machine learning algorithms"
        intents = tool_selector.analyze_thought_intent(thought)

        assert "research" in intents
        assert intents["research"] > 0.0
        assert intents["research"] > intents["creation"]  # Should prioritize research

    def test_analyze_thought_intent_creation(self, tool_selector):
        """Test intent analysis for creation-focused thoughts."""
        thought = "Create a new Python script to implement the sorting algorithm"
        intents = tool_selector.analyze_thought_intent(thought)

        assert "creation" in intents
        assert intents["creation"] > 0.0
        assert intents["creation"] > intents["research"]  # Should prioritize creation

    def test_analyze_thought_intent_mixed(self, tool_selector):
        """Test intent analysis for mixed-intent thoughts."""
        thought = (
            "Analyze the code and then create a test suite to verify functionality"
        )
        intents = tool_selector.analyze_thought_intent(thought)

        assert intents["analysis"] > 0.0
        assert intents["creation"] > 0.0
        assert intents["validation"] > 0.0

    def test_calculate_tool_relevance_basic(self, tool_selector):
        """Test basic tool relevance calculation."""
        profile = tool_selector.tool_profiles["ThinkingTools"]
        thought = "I need to think deeply about this problem"
        intents = {"planning": 0.5, "analysis": 0.3, "research": 0.2}

        relevance = tool_selector.calculate_tool_relevance(profile, thought, intents)

        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # Should be reasonably relevant

    def test_calculate_tool_relevance_with_context(self, tool_selector):
        """Test tool relevance with domain context."""
        profile = tool_selector.tool_profiles["code_analysis"]
        thought = "Analyze the performance of this function"
        intents = {"analysis": 0.8, "validation": 0.2}
        context = {"domain": DomainType.TECHNICAL}

        relevance = tool_selector.calculate_tool_relevance(
            profile, thought, intents, context
        )

        assert relevance > 0.7  # Should be highly relevant for technical domain

    def test_recommend_tools_basic(self, tool_selector):
        """Test basic tool recommendation."""
        thought_data = create_test_thought_data(
            thought="Search for best practices in API design",
            domain=DomainType.TECHNICAL,
            keywords=["search", "api", "design", "best practices"],
        )

        recommendations = tool_selector.recommend_tools(thought_data)

        assert isinstance(recommendations, StepRecommendation)
        assert len(recommendations.recommended_tools) > 0
        assert len(recommendations.recommended_tools) <= 3  # Max recommendations

        # Check first recommendation
        first_rec = recommendations.recommended_tools[0]
        assert isinstance(first_rec, ToolRecommendation)
        assert first_rec.priority == 1
        assert first_rec.confidence > 0.0

    def test_recommend_tools_with_available_filter(self, tool_selector):
        """Test tool recommendation with available tools filter."""
        thought_data = create_test_thought_data(
            thought="Read and analyze the configuration file",
        )

        available_tools = ["file_read", "code_analysis"]
        recommendations = tool_selector.recommend_tools(
            thought_data, available_tools=available_tools
        )

        # Should only recommend from available tools
        recommended_names = [r.tool_name for r in recommendations.recommended_tools]
        assert all(name in available_tools for name in recommended_names)

    def test_recommend_tools_no_matches(self, tool_selector):
        """Test recommendation when no tools match well."""
        thought_data = create_test_thought_data(
            thought="Perform quantum computing calculations",  # Unlikely to match
            keywords=["quantum", "qubits", "superposition"],
        )

        recommendations = tool_selector.recommend_tools(
            thought_data, available_tools=["file_read", "file_write"]
        )

        # Should still provide some recommendations
        assert len(recommendations.recommended_tools) >= 0
        if recommendations.recommended_tools:
            # But with lower confidence
            assert all(r.confidence < 0.5 for r in recommendations.recommended_tools)

    def test_tool_effectiveness_tracking(self, tool_selector):
        """Test tracking tool effectiveness."""
        # Record some tool usage
        tool_selector.record_tool_effectiveness("code_analysis", 0.9)
        tool_selector.record_tool_effectiveness("code_analysis", 0.8)
        tool_selector.record_tool_effectiveness("file_read", 0.6)

        stats = tool_selector.get_tool_statistics()

        assert "code_analysis" in stats
        assert stats["code_analysis"]["usage_count"] == 2
        assert stats["code_analysis"]["avg_effectiveness"] == 0.85
        assert stats["code_analysis"]["max_effectiveness"] == 0.9

        assert "file_read" in stats
        assert stats["file_read"]["usage_count"] == 1

    def test_tool_history_influences_recommendations(self, tool_selector):
        """Test that tool usage history influences future recommendations."""
        thought_data = create_test_thought_data(
            thought="Analyze code structure",
        )

        # Get initial recommendation
        initial_rec = tool_selector.recommend_tools(thought_data)
        initial_confidence = {
            r.tool_name: r.confidence for r in initial_rec.recommended_tools
        }

        # Record positive effectiveness for a tool
        if initial_rec.recommended_tools:
            tool_name = initial_rec.recommended_tools[0].tool_name
            tool_selector.record_tool_effectiveness(tool_name, 0.95)

            # Get new recommendation
            new_rec = tool_selector.recommend_tools(thought_data)
            new_confidence = {
                r.tool_name: r.confidence for r in new_rec.recommended_tools
            }

            # Confidence should increase for the effective tool
            if tool_name in new_confidence and tool_name in initial_confidence:
                assert new_confidence[tool_name] >= initial_confidence[tool_name]

    def test_custom_tool_profiles(self, tool_selector, custom_tool_profiles):
        """Test adding and using custom tool profiles."""
        # Add custom tools
        tool_selector.tool_profiles.update(custom_tool_profiles)

        thought_data = create_test_thought_data(
            thought="Process and transform the customer data",
            keywords=["process", "transform", "data"],
        )

        recommendations = tool_selector.recommend_tools(thought_data)
        recommended_names = [r.tool_name for r in recommendations.recommended_tools]

        # Should recommend the custom data processor
        assert "data_processor" in recommended_names

    def test_generate_rationale(self, tool_selector):
        """Test rationale generation for different tool categories."""
        # Research tool
        research_profile = tool_selector.tool_profiles["ExaTools"]
        research_rationale = tool_selector._generate_rationale(
            research_profile, "Find information about X", {"research": 0.8}
        )
        assert "gather information" in research_rationale.lower()

        # Analysis tool
        if "code_analysis" in tool_selector.tool_profiles:
            analysis_profile = tool_selector.tool_profiles["code_analysis"]
            analysis_rationale = tool_selector._generate_rationale(
                analysis_profile, "Analyze the code", {"analysis": 0.8}
            )
            assert "analyze" in analysis_rationale.lower()

    def test_suggest_inputs(self, tool_selector):
        """Test input suggestion for tools."""
        thought_data = create_test_thought_data(
            thought="Read the config file at /etc/app/config.yaml",
            keywords=["file", "config", "read"],
        )

        suggestions = tool_selector._suggest_inputs("file_read", thought_data)

        # Should suggest file path
        assert suggestions is not None
        assert "path" in suggestions

    def test_expected_outcome_generation(self, tool_selector):
        """Test generation of expected outcomes."""
        # Research intent
        research_outcome = tool_selector._generate_expected_outcome(
            "research", "Find information"
        )
        assert "information" in research_outcome.lower()

        # Creation intent
        creation_outcome = tool_selector._generate_expected_outcome(
            "creation", "Build something"
        )
        assert (
            "created" in creation_outcome.lower()
            or "solution" in creation_outcome.lower()
        )

    def test_next_conditions_generation(self, tool_selector):
        """Test generation of next step conditions."""
        thought_data = create_test_thought_data(
            thought="Research the topic",
            isRevision=False,
        )

        conditions = tool_selector._generate_next_conditions("research", thought_data)

        assert isinstance(conditions, list)
        assert len(conditions) <= 3
        assert all(isinstance(c, str) for c in conditions)

        # Should include verification step
        assert any("verify" in c.lower() or "check" in c.lower() for c in conditions)

    def test_tool_category_assignment(self, tool_selector):
        """Test that tools have appropriate categories."""
        for name, profile in tool_selector.tool_profiles.items():
            assert isinstance(profile.category, ToolCategory)

            # Verify category matches tool purpose
            if "think" in name.lower():
                assert profile.category == ToolCategory.THINKING
            elif "search" in name.lower():
                assert profile.category == ToolCategory.RESEARCH
            elif "write" in name.lower():
                assert profile.category == ToolCategory.CREATION

    def test_alternatives_recommendation(self, tool_selector):
        """Test that alternatives are properly recommended."""
        thought_data = create_test_thought_data(
            thought="Search for information about Python libraries",
        )

        recommendations = tool_selector.recommend_tools(thought_data)

        # Check if alternatives are provided
        for rec in recommendations.recommended_tools:
            assert isinstance(rec.alternatives, list)
            # Alternatives should be different from main recommendation
            assert rec.tool_name not in rec.alternatives

    def test_priority_ordering(self, tool_selector):
        """Test that tool recommendations are properly prioritized."""
        thought_data = create_test_thought_data(
            thought="Analyze and optimize the code performance",
        )

        recommendations = tool_selector.recommend_tools(thought_data)

        # Check priority ordering
        priorities = [r.priority for r in recommendations.recommended_tools]
        assert priorities == sorted(priorities)  # Should be in order
        assert all(p >= 1 for p in priorities)  # Priorities start at 1

    def test_confidence_threshold(self, tool_selector):
        """Test that low-confidence tools are filtered out."""
        thought_data = create_test_thought_data(
            thought="Perform an obscure task with no clear tool match",
            keywords=["obscure", "unknown", "mysterious"],
        )

        recommendations = tool_selector.recommend_tools(thought_data)

        # All recommended tools should have minimum confidence
        for rec in recommendations.recommended_tools:
            assert rec.confidence > 0.3  # Minimum threshold

    def test_domain_specific_recommendations(self, tool_selector):
        """Test domain-specific tool recommendations."""
        # Technical domain
        tech_thought = create_test_thought_data(
            thought="Analyze algorithm complexity",
            domain=DomainType.TECHNICAL,
        )
        tech_rec = tool_selector.recommend_tools(tech_thought)

        # Creative domain
        creative_thought = create_test_thought_data(
            thought="Design a user interface",
            domain=DomainType.CREATIVE,
        )
        creative_rec = tool_selector.recommend_tools(creative_thought)

        # Different domains should potentially yield different recommendations
        tech_tools = [r.tool_name for r in tech_rec.recommended_tools]
        creative_tools = [r.tool_name for r in creative_rec.recommended_tools]

        # Not necessarily completely different, but confidence may vary
        assert tech_rec.recommended_tools or creative_rec.recommended_tools
