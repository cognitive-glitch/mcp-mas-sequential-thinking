"""
Pytest configuration and fixtures for integration testing.
"""

import pytest
import asyncio
import os
import uuid
from unittest.mock import Mock
from typing import List, Optional

# Import the main components
from src.models.thought_models import (
    ThoughtData,
    SessionContext,
    DomainType,
    ToolRecommendation,
    StepRecommendation,
)
from src.context.shared_context import SharedContext
from src.providers.base import ProviderConfig


class MockModel:
    """Mock LLM model for testing."""

    def __init__(self, model_id: str = "test-model"):
        self.id = model_id
        self.api_key = "test-key"

    async def arun(self, prompt: str) -> Mock:
        """Mock async run method."""
        response = Mock()
        response.content = f"Mock response for: {prompt[:50]}..."
        return response


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, role: str = "test-role"):
        self.name = name
        self.role = role
        self.model = MockModel()

    async def arun(self, input_data: str) -> Mock:
        """Mock async run method."""
        response = Mock()
        response.content = f"[{self.name}] Mock analysis: {input_data[:100]}..."
        return response


class MockTeam:
    """Mock team for testing dual-team architecture."""

    def __init__(
        self, name: str = "test-team", agents: Optional[List[MockAgent]] = None
    ):
        self.name = name
        self.members = agents or []

    async def arun(self, input_data: str) -> Mock:
        """Mock team coordination."""
        response = Mock()

        # Simulate team coordination response
        if "primary" in self.name.lower():
            response.content = f"Primary Team Analysis:\n{input_data[:200]}...\n\nRecommended tools: code_analysis, performance_check"
        elif "reflection" in self.name.lower():
            response.content = "Reflection Team Feedback:\nStrengths: Clear analysis\nWeaknesses: Could be more specific\nSuggestions: Add more context"
        else:
            response.content = f"Team {self.name} response to: {input_data[:100]}..."

        return response


@pytest.fixture
def mock_session_context():
    """Provide a mock session context for testing."""
    return SessionContext(
        session_id=str(uuid.uuid4()),
        available_tools=["ThinkingTools", "ExaTools"],
        session_topic="Test Topic",
        session_domain=DomainType.TECHNICAL,
    )


@pytest.fixture
def sample_thought_data(mock_session_context):
    """Provide sample thought data for testing."""
    return ThoughtData(
        thought="Analyze the performance characteristics of the dual-team architecture",
        thoughtNumber=1,
        totalThoughts=3,
        nextThoughtNeeded=True,
        topic="Performance Analysis",
        subject="Architecture Evaluation",
        domain=DomainType.TECHNICAL,
        keywords=["performance", "architecture", "analysis"],
        session_context=mock_session_context,
        timestamp_ms=1234567890000,
        isRevision=False,
        revisesThought=None,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
        current_step=None,
        reflection_feedback=None,
        confidence_score=0.8,
        processing_time_ms=1000,
    )


@pytest.fixture
def shared_context():
    """Provide a fresh SharedContext for testing."""
    return SharedContext()


@pytest.fixture
def mock_primary_team():
    """Provide a mock primary team."""
    agents = [
        MockAgent("Planner", "Strategic Planner"),
        MockAgent("Researcher", "Information Gatherer"),
        MockAgent("Analyzer", "Core Analyst"),
        MockAgent("Critic", "Quality Controller"),
        MockAgent("Synthesizer", "Integration Specialist"),
    ]
    return MockTeam("PrimaryTeam", agents)


@pytest.fixture
def mock_reflection_team():
    """Provide a mock reflection team."""
    agents = [
        MockAgent("MetaAnalyzer", "Thinking Pattern Analyst"),
        MockAgent("PatternRecognizer", "Bias Detection Specialist"),
        MockAgent("QualityAssessor", "Quality Evaluator"),
        MockAgent("DecisionCritic", "Decision Process Analyst"),
    ]
    return MockTeam("ReflectionTeam", agents)


@pytest.fixture
def mock_app_context(
    mock_primary_team, mock_reflection_team, shared_context, mock_session_context
):
    """Provide a mock enhanced app context."""
    from main_refactored import EnhancedAppContext

    # Create a real context but with mocked teams
    context = EnhancedAppContext()
    context.primary_team = mock_primary_team
    context.reflection_team = mock_reflection_team
    context.shared_context = shared_context
    context.session_context = mock_session_context

    # Mock the provider config
    context.provider_config = Mock()
    context.provider_config.provider_name = "test-provider"

    return context


@pytest.fixture
def sample_tool_recommendation():
    """Provide a sample tool recommendation."""
    return ToolRecommendation(
        tool_name="code_analysis",
        confidence=0.9,
        rationale="Need to understand current performance bottlenecks",
        priority=1,
        expected_outcome="Identify O(n) complexity issues",
        alternatives=["profiling_tool", "benchmark_suite"],
        suggested_inputs={"file_path": "src/"},
        risk_assessment="Low risk analysis tool",
        execution_time_estimate=5000,
    )


@pytest.fixture
def sample_step_recommendation(sample_tool_recommendation):
    """Provide a sample step recommendation."""
    return StepRecommendation(
        step_description="Analyze current algorithm complexity",
        recommended_tools=[sample_tool_recommendation],
        expected_outcome="Understanding of performance characteristics",
        dependencies=["requirements_analysis"],
        validation_criteria=["complexity_identified", "bottlenecks_documented"],
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["LLM_PROVIDER"] = "test"
    os.environ["TEST_API_KEY"] = "test-key"

    yield

    # Clean up
    test_vars = ["LLM_PROVIDER", "TEST_API_KEY"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_provider_config():
    """Provide a mock provider configuration."""
    from unittest.mock import Mock

    mock_model_class = Mock()
    mock_model_class.return_value = MockModel()

    return ProviderConfig(
        provider_name="Test Provider",
        api_key_env="TEST_API_KEY",
        team_model_env="TEST_TEAM_MODEL",
        agent_model_env="TEST_AGENT_MODEL",
        default_team_model="test-team-model",
        default_agent_model="test-agent-model",
        model_class=mock_model_class,  # type: ignore
    )


# Test data constants
SAMPLE_THOUGHTS = [
    "Analyze the core problem structure",
    "Identify potential solutions and trade-offs",
    "Evaluate implementation approaches",
    "Design the optimal solution architecture",
    "Plan the implementation roadmap",
]

SAMPLE_TOPICS = [
    "Performance Optimization",
    "System Architecture",
    "Security Analysis",
    "Code Quality Improvement",
    "User Experience Enhancement",
]

SAMPLE_DOMAINS = [
    DomainType.TECHNICAL,
    DomainType.CREATIVE,
    DomainType.ANALYTICAL,
    DomainType.STRATEGIC,
    DomainType.RESEARCH,
]


def create_test_thought_data(**overrides):
    """Utility function to create ThoughtData with sensible defaults for testing."""
    defaults = {
        "thought": "Test thought content",
        "thoughtNumber": 1,
        "totalThoughts": 5,
        "nextThoughtNeeded": True,
        "topic": None,
        "subject": None,
        "domain": DomainType.GENERAL,
        "keywords": [],
        "isRevision": False,
        "revisesThought": None,
        "branchFromThought": None,
        "branchId": None,
        "needsMoreThoughts": False,
        "current_step": None,
        "reflection_feedback": None,
        "confidence_score": 0.5,
        "timestamp_ms": 1234567890000,
        "processing_time_ms": 1000,
        "session_context": None,
    }
    defaults.update(overrides)
    return ThoughtData(**defaults)


def create_test_tool_recommendation(**overrides):
    """Utility function to create ToolRecommendation with sensible defaults for testing."""
    defaults = {
        "tool_name": "test_tool",
        "confidence": 0.8,
        "rationale": "Test rationale",
        "priority": 1,
        "expected_outcome": "Test outcome",
        "alternatives": [],
        "suggested_inputs": None,
        "risk_assessment": None,
        "execution_time_estimate": None,
    }
    defaults.update(overrides)
    return ToolRecommendation(**defaults)
