"""
Comprehensive tests for EnhancedAppContext.
Following TDD principles - tests written FIRST before extraction.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from context.app_context import EnhancedAppContext
from models.thought_models import ThoughtData, DomainType
from exceptions import ErrorType


class TestEnhancedAppContextInitialization:
    """Test EnhancedAppContext initialization and setup."""

    def test_app_context_initialization(self):
        """Test basic initialization of EnhancedAppContext."""
        context = EnhancedAppContext()
        
        # Core attributes should be initialized
        assert context.start_time is not None
        assert isinstance(context.start_time, datetime)
        assert context.shared_context is not None
        assert context.error_handler is not None
        
        # Team attributes
        assert context.primary_team is None  # Not initialized yet
        assert context.reflection_team is None
        assert context.teams_initialized is False
        
        # Available tools
        assert "ThinkingTools" in context.available_tools
        assert "ExaTools" in context.available_tools
        
        # Topic and domain
        assert context.current_topic == "Reflective Thinking Session"
        assert context.current_domain == DomainType.GENERAL
        
        # Thought tracking
        assert len(context.thought_history) == 0
        assert len(context.branches) == 0
        assert len(context.available_mcp_tools) == 0

    def test_app_context_provider_initialization(self):
        """Test provider configuration during initialization."""
        with patch('main.LLMProviderFactory.get_provider_config') as mock_provider:
            mock_config = Mock()
            mock_config.provider_name = "TestProvider"
            mock_provider.return_value = mock_config
            
            context = EnhancedAppContext()
            
            assert context.provider_initialized is True
            assert context.provider_config == mock_config
            mock_provider.assert_called_once()

    def test_app_context_provider_initialization_failure(self):
        """Test handling of provider initialization failure."""
        with patch('main.LLMProviderFactory.get_provider_config') as mock_provider:
            mock_provider.side_effect = Exception("Provider init failed")
            
            with pytest.raises(Exception) as exc_info:
                context = EnhancedAppContext()
            
            assert "Provider init failed" in str(exc_info.value)


class TestEnhancedAppContextTeamManagement:
    """Test team initialization and management."""

    @pytest.fixture
    def mock_provider_config(self):
        """Create mock provider configuration."""
        config = Mock()
        config.provider_name = "TestProvider"
        config.get_models.return_value = ("team-model-id", "agent-model-id")
        
        # Mock model creation
        mock_model = Mock()
        config.create_model_instance.return_value = mock_model
        
        return config

    @pytest.fixture
    def app_context_with_provider(self, mock_provider_config):
        """Create app context with mocked provider."""
        with patch('main.LLMProviderFactory.get_provider_config', return_value=mock_provider_config):
            context = EnhancedAppContext()
            context.provider_config = mock_provider_config
            return context

    @pytest.mark.asyncio
    async def test_initialize_teams_success(self, app_context_with_provider):
        """Test successful team initialization."""
        context = app_context_with_provider
        
        # Mock the team creation methods
        mock_primary_team = AsyncMock()
        mock_reflection_team = AsyncMock()
        
        with patch.object(context, '_create_primary_team', return_value=mock_primary_team):
            with patch.object(context, '_create_reflection_team', return_value=mock_reflection_team):
                await context.initialize_teams()
        
        assert context.teams_initialized is True
        assert context.primary_team == mock_primary_team
        assert context.reflection_team == mock_reflection_team

    @pytest.mark.asyncio
    async def test_initialize_teams_already_initialized(self, app_context_with_provider):
        """Test that teams are not re-initialized if already done."""
        context = app_context_with_provider
        context.teams_initialized = True
        
        # Set existing teams
        context.primary_team = Mock()
        context.reflection_team = Mock()
        
        # Should return without doing anything
        await context.initialize_teams()
        
        # Teams should remain unchanged
        assert context.teams_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_teams_model_creation_failure(self, app_context_with_provider):
        """Test handling of model creation failure during team initialization."""
        context = app_context_with_provider
        context.provider_config.create_model_instance.side_effect = Exception("Model creation failed")
        
        with pytest.raises(Exception) as exc_info:
            await context.initialize_teams()
        
        assert "Model creation failed" in str(exc_info.value)
        assert context.teams_initialized is False

    @pytest.mark.asyncio
    async def test_create_primary_team(self, app_context_with_provider):
        """Test primary team creation with agents."""
        context = app_context_with_provider
        mock_model = Mock()
        
        # Mock agent creation
        with patch('context.app_context.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Mock AsyncTeam
            with patch('context.app_context.AsyncTeam') as mock_team_class:
                mock_team = AsyncMock()
                mock_team.arun = AsyncMock(return_value="Initialization successful")
                mock_team_class.return_value = mock_team
                
                team = await context._create_primary_team(mock_model, mock_model)
                
                assert team == mock_team
                # Should create 5 agents (planner, researcher, analyzer, critic, synthesizer)
                assert mock_agent_class.call_count == 5
                # Team should be initialized with agents
                mock_team.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_reflection_team(self, app_context_with_provider):
        """Test reflection team creation with agents."""
        context = app_context_with_provider
        mock_model = Mock()
        
        with patch('context.app_context.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            with patch('context.app_context.AsyncTeam') as mock_team_class:
                mock_team = AsyncMock()
                mock_team.arun = AsyncMock(return_value="Initialization successful")
                mock_team_class.return_value = mock_team
                
                team = await context._create_reflection_team(mock_model, mock_model)
                
                assert team == mock_team
                # Should create 4 agents (meta_analyzer, pattern_recognizer, quality_assessor, decision_critic)
                assert mock_agent_class.call_count == 4
                mock_team.arun.assert_called_once()


class TestEnhancedAppContextThoughtManagement:
    """Test thought tracking and context management."""

    @pytest.fixture
    def app_context(self):
        """Create app context for testing."""
        with patch('main.LLMProviderFactory.get_provider_config'):
            return EnhancedAppContext()

    @pytest.fixture
    def sample_thought(self):
        """Create sample thought data."""
        return ThoughtData(
            thought="Analyze the requirements for a distributed system",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            topic="System Design",
            domain=DomainType.TECHNICAL,
            keywords=["distributed", "system", "requirements"],
            confidence_score=0.8,
            timestamp_ms=int(datetime.now().timestamp() * 1000)
        )

    @pytest.mark.asyncio
    async def test_add_thought(self, app_context, sample_thought):
        """Test adding a thought to the context."""
        # Mock shared context
        app_context.shared_context.update_from_thought = AsyncMock()
        
        await app_context.add_thought(sample_thought)
        
        # Thought should be added to history
        assert len(app_context.thought_history) == 1
        assert app_context.thought_history[0] == sample_thought
        
        # Shared context should be updated
        app_context.shared_context.update_from_thought.assert_called_once_with(sample_thought)
        
        # Topic and domain should be updated
        assert app_context.current_topic == "System Design"
        assert app_context.current_domain == DomainType.TECHNICAL

    @pytest.mark.asyncio
    async def test_add_thought_with_branch(self, app_context):
        """Test adding a branched thought."""
        # Create thought with branch info (non-consecutive to avoid validation error)
        branched_thought = ThoughtData(
            thought="Implement feature X with alternative approach exploring a different design pattern",
            thoughtNumber=3,
            totalThoughts=5,
            nextThoughtNeeded=True,
            branchId="feature-branch",
            branchFromThought=1,  # Branching from thought 1 to thought 3 (non-consecutive)
            topic="Feature Implementation",
            timestamp_ms=int(datetime.now().timestamp() * 1000)
        )
        
        app_context.shared_context.update_from_thought = AsyncMock()
        
        await app_context.add_thought(branched_thought)
        
        # Branch should be tracked
        assert "feature-branch" in app_context.branches
        assert len(app_context.branches["feature-branch"]) == 1
        assert app_context.branches["feature-branch"][0] == branched_thought

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, app_context):
        """Test getting relevant context for a thought."""
        mock_context = {"key": "value", "relevance": "high"}
        app_context.shared_context.get_relevant_context = AsyncMock(return_value=mock_context)
        
        result = await app_context.get_relevant_context("test thought")
        
        assert result == mock_context
        app_context.shared_context.get_relevant_context.assert_called_once_with("test thought")

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, app_context):
        """Test getting performance metrics."""
        # Mock dependencies
        app_context.shared_context.get_performance_summary = AsyncMock(
            return_value={"avg_time": 100, "total_thoughts": 5}
        )
        app_context.shared_context.get_memory_usage = Mock(
            return_value={"memory_items": 50, "insights": 10}
        )
        app_context.error_handler.get_error_summary = Mock(
            return_value={"total_errors": 2, "by_type": {"validation": 2}}
        )
        
        # Add some thoughts
        for i in range(3):
            thought = ThoughtData(
                thought=f"Test thought {i}",
                thoughtNumber=i+1,
                totalThoughts=5,
                nextThoughtNeeded=True
            )
            app_context.thought_history.append(thought)
        
        # Add a branch
        app_context.branches["test-branch"] = [app_context.thought_history[0]]
        
        metrics = await app_context.get_performance_metrics()
        
        assert metrics["total_thoughts"] == 3
        assert metrics["total_branches"] == 1
        assert "duration_seconds" in metrics
        assert metrics["performance"]["avg_time"] == 100
        assert metrics["errors"]["total_errors"] == 2
        assert metrics["memory_usage"]["memory_items"] == 50

    def test_update_available_tools(self, app_context):
        """Test updating available tools list."""
        new_tools = ["tool1", "tool2", "tool3"]
        
        app_context.update_available_tools(new_tools)
        
        assert app_context.available_mcp_tools == new_tools
        assert app_context.available_tools == new_tools

    def test_cleanup(self, app_context):
        """Test cleanup of resources."""
        app_context.shared_context.clear = Mock()
        
        app_context.cleanup()
        
        app_context.shared_context.clear.assert_called_once()


class TestEnhancedAppContextInstructionGeneration:
    """Test adaptive instruction generation for teams."""

    @pytest.fixture
    def app_context(self):
        """Create app context for testing."""
        with patch('main.LLMProviderFactory.get_provider_config'):
            return EnhancedAppContext()

    @pytest.mark.asyncio
    async def test_generate_adaptive_coordinator_instructions_basic(self, app_context):
        """Test basic coordinator instruction generation."""
        app_context.shared_context.get_performance_summary = AsyncMock(
            return_value={"processing_time": {"mean": 2000}}
        )
        
        instructions = await app_context._generate_adaptive_coordinator_instructions()
        
        assert isinstance(instructions, list)
        assert len(instructions) > 0
        assert "PRIMARY THINKING TEAM COORDINATOR" in instructions[0]
        # Check for the content in the joined string
        instructions_str = "\n".join(instructions)
        assert "Team Coordination Process:" in instructions_str
        assert "Key Principles:" in instructions_str

    @pytest.mark.asyncio
    async def test_generate_adaptive_coordinator_instructions_with_history(self, app_context):
        """Test coordinator instructions with thought history."""
        # Add technical thoughts
        for i in range(3):
            thought = ThoughtData(
                thought=f"Technical analysis {i}",
                thoughtNumber=i+1,
                totalThoughts=5,
                nextThoughtNeeded=True,
                domain=DomainType.TECHNICAL
            )
            app_context.thought_history.append(thought)
        
        app_context.shared_context.get_performance_summary = AsyncMock(
            return_value={"processing_time": {"mean": 2000}}
        )
        
        instructions = await app_context._generate_adaptive_coordinator_instructions()
        
        # Should include technical focus
        assert any("technical accuracy" in inst.lower() for inst in instructions)

    @pytest.mark.asyncio
    async def test_generate_adaptive_coordinator_instructions_slow_performance(self, app_context):
        """Test coordinator instructions with slow performance."""
        app_context.shared_context.get_performance_summary = AsyncMock(
            return_value={"processing_time": {"mean": 4000}}  # > 3 seconds
        )
        
        instructions = await app_context._generate_adaptive_coordinator_instructions()
        
        # Should include efficiency optimization
        assert any("efficiency" in inst.lower() for inst in instructions)

    def test_generate_planner_instructions_basic(self, app_context):
        """Test basic planner instruction generation."""
        instructions = app_context._generate_planner_instructions()
        
        assert "Strategic Planner" in instructions
        assert "responsibilities" in instructions
        assert "strategic approaches" in instructions

    def test_generate_planner_instructions_for_revision(self, app_context):
        """Test planner instructions for revision thought."""
        thought = ThoughtData(
            thought="After analyzing the initial approach, I realize we need to revise our strategy to better handle edge cases and improve performance",
            thoughtNumber=2,
            totalThoughts=5,
            nextThoughtNeeded=True,
            isRevision=True,
            revisesThought=1,
            timestamp_ms=int(datetime.now().timestamp() * 1000)
        )
        
        instructions = app_context._generate_planner_instructions(thought)
        
        assert "revising the previous strategy" in instructions

    def test_generate_researcher_instructions_with_keywords(self, app_context):
        """Test researcher instructions with keywords."""
        thought = ThoughtData(
            thought="Research distributed systems",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            keywords=["distributed", "scalability", "fault-tolerance", "consensus", "replication"]
        )
        
        instructions = app_context._generate_researcher_instructions(thought)
        
        assert "Information Researcher" in instructions
        assert "Focus your research on:" in instructions
        # Should include first 5 keywords
        assert "distributed" in instructions
        assert "scalability" in instructions