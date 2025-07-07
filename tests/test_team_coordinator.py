"""
Comprehensive tests for TeamCoordinator following TDD principles.
Tests written FIRST before any refactoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers.team_coordinator import TeamCoordinator
from models.protocols import ModelProtocol
from exceptions import ModelInitializationError, ConfigurationError
from config import DEFAULT_MAX_AGENTS_PER_TEAM, REFLECTIVE_LLM_PROVIDER


class TestTeamCoordinatorInitialization:
    """Test TeamCoordinator initialization and setup."""
    
    def test_team_coordinator_basic_init(self):
        """Test basic initialization of TeamCoordinator."""
        coordinator = TeamCoordinator()
        
        assert coordinator.provider == REFLECTIVE_LLM_PROVIDER
        assert coordinator.primary_team is None
        assert coordinator.reflection_team is None
        assert coordinator.teams_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_teams_success(self):
        """Test successful team initialization."""
        coordinator = TeamCoordinator()
        
        with patch.object(coordinator, '_get_team_model') as mock_team_model:
            with patch.object(coordinator, '_get_reflection_model') as mock_reflection_model:
                with patch.object(coordinator, '_create_primary_team') as mock_create_primary:
                    with patch.object(coordinator, '_create_reflection_team') as mock_create_reflection:
                        # Mock return values
                        mock_model = Mock(spec=ModelProtocol)
                        mock_team_model.return_value = mock_model
                        mock_reflection_model.return_value = mock_model
                        
                        mock_primary_team = Mock()
                        mock_reflection_team = Mock()
                        mock_create_primary.return_value = mock_primary_team
                        mock_create_reflection.return_value = mock_reflection_team
                        
                        # Initialize teams
                        await coordinator.initialize_teams()
                        
                        # Verify calls
                        mock_team_model.assert_called_once()
                        mock_reflection_model.assert_called_once()
                        mock_create_primary.assert_called_once_with(mock_model)
                        mock_create_reflection.assert_called_once_with(mock_model)
                        
                        # Verify state
                        assert coordinator.primary_team == mock_primary_team
                        assert coordinator.reflection_team == mock_reflection_team
                        assert coordinator.teams_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_teams_already_initialized(self):
        """Test that teams are not re-initialized if already done."""
        coordinator = TeamCoordinator()
        coordinator.teams_initialized = True
        coordinator.primary_team = Mock()
        coordinator.reflection_team = Mock()
        
        original_primary = coordinator.primary_team
        original_reflection = coordinator.reflection_team
        
        with patch.object(coordinator, '_get_team_model') as mock_get_model:
            await coordinator.initialize_teams()
            
            # Should not call any initialization methods
            mock_get_model.assert_not_called()
            
            # Teams should remain unchanged
            assert coordinator.primary_team == original_primary
            assert coordinator.reflection_team == original_reflection
    
    @pytest.mark.asyncio
    async def test_initialize_teams_model_creation_failure(self):
        """Test handling of model creation failures."""
        coordinator = TeamCoordinator()
        
        with patch.object(coordinator, '_get_team_model') as mock_get_model:
            mock_get_model.side_effect = ModelInitializationError(
                provider="test", model_id="test-model", reason="API key missing"
            )
            
            with pytest.raises(ModelInitializationError) as exc_info:
                await coordinator.initialize_teams()
            
            assert "API key missing" in str(exc_info.value)
            assert coordinator.teams_initialized is False


class TestTeamCoordinatorModelManagement:
    """Test model creation and management."""
    
    @pytest.mark.asyncio
    async def test_get_team_model_success(self):
        """Test successful team model retrieval."""
        coordinator = TeamCoordinator()
        
        with patch('handlers.team_coordinator.LLMProviderFactory') as mock_factory:
            # Mock provider config
            mock_config = Mock()
            mock_config.get_models.return_value = ("team-model-id", "agent-model-id")
            mock_config.validate.return_value = None
            
            mock_model = Mock(spec=ModelProtocol)
            mock_config.create_model_instance.return_value = mock_model
            
            mock_factory.get_provider_config.return_value = mock_config
            
            # Get model
            model = await coordinator._get_team_model()
            
            # Verify
            assert model == mock_model
            mock_config.validate.assert_called_once()
            mock_config.get_models.assert_called_once()
            mock_config.create_model_instance.assert_called_once_with("team-model-id")
    
    @pytest.mark.asyncio
    async def test_get_team_model_validation_failure(self):
        """Test handling of config validation failure."""
        coordinator = TeamCoordinator()
        
        with patch('handlers.team_coordinator.LLMProviderFactory') as mock_factory:
            mock_config = Mock()
            mock_config.validate.side_effect = ConfigurationError("provider", "Invalid config")
            mock_factory.get_provider_config.return_value = mock_config
            
            with pytest.raises(ModelInitializationError) as exc_info:
                await coordinator._get_team_model()
            
            assert "Invalid config" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_reflection_model_different_from_team(self):
        """Test that reflection model can be different from team model."""
        coordinator = TeamCoordinator()
        
        with patch('handlers.team_coordinator.LLMProviderFactory') as mock_factory:
            mock_config = Mock()
            mock_config.get_models.return_value = ("team-model-id", "reflection-model-id")
            mock_config.validate.return_value = None
            
            # Different models for team and reflection
            team_model = Mock(spec=ModelProtocol)
            reflection_model = Mock(spec=ModelProtocol)
            
            def create_model(model_id):
                if model_id == "team-model-id":
                    return team_model
                elif model_id == "reflection-model-id":
                    return reflection_model
                else:
                    raise ValueError(f"Unknown model: {model_id}")
            
            mock_config.create_model_instance.side_effect = create_model
            mock_factory.get_provider_config.return_value = mock_config
            
            # Get models
            team_result = await coordinator._get_team_model()
            reflection_result = await coordinator._get_reflection_model()
            
            assert team_result == team_model
            assert reflection_result == reflection_model
            assert team_result != reflection_result


class TestTeamCoordinatorTeamCreation:
    """Test team creation and agent configuration."""
    
    def test_create_primary_agents_respects_max_limit(self):
        """Test that agent creation respects DEFAULT_MAX_AGENTS_PER_TEAM."""
        coordinator = TeamCoordinator()
        mock_model = Mock(spec=ModelProtocol)
        
        with patch('handlers.team_coordinator.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            agents = coordinator._create_primary_agents(mock_model)
            
            # Should not exceed max agents
            assert len(agents) <= DEFAULT_MAX_AGENTS_PER_TEAM
            
            # Each agent should have required attributes
            for call in mock_agent_class.call_args_list:
                kwargs = call[1]
                assert "name" in kwargs
                assert "role" in kwargs
                assert "goal" in kwargs
                assert "instructions" in kwargs
                assert kwargs["model"] == mock_model
    
    def test_create_reflection_agents_configuration(self):
        """Test reflection team agent configuration."""
        coordinator = TeamCoordinator()
        mock_model = Mock(spec=ModelProtocol)
        
        with patch('handlers.team_coordinator.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            agents = coordinator._create_reflection_agents(mock_model)
            
            # Should create exactly 4 reflection agents
            assert len(agents) == 4
            
            # Verify agent names
            agent_names = [call[1]["name"] for call in mock_agent_class.call_args_list]
            expected_names = ["Meta-Analyzer", "Pattern Recognizer", "Quality Assessor", "Decision Critic"]
            assert all(name in agent_names for name in expected_names)
    
    @pytest.mark.asyncio
    async def test_create_primary_team_full_flow(self):
        """Test complete primary team creation flow."""
        coordinator = TeamCoordinator()
        mock_model = Mock(spec=ModelProtocol)
        
        with patch('handlers.team_coordinator.Agent') as mock_agent_class:
            with patch('handlers.team_coordinator.AsyncTeam') as mock_team_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                
                mock_team = Mock()
                mock_team.name = "PrimaryThinkingTeam"
                mock_team_class.return_value = mock_team
                
                team = await coordinator._create_primary_team(mock_model)
                
                # Verify team creation
                assert team == mock_team
                mock_team_class.assert_called_once()
                
                # Verify team configuration
                team_kwargs = mock_team_class.call_args[1]
                assert team_kwargs["name"] == "PrimaryThinkingTeam"
                # Should create 5 agents (the number of configs available)
                assert len(team_kwargs["members"]) == 5
                assert team_kwargs["model"] == mock_model
                assert isinstance(team_kwargs["instructions"], list)
    
    @pytest.mark.asyncio
    async def test_create_reflection_team_full_flow(self):
        """Test complete reflection team creation flow."""
        coordinator = TeamCoordinator()
        mock_model = Mock(spec=ModelProtocol)
        
        with patch('handlers.team_coordinator.Agent') as mock_agent_class:
            with patch('handlers.team_coordinator.AsyncTeam') as mock_team_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                
                mock_team = Mock()
                mock_team.name = "ReflectionTeam"
                mock_team_class.return_value = mock_team
                
                team = await coordinator._create_reflection_team(mock_model)
                
                # Verify team creation
                assert team == mock_team
                mock_team_class.assert_called_once()
                
                # Verify team configuration
                team_kwargs = mock_team_class.call_args[1]
                assert team_kwargs["name"] == "ReflectionTeam"
                assert len(team_kwargs["members"]) == 4  # Reflection team has 4 members
                assert team_kwargs["model"] == mock_model


class TestTeamCoordinatorInstructionGeneration:
    """Test adaptive instruction generation."""
    
    def test_generate_primary_instructions_basic(self):
        """Test basic primary team instruction generation."""
        coordinator = TeamCoordinator()
        instructions = coordinator._generate_primary_instructions()
        
        assert isinstance(instructions, list)
        assert len(instructions) > 0
        
        # Check for key instruction elements
        instruction_text = " ".join(instructions)
        assert "collaboratively" in instruction_text
        assert "perspectives" in instruction_text
        assert "actionable" in instruction_text
    
    def test_generate_reflection_instructions_basic(self):
        """Test basic reflection team instruction generation."""
        coordinator = TeamCoordinator()
        instructions = coordinator._generate_reflection_instructions()
        
        assert isinstance(instructions, list)
        assert len(instructions) > 0
        
        # Check for key reflection elements
        instruction_text = " ".join(instructions)
        assert "meta-analysis" in instruction_text
        assert "strengths" in instruction_text
        assert "weaknesses" in instruction_text
    
    @pytest.mark.asyncio
    async def test_generate_adaptive_instructions_with_context(self):
        """Test adaptive instruction generation based on context."""
        coordinator = TeamCoordinator()
        
        # Test with technical domain
        tech_context = {
            "domain": "technical",
            "complexity": 0.8,
            "is_revision": False
        }
        
        tech_instructions = await coordinator.generate_adaptive_instructions(tech_context)
        tech_text = " ".join(tech_instructions)
        assert "technical accuracy" in tech_text
        assert "complex problem" in tech_text  # Due to high complexity
        
        # Test with creative domain
        creative_context = {
            "domain": "creative",
            "complexity": 0.3,
            "is_revision": True
        }
        
        creative_instructions = await coordinator.generate_adaptive_instructions(creative_context)
        creative_text = " ".join(creative_instructions)
        assert "innovative" in creative_text
        assert "previous limitations" in creative_text  # Due to revision
    
    @pytest.mark.asyncio
    async def test_adaptive_instructions_edge_cases(self):
        """Test adaptive instructions with edge case contexts."""
        coordinator = TeamCoordinator()
        
        # Empty context
        empty_instructions = await coordinator.generate_adaptive_instructions({})
        assert len(empty_instructions) > 0  # Should still generate base instructions
        
        # Unknown domain
        unknown_context = {"domain": "unknown_domain"}
        unknown_instructions = await coordinator.generate_adaptive_instructions(unknown_context)
        assert len(unknown_instructions) > 0  # Should handle gracefully
        
        # Very high complexity
        high_complex_context = {"complexity": 1.0}
        complex_instructions = await coordinator.generate_adaptive_instructions(high_complex_context)
        assert "complex problem" in " ".join(complex_instructions)


class TestTeamCoordinatorErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_team_initialization_partial_failure(self):
        """Test handling when only one team fails to initialize."""
        coordinator = TeamCoordinator()
        
        with patch.object(coordinator, '_get_team_model') as mock_get_model:
            with patch.object(coordinator, '_create_primary_team') as mock_create_primary:
                with patch.object(coordinator, '_create_reflection_team') as mock_create_reflection:
                    mock_model = Mock(spec=ModelProtocol)
                    mock_get_model.return_value = mock_model
                    
                    # Primary succeeds, reflection fails
                    mock_primary = Mock()
                    mock_create_primary.return_value = mock_primary
                    mock_create_reflection.side_effect = Exception("Reflection team error")
                    
                    with pytest.raises(ModelInitializationError):
                        await coordinator.initialize_teams()
                    
                    # Should not be marked as initialized
                    assert coordinator.teams_initialized is False
                    # Primary team might be set but not usable
                    assert coordinator.reflection_team is None
    
    @pytest.mark.asyncio
    async def test_model_creation_with_missing_env_vars(self):
        """Test handling of missing environment variables."""
        coordinator = TeamCoordinator()
        
        with patch('handlers.team_coordinator.LLMProviderFactory') as mock_factory:
            mock_factory.get_provider_config.side_effect = ConfigurationError(
                "API_KEY", "Missing API key for provider"
            )
            
            with pytest.raises(ModelInitializationError) as exc_info:
                await coordinator._get_team_model()
            
            assert "Missing API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_without_reflection_team(self):
        """Test that system can work with only primary team if needed."""
        coordinator = TeamCoordinator()
        
        with patch.object(coordinator, '_get_team_model') as mock_get_model:
            with patch.object(coordinator, '_create_primary_team') as mock_create_primary:
                with patch.object(coordinator, '_create_reflection_team') as mock_create_reflection:
                    mock_model = Mock(spec=ModelProtocol)
                    mock_get_model.return_value = mock_model
                    
                    # Primary succeeds
                    mock_primary = Mock()
                    mock_create_primary.return_value = mock_primary
                    
                    # Reflection fails but we catch it
                    mock_create_reflection.side_effect = Exception("Reflection error")
                    
                    # Modify initialize_teams to handle partial success
                    try:
                        await coordinator.initialize_teams()
                    except ModelInitializationError:
                        # In real implementation, we might want to allow partial success
                        pass
                    
                    # Primary team should still be available
                    assert coordinator.primary_team == mock_primary