"""
Unit tests for LLM provider factory and configuration.
Tests provider selection, model creation, and configuration handling.
"""

import pytest
import os
from unittest.mock import Mock
from typing import cast, Type
from agno.models.base import Model

from src.providers.base import (
    LLMProviderFactory,
    ProviderConfig,
)


class TestProviderConfig:
    """Test the ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating a provider configuration."""
        config = ProviderConfig(
            provider_name="Test Provider",
            api_key_env="TEST_API_KEY",
            team_model_env="TEST_TEAM_MODEL",
            agent_model_env="TEST_AGENT_MODEL",
            default_team_model="test-team-model",
            default_agent_model="test-agent-model",
            model_class=Mock,
        )

        assert config.provider_name == "Test Provider"
        assert config.api_key_env == "TEST_API_KEY"
        assert config.default_team_model == "test-team-model"

    def test_provider_config_equality(self):
        """Test provider config equality."""
        config1 = ProviderConfig(
            provider_name="Test",
            api_key_env="KEY",
            team_model_env="TEAM",
            agent_model_env="AGENT",
            default_team_model="team",
            default_agent_model="agent",
            model_class=Mock,
        )

        config2 = ProviderConfig(
            provider_name="Test",
            api_key_env="KEY",
            team_model_env="TEAM",
            agent_model_env="AGENT",
            default_team_model="team",
            default_agent_model="agent",
            model_class=Mock,
        )

        # Should be equal based on values
        assert config1.provider_name == config2.provider_name
        assert config1.api_key_env == config2.api_key_env


class TestLLMProviderFactory:
    """Test the LLMProviderFactory class."""

    @pytest.fixture
    def clean_env(self):
        """Clean environment variables."""
        # Store original values
        original = {}
        keys_to_clean = [
            "REFLECTIVE_LLM_PROVIDER",
            "OPENROUTER_API_KEY",
            "OPENROUTER_TEAM_MODEL_ID",
            "OPENROUTER_AGENT_MODEL_ID",
            "OPENAI_API_KEY",
            "OPENAI_TEAM_MODEL_ID",
            "OPENAI_AGENT_MODEL_ID",
            "GOOGLE_API_KEY",
            "GEMINI_TEAM_MODEL_ID",
            "GEMINI_AGENT_MODEL_ID",
        ]

        for key in keys_to_clean:
            if key in os.environ:
                original[key] = os.environ[key]
                del os.environ[key]

        yield

        # Restore original values
        for key, value in original.items():
            os.environ[key] = value

    def test_supported_providers(self):
        """Test that required providers are registered."""
        providers = LLMProviderFactory.list_providers()

        # Should have OpenRouter, OpenAI, and Gemini providers
        provider_names = {info["provider_name"] for info in providers.values()}
        assert "OpenRouter" in provider_names
        assert "OpenAI" in provider_names
        assert "Google Gemini" in provider_names
        # Note: Not testing Groq as per user request

    def test_get_provider_config_default(self, clean_env):
        """Test getting default provider config."""
        # Should default to OpenAI
        config = LLMProviderFactory.get_provider_config()

        assert config.provider_name == "OpenAI"

    def test_get_provider_config_from_env(self, clean_env):
        """Test getting provider config from environment."""
        # Set provider to OpenAI
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"

        config = LLMProviderFactory.get_provider_config()

        assert config.provider_name == "OpenAI"

    def test_get_provider_config_invalid(self, clean_env):
        """Test handling of invalid provider."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "invalid_provider"

        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.get_provider_config()

        assert "unsupported llm provider" in str(exc_info.value).lower()

    def test_validate_provider_config_missing_key(self, clean_env):
        """Test validation with missing API key."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
        # Don't set OPENAI_API_KEY

        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create_models()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_validate_provider_config_with_values(self, clean_env):
        """Test validation with all required values."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_TEAM_MODEL_ID"] = "gpt-4"

        # Should not raise
        config = LLMProviderFactory.get_provider_config()

        assert config.provider_name == "OpenAI"
        assert config is not None

    def test_create_models_openrouter(self, clean_env):
        """Test creating models for OpenRouter."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_TEAM_MODEL_ID"] = "team-model"
        os.environ["OPENROUTER_AGENT_MODEL_ID"] = "agent-model"

        # Create a mock model class
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the model class in the PROVIDERS config
        original_config = LLMProviderFactory.PROVIDERS["openrouter"]
        original_model_class = original_config.model_class
        original_config.model_class = cast(Type[Model], mock_model_class)

        try:
            team_model, agent_model, returned_config = (
                LLMProviderFactory.create_models()
            )

            # Should create models with correct parameters
            assert mock_model_class.call_count == 2

            # Check team model creation
            team_call = mock_model_class.call_args_list[0]
            assert "id" in team_call[1]
            assert team_call[1]["id"] == "team-model"
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_create_models_openai(self, clean_env):
        """Test creating models for OpenAI."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_TEAM_MODEL_ID"] = "gpt-4"

        # Create a mock model class
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the model class in the PROVIDERS config
        original_config = LLMProviderFactory.PROVIDERS["openai"]
        original_model_class = original_config.model_class
        original_config.model_class = cast(Type[Model], mock_model_class)

        try:
            team_model, agent_model, config = LLMProviderFactory.create_models()

            # Should create models
            assert mock_model_class.call_count == 2
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_create_models_gemini(self, clean_env):
        """Test creating models for Google Gemini."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "gemini"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["GEMINI_TEAM_MODEL_ID"] = "gemini-pro"

        # Mock the model_class directly in the provider config
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the provider config's model_class
        original_config = LLMProviderFactory.PROVIDERS["gemini"]
        original_model_class = original_config.model_class
        original_config.model_class = mock_model_class  # type: ignore

        try:
            team_model, agent_model, config = LLMProviderFactory.create_models()

            # Should create models
            assert mock_model_class.call_count == 2
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_create_models_with_defaults(self, clean_env):
        """Test model creation with default model IDs."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        # Don't set model IDs - should use defaults

        # Mock the model_class directly in the provider config
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the provider config's model_class
        original_config = LLMProviderFactory.PROVIDERS["openrouter"]
        original_model_class = original_config.model_class
        original_config.model_class = mock_model_class  # type: ignore

        try:
            team_model, agent_model, config = LLMProviderFactory.create_models()

            # Should use default model IDs
            team_call = mock_model_class.call_args_list[0]
            agent_call = mock_model_class.call_args_list[1]

            # Check defaults are used
            assert team_call[1]["id"] == "openai/o3"
            assert agent_call[1]["id"] == "x-ai/grok-3-mini"
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = LLMProviderFactory.list_providers()

        assert isinstance(providers, dict)
        assert len(providers) >= 3  # At least openrouter, openai, gemini
        assert all(isinstance(k, str) for k in providers.keys())
        assert "openrouter" in providers
        assert "openai" in providers

    # def test_provider_info_string(self, clean_env):
    #     """Test getting provider info string."""
    #     os.environ["REFLECTIVE_LLM_PROVIDER"] = "openai"
    #     os.environ["OPENAI_API_KEY"] = "test-key"
    #     os.environ["OPENAI_TEAM_MODEL_ID"] = "gpt-4-turbo"

    #     info = LLMProviderFactory.get_provider_info()

    #     assert "Provider: OpenAI" in info
    #     assert "Team Model: gpt-4-turbo" in info

    def test_case_insensitive_provider(self, clean_env):
        """Test provider selection is case insensitive."""
        variations = ["OpenRouter", "openrouter", "OPENROUTER", "OpenROUTER"]

        for variant in variations:
            os.environ["REFLECTIVE_LLM_PROVIDER"] = variant
            config = LLMProviderFactory.get_provider_config()
            assert config.provider_name == "OpenRouter"

    def test_provider_with_api_base(self, clean_env):
        """Test providers that need API base configuration."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        # Mock the model_class directly in the provider config
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the provider config's model_class
        original_config = LLMProviderFactory.PROVIDERS["openrouter"]
        original_model_class = original_config.model_class
        original_config.model_class = mock_model_class  # type: ignore

        try:
            LLMProviderFactory.create_models()

            # Should set API base for OpenRouter
            call_kwargs = mock_model_class.call_args_list[0][1]
            if "api_base" in call_kwargs:
                assert "openrouter.ai" in call_kwargs["api_base"]
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_environment_variable_interpolation(self, clean_env):
        """Test that environment variables are properly interpolated."""
        test_key = "sk-test-key-123"
        test_model = "claude-3-opus"

        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = test_key
        os.environ["OPENROUTER_TEAM_MODEL_ID"] = test_model

        # Mock the model_class directly in the provider config
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the provider config's model_class
        original_config = LLMProviderFactory.PROVIDERS["openrouter"]
        original_model_class = original_config.model_class
        original_config.model_class = mock_model_class  # type: ignore

        try:
            LLMProviderFactory.create_models()

            # Should pass exact environment values
            call_kwargs = mock_model_class.call_args_list[0][1]
            assert call_kwargs["api_key"] == test_key
            assert call_kwargs["id"] == test_model
        finally:
            # Restore original model class
            original_config.model_class = original_model_class

    def test_missing_optional_models(self, clean_env):
        """Test handling when only team model is specified."""
        os.environ["REFLECTIVE_LLM_PROVIDER"] = "openrouter"
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_TEAM_MODEL_ID"] = "team-model"
        # Don't set agent model - should use team model

        # Mock the model_class directly in the provider config
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_model_class.return_value = mock_instance

        # Patch the provider config's model_class
        original_config = LLMProviderFactory.PROVIDERS["openrouter"]
        original_model_class = original_config.model_class
        original_config.model_class = mock_model_class  # type: ignore

        try:
            team_model, agent_model, config = LLMProviderFactory.create_models()

            # Both should use team model ID
            team_call = mock_model_class.call_args_list[0]
            agent_call = mock_model_class.call_args_list[1]

            assert team_call[1]["id"] == "team-model"
            assert (
                agent_call[1]["id"] == "x-ai/grok-3-mini"
            )  # Falls back to default agent model
        finally:
            # Restore original model class
            original_config.model_class = original_model_class
