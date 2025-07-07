"""
Base provider interface and factory for LLM providers.
"""

from typing import Dict, Optional, Tuple, Type
from agno.models.base import Model
from agno.models.openrouter import OpenRouter
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.groq import Groq
import os
import logging

logger = logging.getLogger(__name__)


class ProviderConfig:
    """Configuration for an LLM provider."""

    def __init__(
        self,
        provider_name: str,
        api_key_env: str,
        team_model_env: str,
        agent_model_env: str,
        default_team_model: str,
        default_agent_model: str,
        model_class: Type[Model],
    ):
        self.provider_name = provider_name
        self.api_key_env = api_key_env
        self.team_model_env = team_model_env
        self.agent_model_env = agent_model_env
        self.default_team_model = default_team_model
        self.default_agent_model = default_agent_model
        self.model_class = model_class

    def validate(self) -> None:
        """Validates that required environment variables are set."""
        if self.api_key_env not in os.environ:
            raise ValueError(
                f"Missing required API key environment variable: {self.api_key_env} "
                f"for provider {self.provider_name}"
            )

    def get_models(self) -> Tuple[str, str]:
        """Returns the team and agent model IDs."""
        team_model = os.environ.get(self.team_model_env, self.default_team_model)
        agent_model = os.environ.get(self.agent_model_env, self.default_agent_model)
        return team_model, agent_model

    def create_model_instance(self, model_id: str, **kwargs) -> Model:
        """Creates a model instance with the given ID."""
        # Get API key from environment
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {self.api_key_env}")

        # Agno models accept api_key parameter
        return self.model_class(id=model_id, api_key=api_key, **kwargs)  # type: ignore[call-arg]


class LLMProviderFactory:
    """Factory for creating LLM provider configurations."""

    # Provider configurations
    PROVIDERS = {
        "openrouter": ProviderConfig(
            provider_name="OpenRouter",
            api_key_env="OPENROUTER_API_KEY",
            team_model_env="OPENROUTER_TEAM_MODEL_ID",
            agent_model_env="OPENROUTER_AGENT_MODEL_ID",
            default_team_model="openai/o3",
            default_agent_model="x-ai/grok-3-mini",
            model_class=OpenRouter,
        ),
        "openai": ProviderConfig(
            provider_name="OpenAI",
            api_key_env="OPENAI_API_KEY",
            team_model_env="OPENAI_TEAM_MODEL_ID",
            agent_model_env="OPENAI_AGENT_MODEL_ID",
            default_team_model="o3",
            default_agent_model="o4-mini",
            model_class=OpenAIChat,
        ),
        "gemini": ProviderConfig(
            provider_name="Google Gemini",
            api_key_env="GOOGLE_API_KEY",
            team_model_env="GEMINI_TEAM_MODEL_ID",
            agent_model_env="GEMINI_AGENT_MODEL_ID",
            default_team_model="gemini-2.5-pro",
            default_agent_model="gemini-2.5-flash",
            model_class=Gemini,
        ),
        "groq": ProviderConfig(
            provider_name="Groq",
            api_key_env="GROQ_API_KEY",
            team_model_env="GROQ_TEAM_MODEL_ID",
            agent_model_env="GROQ_AGENT_MODEL_ID",
            default_team_model="llama-3.3-70b-versatile",
            default_agent_model="qwen-2.5-32b",
            model_class=Groq,
        ),
    }

    @classmethod
    def get_provider_config(cls, provider_name: Optional[str] = None) -> ProviderConfig:
        """
        Gets the provider configuration for the specified provider.

        Args:
            provider_name: Name of the provider. If None, uses REFLECTIVE_LLM_PROVIDER env var.
                         Defaults to 'openai' if not set.

        Returns:
            ProviderConfig instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name is None:
            provider_name = os.environ.get("REFLECTIVE_LLM_PROVIDER", "openai").lower()
        else:
            provider_name = provider_name.lower()

        if provider_name not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        config = cls.PROVIDERS[provider_name]
        logger.info(f"Selected LLM Provider: {config.provider_name}")

        return config

    @classmethod
    def create_models(
        cls, provider_name: Optional[str] = None
    ) -> Tuple[Model, Model, ProviderConfig]:
        """
        Creates team and agent model instances.

        Returns:
            Tuple of (team_model, agent_model, provider_config)
        """
        config = cls.get_provider_config(provider_name)

        # Validate configuration
        config.validate()

        # Get model IDs
        team_model_id, agent_model_id = config.get_models()
        logger.info(
            f"Using {config.provider_name}: "
            f"Team Model='{team_model_id}', Agent Model='{agent_model_id}'"
        )

        # Create model instances
        try:
            team_model = config.create_model_instance(team_model_id)
            agent_model = config.create_model_instance(agent_model_id)
        except Exception as e:
            logger.error(f"Failed to create model instances: {e}")
            raise ValueError(
                f"Failed to initialize {config.provider_name} models: {e}. "
                f"Please check your API key and model IDs."
            )

        return team_model, agent_model, config

    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, str]]:
        """Lists all available providers and their configurations."""
        return {
            name: {
                "provider_name": config.provider_name,
                "api_key_env": config.api_key_env,
                "default_team_model": config.default_team_model,
                "default_agent_model": config.default_agent_model,
            }
            for name, config in cls.PROVIDERS.items()
        }
