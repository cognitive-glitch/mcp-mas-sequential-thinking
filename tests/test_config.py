"""
Tests for configuration module.
"""

import pytest
import os
from unittest.mock import patch

import src.config as config


class TestConfigConstants:
    """Test configuration constants."""

    def test_team_configuration_constants(self):
        """Test team configuration constants."""
        assert config.DEFAULT_MAX_AGENTS_PER_TEAM == 8
        assert config.DEFAULT_AGENT_TIMEOUT == 30.0
        assert config.AGENT_CONCURRENCY_LIMIT == 3

    def test_model_configuration_constants(self):
        """Test model configuration constants."""
        assert config.DEFAULT_TEAM_MODEL == "gpt-4o-mini"
        assert config.DEFAULT_REFLECTION_MODEL == "gpt-4o-mini"

    def test_thought_validation_constants(self):
        """Test thought validation constants."""
        assert config.MIN_THOUGHT_LENGTH == 10
        assert config.MIN_TOTAL_THOUGHTS == 5
        assert config.MAX_KEYWORD_LENGTH == 20

    def test_context_management_constants(self):
        """Test context management constants."""
        assert config.MAX_CONTEXT_MEMORY_ITEMS == 100
        assert config.RELEVANT_CONTEXT_LIMIT == 5
        assert config.GRAPH_EDGE_SIMILARITY_THRESHOLD == 0.7

    def test_performance_settings_constants(self):
        """Test performance settings constants."""
        assert config.DEFAULT_PROCESSING_TIMEOUT == 120.0
        assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5
        assert config.CIRCUIT_BREAKER_TIMEOUT == 60.0

    def test_logging_constants(self):
        """Test logging constants."""
        assert (
            config.LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def test_error_messages_structure(self):
        """Test error messages dictionary structure."""
        expected_keys = {
            "invalid_thought_length",
            "invalid_total_thoughts",
            "ending_too_early",
            "consecutive_branching",
            "team_not_initialized",
            "model_error",
            "circuit_breaker_open",
        }

        assert set(config.ERROR_MESSAGES.keys()) == expected_keys

        # Verify error messages contain expected content
        assert (
            str(config.MIN_THOUGHT_LENGTH)
            in config.ERROR_MESSAGES["invalid_thought_length"]
        )
        assert (
            str(config.MIN_TOTAL_THOUGHTS)
            in config.ERROR_MESSAGES["invalid_total_thoughts"]
        )

    def test_team_instruction_template(self):
        """Test team instruction template."""
        template = config.TEAM_INSTRUCTION_TEMPLATE

        assert "{team_type}" in template
        assert "{role_description}" in template
        assert "{responsibilities}" in template
        assert "systematic" in template
        assert "perspectives" in template


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_log_level_default(self):
        """Test LOG_LEVEL default value."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload config to get default
            import importlib

            importlib.reload(config)
            assert config.LOG_LEVEL == "INFO"

    def test_log_level_from_env(self):
        """Test LOG_LEVEL from environment variable."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import importlib

            importlib.reload(config)
            assert config.LOG_LEVEL == "DEBUG"

    def test_reflective_llm_provider_default(self):
        """Test REFLECTIVE_LLM_PROVIDER default value."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            importlib.reload(config)
            assert config.REFLECTIVE_LLM_PROVIDER == "openrouter"

    def test_reflective_llm_provider_from_env(self):
        """Test REFLECTIVE_LLM_PROVIDER from environment variable."""
        with patch.dict(os.environ, {"REFLECTIVE_LLM_PROVIDER": "openai"}):
            import importlib

            importlib.reload(config)
            assert config.REFLECTIVE_LLM_PROVIDER == "openai"

    def test_enable_reflection_default_true(self):
        """Test ENABLE_REFLECTION default value (true)."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_REFLECTION is True

    def test_enable_reflection_false(self):
        """Test ENABLE_REFLECTION set to false."""
        with patch.dict(os.environ, {"ENABLE_REFLECTION": "false"}):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_REFLECTION is False

    def test_enable_reflection_case_insensitive(self):
        """Test ENABLE_REFLECTION is case insensitive."""
        with patch.dict(os.environ, {"ENABLE_REFLECTION": "FALSE"}):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_REFLECTION is False

        with patch.dict(os.environ, {"ENABLE_REFLECTION": "True"}):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_REFLECTION is True

    def test_enable_shared_context_default_true(self):
        """Test ENABLE_SHARED_CONTEXT default value (true)."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_SHARED_CONTEXT is True

    def test_enable_shared_context_false(self):
        """Test ENABLE_SHARED_CONTEXT set to false."""
        with patch.dict(os.environ, {"ENABLE_SHARED_CONTEXT": "false"}):
            import importlib

            importlib.reload(config)
            assert config.ENABLE_SHARED_CONTEXT is False

    def test_reflection_delay_ms_default(self):
        """Test REFLECTION_DELAY_MS default value."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            importlib.reload(config)
            assert config.REFLECTION_DELAY_MS == 500

    def test_reflection_delay_ms_from_env(self):
        """Test REFLECTION_DELAY_MS from environment variable."""
        with patch.dict(os.environ, {"REFLECTION_DELAY_MS": "1000"}):
            import importlib

            importlib.reload(config)
            assert config.REFLECTION_DELAY_MS == 1000


class TestGetModelId:
    """Test get_model_id function."""

    def test_get_model_id_with_env_var(self):
        """Test get_model_id when environment variable exists."""
        with patch.dict(os.environ, {"OPENAI_TEAM_MODEL_ID": "gpt-4"}):
            result = config.get_model_id("openai", "team")
            assert result == "gpt-4"

    def test_get_model_id_with_default(self):
        """Test get_model_id falls back to default when env var missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = config.get_model_id("nonexistent", "team")
            assert result == config.DEFAULT_TEAM_MODEL

    def test_get_model_id_reflection_model(self):
        """Test get_model_id for reflection model type."""
        with patch.dict(os.environ, {"OPENAI_REFLECTION_MODEL_ID": "gpt-4-turbo"}):
            result = config.get_model_id("openai", "reflection")
            assert result == "gpt-4-turbo"

    def test_get_model_id_reflection_default(self):
        """Test get_model_id reflection model default."""
        with patch.dict(os.environ, {}, clear=True):
            result = config.get_model_id("nonexistent", "reflection")
            assert result == config.DEFAULT_REFLECTION_MODEL

    def test_get_model_id_case_handling(self):
        """Test get_model_id handles case conversion correctly."""
        with patch.dict(os.environ, {"GEMINI_TEAM_MODEL_ID": "gemini-pro"}):
            result = config.get_model_id("gemini", "team")
            assert result == "gemini-pro"


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_success(self):
        """Test validate_config with valid configuration."""
        # Should not raise any exception
        config.validate_config()

    def test_validate_config_invalid_min_thought_length(self):
        """Test validate_config with invalid MIN_THOUGHT_LENGTH."""
        original_value = config.MIN_THOUGHT_LENGTH
        try:
            config.MIN_THOUGHT_LENGTH = 0
            with pytest.raises(ValueError, match="MIN_THOUGHT_LENGTH must be positive"):
                config.validate_config()
        finally:
            config.MIN_THOUGHT_LENGTH = original_value

    def test_validate_config_invalid_min_total_thoughts(self):
        """Test validate_config with invalid MIN_TOTAL_THOUGHTS."""
        original_value = config.MIN_TOTAL_THOUGHTS
        try:
            config.MIN_TOTAL_THOUGHTS = 0
            with pytest.raises(ValueError, match="MIN_TOTAL_THOUGHTS must be positive"):
                config.validate_config()
        finally:
            config.MIN_TOTAL_THOUGHTS = original_value

    def test_validate_config_invalid_agent_concurrency_limit(self):
        """Test validate_config with invalid AGENT_CONCURRENCY_LIMIT."""
        original_value = config.AGENT_CONCURRENCY_LIMIT
        try:
            config.AGENT_CONCURRENCY_LIMIT = 0
            with pytest.raises(
                ValueError, match="AGENT_CONCURRENCY_LIMIT must be positive"
            ):
                config.validate_config()
        finally:
            config.AGENT_CONCURRENCY_LIMIT = original_value

    def test_validate_config_invalid_max_context_memory_items(self):
        """Test validate_config with invalid MAX_CONTEXT_MEMORY_ITEMS."""
        original_value = config.MAX_CONTEXT_MEMORY_ITEMS
        try:
            config.MAX_CONTEXT_MEMORY_ITEMS = 5
            with pytest.raises(
                ValueError, match="MAX_CONTEXT_MEMORY_ITEMS must be at least 10"
            ):
                config.validate_config()
        finally:
            config.MAX_CONTEXT_MEMORY_ITEMS = original_value

    def test_validate_config_negative_min_thought_length(self):
        """Test validate_config with negative MIN_THOUGHT_LENGTH."""
        original_value = config.MIN_THOUGHT_LENGTH
        try:
            config.MIN_THOUGHT_LENGTH = -5
            with pytest.raises(ValueError, match="MIN_THOUGHT_LENGTH must be positive"):
                config.validate_config()
        finally:
            config.MIN_THOUGHT_LENGTH = original_value

    def test_validate_config_edge_case_values(self):
        """Test validate_config with edge case values."""
        # Test minimum acceptable values
        original_values = {
            "MIN_THOUGHT_LENGTH": config.MIN_THOUGHT_LENGTH,
            "MIN_TOTAL_THOUGHTS": config.MIN_TOTAL_THOUGHTS,
            "AGENT_CONCURRENCY_LIMIT": config.AGENT_CONCURRENCY_LIMIT,
            "MAX_CONTEXT_MEMORY_ITEMS": config.MAX_CONTEXT_MEMORY_ITEMS,
        }

        try:
            # Set to minimum acceptable values
            config.MIN_THOUGHT_LENGTH = 1
            config.MIN_TOTAL_THOUGHTS = 1
            config.AGENT_CONCURRENCY_LIMIT = 1
            config.MAX_CONTEXT_MEMORY_ITEMS = 10

            # Should not raise any exception
            config.validate_config()

        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(config, key, value)


class TestConfigIntegration:
    """Test configuration integration and edge cases."""

    def test_error_message_template_interpolation(self):
        """Test that error messages properly interpolate values."""
        invalid_length_msg = config.ERROR_MESSAGES["invalid_thought_length"]
        assert str(config.MIN_THOUGHT_LENGTH) in invalid_length_msg

        invalid_total_msg = config.ERROR_MESSAGES["invalid_total_thoughts"]
        assert str(config.MIN_TOTAL_THOUGHTS) in invalid_total_msg

    def test_team_instruction_template_format(self):
        """Test team instruction template formatting."""
        template = config.TEAM_INSTRUCTION_TEMPLATE

        # Test that template can be formatted with expected parameters
        formatted = template.format(
            team_type="Primary",
            role_description="analyze complex problems",
            responsibilities="- Task 1\n- Task 2",
        )

        assert "Primary" in formatted
        assert "analyze complex problems" in formatted
        assert "Task 1" in formatted
        assert "Task 2" in formatted

    def test_boolean_environment_variable_parsing(self):
        """Test boolean environment variable parsing edge cases."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
            ("yes", False),  # Only "true" should be True
            ("1", False),  # Only "true" should be True
            ("", False),  # Empty string should be False
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ENABLE_REFLECTION": env_value}):
                import importlib

                importlib.reload(config)
                assert config.ENABLE_REFLECTION is expected, (
                    f"Failed for env_value: '{env_value}'"
                )
