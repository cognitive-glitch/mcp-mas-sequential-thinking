"""
Custom exception classes for the Reflective Sequential Thinking MCP server.
"""

from typing import Any, Dict, Optional
from enum import Enum


class ErrorType(str, Enum):
    """Types of errors that can occur."""

    VALIDATION = "validation"
    VALIDATION_ERROR = "validation_error"  # Legacy alias for validation
    MODEL_INITIALIZATION = "model_initialization"
    TEAM_INITIALIZATION = "team_initialization"  # Team setup errors
    TEAM_PROCESSING = "team_processing"
    MODEL_COMMUNICATION = "model_communication"  # LLM communication errors
    CONTEXT_MANAGEMENT = "context_management"
    CONTEXT_ERROR = "context_error"  # Legacy alias for context_management
    PROVIDER_ERROR = "provider_error"
    CONFIGURATION = "configuration"
    TOOL_EXECUTION = "tool_execution"
    TIMEOUT_ERROR = "timeout_error"  # Timeout-related errors


class ReflectiveThinkingError(Exception):
    """Base exception for all reflective thinking errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.VALIDATION,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class ValidationError(ReflectiveThinkingError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"Validation failed for {field}: {reason}",
            ErrorType.VALIDATION,
            {"field": field, "value": value, "reason": reason},
        )


class ModelInitializationError(ReflectiveThinkingError):
    """Raised when model initialization fails."""

    def __init__(self, provider: str, model_id: str, reason: str):
        self.provider = provider
        self.model_id = model_id
        super().__init__(
            f"Failed to initialize {provider} model '{model_id}': {reason}",
            ErrorType.MODEL_INITIALIZATION,
            {"provider": provider, "model_id": model_id},
        )


class TeamProcessingError(ReflectiveThinkingError):
    """Raised when team processing fails."""

    def __init__(self, team_name: str, reason: str, stage: Optional[str] = None):
        self.team_name = team_name
        self.stage = stage
        super().__init__(
            f"Team '{team_name}' processing failed{f' at {stage}' if stage else ''}: {reason}",
            ErrorType.TEAM_PROCESSING,
            {"team_name": team_name, "stage": stage},
        )


class ContextManagementError(ReflectiveThinkingError):
    """Raised when context management operations fail."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(
            f"Context operation '{operation}' failed: {reason}",
            ErrorType.CONTEXT_MANAGEMENT,
            {"operation": operation},
        )


class ProviderError(ReflectiveThinkingError):
    """Raised when LLM provider operations fail."""

    def __init__(self, provider: str, operation: str, reason: str):
        self.provider = provider
        self.operation = operation
        super().__init__(
            f"{provider} provider {operation} failed: {reason}",
            ErrorType.PROVIDER_ERROR,
            {"provider": provider, "operation": operation},
        )


class ConfigurationError(ReflectiveThinkingError):
    """Raised when configuration is invalid."""

    def __init__(self, setting: str, reason: str):
        self.setting = setting
        super().__init__(
            f"Invalid configuration for '{setting}': {reason}",
            ErrorType.CONFIGURATION,
            {"setting": setting},
        )


class ToolExecutionError(ReflectiveThinkingError):
    """Raised when MCP tool execution fails."""

    def __init__(
        self, tool_name: str, reason: str, input_data: Optional[Dict[str, Any]] = None
    ):
        self.tool_name = tool_name
        self.input_data = input_data
        super().__init__(
            f"Tool '{tool_name}' execution failed: {reason}",
            ErrorType.TOOL_EXECUTION,
            {"tool_name": tool_name, "input_data": input_data},
        )


class CircuitBreakerOpen(ReflectiveThinkingError):
    """Raised when circuit breaker is open due to too many failures."""

    def __init__(self, service: str, failure_count: int, timeout_remaining: float):
        self.service = service
        self.failure_count = failure_count
        self.timeout_remaining = timeout_remaining
        super().__init__(
            f"Circuit breaker for '{service}' is open (failures: {failure_count}, timeout: {timeout_remaining:.1f}s)",
            ErrorType.TEAM_PROCESSING,
            {
                "service": service,
                "failure_count": failure_count,
                "timeout_remaining": timeout_remaining,
            },
        )
