"""
Enhanced error handler with comprehensive error management and recovery strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from exceptions import ErrorType
from .circuit_breaker import CircuitBreaker
from .error_context import ErrorContext, ErrorSeverity

logger = logging.getLogger(__name__)


class EnhancedErrorHandler:
    """Comprehensive error handling with recovery strategies."""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "team_processing": CircuitBreaker(),
            "model_communication": CircuitBreaker(),
        }

    def handle_error(
        self,
        error: Exception,
        error_type: ErrorType,
        thought_number: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Handle errors with appropriate recovery strategies.

        Returns:
            Recovery message or None if unrecoverable
        """
        severity = self._assess_severity(error, error_type)

        error_context = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            timestamp=datetime.now(),
            thought_number=thought_number,
            additional_info=context or {},
        )

        self.error_history.append(error_context)

        # Log error with appropriate level
        if severity == ErrorSeverity.CRITICAL:
            logger.error(f"Critical error: {error_context}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {error_context}")
        else:
            logger.warning(f"Error occurred: {error_context}")

        # Apply recovery strategy
        recovery_message = self._apply_recovery_strategy(error_context)

        if recovery_message:
            error_context.recovery_attempted = True

        return recovery_message

    def _assess_severity(
        self, error: Exception, error_type: ErrorType
    ) -> ErrorSeverity:
        """Assess error severity based on type and content."""
        if isinstance(error, ValidationError) or error_type in (
            ErrorType.VALIDATION,
            ErrorType.VALIDATION_ERROR,
        ):
            return ErrorSeverity.LOW
        elif error_type == ErrorType.TEAM_INITIALIZATION:
            return ErrorSeverity.CRITICAL
        elif error_type == ErrorType.MODEL_COMMUNICATION:
            return ErrorSeverity.HIGH
        elif "token" in str(error).lower() or "api" in str(error).lower():
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM

    def _apply_recovery_strategy(self, error_context: ErrorContext) -> Optional[str]:
        """Apply appropriate recovery strategy based on error type."""
        if error_context.error_type in (
            ErrorType.VALIDATION,
            ErrorType.VALIDATION_ERROR,
        ):
            return "Input validation failed. Please check the format and try again."

        elif error_context.error_type == ErrorType.TEAM_PROCESSING:
            breaker = self.circuit_breakers.get("team_processing")
            if breaker and not breaker.can_proceed():
                return (
                    "Team processing temporarily unavailable. Please try again later."
                )
            return "Team processing error. Attempting with reduced complexity."

        elif error_context.error_type == ErrorType.MODEL_COMMUNICATION:
            breaker = self.circuit_breakers.get("model_communication")
            if breaker and not breaker.can_proceed():
                return "Model communication temporarily unavailable. Please try again later."
            return "Communication error with AI model. Retrying with fallback settings."

        elif error_context.error_type == ErrorType.TOOL_EXECUTION:
            return "Tool execution failed. Please check parameters and try again."

        return None

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history."""
        if not self.error_history:
            return {"total_errors": 0}

        summary = {
            "total_errors": len(self.error_history),
            "by_type": {},
            "by_severity": {},
            "recent_errors": [],
        }

        for error in self.error_history:
            # Count by type
            error_type_str = error.error_type.value
            summary["by_type"][error_type_str] = (
                summary["by_type"].get(error_type_str, 0) + 1
            )

            # Count by severity
            severity_str = error.severity.value
            summary["by_severity"][severity_str] = (
                summary["by_severity"].get(severity_str, 0) + 1
            )

        # Recent errors
        summary["recent_errors"] = [
            {
                "type": err.error_type.value,
                "message": err.message,
                "timestamp": err.timestamp.isoformat(),
            }
            for err in self.error_history[-5:]
        ]

        return summary
